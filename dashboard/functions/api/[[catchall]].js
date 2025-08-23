// Cloudflare Pages Function: API Gateway
export async function onRequest(context) {
    const { request, env } = context;
    const url = new URL(request.url);
    const apiPath = url.pathname.replace('/api/', '');

    const SUPABASE_URL = env.SUPABASE_URL;
    const SUPABASE_KEY = env.SUPABASE_ANON_KEY;
    const ANALYZER_URL = 'https://gandalf-analyzer.psnedujime.workers.dev';

    const corsHeaders = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PATCH, OPTIONS, HEAD',
        'Access-Control-Allow-Headers': 'Content-Type, apikey, Authorization, x-admin-token'
    };

    // CORS preflight
    if (request.method === 'OPTIONS') {
        return new Response(null, { status: 204, headers: corsHeaders });
    }

    try {
        // Protected: run analyzer now
        if (apiPath === 'admin/analyzer/run') {
            const token = request.headers.get('x-admin-token') || url.searchParams.get('token');
            if (!env.ADMIN_TOKEN || token !== env.ADMIN_TOKEN) {
                return new Response(JSON.stringify({ error: 'Unauthorized' }), {
                    status: 401,
                    headers: { ...corsHeaders, 'Content-Type': 'application/json' }
                });
            }
            const targetUrl = `${ANALYZER_URL}/api/run-now${url.search}`;
            const resp = await fetch(targetUrl, { headers: { 'User-Agent': 'Gandalf-API-Gateway/1.0' } });
            return new Response(resp.body, { status: resp.status, headers: { ...corsHeaders, ...Object.fromEntries(resp.headers) } });
        }

        // 1) ML predictions direct from Supabase
        if (apiPath.startsWith('ml-predictions')) {
            const targetUrl = `${SUPABASE_URL}/rest/v1/predictions?order=predicted_at.desc&limit=100${url.search}`;
            const resp = await fetch(targetUrl, {
                headers: {
                    'User-Agent': 'Gandalf-API-Gateway/1.0',
                    'apikey': SUPABASE_KEY,
                    'Authorization': `Bearer ${SUPABASE_KEY}`,
                    'Content-Type': 'application/json'
                }
            });
            return new Response(resp.body, { status: resp.status, headers: { ...corsHeaders, ...Object.fromEntries(resp.headers) } });
        }

        // 2) Signals from DB (persisted)
        if (apiPath.startsWith('signals-db')) {
            const targetUrl = `${SUPABASE_URL}/rest/v1/signals?order=created_at.desc&limit=100${url.search}`;
            const resp = await fetch(targetUrl, {
                headers: {
                    'User-Agent': 'Gandalf-API-Gateway/1.0',
                    'apikey': SUPABASE_KEY,
                    'Authorization': `Bearer ${SUPABASE_KEY}`,
                    'Content-Type': 'application/json'
                }
            });
            return new Response(resp.body, { status: resp.status, headers: { ...corsHeaders, ...Object.fromEntries(resp.headers) } });
        }

        // 3) Analyzer-backed endpoints (KV cache)
        if (apiPath.startsWith('patterns') || apiPath.startsWith('predictions') || apiPath.startsWith('signals')) {
            const targetUrl = `${ANALYZER_URL}/api/${apiPath}`;
            const resp = await fetch(targetUrl, { headers: { 'User-Agent': 'Gandalf-API-Gateway/1.0' } });
            return new Response(resp.body, { status: resp.status, headers: { ...corsHeaders, ...Object.fromEntries(resp.headers) } });
        }

        // 4) Default: forward to Supabase REST
        const targetUrl = `${SUPABASE_URL}/rest/v1/${apiPath}${url.search}`;
        const requestOptions = {
            method: request.method,
            headers: {
                'User-Agent': 'Gandalf-API-Gateway/1.0',
                'apikey': SUPABASE_KEY,
                'Authorization': `Bearer ${SUPABASE_KEY}`,
                'Content-Type': 'application/json',
                'Prefer': 'return-minimal'
            }
        };

        if (request.method === 'POST' || request.method === 'PATCH') {
            requestOptions.body = await request.text();
        }

        const resp = await fetch(targetUrl, requestOptions);
        const headersCombined = { ...corsHeaders, ...Object.fromEntries(resp.headers) };

        if (!resp.ok) {
            const errorText = await resp.text();
            console.error(`Upstream error ${resp.status}: ${errorText}`);
            return new Response(JSON.stringify({ error: `Upstream service error: ${resp.status}` }), {
                status: resp.status,
                headers: { ...headersCombined, 'Content-Type': 'application/json' }
            });
        }

        return new Response(resp.body, { status: resp.status, headers: headersCombined });

    } catch (error) {
        console.error('Gateway fetch error:', error);
        return new Response(JSON.stringify({ error: 'Gateway internal error' }), {
            status: 500,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
    }
}