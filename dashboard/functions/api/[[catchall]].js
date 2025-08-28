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

    if (request.method === 'OPTIONS') {
        return new Response(null, { status: 204, headers: corsHeaders });
    }

    function sanitizeHeaders(h) {
        const drop = new Set(['connection', 'transfer-encoding', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailer', 'upgrade']);
        const out = {};
        for (const [k, v] of h.entries()) {
            if (!drop.has(k.toLowerCase())) out[k] = v;
        }
        return out;
    }

    // Read-only tables exposed via GET
    const READ_ONLY = new Set([
        'market_data',
        'predictions',
        'signals',
        'item_parts',
        'part_relic_drops',
        'worldstate_fissures',
        'worldstate_flags',
        'tracked_items',
        'daily_performance',
        'model_performance',
        'backtest_results',
        'portfolio'
    ]);

    // RPCs allowed from browser
    const RPC_ALLOW = new Set([
        'get_system_metrics',
        'get_enhanced_metrics'
    ]);

    try {
        // Admin route
        if (apiPath === 'admin/analyzer/run') {
            const token = request.headers.get('x-admin-token') || url.searchParams.get('token');
            if (!env.ADMIN_TOKEN || token !== env.ADMIN_TOKEN) {
                return new Response(JSON.stringify({ error: 'Unauthorized' }), {
                    status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
                });
            }
            const targetUrl = `${ANALYZER_URL}/api/run-now${url.search}`;
            const resp = await fetch(targetUrl, { headers: { 'User-Agent': 'Gandalf-API-Gateway/1.0' } });
            return new Response(resp.body, { status: resp.status, headers: { ...corsHeaders, ...sanitizeHeaders(resp.headers) } });
        }

        // Analyzer namespace (explicit)
        if (apiPath.startsWith('analyzer/')) {
            const sub = apiPath.substring('analyzer/'.length);
            const targetUrl = `${ANALYZER_URL}/api/${sub}${url.search ? '?' + url.searchParams.toString() : ''}`;
            const resp = await fetch(targetUrl, { headers: { 'User-Agent': 'Gandalf-API-Gateway/1.0' } });
            return new Response(resp.body, { status: resp.status, headers: { ...corsHeaders, ...sanitizeHeaders(resp.headers) } });
        }

        // Analyzer-backed short endpoints (keep signals and patterns only)
        if (apiPath.startsWith('signals') || apiPath.startsWith('patterns')) {
            const targetUrl = `${ANALYZER_URL}/api/${apiPath}`;
            const resp = await fetch(targetUrl, { headers: { 'User-Agent': 'Gandalf-API-Gateway/1.0' } });
            return new Response(resp.body, { status: resp.status, headers: { ...corsHeaders, ...sanitizeHeaders(resp.headers) } });
        }

        // RPC browser-safe forwarding (POST only)
        if (apiPath.startsWith('rpc/')) {
            const fn = apiPath.split('/')[1] || '';
            if (!RPC_ALLOW.has(fn) || request.method !== 'POST') {
                return new Response(JSON.stringify({ error: 'Forbidden' }), {
                    status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
                });
            }
            const targetUrl = `${SUPABASE_URL}/rest/v1/rpc/${fn}${url.search}`;
            const resp = await fetch(targetUrl, {
                method: 'POST',
                headers: {
                    'User-Agent': 'Gandalf-API-Gateway/1.0',
                    'apikey': SUPABASE_KEY,
                    'Authorization': `Bearer ${SUPABASE_KEY}`,
                    'Content-Type': 'application/json'
                },
                body: await request.text()
            });
            return new Response(resp.body, { status: resp.status, headers: { ...corsHeaders, ...sanitizeHeaders(resp.headers) } });
        }

        // Default: read-only REST passthrough
        const [root] = apiPath.split('?', 1);
        const table = (root || '').split('/')[0];
        if (request.method !== 'GET' || !READ_ONLY.has(table)) {
            return new Response(JSON.stringify({ error: 'Forbidden' }), {
                status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
            });
        }
        const targetUrl = `${SUPABASE_URL}/rest/v1/${apiPath}${url.search}`;
        const resp = await fetch(targetUrl, {
            headers: {
                'User-Agent': 'Gandalf-API-Gateway/1.0',
                'apikey': SUPABASE_KEY,
                'Authorization': `Bearer ${SUPABASE_KEY}`,
                'Content-Type': 'application/json'
            }
        });
        const headersCombined = { ...corsHeaders, ...sanitizeHeaders(resp.headers) };

        if (!resp.ok) {
            const errorText = await resp.text().catch(() => '');
            return new Response(JSON.stringify({ error: `Upstream service error: ${resp.status}`, body: errorText }), {
                status: resp.status, headers: { ...headersCombined, 'Content-Type': 'application/json' }
            });
        }

        return new Response(resp.body, { status: resp.status, headers: headersCombined });
    } catch (error) {
        return new Response(JSON.stringify({ error: 'Gateway internal error' }), {
            status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
    }
}