// This is your new secure API gateway.
// It runs on the same domain as your dashboard, avoiding all CORS issues.

export async function onRequest(context) {
    // Get the original path the user requested (e.g., /api/signals)
    const url = new URL(context.request.url);
    const apiPath = url.pathname.replace('/api/', '');

    // Get secure environment variables from Cloudflare dashboard
    const SUPABASE_URL = context.env.SUPABASE_URL;
    const SUPABASE_KEY = context.env.SUPABASE_ANON_KEY;
    const ANALYZER_URL = 'https://gandalf-analyzer.psnedujime.workers.dev';

    let targetUrl;
    let requestOptions = {
        headers: {
            'User-Agent': 'Gandalf-API-Gateway/1.0',
        }
    };

    // Add this near the top of onRequest after apiPath is computed
    if (apiPath.startsWith('ml-predictions')) {
        const SUPABASE_URL = context.env.SUPABASE_URL;
        const SUPABASE_KEY = context.env.SUPABASE_ANON_KEY;
        const targetUrl = `${SUPABASE_URL}/rest/v1/predictions?order=predicted_at.desc&limit=100${url.search}`;
        const requestOptions = {
            headers: {
                'User-Agent': 'Gandalf-API-Gateway/1.0',
                'apikey': SUPABASE_KEY,
                'Authorization': `Bearer ${SUPABASE_KEY}`,
                'Content-Type': 'application/json'
            }
        };
        const resp = await fetch(targetUrl, requestOptions);
        const newHeaders = new Headers(resp.headers);
        newHeaders.set('Access-Control-Allow-Origin', '*');
        newHeaders.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
        newHeaders.set('Access-Control-Allow-Headers', 'Content-Type, apikey, Authorization');
        return new Response(resp.body, { status: resp.status, headers: newHeaders });
    }

    // Route the request to the correct destination
    if (apiPath.startsWith('patterns') || apiPath.startsWith('predictions') || apiPath.startsWith('signals')) {
        // Forward to the Analyzer worker
        targetUrl = `${ANALYZER_URL}/api/${apiPath}`;
    } else {
        // Forward to Supabase
        targetUrl = `${SUPABASE_URL}/rest/v1/${apiPath}${url.search}`;
        requestOptions.headers['apikey'] = SUPABASE_KEY;
        requestOptions.headers['Authorization'] = `Bearer ${SUPABASE_KEY}`;

        // Pass through method and body for POST requests (like executing trades)
        if (context.request.method === 'POST') {
            requestOptions.method = 'POST';
            requestOptions.headers['Content-Type'] = 'application/json';
            requestOptions.headers['Prefer'] = 'return=minimal';
            requestOptions.body = await context.request.text();
        }
    }

    try {
        // Fetch the data from the target
        const response = await fetch(targetUrl, requestOptions);

        // Create a new response with CORS headers to avoid being blocked by browsers
        const newHeaders = new Headers(response.headers);
        newHeaders.set('Access-Control-Allow-Origin', '*');
        newHeaders.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
        newHeaders.set('Access-Control-Allow-Headers', 'Content-Type, apikey, Authorization');

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Error fetching from ${targetUrl}: ${errorText}`);
            return new Response(JSON.stringify({ error: `Upstream service error: ${response.status}` }), {
                status: response.status,
                headers: newHeaders
            });
        }

        return new Response(response.body, {
            status: response.status,
            headers: newHeaders
        });

    } catch (error) {
        console.error(`Gateway fetch error: ${error.message}`);
        return new Response(JSON.stringify({ error: 'Gateway internal error' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }
        });
    }
}