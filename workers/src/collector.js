export default {
    async scheduled(event, env, ctx) {
        console.log('Collector: scheduled collection…');
        ctx.waitUntil(collectMarketData(env));
    },

    async fetch(request, env, ctx) {
        const url = new URL(request.url);
        const cors = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        };
        if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });

        if (url.pathname === '/health') {
            const latest = await env.MARKET_CACHE.get('latest_snapshot');
            let ts = null, ageM = null;
            if (latest) {
                try {
                    const arr = JSON.parse(latest);
                    ts = Array.isArray(arr) && arr[0] ? arr[0].timestamp : null;
                    if (ts) ageM = Math.round((Date.now() - new Date(ts).getTime()) / 60000);
                } catch { }
            }
            return new Response(JSON.stringify({
                status: latest ? 'healthy' : 'no_cache_data',
                last_collection_timestamp: ts,
                last_collection_age_minutes: ageM
            }), { headers: { ...cors, 'Content-Type': 'application/json' } });
        }

        if (url.pathname === '/collect') {
            ctx.waitUntil(collectMarketData(env));
            return new Response('Collection started in background.', { status: 202, headers: cors });
        }

        return new Response('Gandalf Collector v1.4 is online.', { status: 200, headers: cors });
    }
};

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function fetchWithRetry(url, opts = {}, retries = 3, backoffMs = 700) {
    for (let i = 0; i < retries; i++) {
        try {
            const resp = await fetch(url, opts);
            // Respect WM rate limits and transient issues
            if (resp.status === 429 || resp.status >= 500) {
                await sleep(backoffMs * (i + 1));
                continue;
            }
            if (!resp.ok) {
                console.error(`Fetch failed ${resp.status} ${url}`);
                return null;
            }
            return await resp.json();
        } catch (e) {
            if (i === retries - 1) return null;
            await sleep(backoffMs * (i + 1));
        }
    }
    return null;
}

async function getTrackedItems(env) {
    const cached = await env.MARKET_CACHE.get('tracked_items');
    if (cached) { try { return JSON.parse(cached); } catch { } }

    try {
        const url = `${env.SUPABASE_URL}/rest/v1/tracked_items?select=item,score,active&active=eq.true&order=score.desc&limit=50`;
        const resp = await fetch(url, {
            headers: {
                'apikey': env.SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}`
            }
        });
        if (resp.ok) {
            const rows = await resp.json();
            const items = rows.map(r => r.item);
            if (items.length) {
                await env.MARKET_CACHE.put('tracked_items', JSON.stringify(items), { expirationTtl: 600 });
                return items;
            }
        } else {
            console.error('tracked_items error:', resp.status);
        }
    } catch (e) {
        console.error('tracked_items fetch failed:', e.message);
    }

    // conservative fallback list
    return [
        'octavia_prime_set', 'wisp_prime_set', 'volt_prime_set', 'saryn_prime_set',
        'mesa_prime_set', 'nekros_prime_set', 'nova_prime_set', 'rhino_prime_set',
        'frost_prime_set', 'ember_prime_set'
    ];
}

function median(arr) {
    if (!arr || !arr.length) return null;
    const s = [...arr].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

async function collectMarketData(env) {
    const items = await getTrackedItems(env);
    const results = [];

    // 3 rps limit: keep concurrency 1 and add ~900ms pacing (safe)
    for (const item of items) {
        try {
            const data = await fetchWithRetry(
                `https://api.warframe.market/v1/items/${item}/orders`,
                { headers: { 'User-Agent': 'Gandalf/1.0', 'Accept': 'application/json' } },
                3,
                800
            );
            if (!data || !data.payload || !Array.isArray(data.payload.orders)) {
                console.error(`No orders for ${item}`);
                await sleep(900);
                continue;
            }
            const orders = data.payload.orders.filter(o => o?.user?.status === 'ingame');
            const buys = orders.filter(o => o.order_type === 'buy').map(o => o.platinum).filter(v => typeof v === 'number');
            const sells = orders.filter(o => o.order_type === 'sell').map(o => o.platinum).filter(v => typeof v === 'number');

            const rec = {
                timestamp: new Date().toISOString(),
                item,
                buy_orders: buys.length,
                sell_orders: sells.length,
                buy_median: median(buys),
                sell_median: median(sells),
                spread: null
            };
            if (rec.buy_median != null && rec.sell_median != null) {
                rec.spread = rec.sell_median - rec.buy_median;
            }
            results.push(rec);
        } catch (e) {
            console.error(`Error ${item}:`, e);
        }
        // gentle pacing to remain under rate limit
        await sleep(900);
    }

    if (results.length) await storeInSupabase(env, results);
    await env.MARKET_CACHE.put('latest_snapshot', JSON.stringify(results), { expirationTtl: 300 });
    console.log(`Collected ${results.length}/${items.length}`);
}

async function storeInSupabase(env, data) {
    try {
        const resp = await fetch(`${env.SUPABASE_URL}/rest/v1/market_data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'apikey': env.SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}`,
                'Prefer': 'return=minimal'
            },
            body: JSON.stringify(data)
        });
        if (!resp.ok) {
            const txt = await resp.text().catch(() => '');
            console.error('Supabase write error:', resp.status, txt);
        }
    } catch (e) {
        console.error('Supabase store error:', e);
    }
}