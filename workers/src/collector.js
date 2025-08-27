export default {
    async scheduled(event, env, ctx) {
        console.log('Collector: scheduled collection...');
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
            return new Response(JSON.stringify({ status: latest ? 'healthy' : 'no_cache_data', last_collection_timestamp: ts, last_collection_age_minutes: ageM }), { headers: { ...cors, 'Content-Type': 'application/json' } });
        }

        if (url.pathname === '/collect') {
            ctx.waitUntil(collectMarketData(env));
            return new Response('Collection started in background.', { status: 202, headers: cors });
        }

        return new Response('Gandalf Collector v1.3 is online.', { status: 200, headers: cors });
    }
};

async function getTrackedItems(env) {
    const cached = await env.MARKET_CACHE.get('tracked_items');
    if (cached) { try { return JSON.parse(cached); } catch { } }
    try {
        const url = `${env.SUPABASE_URL}/rest/v1/tracked_items?select=item,score&active=eq.true&order=score.desc&limit=50`;
        const resp = await fetch(url, { headers: { 'apikey': env.SUPABASE_ANON_KEY, 'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}` } });
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
    return ['octavia_prime_set', 'wisp_prime_set', 'volt_prime_set', 'saryn_prime_set', 'mesa_prime_set', 'nekros_prime_set', 'nova_prime_set', 'rhino_prime_set', 'frost_prime_set', 'ember_prime_set'];
}

async function collectMarketData(env) {
    const items = await getTrackedItems(env);
    const results = [];
    for (const item of items) {
        try {
            const resp = await fetch(`https://api.warframe.market/v1/items/${item}/orders`, { headers: { 'User-Agent': 'Gandalf/1.0', 'Accept': 'application/json' } });
            if (!resp.ok) { console.error(`Fetch ${item} failed:`, resp.status); continue; }
            const data = await resp.json();
            const orders = (data?.payload?.orders || []).filter(o => o?.user?.status === 'ingame');
            const buys = orders.filter(o => o.order_type === 'buy').map(o => o.platinum).filter(v => typeof v === 'number');
            const sells = orders.filter(o => o.order_type === 'sell').map(o => o.platinum).filter(v => typeof v === 'number');
            const buyCount = buys.length, sellCount = sells.length;

            const rec = {
                timestamp: new Date().toISOString(),
                item,
                buy_orders: buyCount,
                sell_orders: sellCount,
                buy_median: median(buys),
                sell_median: median(sells),
                spread: 0
            };
            if (rec.buy_median != null && rec.sell_median != null) rec.spread = rec.sell_median - rec.buy_median;
            results.push(rec);
            await new Promise(r => setTimeout(r, 900));
        } catch (e) {
            console.error(`Error ${item}:`, e);
        }
    }
    if (results.length) await storeInSupabase(env, results);
    await env.MARKET_CACHE.put('latest_snapshot', JSON.stringify(results), { expirationTtl: 300 });
    console.log(`Collected ${results.length} items`);
}

async function storeInSupabase(env, data) {
    try {
        const resp = await fetch(`${env.SUPABASE_URL}/rest/v1/market_data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'apikey': env.SUPABASE_ANON_KEY, 'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}`, 'Prefer': 'return=minimal' },
            body: JSON.stringify(data)
        });
        if (!resp.ok) console.error('Supabase write error:', await resp.text());
    } catch (e) {
        console.error('Supabase store error:', e);
    }
}

function median(arr) {
    if (!arr || !arr.length) return null;
    const s = [...arr].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; 
}