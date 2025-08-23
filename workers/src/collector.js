// collector.js - v1.3 - Dynamic tracked items + CORS + robustness

export default {
    async scheduled(event, env, ctx) {
        console.log('Starting scheduled collection cycle...');
        ctx.waitUntil(collectMarketData(env));
    },

    async fetch(request, env, ctx) {
        const url = new URL(request.url);

        const corsHeaders = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        };

        if (request.method === 'OPTIONS') {
            return new Response(null, { status: 204, headers: corsHeaders });
        }

        if (url.pathname === '/health') {
            const latestSnapshot = await env.MARKET_CACHE.get('latest_snapshot');
            const status = latestSnapshot ? 'healthy' : 'no_cache_data';
            let timestamp = null;
            let ageMinutes = null;

            if (latestSnapshot) {
                try {
                    const snapshotData = JSON.parse(latestSnapshot);
                    timestamp = Array.isArray(snapshotData) && snapshotData[0] ? snapshotData[0].timestamp : null;
                    if (timestamp) {
                        ageMinutes = Math.round((Date.now() - new Date(timestamp).getTime()) / 60000);
                    }
                } catch (_) { }
            }

            const healthData = {
                status,
                last_collection_age_minutes: ageMinutes,
                last_collection_timestamp: timestamp,
            };

            return new Response(JSON.stringify(healthData), {
                headers: { ...corsHeaders, 'Content-Type': 'application/json' }
            });
        }

        if (url.pathname === '/collect') {
            ctx.waitUntil(collectMarketData(env));
            return new Response('Collection started in background.', { status: 202, headers: corsHeaders });
        }

        return new Response('Gandalf Collector v1.3 is online.', { status: 200, headers: corsHeaders });
    }
};

// Fetch tracked items from Supabase with KV cache, fallback to static
async function getTrackedItems(env) {
    // 1) KV cache
    const cached = await env.MARKET_CACHE.get('tracked_items');
    if (cached) {
        try { return JSON.parse(cached); } catch (_) { }
    }

    // 2) Supabase tracked_items (active=true)
    try {
        const url = `${env.SUPABASE_URL}/rest/v1/tracked_items?select=item,score&active=eq.true&order=score.desc&limit=50`;
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
                await env.MARKET_CACHE.put('tracked_items', JSON.stringify(items), { expirationTtl: 600 }); // 10 min
                return items;
            }
        } else {
            console.error('Supabase tracked_items error:', resp.status);
        }
    } catch (e) {
        console.error('Failed to load tracked_items from Supabase:', e.message);
    }

    // 3) Fallback
    return [
        'octavia_prime_set', 'wisp_prime_set', 'volt_prime_set', 'saryn_prime_set',
        'mesa_prime_set', 'nekros_prime_set', 'nova_prime_set', 'rhino_prime_set',
        'frost_prime_set', 'ember_prime_set'
    ];
}

async function collectMarketData(env) {
    const items = await getTrackedItems(env);
    const results = [];

    for (const item of items) {
        try {
            const response = await fetch(`https://api.warframe.market/v1/items/${item}/orders`, {
                headers: { 'User-Agent': 'Gandalf/1.0', 'Accept': 'application/json' }
            });

            if (!response.ok) {
                console.error(`Failed to fetch ${item}: ${response.status}`);
                continue;
            }

            const data = await response.json();
            const orders = (data?.payload?.orders || []).filter(order => order?.user?.status === 'ingame');
            const buyOrders = orders.filter(o => o.order_type === 'buy');
            const sellOrders = orders.filter(o => o.order_type === 'sell');

            const buyPrices = buyOrders.map(o => o.platinum).filter(v => typeof v === 'number');
            const sellPrices = sellOrders.map(o => o.platinum).filter(v => typeof v === 'number');

            const marketData = {
                timestamp: new Date().toISOString(),
                item,
                buy_orders: buyOrders.length,
                sell_orders: sellOrders.length,
                buy_median: calculateMedian(buyPrices),
                sell_median: calculateMedian(sellPrices),
                spread: 0
            };

            if (marketData.buy_median != null && marketData.sell_median != null) {
                marketData.spread = marketData.sell_median - marketData.buy_median;
            }

            results.push(marketData);

            // rate limit
            await new Promise(resolve => setTimeout(resolve, 1000));
        } catch (error) {
            console.error(`Error processing ${item}:`, error);
        }
    }

    if (results.length > 0) {
        await storeInSupabase(env, results);
    }

    await env.MARKET_CACHE.put('latest_snapshot', JSON.stringify(results), { expirationTtl: 300 });
    console.log(`Collected data for ${results.length} items`);
}

async function storeInSupabase(env, data) {
    try {
        const response = await fetch(`${env.SUPABASE_URL}/rest/v1/market_data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'apikey': env.SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}`,
                'Prefer': 'return=minimal'
            },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            const error = await response.text();
            console.error('Supabase error:', error);
        }
    } catch (error) {
        console.error('Failed to store in Supabase:', error);
    }
}

function calculateMedian(values) {
    if (!values || values.length === 0) return null;
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];
}