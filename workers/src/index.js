// Market data collector that runs 24/7
export default {
    // Scheduled trigger (runs every 5 minutes)
    async scheduled(event, env, ctx) {
        console.log('Starting scheduled collection...');
        await collectMarketData(env);
    },

    // HTTP trigger (for testing and monitoring)
    async fetch(request, env, ctx) {
        const url = new URL(request.url);

        if (url.pathname === '/health') {
            return new Response(JSON.stringify({
                status: 'healthy',
                timestamp: new Date().toISOString()
            }), {
                headers: { 'Content-Type': 'application/json' }
            });
        }

        if (url.pathname === '/collect') {
            await collectMarketData(env);
            return new Response('Collection started', { status: 200 });
        }

        return new Response('Gandalf Collector v1.0', { status: 200 });
    }
};

async function collectMarketData(env) {
    // Items to track (start small)
    const items = [
        'octavia_prime_set',
        'wisp_prime_set',
        'volt_prime_set',
        'saryn_prime_set',
        'mesa_prime_set'
    ];

    const results = [];

    for (const item of items) {
        try {
            // Fetch from Warframe Market API
            const response = await fetch(
                `https://api.warframe.market/v1/items/${item}/orders`,
                {
                    headers: {
                        'User-Agent': 'Gandalf/1.0',
                        'Accept': 'application/json'
                    }
                }
            );

            if (!response.ok) {
                console.error(`Failed to fetch ${item}: ${response.status}`);
                continue;
            }

            const data = await response.json();

            // Process orders
            const orders = data.payload.orders.filter(
                order => order.user.status === 'ingame'
            );

            const buyOrders = orders.filter(o => o.order_type === 'buy');
            const sellOrders = orders.filter(o => o.order_type === 'sell');

            const marketData = {
                timestamp: new Date().toISOString(),
                item: item,
                buy_orders: buyOrders.length,
                sell_orders: sellOrders.length,
                buy_median: calculateMedian(buyOrders.map(o => o.platinum)),
                sell_median: calculateMedian(sellOrders.map(o => o.platinum)),
                spread: 0 // Calculate later
            };

            if (marketData.buy_median && marketData.sell_median) {
                marketData.spread = marketData.sell_median - marketData.buy_median;
            }

            results.push(marketData);

            // Rate limit
            await new Promise(resolve => setTimeout(resolve, 1000));

        } catch (error) {
            console.error(`Error processing ${item}:`, error);
        }
    }

    // Store in Supabase
    if (results.length > 0) {
        await storeInSupabase(env, results);
    }

    // Cache latest data in KV
    await env.MARKET_CACHE.put(
        'latest_snapshot',
        JSON.stringify(results),
        { expirationTtl: 300 } // 5 minutes
    );

    console.log(`Collected data for ${results.length} items`);
}

async function storeInSupabase(env, data) {
    try {
        const response = await fetch(
            `${env.SUPABASE_URL}/rest/v1/market_data`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'apikey': env.SUPABASE_ANON_KEY,
                    'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}`
                },
                body: JSON.stringify(data)
            }
        );

        if (!response.ok) {
            const error = await response.text();
            console.error('Supabase error:', error);
        }
    } catch (error) {
        console.error('Failed to store in Supabase:', error);
    }
}

function calculateMedian(values) {
    if (values.length === 0) return null;

    const sorted = values.sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2;
    }

    return sorted[mid];
}