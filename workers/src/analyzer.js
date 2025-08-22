// This runs every hour to find patterns
export default {
    async scheduled(event, env, ctx) {
        console.log('Running pattern analysis...');

        try {
            // Fetch recent data from Supabase
            const data = await fetchRecentData(env);

            // Find correlations
            const patterns = await findPatterns(data);

            // Store discoveries
            await storePatterns(env, patterns);

            console.log(`Found ${patterns.length} patterns`);
        } catch (error) {
            console.error('Analysis error:', error);
        }
    },

    async fetch(request, env) {
        const url = new URL(request.url);

        if (url.pathname === '/api/patterns') {
            const patterns = await env.PATTERN_CACHE.get('latest_patterns');
            return new Response(patterns || '[]', {
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            });
        }

        if (url.pathname === '/api/predictions') {
            const predictions = await generatePredictions(env);
            return new Response(JSON.stringify(predictions), {
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            });
        }

        return new Response('Gandalf Analyzer API', { status: 200 });
    }
};

async function fetchRecentData(env) {
    // Get last 24 hours of data
    const response = await fetch(
        `${env.SUPABASE_URL}/rest/v1/market_data?order=timestamp.desc&limit=1000`,
        {
            headers: {
                'apikey': env.SUPABASE_ANON_KEY
            }
        }
    );

    return await response.json();
}

async function findPatterns(data) {
    const patterns = [];

    // Group by item
    const itemData = {};
    data.forEach(row => {
        if (!itemData[row.item]) itemData[row.item] = [];
        itemData[row.item].push(row);
    });

    // Find price movements
    Object.entries(itemData).forEach(([item, prices]) => {
        if (prices.length < 10) return;

        // Sort by timestamp
        prices.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

        // Calculate statistics
        const sellPrices = prices.map(p => p.sell_median).filter(p => p);
        const avgPrice = sellPrices.reduce((a, b) => a + b, 0) / sellPrices.length;
        const currentPrice = sellPrices[sellPrices.length - 1];
        const oldPrice = sellPrices[0];

        const change = ((currentPrice - oldPrice) / oldPrice) * 100;
        const volatility = calculateVolatility(sellPrices);

        // Identify patterns
        if (Math.abs(change) > 10) {
            patterns.push({
                type: change > 0 ? 'surge' : 'crash',
                item: item,
                change: change.toFixed(2),
                from: oldPrice,
                to: currentPrice,
                volatility: volatility,
                confidence: prices.length / 288 // How much data we have (288 = 24h of 5min intervals)
            });
        }

        // Moving average crossover
        if (sellPrices.length >= 20) {
            const ma5 = movingAverage(sellPrices.slice(-5));
            const ma20 = movingAverage(sellPrices.slice(-20));
            const prevMa5 = movingAverage(sellPrices.slice(-6, -1));
            const prevMa20 = movingAverage(sellPrices.slice(-21, -1));

            if (prevMa5 < prevMa20 && ma5 > ma20) {
                patterns.push({
                    type: 'golden_cross',
                    item: item,
                    signal: 'bullish',
                    ma5: ma5,
                    ma20: ma20,
                    currentPrice: currentPrice
                });
            }
        }
    });

    // Find correlations between items
    const items = Object.keys(itemData);
    for (let i = 0; i < items.length; i++) {
        for (let j = i + 1; j < items.length; j++) {
            const correlation = calculateCorrelation(
                itemData[items[i]],
                itemData[items[j]]
            );

            if (Math.abs(correlation) > 0.7) {
                patterns.push({
                    type: 'correlation',
                    item1: items[i],
                    item2: items[j],
                    strength: correlation,
                    direction: correlation > 0 ? 'positive' : 'negative'
                });
            }
        }
    }

    return patterns;
}

function calculateVolatility(prices) {
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
        returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }

    const avg = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avg, 2), 0) / returns.length;
    return Math.sqrt(variance) * 100; // As percentage
}

function movingAverage(prices) {
    return prices.reduce((a, b) => a + b, 0) / prices.length;
}

function calculateCorrelation(data1, data2) {
    // Simple correlation calculation
    // Match timestamps
    const prices1 = [];
    const prices2 = [];

    data1.forEach(d1 => {
        const match = data2.find(d2 =>
            Math.abs(new Date(d1.timestamp) - new Date(d2.timestamp)) < 300000 // 5 min tolerance
        );
        if (match && d1.sell_median && match.sell_median) {
            prices1.push(d1.sell_median);
            prices2.push(match.sell_median);
        }
    });

    if (prices1.length < 10) return 0;

    // Pearson correlation
    const n = prices1.length;
    const sum1 = prices1.reduce((a, b) => a + b, 0);
    const sum2 = prices2.reduce((a, b) => a + b, 0);
    const sum1Sq = prices1.reduce((a, b) => a + b * b, 0);
    const sum2Sq = prices2.reduce((a, b) => a + b * b, 0);
    const pSum = prices1.reduce((sum, p1, i) => sum + p1 * prices2[i], 0);

    const num = pSum - (sum1 * sum2 / n);
    const den = Math.sqrt((sum1Sq - sum1 * sum1 / n) * (sum2Sq - sum2 * sum2 / n));

    return den === 0 ? 0 : num / den;
}

async function storePatterns(env, patterns) {
    // Store in KV
    await env.PATTERN_CACHE.put(
        'latest_patterns',
        JSON.stringify(patterns),
        { expirationTtl: 3600 } // 1 hour
    );

    // Store significant patterns in Supabase
    const significant = patterns.filter(p =>
        (p.type === 'surge' || p.type === 'crash') && Math.abs(p.change) > 20
    );

    if (significant.length > 0) {
        await fetch(`${env.SUPABASE_URL}/rest/v1/discovered_patterns`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'apikey': env.SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}`
            },
            body: JSON.stringify(significant.map(p => ({
                pattern_type: p.type,
                pattern_data: p,
                confidence: p.confidence || 0.5,
                discovered_at: new Date().toISOString()
            })))
        });
    }
}

async function generatePredictions(env) {
    const data = await fetchRecentData(env);
    const predictions = [];

    // Group by item
    const itemData = {};
    data.forEach(row => {
        if (!itemData[row.item]) itemData[row.item] = [];
        itemData[row.item].push(row);
    });

    Object.entries(itemData).forEach(([item, prices]) => {
        if (prices.length < 20) return;

        prices.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        const sellPrices = prices.map(p => p.sell_median).filter(p => p);

        if (sellPrices.length < 10) return;

        // Simple linear regression for trend
        const n = sellPrices.length;
        const x = Array.from({ length: n }, (_, i) => i);
        const y = sellPrices;

        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        // Predict next value
        const nextPrice = slope * n + intercept;
        const currentPrice = sellPrices[sellPrices.length - 1];
        const change = ((nextPrice - currentPrice) / currentPrice) * 100;

        predictions.push({
            item: item,
            current_price: currentPrice,
            predicted_price: Math.round(nextPrice),
            predicted_change: change.toFixed(2),
            trend: slope > 0 ? 'up' : 'down',
            confidence: Math.min(0.9, prices.length / 100), // More data = more confidence
            based_on_hours: (prices.length * 5) / 60 // Convert 5-min intervals to hours
        });
    });

    // Sort by biggest predicted moves
    predictions.sort((a, b) => Math.abs(b.predicted_change) - Math.abs(a.predicted_change));

    return predictions.slice(0, 10); // Top 10
}