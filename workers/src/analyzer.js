// Complete analyzer.js with Discord alerts and trading signals
export default {
    async scheduled(event, env, ctx) {
        console.log('Running pattern analysis...');
        try {
            const data = await fetchRecentData(env);
            const patterns = await findPatterns(data);
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

        // NEW: Signals endpoint
        if (url.pathname === '/api/signals') {
            const signals = await env.PATTERN_CACHE.get('latest_signals');
            return new Response(signals || '[]', {
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

// NEW: Discord notification function
async function sendDiscordAlert(env, alert) {
    if (!env.DISCORD_WEBHOOK) return; // Skip if no webhook configured

    const embed = {
        title: alert.title || "🧙‍♂️ Gandalf Alert",
        description: alert.description,
        color: alert.color || 3447003, // Blue default
        fields: alert.fields || [],
        timestamp: new Date().toISOString(),
        footer: {
            text: "Gandalf Market Intelligence"
        }
    };

    try {
        await fetch(env.DISCORD_WEBHOOK, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ embeds: [embed] })
        });
    } catch (error) {
        console.error('Discord webhook error:', error);
    }
}

// NEW: Trading signal generator
async function generateTradingSignals(patterns, data) {
    const signals = [];

    // Process each pattern for trading opportunities
    for (const pattern of patterns) {
        let signal = null;

        if (pattern.type === 'surge' && pattern.change > 15) {
            signal = {
                type: 'SELL',
                item: pattern.item,
                reason: `Price surged ${pattern.change}% - potential peak`,
                confidence: Math.min(pattern.confidence * 1.2, 0.95),
                current_price: pattern.to,
                action: `Consider selling ${pattern.item} at ${pattern.to}p`
            };
        } else if (pattern.type === 'crash' && pattern.change < -15) {
            signal = {
                type: 'BUY',
                item: pattern.item,
                reason: `Price crashed ${Math.abs(pattern.change)}% - potential bottom`,
                confidence: Math.min(pattern.confidence * 1.2, 0.95),
                current_price: pattern.to,
                action: `Consider buying ${pattern.item} at ${pattern.to}p`
            };
        } else if (pattern.type === 'golden_cross') {
            signal = {
                type: 'BUY',
                item: pattern.item,
                reason: 'Golden cross detected - bullish signal',
                confidence: 0.75,
                current_price: pattern.currentPrice,
                action: `Uptrend starting for ${pattern.item}`
            };
        }

        if (signal && signal.confidence > 0.7) {
            signals.push(signal);
        }
    }

    // Look for arbitrage opportunities
    const arbitrage = findArbitrageOpportunities(data);
    signals.push(...arbitrage);

    return signals;
}

// NEW: Arbitrage finder
function findArbitrageOpportunities(data) {
    const opportunities = [];
    const itemGroups = {};

    // Group by item
    data.forEach(row => {
        if (!itemGroups[row.item]) {
            itemGroups[row.item] = [];
        }
        itemGroups[row.item].push(row);
    });

    // Check each item for arbitrage
    Object.entries(itemGroups).forEach(([item, rows]) => {
        const latestRow = rows[rows.length - 1];

        if (latestRow.buy_median && latestRow.sell_median) {
            const spread = latestRow.sell_median - latestRow.buy_median;
            const spreadPct = (spread / latestRow.sell_median) * 100;

            // High spread = opportunity
            if (spreadPct > 20 && spread > 5) {
                opportunities.push({
                    type: 'ARBITRAGE',
                    item: item,
                    buy_price: latestRow.sell_median,
                    sell_price: latestRow.buy_median,
                    profit: spread,
                    profit_pct: spreadPct,
                    confidence: 0.9,
                    action: `Arbitrage: Buy at ${latestRow.sell_median}p, sell at ${latestRow.buy_median}p for ${spread}p profit`
                });
            }
        }
    });

    return opportunities;
}

// UPDATED: storePatterns with Discord alerts and signals
async function storePatterns(env, patterns) {
    // Store in KV
    await env.PATTERN_CACHE.put(
        'latest_patterns',
        JSON.stringify(patterns),
        { expirationTtl: 3600 } // 1 hour
    );

    // Generate trading signals
    const data = await fetchRecentData(env);
    const signals = await generateTradingSignals(patterns, data);

    // Store signals
    await env.PATTERN_CACHE.put(
        'latest_signals',
        JSON.stringify(signals),
        { expirationTtl: 3600 }
    );

    // Send Discord alerts for HIGH confidence signals
    for (const signal of signals) {
        if (signal.confidence > 0.8) {
            const alertColor = signal.type === 'BUY' ? 3066993 : // Green
                signal.type === 'SELL' ? 15158332 : // Red
                    3447003; // Blue for arbitrage

            await sendDiscordAlert(env, {
                title: `💎 ${signal.type} Signal: ${signal.item.replace(/_/g, ' ').toUpperCase()}`,
                description: signal.action,
                color: alertColor,
                fields: [
                    {
                        name: "Confidence",
                        value: `${(signal.confidence * 100).toFixed(0)}%`,
                        inline: true
                    },
                    {
                        name: "Reason",
                        value: signal.reason || "Pattern detected",
                        inline: false
                    },
                    ...(signal.profit ? [{
                        name: "Potential Profit",
                        value: `${signal.profit}p (${signal.profit_pct.toFixed(1)}%)`,
                        inline: true
                    }] : [])
                ]
            });

            // Rate limit Discord messages
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    }

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