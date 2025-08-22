// FINAL & FIXED analyzer.js - v1.1
export default {
    async scheduled(event, env, ctx) {
        console.log('Running full analysis cycle...');
        try {
            const data = await fetchRecentData(env);
            if (!data || data.length < 50) { // Need some data to work with
                console.log("Not enough data to run analysis. Skipping cycle.");
                return;
            }
            const patterns = findPatterns(data);
            const predictions = generatePredictions(data);

            // This now stores everything correctly
            await storeAnalysisResults(env, patterns, predictions, data);

            console.log(`Analysis complete. Found ${patterns.length} patterns and ${predictions.length} predictions.`);
        } catch (error) {
            console.error('Analysis cycle error:', error);
        }
    },

    async fetch(request, env) {
        const url = new URL(request.url);
        // CRITICAL FIX FOR DASHBOARD: Add CORS headers to all API responses
        const headers = {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        };

        if (url.pathname === '/api/patterns') {
            const patterns = await env.PATTERN_CACHE.get('latest_patterns');
            return new Response(patterns || '[]', { headers });
        }

        if (url.pathname === '/api/predictions') {
            const predictions = await env.PATTERN_CACHE.get('latest_predictions');
            return new Response(predictions || '[]', { headers });
        }

        if (url.pathname === '/api/signals') {
            const signals = await env.PATTERN_CACHE.get('latest_signals');
            return new Response(signals || '[]', { headers });
        }

        return new Response('Gandalf Analyzer API is online.', { headers });
    }
};

// =================================================================
// ==                      DATA & HELPERS                           ==
// =================================================================

async function fetchRecentData(env) {
    const response = await fetch(`${env.SUPABASE_URL}/rest/v1/market_data?order=timestamp.desc&limit=2500`, { headers: { 'apikey': env.SUPABASE_ANON_KEY } });
    if (!response.ok) return [];
    return await response.json();
}

// =================================================================
// ==                   STORAGE & NOTIFICATIONS                     ==
// =================================================================

async function storeAnalysisResults(env, patterns, predictions, data) {
    const signals = generateTradingSignals(patterns, data);

    // --- Cache everything in KV for fast API access ---
    await env.PATTERN_CACHE.put('latest_patterns', JSON.stringify(patterns), { expirationTtl: 3600 });
    await env.PATTERN_CACHE.put('latest_predictions', JSON.stringify(predictions), { expirationTtl: 3600 });
    await env.PATTERN_CACHE.put('latest_signals', JSON.stringify(signals), { expirationTtl: 3600 });

    // --- Send Discord alerts for high-confidence signals ---
    for (const signal of signals) {
        if (signal.confidence > 0.8) {
            const alertColor = signal.type === 'BUY' ? 3066993 : signal.type === 'SELL' ? 15158332 : 16776960; // Yellow for arbitrage
            await sendDiscordAlert(env, {
                title: `💎 ${signal.type} Signal: ${signal.item.replace(/_/g, ' ').toUpperCase()}`,
                description: signal.action,
                color: alertColor,
                fields: [
                    { name: "Confidence", value: `${(signal.confidence * 100).toFixed(0)}%`, inline: true },
                    { name: "Reason", value: signal.reason || "Pattern detected", inline: false },
                    ...(signal.profit ? [{ name: "Potential Profit", value: `${signal.profit}p (${signal.profit_pct.toFixed(1)}%)`, inline: true }] : [])
                ]
            });
            await new Promise(resolve => setTimeout(resolve, 1500)); // Rate limit Discord messages
        }
    }

    // --- Store significant findings in Supabase for long-term storage ---
    const supabaseHeaders = {
        'Content-Type': 'application/json', 'apikey': env.SUPABASE_ANON_KEY,
        'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}`, 'Prefer': 'return=minimal'
    };

    // 1. Store patterns
    if (patterns.length > 0) {
        await fetch(`${env.SUPABASE_URL}/rest/v1/discovered_patterns`, {
            method: 'POST', headers: supabaseHeaders,
            body: JSON.stringify(patterns.map(p => ({
                pattern_type: p.type, pattern_data: p, confidence: p.confidence || 0.5,
            })))
        });
    }

    // 2. Store predictions
    if (predictions.length > 0) {
        await fetch(`${env.SUPABASE_URL}/rest/v1/predictions`, {
            method: 'POST', headers: supabaseHeaders,
            body: JSON.stringify(predictions.map(p => ({
                item: p.item, predicted_at: new Date().toISOString(),
                current_price: p.current_price, predicted_price: p.predicted_price,
                confidence: p.confidence
            })))
        });
    }
}

async function sendDiscordAlert(env, alert) {
    if (!env.DISCORD_WEBHOOK) return;
    const embed = {
        title: alert.title || "🧙‍♂️ Gandalf Alert", description: alert.description, color: alert.color || 3447003,
        fields: alert.fields || [], timestamp: new Date().toISOString(), footer: { text: "Gandalf Market Intelligence" }
    };
    try {
        await fetch(env.DISCORD_WEBHOOK, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ embeds: [embed] })
        });
    } catch (error) { console.error('Discord webhook error:', error); }
}

// =================================================================
// ==           UNCHANGED ANALYSIS HELPER FUNCTIONS               ==
// =================================================================

function calculateVolatility(prices) {
    if (prices.length < 2) return 0; const returns = [];
    for (let i = 1; i < prices.length; i++) { if (prices[i - 1] !== 0) returns.push((prices[i] - prices[i - 1]) / prices[i - 1]); }
    if (returns.length === 0) return 0;
    const avg = returns.reduce((a, b) => a + b, 0) / returns.length; if (isNaN(avg)) return 0;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avg, 2), 0) / returns.length;
    return Math.sqrt(variance) * 100;
}
function movingAverage(prices) { if (prices.length === 0) return 0; return prices.reduce((a, b) => a + b, 0) / prices.length; }
function calculateCorrelation(data1, data2) {
    const prices1 = [], prices2 = [];
    data1.forEach(d1 => {
        const match = data2.find(d2 => Math.abs(new Date(d1.timestamp) - new Date(d2.timestamp)) < 300000);
        if (match && d1.sell_median != null && match.sell_median != null) { prices1.push(d1.sell_median); prices2.push(match.sell_median); }
    });
    if (prices1.length < 10) return 0;
    const n = prices1.length; const sum1 = prices1.reduce((a, b) => a + b, 0); const sum2 = prices2.reduce((a, b) => a + b, 0);
    const sum1Sq = prices1.reduce((a, b) => a + b * b, 0); const sum2Sq = prices2.reduce((a, b) => a + b * b, 0);
    const pSum = prices1.reduce((sum, p1, i) => sum + p1 * prices2[i], 0);
    const num = pSum - (sum1 * sum2 / n); const den = Math.sqrt((sum1Sq - sum1 * sum1 / n) * (sum2Sq - sum2 * sum2 / n));
    return den === 0 ? 0 : num / den;
}
function findPatterns(data) {
    const patterns = []; const itemData = {}; data.forEach(row => { if (!itemData[row.item]) itemData[row.item] = []; itemData[row.item].push(row); });
    Object.entries(itemData).forEach(([item, prices]) => {
        if (prices.length < 10) return; prices.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        const sellPrices = prices.map(p => p.sell_median).filter(p => p != null); if (sellPrices.length < 2) return;
        const currentPrice = sellPrices[sellPrices.length - 1]; const oldPrice = sellPrices[0];
        const change = oldPrice === 0 ? 0 : ((currentPrice - oldPrice) / oldPrice) * 100;
        if (Math.abs(change) > 10) { patterns.push({ type: change > 0 ? 'surge' : 'crash', item, change: change.toFixed(2), from: oldPrice, to: currentPrice, confidence: Math.min(1, prices.length / 288) }); }
        if (sellPrices.length >= 20) {
            const ma5 = movingAverage(sellPrices.slice(-5)); const ma20 = movingAverage(sellPrices.slice(-20));
            const prevMa5 = movingAverage(sellPrices.slice(-6, -1)); const prevMa20 = movingAverage(sellPrices.slice(-21, -1));
            if (prevMa5 < prevMa20 && ma5 > ma20) { patterns.push({ type: 'golden_cross', item, signal: 'bullish', ma5, ma20, currentPrice, confidence: 0.75 }); }
        }
    });
    const items = Object.keys(itemData);
    for (let i = 0; i < items.length; i++) {
        for (let j = i + 1; j < items.length; j++) {
            const correlation = calculateCorrelation(itemData[items[i]], itemData[items[j]]);
            if (Math.abs(correlation) > 0.7) { patterns.push({ type: 'correlation', item1: items[i], item2: items[j], strength: correlation, direction: correlation > 0 ? 'positive' : 'negative', confidence: Math.abs(correlation) }); }
        }
    }
    return patterns;
}
function generatePredictions(data) {
    const predictions = []; const itemData = {}; data.forEach(row => { if (!itemData[row.item]) itemData[row.item] = []; itemData[row.item].push(row); });
    Object.entries(itemData).forEach(([item, prices]) => {
        if (prices.length < 20) return; prices.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        const sellPrices = prices.map(p => p.sell_median).filter(p => p != null); if (sellPrices.length < 10) return;
        const n = sellPrices.length; const x = Array.from({ length: n }, (_, i) => i); const y = sellPrices;
        const sumX = x.reduce((a, b) => a + b, 0); const sumY = y.reduce((a, b) => a + b, 0); const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0); const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n; if (isNaN(slope) || isNaN(intercept)) return;
        const nextPrice = slope * n + intercept; const currentPrice = sellPrices[sellPrices.length - 1]; const change = currentPrice === 0 ? 0 : ((nextPrice - currentPrice) / currentPrice) * 100;
        predictions.push({ item, current_price: currentPrice, predicted_price: Math.round(nextPrice), predicted_change: change.toFixed(2), trend: slope > 0 ? 'up' : 'down', confidence: Math.min(0.9, prices.length / 100), based_on_hours: (prices.length * 5) / 60 });
    });
    predictions.sort((a, b) => Math.abs(b.predicted_change) - Math.abs(a.predicted_change));
    return predictions.slice(0, 10);
}
function generateTradingSignals(patterns, data) {
    const signals = [];
    for (const pattern of patterns) {
        let signal = null;
        if (pattern.type === 'surge' && pattern.change > 15) { signal = { type: 'SELL', item: pattern.item, reason: `Price surged ${pattern.change}% - potential peak`, confidence: Math.min(pattern.confidence * 1.2, 0.95), current_price: pattern.to, action: `Consider selling ${pattern.item} at ${pattern.to}p` }; }
        else if (pattern.type === 'crash' && pattern.change < -15) { signal = { type: 'BUY', item: pattern.item, reason: `Price crashed ${Math.abs(pattern.change)}% - potential bottom`, confidence: Math.min(pattern.confidence * 1.2, 0.95), current_price: pattern.to, action: `Consider buying ${pattern.item} at ${pattern.to}p` }; }
        else if (pattern.type === 'golden_cross') { signal = { type: 'BUY', item: pattern.item, reason: 'Golden cross detected - bullish signal', confidence: 0.75, current_price: pattern.currentPrice, action: `Uptrend starting for ${pattern.item}` }; }
        if (signal && signal.confidence > 0.7) { signals.push(signal); }
    }
    const arbitrage = findArbitrageOpportunities(data); signals.push(...arbitrage);
    return signals;
}
function findArbitrageOpportunities(data) {
    const opportunities = []; const itemGroups = {}; data.forEach(row => { if (!itemGroups[row.item]) { itemGroups[row.item] = []; } itemGroups[row.item].push(row); });
    Object.entries(itemGroups).forEach(([item, rows]) => {
        const latestRow = rows.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
        if (latestRow && latestRow.buy_median != null && latestRow.sell_median != null) {
            const profit = latestRow.buy_median - latestRow.sell_median;
            if (profit > 5) {
                const profitPct = latestRow.sell_median === 0 ? Infinity : (profit / latestRow.sell_median) * 100;
                if (profitPct > 15) {
                    opportunities.push({ type: 'ARBITRAGE', item: item, buy_price: latestRow.sell_median, sell_price: latestRow.buy_median, profit, profit_pct: profitPct, confidence: 0.9, action: `Arbitrage: Buy at ${latestRow.sell_median}p, sell at ${latestRow.buy_median}p for ${profit}p profit` });
                }
            }
        }
    });
    return opportunities;
}