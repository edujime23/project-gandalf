// analyzer.js - v1.4 - Persist signals/patterns/predictions with service role + error logging

export default {
    async scheduled(event, env, ctx) {
        console.log('Analyzer: running full cycle...');
        try {
            const data = await fetchRecentData(env);
            if (!data || data.length < 50) {
                console.log('Analyzer: not enough data; skipping.');
                return;
            }
            const patterns = findPatterns(data);
            const predictions = generatePredictions(data);
            await storeAnalysisResults(env, patterns, predictions, data);
            console.log(`Analyzer: patterns=${patterns.length}, predictions=${predictions.length}`);
        } catch (err) {
            console.error('Analyzer scheduled error:', err);
        }
    },

    async fetch(request, env) {
        const url = new URL(request.url);
        const headers = { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' };

        if (url.pathname === '/' || url.pathname === '/api/health') {
            const signals = await env.PATTERN_CACHE.get('latest_signals');
            return new Response(JSON.stringify({ status: 'healthy', cached: !!signals }), { headers });
        }

        if (url.pathname.startsWith('/api/')) {
            if (url.pathname === '/api/patterns') {
                const patterns = await env.PATTERN_CACHE.get('latest_patterns');
                return new Response(patterns || '[]', { headers });
            }
            if (url.pathname === '/api/predictions') {
                const preds = await env.PATTERN_CACHE.get('latest_predictions');
                return new Response(preds || '[]', { headers });
            }
            if (url.pathname === '/api/signals') {
                const sigs = await env.PATTERN_CACHE.get('latest_signals');
                return new Response(sigs || '[]', { headers });
            }
        }

        return new Response(JSON.stringify({ error: 'Not found' }), { status: 404, headers });
    }
};

// ---------- Helpers ----------
async function fetchRecentData(env) {
    const resp = await fetch(
        `${env.SUPABASE_URL}/rest/v1/market_data?order=timestamp.desc&limit=2500`,
        { headers: { 'apikey': env.SUPABASE_ANON_KEY, 'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}` } }
    );
    if (!resp.ok) return [];
    return await resp.json();
}

async function postJSON(url, key, payload, label) {
    const headers = {
        'Content-Type': 'application/json',
        'apikey': key,
        'Authorization': `Bearer ${key}`,
        'Prefer': 'return=minimal,resolution=merge-duplicates'
    };
    const resp = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) });
    if (!resp.ok) {
        const txt = await resp.text();
        console.error(`Supabase write failed [${label}] ${resp.status}: ${txt}`);
    } else {
        console.log(`Supabase write ok [${label}] count=${Array.isArray(payload) ? payload.length : 1}`);
    }
}

async function storeAnalysisResults(env, patterns, predictions, data) {
    const signals = generateTradingSignals(patterns, data);

    // KV cache
    await env.PATTERN_CACHE.put('latest_patterns', JSON.stringify(patterns), { expirationTtl: 3600 });
    await env.PATTERN_CACHE.put('latest_predictions', JSON.stringify(predictions), { expirationTtl: 3600 });
    await env.PATTERN_CACHE.put('latest_signals', JSON.stringify(signals), { expirationTtl: 3600 });

    // Discord alerts
    for (const s of signals) {
        if (s.confidence > 0.8) {
            const color = s.type === 'BUY' ? 3066993 : (s.type === 'SELL' ? 15158332 : 16776960);
            await sendDiscordAlert(env, {
                title: `${s.type} • ${s.item.replace(/_/g, ' ').toUpperCase()}`,
                description: s.reason || 'Signal',
                color,
                fields: [
                    { name: 'Confidence', value: `${Math.round((s.confidence ?? 0) * 100)}%`, inline: true },
                    ...(s.profit ? [{ name: 'Potential Profit', value: `${s.profit}p (${(s.profit_pct ?? 0).toFixed(1)}%)`, inline: true }] : [])
                ]
            });
            await new Promise(r => setTimeout(r, 1200));
        }
    }

    // DB writes using service key if present
    const key = env.SUPABASE_SERVICE_ROLE || env.SUPABASE_ANON_KEY;

    // 1) Patterns
    if (patterns.length) {
        const payload = patterns.map(p => ({
            pattern_type: p.type,
            pattern_data: p,
            confidence: p.confidence ?? null
        }));
        await postJSON(`${env.SUPABASE_URL}/rest/v1/discovered_patterns`, key, payload, 'patterns');
    }

    // 2) Predictions
    if (predictions.length) {
        const payload = predictions.map(p => ({
            item: p.item,
            predicted_at: new Date().toISOString(),
            current_price: p.current_price,
            predicted_price: p.predicted_price,
            confidence: p.confidence ?? null
        }));
        await postJSON(`${env.SUPABASE_URL}/rest/v1/predictions`, key, payload, 'predictions');
    }

    // 3) Signals
    if (signals.length) {
        const payload = signals.map(s => ({
            source: 'analyzer',
            item: s.item,
            type: s.type,
            reason: s.reason ?? null,
            confidence: s.confidence ?? null,
            current_price: s.current_price ?? null,
            extra: {
                action: s.action ?? null,
                profit: s.profit ?? null,
                profit_pct: s.profit_pct ?? null
            }
        }));
        await postJSON(`${env.SUPABASE_URL}/rest/v1/signals`, key, payload, 'signals');
    }
}

async function sendDiscordAlert(env, alert) {
    if (!env.DISCORD_WEBHOOK) return;
    const embed = {
        title: alert.title || 'Gandalf Alert',
        description: alert.description || '',
        color: alert.color || 3447003,
        fields: alert.fields || [],
        timestamp: new Date().toISOString(),
        footer: { text: 'Gandalf Analyzer' }
    };
    try {
        await fetch(env.DISCORD_WEBHOOK, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ embeds: [embed] })
        });
    } catch (e) {
        console.error('Discord webhook error:', e);
    }
}

// ---- Analysis helpers (unchanged core) ----
function movingAverage(arr) { if (!arr.length) return 0; return arr.reduce((a, b) => a + b, 0) / arr.length; }

function calculateCorrelation(set1, set2) {
    const p1 = [], p2 = [];
    set1.forEach(d1 => {
        const match = set2.find(d2 => Math.abs(new Date(d1.timestamp) - new Date(d2.timestamp)) < 300000);
        if (match && d1.sell_median != null && match.sell_median != null) { p1.push(d1.sell_median); p2.push(match.sell_median); }
    });
    if (p1.length < 10) return 0;
    const n = p1.length, sum1 = p1.reduce((a, b) => a + b, 0), sum2 = p2.reduce((a, b) => a + b, 0);
    const sum1Sq = p1.reduce((a, b) => a + b * b, 0), sum2Sq = p2.reduce((a, b) => a + b * b, 0);
    const pSum = p1.reduce((s, v, i) => s + v * p2[i], 0);
    const num = pSum - (sum1 * sum2 / n);
    const den = Math.sqrt((sum1Sq - sum1 * sum1 / n) * (sum2Sq - sum2 * sum2 / n));
    return den === 0 ? 0 : num / den;
}

function findPatterns(data) {
    const patterns = []; const byItem = {};
    data.forEach(r => { (byItem[r.item] ||= []).push(r); });

    for (const [item, arr] of Object.entries(byItem)) {
        if (arr.length < 10) continue;
        arr.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        const sells = arr.map(p => p.sell_median).filter(v => v != null);
        if (sells.length < 2) continue;

        const cur = sells[sells.length - 1], old = sells[0];
        const ch = old === 0 ? 0 : ((cur - old) / old) * 100;
        if (Math.abs(ch) > 10) {
            patterns.push({ type: ch > 0 ? 'surge' : 'crash', item, change: Number(ch.toFixed(2)), from: old, to: cur, confidence: Math.min(1, arr.length / 288) });
        }

        if (sells.length >= 20) {
            const ma5 = movingAverage(sells.slice(-5)), ma20 = movingAverage(sells.slice(-20));
            const prev5 = movingAverage(sells.slice(-6, -1)), prev20 = movingAverage(sells.slice(-21, -1));
            if (prev5 < prev20 && ma5 > ma20) {
                patterns.push({ type: 'golden_cross', item, signal: 'bullish', ma5, ma20, currentPrice: cur, confidence: 0.75 });
            }
        }
    }

    const items = Object.keys(byItem);
    for (let i = 0; i < items.length; i++) {
        for (let j = i + 1; j < items.length; j++) {
            const c = calculateCorrelation(byItem[items[i]], byItem[items[j]]);
            if (Math.abs(c) > 0.7) {
                patterns.push({ type: 'correlation', item1: items[i], item2: items[j], strength: c, direction: c > 0 ? 'positive' : 'negative', confidence: Math.abs(c) });
            }
        }
    }
    return patterns;
}

function generatePredictions(data) {
    const preds = []; const byItem = {};
    data.forEach(r => { (byItem[r.item] ||= []).push(r); });

    for (const [item, arr] of Object.entries(byItem)) {
        if (arr.length < 20) continue;
        arr.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        const sells = arr.map(p => p.sell_median).filter(v => v != null);
        if (sells.length < 10) continue;

        const n = sells.length, x = [...Array(n).keys()], y = sells;
        const sumX = x.reduce((a, b) => a + b, 0), sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((s, xi, i) => s + xi * y[i], 0), sumX2 = x.reduce((s, xi) => s + xi * xi, 0);
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        if (isNaN(slope) || isNaN(intercept)) continue;

        const next = slope * n + intercept;
        const cur = y[y.length - 1];
        const ch = cur === 0 ? 0 : ((next - cur) / cur) * 100;
        preds.push({ item, current_price: cur, predicted_price: Math.round(next), predicted_change: Number(ch.toFixed(2)), trend: slope > 0 ? 'up' : 'down', confidence: Math.min(0.9, sells.length / 100) });
    }

    preds.sort((a, b) => Math.abs(b.predicted_change) - Math.abs(a.predicted_change));
    return preds.slice(0, 10);
}

function generateTradingSignals(patterns, data) {
    const signals = [];
    for (const p of patterns) {
        let s = null;
        if (p.type === 'surge' && p.change > 15) {
            s = { type: 'SELL', item: p.item, reason: `Price surged ${p.change}%`, confidence: Math.min((p.confidence ?? 0.7) * 1.1, 0.95), current_price: p.to, action: `Consider selling ${p.item} at ${p.to}p` };
        } else if (p.type === 'crash' && p.change < -15) {
            s = { type: 'BUY', item: p.item, reason: `Price crashed ${Math.abs(p.change)}%`, confidence: Math.min((p.confidence ?? 0.7) * 1.1, 0.95), current_price: p.to, action: `Consider buying ${p.item} at ${p.to}p` };
        } else if (p.type === 'golden_cross') {
            s = { type: 'BUY', item: p.item, reason: 'Golden cross detected - bullish', confidence: 0.75, current_price: p.currentPrice, action: `Uptrend for ${p.item}` };
        }
        if (s && s.confidence > 0.7) signals.push(s);
    }
    signals.push(...findArbitrageOpportunities(data));
    return signals;
}

function findArbitrageOpportunities(data) {
    const ops = []; const groups = {};
    data.forEach(r => { (groups[r.item] ||= []).push(r); });
    for (const [item, rows] of Object.entries(groups)) {
        const latest = rows.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
        if (latest && latest.buy_median != null && latest.sell_median != null) {
            const profit = latest.buy_median - latest.sell_median;
            if (profit > 5) {
                const pct = latest.sell_median === 0 ? Infinity : (profit / latest.sell_median) * 100;
                if (pct > 15) {
                    ops.push({ type: 'ARBITRAGE', item, buy_price: latest.sell_median, sell_price: latest.buy_median, profit, profit_pct: pct, confidence: 0.9, action: `Arbitrage: Buy at ${latest.sell_median}p, sell at ${latest.buy_median}p for ${profit}p` });
                }
            }
        }
    }
    return ops;
}