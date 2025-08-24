export default {
    async scheduled(event, env, ctx) {
        console.log('Analyzer: running scheduled cycle...');
        ctx.waitUntil(runAnalysis(env, { force: false }));
    },

    async fetch(request, env) {
        const url = new URL(request.url);
        const headers = { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' };

        if (url.pathname === '/' || url.pathname === '/api/health') {
            const signals = env.PATTERN_CACHE ? await env.PATTERN_CACHE.get('latest_signals') : null;
            return new Response(JSON.stringify({ status: 'healthy', cached: !!signals }), { headers });
        }

        if (url.pathname === '/api/run-now') {
            const force = url.searchParams.get('force') === '1';
            try {
                assertEnv(env);
                const result = await runAnalysis(env, { force });
                return new Response(JSON.stringify({ ok: true, ...result }), { headers });
            } catch (e) {
                console.error('run-now error:', e);
                return new Response(JSON.stringify({ ok: false, error: e.message }), { status: 500, headers });
            }
        }

        if (url.pathname === '/api/debug-config') {
            const cfg = {
                has_service_role: !!env.SUPABASE_SERVICE_ROLE,
                has_anon: !!env.SUPABASE_ANON_KEY,
                supabase_url_set: !!env.SUPABASE_URL,
                has_kv: !!env.PATTERN_CACHE
            };
            return new Response(JSON.stringify(cfg), { headers });
        }

        if (url.pathname === '/api/patterns') {
            const patterns = env.PATTERN_CACHE ? await env.PATTERN_CACHE.get('latest_patterns') : null;
            return new Response(patterns || '[]', { headers });
        }
        if (url.pathname === '/api/predictions') {
            const preds = env.PATTERN_CACHE ? await env.PATTERN_CACHE.get('latest_predictions') : null;
            return new Response(preds || '[]', { headers });
        }
        if (url.pathname === '/api/signals') {
            const sigs = env.PATTERN_CACHE ? await env.PATTERN_CACHE.get('latest_signals') : null;
            return new Response(sigs || '[]', { headers });
        }

        return new Response(JSON.stringify({ error: 'Not found' }), { status: 404, headers });
    }
};

function assertEnv(env) {
    const missing = [];
    if (!env.SUPABASE_URL) missing.push('SUPABASE_URL');
    if (!env.SUPABASE_ANON_KEY) missing.push('SUPABASE_ANON_KEY');
    if (!env.PATTERN_CACHE) missing.push('PATTERN_CACHE (KV binding)');
    if (missing.length) throw new Error(`Missing env: ${missing.join(', ')}`);
}

async function runAnalysis(env, { force = false } = {}) {
    const data = await fetchRecentData(env);
    if (!data || data.length < 50) {
        console.log('Analyzer: not enough data; skipping.');
        return { patterns: 0, predictions: 0, reason: 'not_enough_data' };
    }
    const patterns = findPatterns(data);
    const predictions = generatePredictions(data);
    await storeAnalysisResults(env, patterns, predictions, data, { force });
    console.log(`Analyzer: patterns=${patterns.length}, predictions=${predictions.length}`);
    return { patterns: patterns.length, predictions: predictions.length };
}

async function fetchRecentData(env) {
    const url = `${env.SUPABASE_URL}/rest/v1/market_data?order=timestamp.desc&limit=2500`;
    const resp = await fetch(url, {
        headers: { 'apikey': env.SUPABASE_ANON_KEY, 'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}` }
    });
    if (!resp.ok) {
        const txt = await resp.text().catch(() => '');
        console.error('fetchRecentData failed:', resp.status, txt);
        return [];
    }
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
        const txt = await resp.text().catch(() => '');
        console.error(`Supabase write failed [${label}] ${resp.status}: ${txt}`);
    } else {
        console.log(`Supabase write ok [${label}] count=${Array.isArray(payload) ? payload.length : 1}`);
    }
}

async function storeAnalysisResults(env, patterns, predictions, data, { force = false } = {}) {
    const signals = generateTradingSignals(patterns, data, { force });

    if (env.PATTERN_CACHE) {
        await env.PATTERN_CACHE.put('latest_patterns', JSON.stringify(patterns), { expirationTtl: 3600 });
        await env.PATTERN_CACHE.put('latest_predictions', JSON.stringify(predictions), { expirationTtl: 3600 });
        await env.PATTERN_CACHE.put('latest_signals', JSON.stringify(signals), { expirationTtl: 3600 });
    }

    for (const s of signals) {
        if (s.confidence > 0.8) {
            const color = s.type === 'BUY' ? 3066993 : (s.type === 'SELL' ? 15158332 : 16776960);
            await sendDiscordAlert(env, {
                title: `${s.type} • ${s.item.replace(/_/g, ' ').toUpperCase()}`,
                description: s.reason || 'Signal',
                color,
                fields: [
                    { name: 'Confidence', value: `${Math.round((s.confidence ?? 0) * 100)}%`, inline: true }
                ]
            });
            await new Promise(r => setTimeout(r, 1200));
        }
    }

    const key = env.SUPABASE_SERVICE_ROLE || env.SUPABASE_ANON_KEY || '';
    if (!key) console.warn('No Supabase key available in analyzer env');

    if (patterns.length && env.SUPABASE_URL) {
        const payload = patterns.map(p => ({ pattern_type: p.type, pattern_data: p, confidence: p.confidence ?? null }));
        await postJSON(`${env.SUPABASE_URL}/rest/v1/discovered_patterns`, key, payload, 'patterns');
    }

    if (predictions.length && env.SUPABASE_URL) {
        const payload = predictions.map(p => ({
            item: p.item, predicted_at: new Date().toISOString(),
            current_price: p.current_price, predicted_price: p.predicted_price, confidence: p.confidence ?? null
        }));
        await postJSON(`${env.SUPABASE_URL}/rest/v1/predictions`, key, payload, 'predictions');
    }

    if (signals.length && env.SUPABASE_URL) {
        const payload = signals.map(s => ({
            source: 'analyzer', item: s.item, type: s.type, reason: s.reason ?? null,
            confidence: s.confidence ?? null, current_price: s.current_price ?? null
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
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ embeds: [embed] })
        });
    } catch (e) {
        console.error('Discord webhook error:', e);
    }
}

// Patterns and predictors (unchanged logic, trimmed for brevity)
function movingAverage(arr) { if (!arr.length) return 0; return arr.reduce((a, b) => a + b, 0) / arr.length; }
function calculateCorrelation(a, b) {
    const p1 = [], p2 = [];
    a.forEach(d1 => {
        const m = b.find(d2 => Math.abs(new Date(d1.timestamp) - new Date(d2.timestamp)) < 300000);
        if (m && d1.sell_median != null && m.sell_median != null) { p1.push(d1.sell_median); p2.push(m.sell_median); }
    });
    if (p1.length < 10) return 0;
    const n = p1.length, sum1 = p1.reduce((x, y) => x + y, 0), sum2 = p2.reduce((x, y) => x + y, 0);
    const sum1Sq = p1.reduce((x, y) => x + y * y, 0), sum2Sq = p2.reduce((x, y) => x + y * y, 0);
    const pSum = p1.reduce((s, v, i) => s + v * p2[i], 0);
    const num = pSum - (sum1 * sum2 / n);
    const den = Math.sqrt((sum1Sq - sum1 * sum1 / n) * (sum2Sq - sum2 * sum2 / n));
    return den === 0 ? 0 : num / den;
}
function findPatterns(data) {
    const patterns = [], byItem = {};
    data.forEach(r => { (byItem[r.item] ||= []).push(r); });
    for (const [item, arr] of Object.entries(byItem)) {
        if (arr.length < 10) continue;
        arr.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        const sells = arr.map(p => p.sell_median).filter(v => v != null);
        if (sells.length < 2) continue;
        const cur = sells[sells.length - 1], old = sells[0];
        const ch = old === 0 ? 0 : ((cur - old) / old) * 100;
        if (Math.abs(ch) > 10) patterns.push({ type: ch > 0 ? 'surge' : 'crash', item, change: +ch.toFixed(2), from: old, to: cur, confidence: Math.min(1, arr.length / 288) });
        if (sells.length >= 20) {
            const ma5 = movingAverage(sells.slice(-5)), ma20 = movingAverage(sells.slice(-20));
            const p5 = movingAverage(sells.slice(-6, -1)), p20 = movingAverage(sells.slice(-21, -1));
            if (p5 < p20 && ma5 > ma20) patterns.push({ type: 'golden_cross', item, signal: 'bullish', ma5, ma20, currentPrice: cur, confidence: 0.75 });
        }
    }
    const items = Object.keys(byItem);
    for (let i = 0; i < items.length; i++) {
        for (let j = i + 1; j < items.length; j++) {
            const c = calculateCorrelation(byItem[items[i]], byItem[items[j]]);
            if (Math.abs(c) > 0.7) patterns.push({ type: 'correlation', item1: items[i], item2: items[j], strength: c, direction: c > 0 ? 'positive' : 'negative', confidence: Math.abs(c) });
        }
    }
    return patterns;
}
function generatePredictions(data) {
    const preds = [], byItem = {}; data.forEach(r => { (byItem[r.item] ||= []).push(r); });
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
        preds.push({ item, current_price: cur, predicted_price: Math.round(next), predicted_change: +ch.toFixed(2), trend: slope > 0 ? 'up' : 'down', confidence: Math.min(0.9, sells.length / 100) });
    }
    preds.sort((a, b) => Math.abs(b.predicted_change) - Math.abs(a.predicted_change));
    return preds.slice(0, 10);
}
function generateTradingSignals(patterns, data, { force = false } = {}) {
    const gate = force ? 0.1 : 0.7;
    const signals = [];
    for (const p of patterns) {
        let s = null;
        if (p.type === 'surge' && p.change > 15) s = { type: 'SELL', item: p.item, reason: `Price surged ${p.change}%`, confidence: Math.min((p.confidence ?? 0.7) * 1.1, 0.95), current_price: p.to };
        else if (p.type === 'crash' && p.change < -15) s = { type: 'BUY', item: p.item, reason: `Price crashed ${Math.abs(p.change)}%`, confidence: Math.min((p.confidence ?? 0.7) * 1.1, 0.95), current_price: p.to };
        else if (p.type === 'golden_cross') s = { type: 'BUY', item: p.item, reason: 'Golden cross detected', confidence: 0.75, current_price: p.currentPrice };
        if (s && s.confidence > gate) signals.push(s);
    }
    return signals;
}