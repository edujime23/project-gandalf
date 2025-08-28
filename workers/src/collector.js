// src/collector.js

export default {
    async scheduled(event, env, ctx) {
        console.log('Collector: scheduled collection…');
        ctx.waitUntil(collectMarketData(env, {}));
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
                status: ts ? 'healthy' : 'no_cache_data',
                last_collection_timestamp: ts,
                last_collection_age_minutes: ageM
            }), { headers: { ...cors, 'Content-Type': 'application/json' } });
        }

        if (url.pathname === '/collect') {
            // Optional controls for manual runs (HTTP has ~30s wall cap; keep it small)
            // ?limit=30           -> limit number of items processed
            // ?shards=5&shard=2   -> process 1 shard of the list
            // ?items=a,b,c        -> override items list completely
            const params = url.searchParams;
            const limit = parseInt(params.get('limit') || '', 10);
            const shards = parseInt(params.get('shards') || '', 10);
            const shard = parseInt(params.get('shard') || '', 10);
            const itemsOverride = (params.get('items') || '')
                .split(',')
                .map(s => s.trim())
                .filter(Boolean);

            ctx.waitUntil(collectMarketData(env, {
                limit: Number.isFinite(limit) ? limit : undefined,
                shards: Number.isFinite(shards) ? shards : undefined,
                shard: Number.isFinite(shard) ? shard : undefined,
                itemsOverride: itemsOverride.length ? itemsOverride : undefined
            }));
            return new Response('Collection started in background.', { status: 202, headers: cors });
        }

        return new Response('Gandalf Collector v1.5 is online.', { status: 200, headers: cors });
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
        const limit = Number.parseInt(env.TRACK_LIMIT || '500', 10);
        const url = `${env.SUPABASE_URL}/rest/v1/tracked_items?select=item,score,active&active=eq.true&order=score.desc&limit=${limit}`;
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

    // fallback
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

function chunk(arr, size) {
    const out = [];
    for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
    return out;
}

function defaultShardIndex() {
    // round-robin shard by 5-minute window
    const window = Math.floor(Date.now() / (5 * 60 * 1000));
    return window % 5; // default 5 shards if not specified
}

async function collectOne(item) {
    const data = await fetchWithRetry(
        `https://api.warframe.market/v1/items/${encodeURIComponent(item)}/orders`,
        {
            headers: {
                'User-Agent': 'Gandalf/1.0',
                'Accept': 'application/json',
                'platform': 'pc',
                'language': 'en'
            }
        },
        3, 800
    );
    if (!data || !data.payload || !Array.isArray(data.payload.orders)) {
        console.error(`No orders for ${item}`);
        return null;
    }
    const orders = data.payload.orders.filter(o => o?.user?.status === 'ingame');
    const buys = orders.filter(o => o.order_type === 'buy').map(o => o.platinum).filter(v => typeof v === 'number');
    const sells = orders.filter(o => o.order_type === 'sell').map(o => o.platinum).filter(v => typeof v === 'number');

    const buyMed = median(buys);
    const sellMed = median(sells);

    return {
        timestamp: new Date().toISOString(),
        item,
        buy_orders: buys.length,
        sell_orders: sells.length,
        buy_median: buyMed,
        sell_median: sellMed,
        spread: (buyMed != null && sellMed != null) ? (sellMed - buyMed) : null
    };
}

// Tuning: keep below WM 3 rps
const BATCH_SIZE = 3;        // 3 concurrent requests
const BATCH_DELAY_MS = 1100; // ~1s between batches
const FLUSH_EVERY = 20;      // write partial results periodically

async function collectMarketData(env, opts = {}) {
    console.log('ENV CHECK', {
        has_SUPABASE_URL: !!env.SUPABASE_URL,
        has_ANON: !!env.SUPABASE_ANON_KEY,
        has_SERVICE: !!env.SUPABASE_SERVICE_ROLE,
        has_KV: !!env.MARKET_CACHE
    });

    let items = [];
    if (opts.itemsOverride?.length) {
        items = opts.itemsOverride;
    } else {
        items = await getTrackedItems(env);
    }

    if (opts.limit && opts.limit > 0) {
        items = items.slice(0, opts.limit);
    }

    if (opts.shards && opts.shards > 1) {
        const shardIdx = Number.isFinite(opts.shard) ? opts.shard : defaultShardIndex();
        items = items.filter((_, i) => (i % opts.shards) === shardIdx);
        console.log(`Sharding: ${shardIdx + 1}/${opts.shards} -> ${items.length} items`);
    }

    const pending = [];
    const kvSample = []; // keep last ~50 records for /health

    let processed = 0;

    for (const batch of chunk(items, BATCH_SIZE)) {
        const recs = await Promise.all(batch.map(collectOne));
        for (const r of recs) {
            if (!r) continue;
            pending.push(r);
            kvSample.push(r);
            if (kvSample.length > 50) kvSample.splice(0, kvSample.length - 50); // keep tail 50
        }
        processed += batch.length;

        if (pending.length >= FLUSH_EVERY) {
            await storeInSupabase(env, pending.splice(0));
        }
        await sleep(BATCH_DELAY_MS);
    }

    if (pending.length) await storeInSupabase(env, pending);

    // Snapshot for /health (guaranteed to have data if anything was collected)
    await env.MARKET_CACHE.put('latest_snapshot', JSON.stringify(kvSample), { expirationTtl: 300 });

    console.log(`Collected ${processed} items; wrote snapshot(${kvSample.length}).`);
}

async function storeInSupabase(env, data) {
    if (!data?.length) return;
    if (!env.SUPABASE_URL) {
        console.error('SUPABASE_URL missing');
        return;
    }
    const auth = env.SUPABASE_SERVICE_ROLE || env.SUPABASE_ANON_KEY;
    try {
        const resp = await fetch(`${env.SUPABASE_URL}/rest/v1/market_data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'apikey': env.SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${auth}`, // service role recommended (bypasses RLS)
                'Prefer': 'return=representation'   // helpful for logging rows inserted
            },
            body: JSON.stringify(data)
        });
        const text = await resp.text();
        if (!resp.ok) {
            console.error('Supabase write error:', resp.status, text);
        } else {
            let rows = 'unknown';
            try { rows = JSON.parse(text).length; } catch { }
            console.log(`Supabase wrote ${rows} rows`);
        }
    } catch (e) {
        console.error('Supabase store error:', e);
    }
}