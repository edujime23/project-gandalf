// src/collector.js

export default {
    async scheduled(event, env, ctx) {
        console.log('Collector: scheduled collection…');
        // Kick off the chain from cursor=0
        ctx.waitUntil(startChain(env));
    },

    async fetch(request, env, ctx) {
        const url = new URL(request.url);
        const cors = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, x-chain-token'
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
            // Self-chain authorization
            const chainMode = url.searchParams.get('chain') === '1';
            const token = request.headers.get('x-chain-token') || url.searchParams.get('token');
            const isChainAuthorized = chainMode && env.CHAIN_TOKEN && token === env.CHAIN_TOKEN;

            // Optional manual controls (small tests only; no chaining without token)
            const params = url.searchParams;
            const limit = parseInt(params.get('limit') || '', 10);        // manual slice
            const shards = parseInt(params.get('shards') || '', 10);      // manual shard count
            const shard = parseInt(params.get('shard') || '', 10);        // manual shard index
            const itemsOverride = (params.get('items') || '')
                .split(',').map(s => s.trim()).filter(Boolean);

            const cursor = Math.max(0, parseInt(params.get('cursor') || '0', 10));
            const session = params.get('session') || randomId();
            const maxPerInvoke = Number(params.get('max') || env.MAX_ITEMS_PER_INVOCATION || 35);

            const result = await collectMarketDataSegment(env, {
                cursor,
                maxPerInvoke,
                itemsOverride: itemsOverride.length ? itemsOverride : undefined,
                manualLimit: Number.isFinite(limit) ? limit : undefined,
                manualShards: Number.isFinite(shards) ? { shards, shard: Number.isFinite(shard) ? shard : 0 } : undefined,
            });

            // Chain next segment only if authorized and more items remain
            if (isChainAuthorized && result.nextCursor < result.totalItems) {
                const nextUrl = new URL((env.SELF_URL || '').trim() || url.origin + '/collect');
                nextUrl.pathname = '/collect';
                nextUrl.searchParams.set('chain', '1');
                nextUrl.searchParams.set('cursor', String(result.nextCursor));
                nextUrl.searchParams.set('session', session);
                nextUrl.searchParams.set('max', String(maxPerInvoke));
                // if itemsOverride present, forward it
                if (itemsOverride.length) nextUrl.searchParams.set('items', itemsOverride.join(','));

                console.log(`Chaining next segment: ${result.nextCursor}/${result.totalItems}`);
                ctx.waitUntil(fetch(nextUrl.toString(), {
                    headers: { 'x-chain-token': env.CHAIN_TOKEN, 'User-Agent': 'Gandalf-Collector/1.6' }
                }));
            } else {
                if (!isChainAuthorized && result.nextCursor < result.totalItems) {
                    console.log(`Manual run processed ${result.nextCursor}/${result.totalItems}. Use scheduled/chain to complete.`);
                }
            }

            return new Response(JSON.stringify({
                ok: true,
                processed: result.processedCount,
                cursor: result.nextCursor,
                totalItems: result.totalItems,
                chained: isChainAuthorized && result.nextCursor < result.totalItems
            }), { status: 202, headers: { ...cors, 'Content-Type': 'application/json' } });
        }

        return new Response('Gandalf Collector v1.6 is online.', { status: 200, headers: cors });
    }
};

function randomId() {
    return Math.random().toString(16).slice(2) + Date.now().toString(36);
}

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function fetchWithRetry(url, opts = {}, retries = 3, backoffMs = 700) {
    for (let i = 0; i < retries; i++) {
        try {
            const resp = await fetch(url, opts);
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

// Page through ALL active tracked items
async function getTrackedItems(env) {
    const cached = await env.MARKET_CACHE.get('tracked_items_all');
    if (cached) { try { return JSON.parse(cached); } catch { } }

    const items = [];
    try {
        const pageSize = Number(env.ITEMS_PAGE_SIZE || 1000);
        const base = `${env.SUPABASE_URL}/rest/v1/tracked_items?select=item&active=eq.true&order=item.asc`;
        let offset = 0;
        while (true) {
            const resp = await fetch(base, {
                headers: {
                    'apikey': env.SUPABASE_ANON_KEY,
                    'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}`,
                    'Range-Unit': 'items',
                    'Range': `${offset}-${offset + pageSize - 1}`
                }
            });
            if (!resp.ok) {
                console.error('tracked_items page error:', resp.status);
                break;
            }
            const rows = await resp.json();
            if (!Array.isArray(rows) || rows.length === 0) break;
            for (const r of rows) if (r?.item) items.push(r.item);
            if (rows.length < pageSize) break;
            offset += pageSize;
        }
        if (items.length) {
            await env.MARKET_CACHE.put('tracked_items_all', JSON.stringify(items), { expirationTtl: 600 });
            return items;
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
    const window = Math.floor(Date.now() / (5 * 60 * 1000));
    return window % 5;
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

// Segment runner (keeps subrequests under limit)
async function collectMarketDataSegment(env, opts = {}) {
    const BATCH_SIZE = Number(env.BATCH_SIZE || 5);            // per-batch concurrency
    const BATCH_PAUSE_MS = Number(env.BATCH_PAUSE_MS || 5000);     // 5s between batches
    const cursor = Number(opts.cursor || 0);
    const maxPerInvoke = Number(opts.maxPerInvoke || 35);        // keep under CF subrequest limit
    const manualLimit = Number(opts.manualLimit || 0);

    // Build item list
    let items = [];
    if (opts.itemsOverride?.length) items = opts.itemsOverride.slice();
    else items = await getTrackedItems(env);

    if (opts.manualShards && opts.manualShards.shards > 1) {
        const { shards, shard } = opts.manualShards;
        items = items.filter((_, i) => i % shards === shard);
    }
    if (manualLimit > 0) items = items.slice(0, manualLimit);

    const totalItems = items.length;
    if (cursor >= totalItems) {
        console.log(`Nothing to do. cursor=${cursor} >= totalItems=${totalItems}`);
        return { processedCount: 0, nextCursor: cursor, totalItems };
    }

    // Segment slice (subrequest-safe)
    const end = Math.min(cursor + maxPerInvoke, totalItems);
    const segment = items.slice(cursor, end);

    console.log(`Segment: cursor=${cursor} end=${end} (size=${segment.length}) of total=${totalItems}`);

    const rows = [];
    let processed = 0;

    for (const batch of chunk(segment, BATCH_SIZE)) {
        const recs = await Promise.all(batch.map(collectOne));
        for (const r of recs) if (r) rows.push(r);
        processed += batch.length;
        await sleep(BATCH_PAUSE_MS); // polite pause between batches
    }

    if (rows.length) {
        await storeInSupabase(env, rows);
    }

    // Small snapshot for /health
    try {
        await env.MARKET_CACHE.put('latest_snapshot', JSON.stringify(rows.slice(-50)), { expirationTtl: 300 });
    } catch (e) {
        console.error('KV snapshot error:', e);
    }

    console.log(`Segment done: processed=${processed}, nextCursor=${end}/${totalItems}`);
    return { processedCount: processed, nextCursor: end, totalItems };
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
                'Authorization': `Bearer ${auth}`,
                'Prefer': 'return=minimal,resolution=merge-duplicates'
            },
            body: JSON.stringify(data)
        });
        if (!resp.ok) {
            const text = await resp.text().catch(() => '');
            console.error('Supabase store error:', text || resp.status);
        } else {
            console.log(`Supabase wrote ${data.length} rows`);
        }
    } catch (e) {
        console.error('Supabase store error:', e);
    }
}

// Start the chain from a scheduled event
async function startChain(env) {
    if (!env.SELF_URL || !env.CHAIN_TOKEN) {
        console.error('Missing SELF_URL or CHAIN_TOKEN; running a single segment only.');
        // Fallback: run one segment only (no chaining)
        await collectMarketDataSegment(env, { cursor: 0, maxPerInvoke: Number(env.MAX_ITEMS_PER_INVOCATION || 35) });
        return;
    }
    const session = randomId();
    const nextUrl = new URL(env.SELF_URL);
    nextUrl.pathname = '/collect';
    nextUrl.searchParams.set('chain', '1');
    nextUrl.searchParams.set('cursor', '0');
    nextUrl.searchParams.set('session', session);
    nextUrl.searchParams.set('max', String(Number(env.MAX_ITEMS_PER_INVOCATION || 35)));
    console.log(`Starting chain session=${session}`);
    try {
        await fetch(nextUrl.toString(), {
            headers: { 'x-chain-token': env.CHAIN_TOKEN, 'User-Agent': 'Gandalf-Collector/1.6' }
        });
    } catch (e) {
        console.error('Chain start error:', e);
    }
}