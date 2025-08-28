export default {
    async scheduled(event, env, ctx) {
        // 1) Run original health checks
        const health = await checkSystemHealth(env);

        // 2) Ingest worldstate (fissures + Baro)
        ctx.waitUntil(pollWorldstate(env));

        // 3) Alert if unhealthy
        if (!health.healthy) {
            await sendDiscordAlert(env, {
                title: "⚠️ System Health Alert",
                description: `Issue detected: ${health.issue}`,
                color: 15158332
            });
        }
    },

    async fetch(request, env) {
        const cors = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        };
        if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });

        const url = new URL(request.url);

        if (url.pathname === '/worldstate') {
            const latest = await env.MARKET_CACHE.get('fissures_latest');
            const flag = await env.MARKET_CACHE.get('baro_flag');
            return new Response(JSON.stringify({
                fissures: latest ? JSON.parse(latest) : null,
                baro: flag ? JSON.parse(flag) : null
            }), { headers: { ...cors, 'Content-Type': 'application/json' } });
        }

        const health = await checkSystemHealth(env);
        return new Response(JSON.stringify(health), { headers: { ...cors, 'Content-Type': 'application/json' } });
    }
};

async function sendDiscordAlert(env, alert) {
    if (!env.DISCORD_WEBHOOK) return;
    const embed = {
        title: alert.title || "🧙‍♂️ Gandalf Alert",
        description: alert.description || "",
        color: alert.color || 3447003,
        fields: alert.fields || [],
        timestamp: new Date().toISOString(),
        footer: { text: "Gandalf System Monitor" }
    };
    try {
        await fetch(env.DISCORD_WEBHOOK, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ embeds: [embed] })
        });
    } catch (e) { console.error('Discord webhook error:', e); }
}

async function checkSystemHealth(env) {
    const latestData = await env.MARKET_CACHE.get('latest_snapshot');
    if (!latestData) return { healthy: false, issue: "No recent data found in cache" };
    let data; try { data = JSON.parse(latestData); } catch { return { healthy: false, issue: "Corrupted cache data" }; }
    const ts = Array.isArray(data) && data[0]?.timestamp ? new Date(data[0].timestamp).getTime() : (data.timestamp ? new Date(data.timestamp).getTime() : Date.now());
    const age = Date.now() - ts;
    if (age > 30 * 60 * 1000) return { healthy: false, issue: `Data is ${Math.round(age / 60000)} minutes old` };

    try {
        const resp = await fetch(`${env.SUPABASE_URL}/rest/v1/market_data?limit=1`, {
            headers: { 'apikey': env.SUPABASE_ANON_KEY, 'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}` }
        });
        if (!resp.ok) return { healthy: false, issue: "Supabase connection failed" };
    } catch {
        return { healthy: false, issue: "Database connection error" };
    }
    return { healthy: true, lastDataAge: Math.round(age / 60000), message: "All systems operational" };
}

async function pollWorldstate(env) {
    try {
        const fissuresResp = await fetch('https://api.warframestat.us/pc/fissures', {
            headers: { 'User-Agent': 'Gandalf/1.0', 'Accept': 'application/json' },
        });
        const fissures = fissuresResp.ok ? await fissuresResp.json() : [];
        const active = Array.isArray(fissures) ? fissures.filter(f => f && f.active) : [];
        const byEra = { Lith: 0, Meso: 0, Neo: 0, Axi: 0 };
        for (const f of active) {
            const t = f.tier;
            if (t && byEra.hasOwnProperty(t)) byEra[t] += 1;
        }

        const baroResp = await fetch('https://api.warframestat.us/pc/voidTrader', {
            headers: { 'User-Agent': 'Gandalf/1.0', 'Accept': 'application/json' },
        });
        let baro_active = false; let baro_location = null; let baro_eta = null;
        if (baroResp.ok) {
            const baro = await baroResp.json();
            baro_active = !!baro?.active;
            baro_location = baro?.location || null;
            // Prefer remaining string if present
            baro_eta = baro?.endString || baro?.startString || null;
        }

        await env.MARKET_CACHE.put('fissures_latest', JSON.stringify(byEra), { expirationTtl: 1200 });
        await env.MARKET_CACHE.put('baro_flag', JSON.stringify({ active: baro_active, location: baro_location, eta: baro_eta }), { expirationTtl: 1200 });

        // Persist snapshot into Supabase via SECURITY DEFINER RPC
        const body = {
            fissures: byEra,
            baro_active,
            baro_location,
            baro_eta
        };
        const resp = await fetch(`${env.SUPABASE_URL}/rest/v1/rpc/insert_worldstate_snapshot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'apikey': env.SUPABASE_ANON_KEY,
                'Authorization': `Bearer ${env.SUPABASE_ANON_KEY}`
            },
            body: JSON.stringify(body)
        });
        if (!resp.ok) {
            const txt = await resp.text().catch(() => '');
            console.error('insert_worldstate_snapshot failed:', resp.status, txt);
        }
    } catch (e) {
        console.error('pollWorldstate error:', e);
    }
}