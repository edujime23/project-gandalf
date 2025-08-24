export default {
    async scheduled(event, env, ctx) {
        const health = await checkSystemHealth(env);
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
        const health = await checkSystemHealth(env);
        return new Response(JSON.stringify(health), { headers: { ...cors, 'Content-Type': 'application/json' } });
    }
};

async function sendDiscordAlert(env, alert) {
    if (!env.DISCORD_WEBHOOK) return;
    const embed = { title: alert.title || "🧙‍♂️ Gandalf Alert", description: alert.description, color: alert.color || 3447003, fields: alert.fields || [], timestamp: new Date().toISOString(), footer: { text: "Gandalf System Monitor" } };
    try { await fetch(env.DISCORD_WEBHOOK, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ embeds: [embed] }) }); }
    catch (e) { console.error('Discord webhook error:', e); }
}

async function checkSystemHealth(env) {
    const latestData = await env.MARKET_CACHE.get('latest_snapshot');
    if (!latestData) return { healthy: false, issue: "No recent data found in cache" };
    let data; try { data = JSON.parse(latestData); } catch { return { healthy: false, issue: "Corrupted cache data" }; }
    const ts = Array.isArray(data) && data[0]?.timestamp ? new Date(data[0].timestamp).getTime() : (data.timestamp ? new Date(data.timestamp).getTime() : Date.now());
    const age = Date.now() - ts;
    if (age > 30 * 60 * 1000) return { healthy: false, issue: `Data is ${Math.round(age / 60000)} minutes old` };

    try {
        const resp = await fetch(`${env.SUPABASE_URL}/rest/v1/market_data?limit=1`, { headers: { 'apikey': env.SUPABASE_ANON_KEY } });
        if (!resp.ok) return { healthy: false, issue: "Supabase connection failed" };
    } catch {
        return { healthy: false, issue: "Database connection error" };
    }
    return { healthy: true, lastDataAge: Math.round(age / 60000), message: "All systems operational" };
}