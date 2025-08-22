export default {
    async scheduled(event, env, ctx) {
        // Run every 30 minutes
        const health = await checkSystemHealth(env);

        if (!health.healthy) {
            await sendDiscordAlert(env, {
                title: "⚠️ System Health Alert",
                description: `Issue detected: ${health.issue}`,
                color: 15158332 // Red
            });
        }
    },

    async fetch(request, env) {
        // Manual health check endpoint
        const health = await checkSystemHealth(env);
        return new Response(JSON.stringify(health), {
            headers: { 'Content-Type': 'application/json' }
        });
    }
};

async function sendDiscordAlert(env, alert) {
    if (!env.DISCORD_WEBHOOK) return;

    const embed = {
        title: alert.title || "🧙‍♂️ Gandalf Alert",
        description: alert.description,
        color: alert.color || 3447003,
        fields: alert.fields || [],
        timestamp: new Date().toISOString(),
        footer: {
            text: "Gandalf System Monitor"
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

async function checkSystemHealth(env) {
    // Check data freshness
    const latestData = await env.MARKET_CACHE.get('latest_snapshot');
    if (!latestData) {
        return { healthy: false, issue: "No recent data found in cache" };
    }

    const data = JSON.parse(latestData);
    const dataTimestamp = Array.isArray(data) && data[0]?.timestamp
        ? new Date(data[0].timestamp).getTime()
        : new Date(data.timestamp).getTime();

    const age = Date.now() - dataTimestamp;

    if (age > 1800000) { // 30 minutes
        return { healthy: false, issue: `Data is ${Math.round(age / 60000)} minutes old` };
    }

    // Check Supabase connectivity
    try {
        const response = await fetch(
            `${env.SUPABASE_URL}/rest/v1/market_data?limit=1`,
            { headers: { 'apikey': env.SUPABASE_ANON_KEY } }
        );

        if (!response.ok) {
            return { healthy: false, issue: "Supabase connection failed" };
        }
    } catch (error) {
        return { healthy: false, issue: "Database connection error" };
    }

    return {
        healthy: true,
        lastDataAge: Math.round(age / 60000),
        message: "All systems operational"
    };
}