
git add .
git commit -m "feat: Implement secure API gateway for dashboard"
git push origin main

CD workers
npx wrangler deploy --config wrangler.toml
npx wrangler deploy --config wrangler-analyzer.toml
npx wrangler deploy --config wrangler-monitor.toml
CD ..

CD dashboard
npx wrangler pages deploy . --project-name=gandalf-dashboard
CD ..