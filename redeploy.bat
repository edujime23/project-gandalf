
git add .
git commit -m "new commit"
git push origin main

CD workers
npx wrangler deploy --config wrangler.toml
npx wrangler deploy --config wrangler-analyzer.toml
npx wrangler deploy --config wrangler-monitor.toml
CD ..

CD dashboard
npx wrangler pages deploy . --project-name=gandalf-dashboard
CD ..