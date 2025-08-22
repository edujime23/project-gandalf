@ECHO OFF
TITLE Gandalf Full Redeployment Script
COLOR 0A

:: =================================================================
:: ==      🧙‍♂️ Project Gandalf Full Redeployment Script (v4)      ==
:: ==           (Uses Secure API Gateway Method)                  ==
:: =================================================================
ECHO.
ECHO This script will commit all local changes and redeploy all components.
ECHO Make sure you have saved all your files before continuing.
ECHO.
PAUSE
CLS

:: =================================================================
:: == Step 1: Commit and Push all changes to GitHub               ==
:: =================================================================
ECHO [STEP 1/3] Committing and pushing all local changes to GitHub...
ECHO.
git add .
git commit -m "feat: Implement secure API gateway for dashboard"
git push origin main
ECHO.
ECHO Git push complete.
ECHO -----------------------------------------------------------------
ECHO.

:: =================================================================
:: == Step 2: Redeploy all Cloudflare Workers                     ==
:: =================================================================
ECHO [STEP 2/3] Redeploying Cloudflare Workers...
ECHO.
CD workers
ECHO [+] Deploying gandalf-collector...
npx wrangler deploy --config wrangler.toml
ECHO.
ECHO [+] Deploying gandalf-analyzer...
npx wrangler deploy --config wrangler-analyzer.toml
ECHO.
ECHO [+] Deploying gandalf-monitor...
npx wrangler deploy --config wrangler-monitor.toml
ECHO.
CD ..
ECHO All workers redeployed successfully.
ECHO -----------------------------------------------------------------
ECHO.

:: =================================================================
:: == Step 3: Redeploy the Dashboard & API Gateway                ==
:: =================================================================
ECHO [STEP 3/3] Redeploying Dashboard and API Gateway...
ECHO.
CD dashboard
:: This deploys the dashboard and the 'functions' folder as the API
npx wrangler pages deploy . --project-name=gandalf-dashboard
CD ..
ECHO.
ECHO Dashboard and API Gateway redeployed successfully.
ECHO -----------------------------------------------------------------
ECHO.

:: =================================================================
:: == Final Summary                                               ==
:: =================================================================
ECHO.
ECHO =================================================================
ECHO ==                       DEPLOYMENT COMPLETE                   ==
ECHO =================================================================
ECHO.
ECHO Your system is now up to date and SECURE.
ECHO.
ECHO - Dashboard & API URL: https://gandalf-dashboard.pages.dev
ECHO.
PAUSE