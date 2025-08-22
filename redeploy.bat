@ECHO OFF
TITLE Gandalf Full Redeployment Script
COLOR 0A

:: =================================================================
:: ==      🧙‍♂️ Project Gandalf Full Redeployment Script (v3)      ==
:: ==           (Fixes Cloudflare Pages deployment)               ==
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
ECHO [STEP 1/4] Committing and pushing all local changes to GitHub...
ECHO.

git add .
git commit -m "Automated redeployment with final build fix"
git push origin main

ECHO.
ECHO Git push complete.
ECHO -----------------------------------------------------------------
ECHO.

:: =================================================================
:: == Step 2: Redeploy all Cloudflare Workers                     ==
:: =================================================================
ECHO [STEP 2/4] Redeploying Cloudflare Workers...
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
:: == Step 3: Build and Redeploy the Dashboard to Cloudflare Pages ==
:: =================================================================
ECHO [STEP 3/4] Building and Redeploying Dashboard...
ECHO.

CD dashboard

ECHO [+] Creating a clean deployment folder...
IF EXIST dist RMDIR /S /Q dist
MKDIR dist

ECHO [+] Copying necessary files to deployment folder...
COPY *.html dist\
COPY *.js dist\

ECHO [+] Deploying the built dashboard to Cloudflare Pages...
:: CRITICAL FIX: Explicitly deploy the 'dist' folder
npx wrangler pages deploy dist --project-name=gandalf-dashboard

CD ..
ECHO.
ECHO Dashboard redeployed successfully.
ECHO -----------------------------------------------------------------
ECHO.

:: =================================================================
:: == Step 4: Final Summary                                       ==
:: =================================================================
ECHO [STEP 4/4] All components have been redeployed!
ECHO.
ECHO =================================================================
ECHO ==                       DEPLOYMENT COMPLETE                   ==
ECHO =================================================================
ECHO.
ECHO Your system is now up to date. You can check the live URLs:
ECHO.
ECHO - Collector URL: https://gandalf-collector.psnedujime.workers.dev
ECHO - Analyzer URL:  https://gandalf-analyzer.psnedujime.workers.dev
ECHO - Monitor URL:   https://gandalf-monitor.psnedujime.workers.dev
ECHO - Dashboard URL: https://gandalf-dashboard.pages.dev
ECHO.
ECHO IMPORTANT: If you changed the database schema (SQL), you must
ECHO apply those changes manually in the Supabase SQL Editor.
ECHO.
PAUSE