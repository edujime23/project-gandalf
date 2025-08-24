#!/usr/bin/env python
import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE') or os.getenv('SUPABASE_KEY')

def fetch_data(days=7):
    headers = {'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'}
    resp = requests.get(f"{SUPABASE_URL}/rest/v1/market_data?order=timestamp.desc&limit=15000", headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
    cutoff = pd.Timestamp.utcnow() - timedelta(days=days)
    return df[df['timestamp'] > cutoff]

def find_arbitrage_opportunities(df):
    out = []
    df = df.dropna(subset=['buy_median', 'sell_median'])
    for item in df['item'].unique():
        latest = df[df['item'] == item].iloc[0]
        profit = latest['buy_median'] - latest['sell_median']
        if profit > 5:
            profit_pct = (profit / latest['sell_median']) * 100 if latest['sell_median'] else 0
            if profit_pct > 15:
                out.append({'item': item, 'buy_at': latest['sell_median'], 'sell_at': latest['buy_median'], 'profit': profit, 'profit_pct': profit_pct})
    return out

def main():
    print("Fetching market data for deep analysis...")
    df = fetch_data(days=7)
    if df.empty:
        print("No data fetched.")
        return
    print(f"Loaded {len(df)} records.")
    print("Finding arbitrage opportunities...")
    arbitrage = find_arbitrage_opportunities(df)
    print(f"Found {len(arbitrage)} arbitrage opportunities.")
    os.makedirs('output', exist_ok=True)
    with open('output/arbitrage.json', 'w') as f:
        json.dump(arbitrage, f, indent=2, default=str)
    print("Deep analysis complete!")

if __name__ == "__main__":
    main()