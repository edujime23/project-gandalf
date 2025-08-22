# scripts/deep_analysis.py
import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

def fetch_data(days=7):
    headers = {'apikey': SUPABASE_KEY}
    response = requests.get(f"{SUPABASE_URL}/rest/v1/market_data?order=timestamp.desc&limit=15000", headers=headers)
    response.raise_for_status() # Will raise an error for bad responses
    data = response.json()
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    # CRITICAL FIX: Tell pandas to automatically handle different ISO timestamp formats
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    
    cutoff = pd.to_datetime('now', utc=True) - timedelta(days=days)
    return df[df['timestamp'] > cutoff]

def find_arbitrage_opportunities(df):
    opportunities = []
    df_no_na = df.dropna(subset=['buy_median', 'sell_median'])
    for item in df_no_na['item'].unique():
        item_df = df_no_na[df_no_na['item'] == item]
        latest = item_df.iloc[0]
        profit = latest['buy_median'] - latest['sell_median']
        if profit > 5:
            profit_pct = (profit / latest['sell_median']) * 100 if latest['sell_median'] != 0 else float('inf')
            if profit_pct > 15:
                opportunities.append({
                    'item': item, 'buy_at': latest['sell_median'], 'sell_at': latest['buy_median'],
                    'profit': profit, 'profit_pct': profit_pct
                })
    return opportunities

def main():
    print("Fetching market data for deep analysis...")
    df = fetch_data(days=7)
    if df.empty:
        print("No data fetched, exiting.")
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