import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']

def fetch_data(days=7):
    """Fetch market data from Supabase"""
    headers = {'apikey': SUPABASE_KEY}
    
    # Get data from last N days
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/market_data?order=timestamp.desc&limit=10000",
        headers=headers
    )
    
    data = response.json()
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter to last N days
    cutoff = datetime.now() - timedelta(days=days)
    df = df[df['timestamp'] > cutoff]
    
    return df

def find_arbitrage_opportunities(df):
    """Find price inefficiencies"""
    opportunities = []
    
    # Group by timestamp to find simultaneous prices
    for timestamp, group in df.groupby(df['timestamp'].dt.floor('5min')):
        for item in group['item'].unique():
            item_data = group[group['item'] == item]
            
            if len(item_data) > 0:
                buy_max = item_data['buy_median'].max()
                sell_min = item_data['sell_median'].min()
                
                if pd.notna(buy_max) and pd.notna(sell_min) and buy_max > sell_min:
                    opportunities.append({
                        'timestamp': timestamp,
                        'item': item,
                        'buy_at': sell_min,
                        'sell_at': buy_max,
                        'profit': buy_max - sell_min,
                        'profit_pct': ((buy_max - sell_min) / sell_min) * 100
                    })
    
    return opportunities

def analyze_patterns(df):
    """Deep pattern analysis"""
    patterns = []
    
    # Analyze each item
    for item in df['item'].unique():
        item_df = df[df['item'] == item].sort_values('timestamp')
        
        if len(item_df) < 50:
            continue
        
        # Price momentum
        prices = item_df['sell_median'].dropna()
        if len(prices) > 20:
            returns = prices.pct_change().dropna()
            momentum = returns.rolling(20).mean().iloc[-1]
            
            if abs(momentum) > 0.02:  # 2% momentum
                patterns.append({
                    'type': 'momentum',
                    'item': item,
                    'strength': momentum,
                    'direction': 'bullish' if momentum > 0 else 'bearish'
                })
    
    return patterns

def main():
    print("Fetching market data...")
    df = fetch_data(days=7)
    print(f"Loaded {len(df)} records")
    
    # Find arbitrage
    print("Finding arbitrage opportunities...")
    arbitrage = find_arbitrage_opportunities(df)
    print(f"Found {len(arbitrage)} opportunities")
    
    # Analyze patterns
    print("Analyzing patterns...")
    patterns = analyze_patterns(df)
    print(f"Found {len(patterns)} patterns")
    
    # Save results
    os.makedirs('output', exist_ok=True)
    
    with open('output/arbitrage.json', 'w') as f:
        json.dump(arbitrage, f, indent=2, default=str)
    
    with open('output/patterns.json', 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()