import os
import json
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timezone

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

def load_model():
    """Load the unified model"""
    model_path = 'models/unified_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model_data = joblib.load(model_path)
    return model_data

def fetch_latest_data(item):
    """Fetch recent data for an item"""
    url = f"{SUPABASE_URL}/rest/v1/market_data"
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}'
    }
    params = {
        'select': '*',
        'item': f'eq.{item}',
        'order': 'timestamp.desc',
        'limit': '100'
    }
    
    resp = requests.get(url, headers=headers, params=params)
    if not resp.ok:
        raise Exception(f"Failed to fetch data: {resp.status_code}")
    
    data = resp.json()
    if not data:
        return None
        
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert numeric columns
    for col in ['sell_median', 'buy_median', 'spread', 'sell_orders', 'buy_orders']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.sort_values('timestamp')

def create_features_for_item(df, item_encoded, feature_columns):
    """Create features matching training format"""
    features = pd.DataFrame()
    
    # Item encoding
    features['item_encoded'] = item_encoded
    
    # Time features
    latest = df.iloc[-1]
    features['hour'] = latest['timestamp'].hour
    features['day_of_week'] = latest['timestamp'].dayofweek
    features['day_of_month'] = latest['timestamp'].day
    features['is_weekend'] = int(latest['timestamp'].dayofweek >= 5)
    
    # Price features
    price = df['sell_median'].fillna(method='ffill')
    features['price'] = price.iloc[-1]
    features['price_ma_6'] = price.rolling(6, min_periods=1).mean().iloc[-1]
    features['price_ma_24'] = price.rolling(24, min_periods=1).mean().iloc[-1]
    features['price_std_6'] = price.rolling(6, min_periods=1).std().iloc[-1]
    features['price_std_24'] = price.rolling(24, min_periods=1).std().iloc[-1]
    
    # Momentum
    features['momentum_6'] = price.pct_change(6).iloc[-1]
    features['momentum_24'] = price.pct_change(24).iloc[-1]
    
    # RSI
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
    
    # Volume
    volume = df['sell_orders'].fillna(0) + df['buy_orders'].fillna(0)
    features['volume'] = volume.iloc[-1]
    features['volume_ma_24'] = volume.rolling(24, min_periods=1).mean().iloc[-1]
    features['volume_ratio'] = volume.iloc[-1] / (features['volume_ma_24'] + 1)
    
    # Spread
    if 'spread' in df.columns:
        features['spread'] = df['spread'].fillna(0).iloc[-1]
        features['spread_ma'] = df['spread'].fillna(0).rolling(12, min_periods=1).mean().iloc[-1]
        features['spread_ratio'] = features['spread'] / (features['price'] + 1)
    else:
        features['spread'] = 0
        features['spread_ma'] = 0
        features['spread_ratio'] = 0
    
    # Order imbalance
    features['order_imbalance'] = (
        (df['buy_orders'].fillna(0).iloc[-1] - df['sell_orders'].fillna(0).iloc[-1]) /
        (df['buy_orders'].fillna(0).iloc[-1] + df['sell_orders'].fillna(0).iloc[-1] + 1)
    )
    
    # Ensure all features are present and in correct order
    features_df = pd.DataFrame([features])
    features_df = features_df.reindex(columns=feature_columns, fill_value=0)
    
    return features_df

def main():
    print("Loading unified model...")
    model_data = load_model()
    
    model = model_data['model']
    item_encoder = model_data['item_encoder']
    feature_columns = model_data['feature_columns']
    
    # Get tracked items
    url = f"{SUPABASE_URL}/rest/v1/tracked_items"
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}'
    }
    params = {
        'select': 'item',
        'active': 'eq.true',
        'order': 'score.desc',
        'limit': '20'
    }
    
    resp = requests.get(url, headers=headers, params=params)
    if not resp.ok:
        print(f"Failed to fetch tracked items: {resp.status_code}")
        return
    
    tracked_items = [row['item'] for row in resp.json()]
    
    predictions = []
    
    for item in tracked_items:
        print(f"Predicting for {item}...")
        
        # Fetch latest data
        df = fetch_latest_data(item)
        if df is None or len(df) < 10:
            print(f"  Skipping {item}: insufficient data")
            continue
        
        # Encode item
        try:
            item_encoded = item_encoder.transform([item])[0]
        except:
            print(f"  Skipping {item}: not in training set")
            continue
        
        # Create features
        X = create_features_for_item(df, item_encoded, feature_columns)
        
        # Make prediction
        current_price = df['sell_median'].iloc[-1]
        predicted_price = model.predict(X)[0]
        
        # Calculate confidence based on recent stability
        price_std = df['sell_median'].rolling(24, min_periods=1).std().iloc[-1]
        confidence = max(0.5, min(0.95, 1 - (price_std / (current_price + 1)) * 2))
        
        predictions.append({
            'item': item,
            'predicted_at': datetime.now(timezone.utc).isoformat(),
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'confidence': float(confidence)
        })
    
    print(f"\nGenerated {len(predictions)} predictions")
    
    # Store predictions
    if predictions:
        headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal,resolution=merge-duplicates'
        }
        
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/predictions?on_conflict=item,predicted_at",
            headers=headers,
            json=predictions
        )
        
        if resp.ok:
            print("Predictions stored successfully")
        else:
            print(f"Failed to store predictions: {resp.status_code}")

if __name__ == "__main__":
    main()