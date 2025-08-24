#!/usr/bin/env python
import os
import json
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timezone

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE') or os.getenv('SUPABASE_KEY')

MODEL_PATH = 'models/unified_model.pkl'

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def fetch_latest_data(item):
    url = f"{SUPABASE_URL}/rest/v1/market_data"
    headers = {'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'}
    params = {'select': '*', 'item': f'eq.{item}', 'order': 'timestamp.desc', 'limit': '100'}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    df = pd.DataFrame(data).sort_values('timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    for col in ['sell_median', 'buy_median', 'spread', 'sell_orders', 'buy_orders']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def create_features_for_item(df, item_encoded, feature_columns):
    features = pd.DataFrame()
    latest = df.iloc[-1]
    features['item_encoded'] = [item_encoded]
    features['hour'] = [latest['timestamp'].hour]
    features['day_of_week'] = [latest['timestamp'].dayofweek]
    features['day_of_month'] = [latest['timestamp'].day]
    features['is_weekend'] = [int(latest['timestamp'].dayofweek >= 5)]

    price = df['sell_median'].ffill()
    features['price'] = [price.iloc[-1]]
    features['price_ma_6'] = [price.rolling(6, min_periods=1).mean().iloc[-1]]
    features['price_ma_24'] = [price.rolling(24, min_periods=1).mean().iloc[-1]]
    features['price_std_6'] = [price.rolling(6, min_periods=1).std().iloc[-1]]
    features['price_std_24'] = [price.rolling(24, min_periods=1).std().iloc[-1]]

    features['momentum_6'] = [price.pct_change(6).iloc[-1]]
    features['momentum_24'] = [price.pct_change(24).iloc[-1]]

    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = [(100 - (100 / (1 + rs))).iloc[-1]]

    volume = df['sell_orders'].fillna(0) + df['buy_orders'].fillna(0)
    features['volume'] = [volume.iloc[-1]]
    features['volume_ma_24'] = [volume.rolling(24, min_periods=1).mean().iloc[-1]]
    features['volume_ratio'] = [volume.iloc[-1] / (features['volume_ma_24'][0] + 1)]

    features['spread'] = [df['spread'].fillna(0).iloc[-1] if 'spread' in df.columns else 0]
    features['spread_ma'] = [df['spread'].fillna(0).rolling(12, min_periods=1).mean().iloc[-1] if 'spread' in df.columns else 0]
    features['spread_ratio'] = [features['spread'][0] / (features['price'][0] + 1)]

    features['order_imbalance'] = [
        (df['buy_orders'].fillna(0).iloc[-1] - df['sell_orders'].fillna(0).iloc[-1]) /
        (df['buy_orders'].fillna(0).iloc[-1] + df['sell_orders'].fillna(0).iloc[-1] + 1)
    ]

    # Ensure correct order
    features = features.reindex(columns=feature_columns, fill_value=0)
    return features

def store_predictions(preds):
    if not preds:
        return
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal,resolution=merge-duplicates'
    }
    url = f"{SUPABASE_URL}/rest/v1/predictions?on_conflict=item,predicted_at"
    resp = requests.post(url, headers=headers, data=json.dumps(preds), timeout=30)
    resp.raise_for_status()

def main():
    print("Loading unified model...")
    model_data = load_model()
    model = model_data['model']
    item_encoder = model_data['item_encoder']
    feature_columns = model_data['feature_columns']

    # Fetch tracked items
    headers = {'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'}
    params = {'select': 'item', 'active': 'eq.true', 'order': 'score.desc', 'limit': '20'}
    resp = requests.get(f"{SUPABASE_URL}/rest/v1/tracked_items", headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    tracked_items = [row['item'] for row in resp.json()]

    predictions = []
    for item in tracked_items:
        try:
            df = fetch_latest_data(item)
            if df is None or len(df) < 10:
                continue
            try:
                item_encoded = item_encoder.transform([item])[0]
            except:
                # Item not seen in training
                continue
            X = create_features_for_item(df, item_encoded, feature_columns)
            current_price = float(df['sell_median'].iloc[-1])
            predicted_price = float(model.predict(X)[0])

            price_std = df['sell_median'].rolling(24, min_periods=1).std().iloc[-1]
            confidence = max(0.5, min(0.95, 1 - (float(price_std) / (current_price + 1)) * 2))

            predictions.append({
                'item': item,
                'predicted_at': datetime.now(timezone.utc).isoformat(),
                'current_price': current_price,
                'predicted_price': predicted_price,
                'confidence': float(confidence)
            })
        except Exception as e:
            print(f"Predict failed for {item}: {e}")

    print(f"Generated {len(predictions)} predictions")
    if predictions:
        store_predictions(predictions)
        print("Stored predictions")

if __name__ == "__main__":
    main()