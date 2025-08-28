#!/usr/bin/env python
import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE') or os.getenv('SUPABASE_KEY')

MODEL_PATH = 'models/unified_model.pkl'
HORIZON_STEPS = int(os.getenv("PRED_HORIZON_STEPS", "6"))

def sb_get(path, params=None):
    headers = {'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'}
    r = requests.get(f"{SUPABASE_URL}{path}", headers=headers, params=params or {}, timeout=30)
    r.raise_for_status()
    return r

def sb_post(path, body):
    headers = {'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}', 'Content-Type': 'application/json'}
    r = requests.post(f"{SUPABASE_URL}{path}", headers=headers, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    return r

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def fetch_tracked_items(limit=40):
    r = sb_get("/rest/v1/tracked_items", params={"select":"item,score,active","active":"eq.true","order":"score.desc","limit":str(limit)})
    rows = r.json()
    return [x["item"] for x in rows]

def fetch_latest_data(item, limit=120):
    params = {'select': '*', 'item': f'eq.{item}', 'order': 'timestamp.desc', 'limit': str(limit)}
    r = sb_get('/rest/v1/market_data', params=params)
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    for col in ['sell_median','buy_median','spread','sell_orders','buy_orders']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_recent_era_weights(hours=6):
    r = sb_post('/rest/v1/rpc/get_recent_era_weights', {"hours": hours})
    w = r.json()
    out = {k: float(w.get(k, 0.25)) for k in ['Lith','Meso','Neo','Axi']}
    s = sum(out.values()) or 1.0
    for k in out:
        out[k] = out[k] / s
    return out

def get_supply_index_bulk(era_weights, only_active=True, top_n=200):
    r = sb_post('/rest/v1/rpc/get_supply_index_bulk', {"era_weights": era_weights, "only_active": only_active, "top_n": top_n})
    rows = r.json() or []
    return {row["set_item"]: float(row["supply_index"] or 0) for row in rows}

def linear_projection(price: pd.Series, steps: int) -> float:
    # Simple baseline: fit y ~ a + b*t over last n points; predict at t+steps
    if len(price) < 8:
        return float(price.iloc[-1])
    y = price.values
    x = np.arange(len(y))
    # least squares
    A = np.vstack([x, np.ones(len(x))]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    next_x = len(y) + steps
    return float(a + b * next_x)

def create_features(df: pd.DataFrame, item_encoded: int, feature_columns: list, era_w: dict, supply_index: float, baro_active: int) -> pd.DataFrame:
    price = df['sell_median'].astype(float).ffill()
    volume = (df.get('sell_orders',0).fillna(0) + df.get('buy_orders',0).fillna(0)).astype(float)
    spread = df.get('spread',0).fillna(0).astype(float)
    buys = df.get('buy_orders',0).fillna(0).astype(float)
    sells = df.get('sell_orders',0).fillna(0).astype(float)

    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    feat = {}
    ts = df['timestamp'].iloc[-1]
    feat['item_encoded'] = item_encoded
    feat['hour'] = ts.hour
    feat['day_of_week'] = ts.dayofweek
    feat['day_of_month'] = ts.day
    feat['is_weekend'] = int(ts.dayofweek >= 5)

    feat['price'] = float(price.iloc[-1])
    feat['ema_6'] = float(ema(price, 6).iloc[-1])
    feat['ema_24'] = float(ema(price, 24).iloc[-1])
    feat['ema_72'] = float(ema(price, 72).iloc[-1])

    def safe(series, n):
        return float(series.iloc[-n]) if len(series) >= n else float(series.iloc[0])

    feat['lag_1'] = float(price.iloc[-2]) if len(price) >= 2 else float(price.iloc[-1])
    feat['lag_6'] = safe(price, min(6, len(price)))
    feat['lag_24'] = safe(price, min(24, len(price)))

    def pct_change(series, n):
        if len(series) <= n:
            return 0.0
        a = float(series.iloc[-1]); b = float(series.iloc[-n-1])
        return (a - b) / b if b != 0 else 0.0

    feat['ret_1'] = pct_change(price, 1)
    feat['ret_6'] = pct_change(price, 6)
    feat['ret_24'] = pct_change(price, 24)

    roll = price.rolling(24, min_periods=1)
    mu = float(roll.mean().iloc[-1])
    sd = float(roll.std().iloc[-1] or 0)
    feat['zscore_24'] = (float(price.iloc[-1]) - mu) / sd if sd > 0 else 0.0

    feat['volume'] = float(volume.iloc[-1])
    feat['volume_ma_24'] = float(volume.rolling(24, min_periods=1).mean().iloc[-1])
    feat['volume_ratio'] = feat['volume'] / (feat['volume_ma_24'] + 1e-9)

    feat['spread'] = float(spread.iloc[-1])
    feat['spread_ma'] = float(spread.rolling(24, min_periods=1).mean().iloc[-1])
    feat['spread_std'] = float(spread.rolling(24, min_periods=1).std().iloc[-1])

    feat['depth'] = float((buys+sells).iloc[-1])
    feat['imbalance'] = float((buys.iloc[-1] - sells.iloc[-1]) / (buys.iloc[-1] + sells.iloc[-1] + 1e-9))

    feat['w_lith'] = float(era_w.get('Lith', 0.25))
    feat['w_meso'] = float(era_w.get('Meso', 0.25))
    feat['w_neo']  = float(era_w.get('Neo', 0.25))
    feat['w_axi']  = float(era_w.get('Axi', 0.25))
    feat['baro_active'] = int(baro_active)
    feat['supply_index'] = float(supply_index)

    X = pd.DataFrame([{k: feat.get(k, 0) for k in feature_columns}])
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    return X

def store_predictions(preds):
    if not preds:
        return
    headers = {'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}', 'Content-Type': 'application/json', 'Prefer': 'return=minimal,resolution=merge-duplicates'}
    url = f"{SUPABASE_URL}/rest/v1/predictions?on_conflict=item,predicted_at"
    r = requests.post(url, headers=headers, data=json.dumps(preds), timeout=60)
    r.raise_for_status()

def get_baro_active():
    r = sb_get("/rest/v1/worldstate_flags", params={"select":"baro_active,taken_at","order":"taken_at.desc","limit":"1"})
    js = r.json()
    if js:
        return 1 if js[0].get("baro_active") else 0
    return 0

def main():
    print("Loading unified model…")
    model_data = load_model()
    model = model_data['model']
    item_encoder = model_data['item_encoder']
    feature_columns = model_data['feature_columns']

    tracked_items = fetch_tracked_items(limit=40)
    print(f"Active tracked items: {len(tracked_items)}")

    era_w = get_recent_era_weights(hours=6)
    supply_map = get_supply_index_bulk(era_weights=era_w, only_active=True, top_n=500)
    baro_active = get_baro_active()

    predictions = []
    for item in tracked_items:
        try:
            df = fetch_latest_data(item, limit=120)
            if df is None or len(df) < 10:
                continue
            try:
                item_encoded = int(item_encoder.transform([item])[0])
            except Exception:
                # item unseen in training
                continue

            current_price = float(df['sell_median'].iloc[-1])

            X = create_features(
                df=df,
                item_encoded=item_encoded,
                feature_columns=feature_columns,
                era_w=era_w,
                supply_index=float(supply_map.get(item, 0.0)),
                baro_active=baro_active
            )
            model_pred = float(model.predict(X)[0])

            # Baseline: linear projection
            lin_pred = linear_projection(df['sell_median'].astype(float), steps=HORIZON_STEPS)

            # Blend (conservative): 70% model, 30% baseline
            final_pred = 0.7 * model_pred + 0.3 * lin_pred

            # Confidence: higher when recent volatility lower and model/baseline agree
            price_std = float(df['sell_median'].rolling(24, min_periods=1).std().iloc[-1] or 0)
            agree = 1.0 - min(1.0, abs(model_pred - lin_pred) / (abs(current_price) + 1e-9))
            conf = max(0.4, min(0.98, agree * (1.0 - min(1.0, price_std / (current_price + 1e-9)))))

            predictions.append({
                'item': item,
                'predicted_at': datetime.now(timezone.utc).isoformat(),
                'current_price': current_price,
                'predicted_price': float(final_pred),
                'confidence': float(conf)
            })
        except Exception as e:
            print(f"Infer failed for {item}: {e}")

    print(f"Generated {len(predictions)} predictions")
    if predictions:
        store_predictions(predictions)
        print("Stored predictions")

if __name__ == "__main__":
    main()