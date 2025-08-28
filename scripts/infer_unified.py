#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE = os.getenv('SUPABASE_SERVICE_ROLE') or os.getenv('SUPABASE_KEY')
SUPABASE_ANON = os.getenv('SUPABASE_ANON_KEY')

MODEL_PATH = 'models/unified_model.pkl'
HORIZON_STEPS = int(os.getenv("PRED_HORIZON_STEPS", "6"))
BLEND_WEIGHT = float(os.getenv("BLEND_WEIGHT", "0.7"))  # model weight; baseline = 1-w

def sb_get(path, params=None, headers=None):
    key = SUPABASE_ANON or SUPABASE_SERVICE
    headers_all = {'apikey': key, 'Authorization': f'Bearer {key}'}
    if headers: headers_all.update(headers)
    r = requests.get(f"{SUPABASE_URL}{path}", headers=headers_all, params=params or {}, timeout=30)
    r.raise_for_status()
    return r

def sb_post(path, body):
    key = SUPABASE_ANON or SUPABASE_SERVICE
    headers = {'apikey': key, 'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    r = requests.post(f"{SUPABASE_URL}{path}", headers=headers, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    return r

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def fetch_tracked_items(limit=40):
    r = sb_get("/rest/v1/tracked_items", params={"select":"item,score,active","active":"eq.true","order":"score.desc","limit":str(limit)})
    return [x["item"] for x in r.json()]

def fetch_latest_per_item(item, limit=200):
    r = sb_get('/rest/v1/market_data', params={'select': '*', 'item': f'eq.{item}', 'order': 'timestamp.desc', 'limit': str(limit)})
    data = r.json()
    if not data: return None
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    for col in ['sell_median','buy_median','spread','sell_orders','buy_orders']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def fetch_recent_all(limit=2500):
    r = sb_get('/rest/v1/market_data', params={'select':'timestamp,sell_median,item','order':'timestamp.desc','limit':str(limit)})
    df = pd.DataFrame(r.json())
    if df.empty: return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp','sell_median']).sort_values('timestamp')
    df['sell_median'] = pd.to_numeric(df['sell_median'], errors='coerce')
    return df

def compute_market_index(df_all):
    if df_all.empty:
        return pd.DataFrame(columns=["timestamp","index_price","index_ret_6","index_ret_24","index_zscore_24"])
    idx = df_all.groupby("timestamp")["sell_median"].mean().reset_index().rename(columns={"sell_median":"index_price"})
    idx = idx.sort_values("timestamp")
    idx["index_ret_6"] = idx["index_price"].pct_change(6)
    idx["index_ret_24"] = idx["index_price"].pct_change(24)
    roll24 = idx["index_price"].rolling(24, min_periods=1)
    mu24 = roll24.mean(); sd24 = roll24.std().replace(0, np.nan)
    idx["index_zscore_24"] = (idx["index_price"] - mu24) / sd24
    idx = idx.replace([np.inf, -np.inf], 0).fillna(0)
    return idx

def get_recent_era_weights(hours=6):
    r = sb_post('/rest/v1/rpc/get_recent_era_weights', {"hours": hours})
    w = r.json()
    out = {k: float(w.get(k, 0.25)) for k in ['Lith','Meso','Neo','Axi']}
    s = sum(out.values()) or 1.0
    for k in out: out[k] = out[k] / s
    return out

def get_baro_active():
    r = sb_get("/rest/v1/worldstate_flags", params={"select":"baro_active,taken_at","order":"taken_at.desc","limit":"1"})
    js = r.json()
    if js: return 1 if js[0].get("baro_active") else 0
    return 0

def get_supply_index_bulk(era_weights, only_active=True, top_n=500):
    r = sb_post('/rest/v1/rpc/get_supply_index_bulk', {"era_weights": era_weights, "only_active": only_active, "top_n": top_n})
    rows = r.json() or []
    return {row["set_item"]: float(row["supply_index"] or 0) for row in rows}

def linear_projection(price: pd.Series, steps: int) -> float:
    if len(price) < 8: return float(price.iloc[-1])
    y = price.values; x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    next_x = len(y) + steps
    return float(a + b * next_x)

def create_features(df: pd.DataFrame, item_encoded: int, feature_columns: list, era_w: dict,
                    supply_index: float, baro_active: int, index_df: pd.DataFrame) -> pd.DataFrame:
    price = df['sell_median'].astype(float).ffill()
    volume = (df.get('sell_orders',0).fillna(0) + df.get('buy_orders',0).fillna(0)).astype(float)
    spread = df.get('spread',0).fillna(0).astype(float)
    buys = df.get('buy_orders',0).fillna(0).astype(float)
    sells = df.get('sell_orders',0).fillna(0).astype(float)

    def ema(series, span): return series.ewm(span=span, adjust=False).mean()

    ts = df['timestamp'].iloc[-1]
    hour = ts.hour; dow = ts.dayofweek

    if index_df is not None and not index_df.empty:
        merge = pd.merge_asof(
            df[['timestamp']].sort_values('timestamp'),
            index_df.sort_values('timestamp'),
            on='timestamp', direction='backward', tolerance=pd.Timedelta('1H')
        )
        idx_price = float(merge['index_price'].iloc[-1] or 0)
        idx_ret6 = float(merge['index_ret_6'].iloc[-1] or 0)
        idx_ret24 = float(merge['index_ret_24'].iloc[-1] or 0)
        idx_z = float(merge['index_zscore_24'].iloc[-1] or 0)
    else:
        idx_price = idx_ret6 = idx_ret24 = idx_z = 0.0

    feat = {
        "item_encoded": item_encoded,
        "hour": hour, "day_of_week": dow, "day_of_month": ts.day, "is_weekend": int(dow>=5),
        "hour_sin": math.sin(2*math.pi*hour/24), "hour_cos": math.cos(2*math.pi*hour/24),
        "dow_sin": math.sin(2*math.pi*dow/7), "dow_cos": math.cos(2*math.pi*dow/7),

        "price": float(price.iloc[-1]),
        "ema_6": float(ema(price,6).iloc[-1]),
        "ema_24": float(ema(price,24).iloc[-1]),
        "ema_72": float(ema(price,72).iloc[-1]),

        "lag_1": float(price.iloc[-2]) if len(price)>=2 else float(price.iloc[-1]),
        "lag_6": float(price.iloc[-6]) if len(price)>=6 else float(price.iloc[0]),
        "lag_24": float(price.iloc[-24]) if len(price)>=24 else float(price.iloc[0]),
        "lag_48": float(price.iloc[-48]) if len(price)>=48 else float(price.iloc[0]),
        "lag_168": float(price.iloc[-168]) if len(price)>=168 else float(price.iloc[0]),

        "ret_1": float(price.pct_change(1).iloc[-1] if len(price)>1 else 0),
        "ret_6": float(price.pct_change(6).iloc[-1] if len(price)>6 else 0),
        "ret_24": float(price.pct_change(24).iloc[-1] if len(price)>24 else 0),
        "ret_48": float(price.pct_change(48).iloc[-1] if len(price)>48 else 0),
        "ret_168": float(price.pct_change(168).iloc[-1] if len(price)>168 else 0),

        "zscore_24": 0.0,
        "volume": float(volume.iloc[-1]),
        "volume_ma_24": float(volume.rolling(24, min_periods=1).mean().iloc[-1]),
        "volume_ratio": 0.0,
        "spread": float(spread.iloc[-1]),
        "spread_ma": float(spread.rolling(24, min_periods=1).mean().iloc[-1]),
        "spread_std": float(spread.rolling(24, min_periods=1).std().iloc[-1]),
        "depth": float((buys+sells).iloc[-1]),
        "imbalance": float((buys.iloc[-1]-sells.iloc[-1])/(buys.iloc[-1]+sells.iloc[-1]+1e-9)),

        "w_lith": float(era_w.get('Lith',0.25)),
        "w_meso": float(era_w.get('Meso',0.25)),
        "w_neo": float(era_w.get('Neo',0.25)),
        "w_axi": float(era_w.get('Axi',0.25)),
        "fissure_total": 0.0,
        "baro_active": int(baro_active),
        "supply_index": float(supply_index),

        "index_price": idx_price, "index_ret_6": idx_ret6, "index_ret_24": idx_ret24, "index_zscore_24": idx_z
    }

    roll24 = df['sell_median'].astype(float).rolling(24, min_periods=1)
    mu24 = float(roll24.mean().iloc[-1]); sd24 = float(roll24.std().iloc[-1] or 0)
    feat["zscore_24"] = ((feat["price"] - mu24)/sd24) if sd24>0 else 0.0
    feat["volume_ratio"] = feat["volume"] / (feat["volume_ma_24"] + 1e-9)

    X = pd.DataFrame([{k: feat.get(k, 0) for k in feature_columns}]).replace([np.inf, -np.inf], 0).fillna(0)
    return X

def store_predictions(preds):
    if not preds:
        return
    key = SUPABASE_ANON or SUPABASE_SERVICE
    headers = {'apikey': key, 'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    url = f"{SUPABASE_URL}/rest/v1/rpc/insert_predictions_bulk"
    body = {"rows": preds}
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
    r.raise_for_status()

def main():
    print("Loading unified model...")
    bundle = load_model()
    model = bundle['model']
    q_low = bundle.get('quantile_low', None)
    q_high = bundle.get('quantile_high', None)
    item_encoder = bundle['item_encoder']
    feature_columns = bundle['feature_columns']
    meta = bundle.get("metadata", {})
    model_name = meta.get("model_name", "unknown")

    items = fetch_tracked_items(limit=40)
    print(f"Active tracked items: {len(items)}")

    era_w = get_recent_era_weights(hours=6)
    baro_active = 1 if get_baro_active() else 0
    supply_map = get_supply_index_bulk(era_weights=era_w, only_active=True, top_n=1000)
    df_all_recent = fetch_recent_all(limit=2500)
    index_df = compute_market_index(df_all_recent)

    results = []
    for it in items:
        try:
            df = fetch_latest_per_item(it, limit=200)
            if df is None or len(df) < 20: continue
            try:
                code = int(item_encoder.transform([it])[0])
            except Exception:
                continue

            X = create_features(df, code, feature_columns, era_w, float(supply_map.get(it,0.0)), baro_active, index_df)

            model_pred = float(model.predict(X)[0])
            lin_pred = float(linear_projection(df['sell_median'].astype(float), steps=HORIZON_STEPS))
            final_pred = BLEND_WEIGHT * model_pred + (1.0 - BLEND_WEIGHT) * lin_pred

            if q_low is not None and q_high is not None:
                lo = float(q_low.predict(X)[0]); hi = float(q_high.predict(X)[0])
                lower = min(lo, hi); upper = max(lo, hi)
            else:
                sd = float(df['sell_median'].rolling(24, min_periods=1).std().iloc[-1] or 0)
                lower = final_pred - 1.2*sd; upper = final_pred + 1.2*sd

            results.append({
                "item": it,
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "current_price": float(df['sell_median'].iloc[-1]),
                "predicted_price": float(final_pred),
                "predicted_lower": float(lower),
                "predicted_upper": float(upper),
                "confidence": float(max(0.4, min(0.98, 1.0 - abs(final_pred - lin_pred)/((abs(df['sell_median'].iloc[-1]) + 1e-9))))),
                "model_name": model_name,
                "baseline_price": float(lin_pred),
                "blend_weight": float(BLEND_WEIGHT)
            })
        except Exception as e:
            print(f"Inference failed for {it}: {e}")

    print(f"Generated {len(results)} predictions")
    if results:
        store_predictions(results)
        print("Stored predictions")

if __name__ == "__main__":
    main()