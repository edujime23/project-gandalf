#!/usr/bin/env python
"""
Unified model trainer for Gandalf.
- Trains one model across all items
- Feature schema matches scripts/infer_unified.py
- Saves:
  - models/unified_model.pkl (model, item_encoder, feature_columns, metadata)
  - models/unified_metadata.json
"""

import os
import json
import time
import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional, List, Dict

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

# Training params (configurable via env)
DAYS = int(os.getenv("TRAIN_DAYS", "30"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "1000"))
HORIZON_STEPS = int(os.getenv("PRED_HORIZON_STEPS", "6"))  # 6 steps ≈ 30m if 5m cadence
MIN_POINTS_PER_ITEM = int(os.getenv("MIN_POINTS_PER_ITEM", "60"))

FEATURE_COLUMNS_ORDER = [
    "item_encoded",
    "hour", "day_of_week", "day_of_month", "is_weekend",
    "price",
    "price_ma_6", "price_ma_24",
    "price_std_6", "price_std_24",
    "momentum_6", "momentum_24",
    "rsi",
    "volume", "volume_ma_24", "volume_ratio",
    "spread", "spread_ma", "spread_ratio",
    "order_imbalance",
]

def fetch_training_data(days: int = DAYS, page_size: int = PAGE_SIZE) -> pd.DataFrame:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    base = f"{SUPABASE_URL}/rest/v1/market_data"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Prefer": "count=exact",
        "Range-Unit": "items",
    }
    params = {
        "select": "*",
        "timestamp": f"gte.{cutoff_iso}",
        "order": "timestamp.asc",
    }

    all_rows = []
    offset = 0
    page = 0
    while True:
        h = {**headers, "Range": f"{offset}-{offset+page_size-1}"}
        r = requests.get(base, headers=h, params=params, timeout=30)
        if not r.ok:
            raise requests.HTTPError(f"{r.status_code} {r.text} :: {r.url}")
        chunk = r.json()
        all_rows.extend(chunk)
        content_range = r.headers.get("Content-Range", "unknown")
        print(f"[fetch] page={page} rows={len(chunk)} range={content_range}")
        if len(chunk) < page_size:
            break
        offset += page_size
        page += 1

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    for col in ["sell_median", "buy_median", "spread", "sell_orders", "buy_orders"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "item"]).reset_index(drop=True)
    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_item_features(item_df: pd.DataFrame, item_code: int, horizon_steps: int = HORIZON_STEPS
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    df = item_df.sort_values("timestamp").copy()
    if len(df) < MIN_POINTS_PER_ITEM:
        return None, None

    price = df["sell_median"].ffill()

    feat = pd.DataFrame(index=df.index)
    feat["item_encoded"] = item_code
    feat["hour"] = df["timestamp"].dt.hour
    feat["day_of_week"] = df["timestamp"].dt.dayofweek
    feat["day_of_month"] = df["timestamp"].dt.day
    feat["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)

    feat["price"] = price
    feat["price_ma_6"] = price.rolling(6, min_periods=1).mean()
    feat["price_ma_24"] = price.rolling(24, min_periods=1).mean()
    feat["price_std_6"] = price.rolling(6, min_periods=1).std()
    feat["price_std_24"] = price.rolling(24, min_periods=1).std()

    feat["momentum_6"] = price.pct_change(6)
    feat["momentum_24"] = price.pct_change(24)

    feat["rsi"] = compute_rsi(price, period=14)

    volume = (df.get("sell_orders", 0).fillna(0) + df.get("buy_orders", 0).fillna(0)).astype(float)
    feat["volume"] = volume
    feat["volume_ma_24"] = volume.rolling(24, min_periods=1).mean()
    feat["volume_ratio"] = volume / (feat["volume_ma_24"] + 1)

    spread = df.get("spread", pd.Series(index=df.index, dtype=float)).fillna(0)
    feat["spread"] = spread
    feat["spread_ma"] = spread.rolling(12, min_periods=1).mean()
    feat["spread_ratio"] = spread / (feat["price"] + 1)

    buys = df.get("buy_orders", pd.Series(index=df.index, dtype=float)).fillna(0)
    sells = df.get("sell_orders", pd.Series(index=df.index, dtype=float)).fillna(0)
    feat["order_imbalance"] = (buys - sells) / (buys + sells + 1)

    target = price.shift(-horizon_steps).rename("future_price")

    data = pd.concat([feat, target], axis=1).dropna()
    if data.empty:
        return None, None

    X = data[FEATURE_COLUMNS_ORDER].copy()
    y = data["future_price"].copy()
    return X, y

def train_unified_model(df: pd.DataFrame) -> Tuple[GradientBoostingRegressor, LabelEncoder, List[str], Dict]:
    items = df["item"].dropna().unique().tolist()
    items.sort()
    if not items:
        raise RuntimeError("No items found in training data")

    le = LabelEncoder()
    le.fit(items)

    X_list, y_list = [], []
    kept_items = 0
    for item in items:
        item_df = df[df["item"] == item]
        code = int(le.transform([item])[0])
        Xi, yi = build_item_features(item_df, code)
        if Xi is not None and yi is not None and len(Xi) >= 50:
            X_list.append(Xi)
            y_list.append(yi)
            kept_items += 1

    if not X_list:
        raise RuntimeError("No items produced valid training features")

    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, shuffle=True
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time_s = time.time() - t0

    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    importances = getattr(model, "feature_importances_", None)
    feat_importance = []
    if importances is not None:
        feat_importance = [
            {"feature": FEATURE_COLUMNS_ORDER[i], "importance": float(importances[i])}
            for i in range(len(FEATURE_COLUMNS_ORDER))
        ]
        feat_importance.sort(key=lambda d: d["importance"], reverse=True)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "GradientBoostingRegressor",
        "train_days": DAYS,
        "horizon_steps": HORIZON_STEPS,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_items": int(kept_items),
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "train_time_seconds": round(train_time_s, 2),
        "feature_importance": feat_importance,
    }

    return model, le, FEATURE_COLUMNS_ORDER, metadata

def main():
    print("=== Unified Model Trainer ===")
    print(f"Fetching last {DAYS} days of data...")
    df = fetch_training_data(DAYS, PAGE_SIZE)
    if df.empty:
        print("No data returned. Aborting.")
        return

    print(f"Rows: {len(df):,} | Items: {df['item'].nunique()} | Window: {df['timestamp'].min()} → {df['timestamp'].max()}")

    model, item_encoder, feature_cols, metadata = train_unified_model(df)

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": model, "item_encoder": item_encoder, "feature_columns": feature_cols, "metadata": metadata},
        "models/unified_model.pkl"
    )
    with open("models/unified_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved: models/unified_model.pkl")
    print("Saved: models/unified_metadata.json")
    print(f"R2 (train): {metadata['r2_train']:.4f} | R2 (test): {metadata['r2_test']:.4f}")
    print("=== Done ===")

if __name__ == "__main__":
    main()