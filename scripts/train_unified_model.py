#!/usr/bin/env python
"""
Unified model trainer (enhanced):
- Uses market data + worldstate (fissure era weights, Baro) + supply index (relic/rarity/drop)
- Richer time-series features (EMAs, lags, returns, zscores, spread/depth)
- GradientBoostingRegressor (portable) with feature importance
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
from typing import List, Dict, Tuple, Optional

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

# Training params
DAYS = int(os.getenv("TRAIN_DAYS", "30"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "1000"))
HORIZON_STEPS = int(os.getenv("PRED_HORIZON_STEPS", "6"))  # 6*5m ≈ 30m
MIN_POINTS_PER_ITEM = int(os.getenv("MIN_POINTS_PER_ITEM", "80"))

FEATURE_COLUMNS_ORDER = [
    # Time
    "item_encoded","hour","day_of_week","day_of_month","is_weekend",
    # Price/volume/depth/spread
    "price","ema_6","ema_24","ema_72","lag_1","lag_6","lag_24",
    "ret_1","ret_6","ret_24","zscore_24",
    "volume","volume_ma_24","volume_ratio","spread","spread_ma","spread_std",
    "depth","imbalance",
    # Worldstate + supply
    "w_lith","w_meso","w_neo","w_axi","baro_active","supply_index"
]

def sb_get(path: str, params=None, headers_extra=None):
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    if headers_extra: headers.update(headers_extra)
    r = requests.get(f"{SUPABASE_URL}{path}", headers=headers, params=params or {}, timeout=30)
    r.raise_for_status()
    return r

def fetch_market_data(days=DAYS, page_size=PAGE_SIZE) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).replace(microsecond=0).isoformat().replace("+00:00","Z")
    base = f"/rest/v1/market_data"
    params = {"select": "*", "timestamp": f"gte.{cutoff}", "order": "timestamp.asc"}

    all_rows, offset = [], 0
    while True:
        r = sb_get(base, params=params, headers_extra={"Range":"%d-%d"%(offset, offset+page_size-1), "Range-Unit":"items"})
        chunk = r.json()
        if not chunk:
            break
        all_rows.extend(chunk)
        if len(chunk) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp","item"]).sort_values("timestamp")
    for col in ["sell_median","buy_median","spread","sell_orders","buy_orders"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fetch_worldstate(days=DAYS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).replace(microsecond=0).isoformat().replace("+00:00","Z")
    fiss = sb_get("/rest/v1/worldstate_fissures", params={"select":"*", "taken_at": f"gte.{cutoff}", "order":"taken_at.asc"}).json()
    flags = sb_get("/rest/v1/worldstate_flags", params={"select":"*", "taken_at": f"gte.{cutoff}", "order":"taken_at.asc"}).json()

    df_f = pd.DataFrame(fiss)
    df_b = pd.DataFrame(flags)
    if not df_f.empty:
        df_f["taken_at"] = pd.to_datetime(df_f["taken_at"], utc=True)
    if not df_b.empty:
        df_b["taken_at"] = pd.to_datetime(df_b["taken_at"], utc=True)
    return df_f, df_b

def build_era_weights(df_fiss: pd.DataFrame) -> pd.DataFrame:
    if df_fiss.empty:
        return pd.DataFrame(columns=["taken_at","w_lith","w_meso","w_neo","w_axi","fissure_total"])
    piv = df_fiss.pivot_table(index="taken_at", columns="era", values="count", aggfunc="sum").fillna(0)
    for era in ["Lith","Meso","Neo","Axi"]:
        if era not in piv.columns:
            piv[era] = 0
    piv["sum"] = piv[["Lith","Meso","Neo","Axi"]].sum(axis=1)
    # Avoid divide-by-zero
    for era, col in zip(["Lith","Meso","Neo","Axi"], ["w_lith","w_meso","w_neo","w_axi"]):
        piv[col] = np.where(piv["sum"]>0, piv[era]/piv["sum"], 0.25)
    piv["fissure_total"] = piv["sum"]
    out = piv[["w_lith","w_meso","w_neo","w_axi","fissure_total"]].reset_index()
    return out

def fetch_supply_index_map(items: List[str], df_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Precompute supply_index per item per snapshot by calling SQL locally would be heavy.
    We approximate using the latest weights per hour to reduce density:
      - Downsample weights to 1 snapshot per hour and compute indices per item.
    """
    if not items:
        return pd.DataFrame(columns=["taken_at","item","supply_index"])
    if df_weights.empty:
        # default zeros
        rows = []
        for it in items:
            rows.append({"taken_at": pd.Timestamp.utcnow(), "item": it, "supply_index": 0.0})
        return pd.DataFrame(rows)

    # Downsample weights (hourly)
    W = df_weights.copy()
    W["taken_at_hour"] = W["taken_at"].dt.floor("60min")
    Wh = (W.groupby("taken_at_hour")[["w_lith","w_meso","w_neo","w_axi"]].mean().reset_index()
          .rename(columns={"taken_at_hour":"taken_at"}))

    # Call bulk supply-index via RPC for each snapshot
    rows = []
    for _, rw in Wh.iterrows():
        era_weights = {
            "Lith": float(rw["w_lith"]),
            "Meso": float(rw["w_meso"]),
            "Neo":  float(rw["w_neo"]),
            "Axi":  float(rw["w_axi"]),
        }
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/get_supply_index_bulk",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"},
            data=json.dumps({"era_weights": era_weights, "only_active": False, "top_n": 10000}),
            timeout=60
        )
        if not r.ok:
            continue
        data = r.json() or []
        for d in data:
            rows.append({"taken_at": rw["taken_at"], "item": d["set_item"], "supply_index": float(d["supply_index"] or 0)})

    return pd.DataFrame(rows)

def add_features(item_df: pd.DataFrame, snapshot_join: pd.DataFrame) -> pd.DataFrame:
    df = item_df.sort_values("timestamp").copy()
    price = df["sell_median"].astype(float).ffill()
    volume = (df.get("sell_orders",0).fillna(0) + df.get("buy_orders",0).fillna(0)).astype(float)
    spread = df.get("spread",0).fillna(0).astype(float)
    buys = df.get("buy_orders",0).fillna(0).astype(float)
    sells = df.get("sell_orders",0).fillna(0).astype(float)

    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    feat = pd.DataFrame(index=df.index)
    # Time
    feat["hour"] = df["timestamp"].dt.hour
    feat["day_of_week"] = df["timestamp"].dt.dayofweek
    feat["day_of_month"] = df["timestamp"].dt.day
    feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(int)
    # Price/volume
    feat["price"] = price
    feat["ema_6"] = ema(price, 6)
    feat["ema_24"] = ema(price, 24)
    feat["ema_72"] = ema(price, 72)
    feat["lag_1"] = price.shift(1)
    feat["lag_6"] = price.shift(6)
    feat["lag_24"] = price.shift(24)
    feat["ret_1"] = price.pct_change(1)
    feat["ret_6"] = price.pct_change(6)
    feat["ret_24"] = price.pct_change(24)
    feat["zscore_24"] = (price - price.rolling(24, min_periods=1).mean()) / (price.rolling(24, min_periods=1).std().replace(0,np.nan))
    feat["volume"] = volume
    feat["volume_ma_24"] = volume.rolling(24, min_periods=1).mean()
    feat["volume_ratio"] = volume / (feat["volume_ma_24"] + 1e-9)
    feat["spread"] = spread
    feat["spread_ma"] = spread.rolling(24, min_periods=1).mean()
    feat["spread_std"] = spread.rolling(24, min_periods=1).std()
    feat["depth"] = buys + sells
    feat["imbalance"] = (buys - sells) / (buys + sells + 1e-9)

    # Join worldstate & supply (nearest snapshot)
    # snapshot_join columns: timestamp, w_*, baro_active, supply_index
    merged = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"),
        snapshot_join.sort_values("timestamp"),
        left_on="timestamp", right_on="timestamp", direction="backward", tolerance=pd.Timedelta("6H")
    )

    for c in ["w_lith","w_meso","w_neo","w_axi","baro_active","supply_index"]:
        feat[c] = merged[c].values if c in merged.columns else 0.0

    # Target
    target = price.shift(-HORIZON_STEPS).rename("future_price")

    final = pd.concat([feat, target], axis=1)
    final = final.replace([np.inf, -np.inf], np.nan).dropna()
    return final

def build_snapshot_join(weights_df: pd.DataFrame, flags_df: pd.DataFrame, supply_df: pd.DataFrame, item: str) -> pd.DataFrame:
    if weights_df.empty:
        base = pd.DataFrame({
            "timestamp":[pd.Timestamp.utcnow()],
            "w_lith":[0.25],"w_meso":[0.25],"w_neo":[0.25],"w_axi":[0.25],
            "baro_active":[0],
            "supply_index":[0.0]
        })
        return base
    w = weights_df.copy().rename(columns={"taken_at":"timestamp"})
    # Baro flag nearest
    if not flags_df.empty:
        fb = flags_df[["taken_at","baro_active"]].rename(columns={"taken_at":"timestamp"}).copy()
        fb["baro_active"] = fb["baro_active"].astype(int)
        w = pd.merge_asof(w.sort_values("timestamp"), fb.sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta("12H"))
    else:
        w["baro_active"] = 0

    # Supply index for item
    if not supply_df.empty:
        s = supply_df[supply_df["item"]==item][["taken_at","supply_index"]].rename(columns={"taken_at":"timestamp"})
        w = pd.merge_asof(w.sort_values("timestamp"), s.sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta("12H"))
    else:
        w["supply_index"] = 0.0

    w["supply_index"] = w["supply_index"].fillna(0.0)
    w["baro_active"] = w["baro_active"].fillna(0).astype(int)
    return w[["timestamp","w_lith","w_meso","w_neo","w_axi","baro_active","supply_index"]]

def train_unified_model(df: pd.DataFrame) -> Tuple[GradientBoostingRegressor, LabelEncoder, List[str], Dict]:
    items = df["item"].dropna().unique().tolist()
    items.sort()
    if not items:
        raise RuntimeError("No items found in training data")

    # Worldstate snapshots and supply index precompute
    df_fiss, df_flags = fetch_worldstate(days=DAYS)
    weights_df = build_era_weights(df_fiss)
    supply_df = fetch_supply_index_map(items, weights_df)

    # Encode items
    le = LabelEncoder()
    le.fit(items)

    X_list, y_list = [], []
    kept_items = 0
    for item in items:
        item_df = df[df["item"] == item].copy()
        if len(item_df) < MIN_POINTS_PER_ITEM:
            continue
        code = int(le.transform([item])[0])
        snap = build_snapshot_join(weights_df, df_flags, supply_df, item)
        feat = add_features(item_df, snap)
        if feat.empty or feat["future_price"].isna().all():
            continue
        feat.insert(0, "item_encoded", code)
        X = feat[FEATURE_COLUMNS_ORDER].copy()
        y = feat["future_price"].copy()
        if len(X) >= 50:
            X_list.append(X)
            y_list.append(y)
            kept_items += 1

    if not X_list:
        raise RuntimeError("No items produced valid training features")

    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = GradientBoostingRegressor(
      n_estimators=600,
      learning_rate=0.03,
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
    print("=== Unified Model Trainer (enhanced) ===")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE/KEY")

    print(f"Fetching last {DAYS} days of data…")
    df = fetch_market_data(DAYS, PAGE_SIZE)
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