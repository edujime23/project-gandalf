#!/usr/bin/env python
"""
Unified model trainer (v3):
- Trains ONLY on items that have enough history for the full window
  (lags up to 168, EMA 72, 24-rolling features) + horizon.
- Skips short-history items instead of aborting.
- Time-series-aware CV with RandomizedSearch (LightGBM if available, else GBR).
- Quantile models for intervals (best-effort).
- Writes model metadata and saves bundle (model + encoders + features).
"""

import os
import json
import time
import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

# Training params (tunable via env)
DAYS = int(os.getenv("TRAIN_DAYS", "30"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "1000"))
HORIZON_STEPS = int(os.getenv("PRED_HORIZON_STEPS", "6"))  # 6*5m ~= 30m
N_CV_SPLITS = int(os.getenv("CV_SPLITS", "3"))
SEARCH_ITER = int(os.getenv("SEARCH_ITER", "20"))

# Feature windows used in this trainer (keep long windows as requested)
WINDOW_LAGS = [1, 6, 24, 48, 168]
ROLL_WINDOWS = [24, 72]  # ema_72 requires 72
BASE_WINDOW = max(WINDOW_LAGS + ROLL_WINDOWS)  # 168
# Required history per item to build at least some valid rows
REQUIRED_HISTORY_STEPS = int(os.getenv("REQUIRED_HISTORY_STEPS", str(BASE_WINDOW + HORIZON_STEPS)))
# After trimming early rows, ensure we still have some training rows:
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "64"))

FEATURES_VERSION = 3  # bump on feature changes

def sb_get(path: str, params=None, headers_extra=None):
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    if headers_extra: headers.update(headers_extra)
    r = requests.get(f"{SUPABASE_URL}{path}", headers=headers, params=params or {}, timeout=30)
    r.raise_for_status()
    return r

def sb_post(path: str, body):
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
    r = requests.post(f"{SUPABASE_URL}{path}", headers=headers, data=json.dumps(body), timeout=60)
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
        if not chunk: break
        all_rows.extend(chunk)
        if len(chunk) < page_size: break
        offset += page_size

    df = pd.DataFrame(all_rows)
    if df.empty: return df
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
    if not df_f.empty: df_f["taken_at"] = pd.to_datetime(df_f["taken_at"], utc=True)
    if not df_b.empty: df_b["taken_at"] = pd.to_datetime(df_b["taken_at"], utc=True)
    return df_f, df_b

def build_era_weights(df_fiss: pd.DataFrame) -> pd.DataFrame:
    if df_fiss.empty:
        return pd.DataFrame(columns=["taken_at","w_lith","w_meso","w_neo","w_axi","fissure_total"])
    piv = df_fiss.pivot_table(index="taken_at", columns="era", values="count", aggfunc="sum").fillna(0)
    for era in ["Lith","Meso","Neo","Axi"]:
        if era not in piv.columns: piv[era] = 0
    piv["sum"] = piv[["Lith","Meso","Neo","Axi"]].sum(axis=1)
    for era, col in zip(["Lith","Meso","Neo","Axi"], ["w_lith","w_meso","w_neo","w_axi"]):
        piv[col] = np.where(piv["sum"]>0, piv[era]/piv["sum"], 0.25)
    piv["fissure_total"] = piv["sum"]
    return piv[["w_lith","w_meso","w_neo","w_axi","fissure_total"]].reset_index()

def compute_market_index(df_all: pd.DataFrame) -> pd.DataFrame:
    idx = df_all.groupby("timestamp")["sell_median"].mean().reset_index().rename(columns={"sell_median":"index_price"})
    idx = idx.sort_values("timestamp")
    idx["index_ret_6"] = idx["index_price"].pct_change(6)
    idx["index_ret_24"] = idx["index_price"].pct_change(24)
    roll24 = idx["index_price"].rolling(24, min_periods=2)
    mu24 = roll24.mean()
    sd24 = roll24.std().replace(0, np.nan)
    idx["index_zscore_24"] = (idx["index_price"] - mu24) / sd24
    idx = idx.replace([np.inf, -np.inf], 0).fillna(0)
    return idx

def fetch_supply_index_map(items: List[str], weights_df: pd.DataFrame) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["taken_at","item","supply_index"])
    if weights_df.empty:
        rows = [{"taken_at": pd.Timestamp.utcnow(), "item": it, "supply_index": 0.0} for it in items]
        return pd.DataFrame(rows)

    W = weights_df.copy()
    W["taken_at_hour"] = W["taken_at"].dt.floor("60min")
    Wh = (W.groupby("taken_at_hour")[["w_lith","w_meso","w_neo","w_axi"]].mean().reset_index()
          .rename(columns={"taken_at_hour":"taken_at"}))

    rows = []
    for _, rw in Wh.iterrows():
        era_weights = {"Lith": float(rw["w_lith"]), "Meso": float(rw["w_meso"]), "Neo": float(rw["w_neo"]), "Axi": float(rw["w_axi"])}
        r = sb_post("/rest/v1/rpc/get_supply_index_bulk", {"era_weights": era_weights, "only_active": False, "top_n": 10000})
        data = r.json() or []
        for d in data:
            rows.append({"taken_at": rw["taken_at"], "item": d["set_item"], "supply_index": float(d["supply_index"] or 0)})
    return pd.DataFrame(rows)

def build_snapshot_join(weights_df: pd.DataFrame, flags_df: pd.DataFrame, supply_df: pd.DataFrame, index_df: pd.DataFrame, item: str) -> pd.DataFrame:
    base = weights_df.rename(columns={"taken_at":"timestamp"}).copy()
    if base.empty:
        base = pd.DataFrame({
            "timestamp":[pd.Timestamp.utcnow()],
            "w_lith":[0.25],"w_meso":[0.25],"w_neo":[0.25],"w_axi":[0.25],
            "fissure_total":[0]
        })
    # Baro flag
    if not flags_df.empty:
        fb = flags_df[["taken_at","baro_active"]].rename(columns={"taken_at":"timestamp"}).copy()
        fb["baro_active"] = fb["baro_active"].astype(int)
        base = pd.merge_asof(base.sort_values("timestamp"),
                             fb.sort_values("timestamp"),
                             on="timestamp", direction="backward", tolerance=pd.Timedelta(hours=12))
    else:
        base["baro_active"] = 0

    # Supply per item
    if not supply_df.empty:
        s = supply_df[supply_df["item"]==item][["taken_at","supply_index"]].rename(columns={"taken_at":"timestamp"})
        base = pd.merge_asof(base.sort_values("timestamp"),
                             s.sort_values("timestamp"),
                             on="timestamp", direction="backward", tolerance=pd.Timedelta(hours=12))
    else:
        base["supply_index"] = 0.0

    # Market index
    if not index_df.empty:
        idx = index_df.copy()
        base = pd.merge_asof(base.sort_values("timestamp"),
                             idx.sort_values("timestamp"),
                             on="timestamp", direction="backward", tolerance=pd.Timedelta(hours=1))
    else:
        base["index_price"] = 0.0
        base["index_ret_6"] = 0.0
        base["index_ret_24"] = 0.0
        base["index_zscore_24"] = 0.0

    base = base.fillna({"supply_index":0.0, "baro_active":0,
                        "index_price":0.0, "index_ret_6":0.0, "index_ret_24":0.0, "index_zscore_24":0.0})
    return base

def add_features(item_df: pd.DataFrame, snapshot_join: pd.DataFrame, item: str) -> pd.DataFrame:
    df = item_df.sort_values("timestamp").copy()
    price = df["sell_median"].astype(float).ffill()
    volume = (df.get("sell_orders",0).fillna(0) + df.get("buy_orders",0).fillna(0)).astype(float)
    spread = df.get("spread",0).fillna(0).astype(float)
    buys = df.get("buy_orders",0).fillna(0).astype(float)
    sells = df.get("sell_orders",0).fillna(0).astype(float)

    def ema(series, span): return series.ewm(span=span, adjust=False).mean()

    feat = pd.DataFrame(index=df.index)

    # Time/cyclical
    feat["hour"] = df["timestamp"].dt.hour
    feat["day_of_week"] = df["timestamp"].dt.dayofweek
    feat["day_of_month"] = df["timestamp"].dt.day
    feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(int)
    feat["hour_sin"] = np.sin(2*np.pi*feat["hour"]/24)
    feat["hour_cos"] = np.cos(2*np.pi*feat["hour"]/24)
    feat["dow_sin"] = np.sin(2*np.pi*feat["day_of_week"]/7)
    feat["dow_cos"] = np.cos(2*np.pi*feat["day_of_week"]/7)

    # Price/volume/depth/spread (long windows kept)
    feat["price"] = price
    feat["ema_6"] = ema(price, 6)
    feat["ema_24"] = ema(price, 24)
    feat["ema_72"] = ema(price, 72)

    feat["lag_1"]   = price.shift(1)
    feat["lag_6"]   = price.shift(6)
    feat["lag_24"]  = price.shift(24)
    feat["lag_48"]  = price.shift(48)
    feat["lag_168"] = price.shift(168)

    feat["ret_1"]   = price.pct_change(1)
    feat["ret_6"]   = price.pct_change(6)
    feat["ret_24"]  = price.pct_change(24)
    feat["ret_48"]  = price.pct_change(48)
    feat["ret_168"] = price.pct_change(168)

    mu24 = price.rolling(24, min_periods=2).mean()
    sd24 = price.rolling(24, min_periods=2).std()
    feat["zscore_24"] = ((price - mu24) / sd24.replace(0, np.nan)).fillna(0)

    feat["volume"] = volume
    feat["volume_ma_24"] = volume.rolling(24, min_periods=1).mean()
    feat["volume_ratio"] = volume / (feat["volume_ma_24"] + 1e-9)
    feat["spread"] = spread
    feat["spread_ma"] = spread.rolling(24, min_periods=1).mean()
    feat["spread_std"] = spread.rolling(24, min_periods=2).std()
    feat["depth"] = buys + sells
    feat["imbalance"] = (buys - sells) / (buys + sells + 1e-9)

    # Snapshot join (worldstate + supply + market index)
    merged = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"),
        snapshot_join.sort_values("timestamp"),
        on="timestamp", direction="backward", tolerance=pd.Timedelta(hours=6)
    )
    for c in ["w_lith","w_meso","w_neo","w_axi","fissure_total","baro_active",
              "supply_index","index_price","index_ret_6","index_ret_24","index_zscore_24"]:
        feat[c] = merged[c].values if c in merged.columns else 0.0

    # Target
    feat["future_price"] = price.shift(-HORIZON_STEPS)

    # Trim early rows so all lags are valid, then drop rows missing target
    trim = max(BASE_WINDOW, 72, 24)  # safe trim: 168
    feat = feat.iloc[trim:].copy()
    feat = feat[feat["future_price"].notna()]
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)

    # Tag item for later encoding
    feat["__item__"] = item
    return feat

def time_series_train_test_split(X, y, test_size=0.2):
    split_point = int(len(X) * (1 - test_size))
    return X.iloc[:split_point], X.iloc[split_point:], y.iloc[:split_point], y.iloc[split_point:]

def main():
    print("=== Unified Model Trainer v3 ===")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE/KEY")

    print(f"Fetching last {DAYS} days of data…")
    df_all = fetch_market_data(DAYS, PAGE_SIZE)
    if df_all.empty:
        print("No data returned. Aborting.")
        return
    print(f"Rows: {len(df_all):,} | Items: {df_all['item'].nunique()} | Window: {df_all['timestamp'].min()} → {df_all['timestamp'].max()}")

    # Worldstate & market index
    df_fiss, df_flags = fetch_worldstate(days=DAYS)
    weights_df = build_era_weights(df_fiss)
    index_df = compute_market_index(df_all)

    items = sorted(df_all["item"].dropna().unique().tolist())

    # Precompute supply index per item/time
    supply_df = fetch_supply_index_map(items, weights_df)

    # Build per-item features but keep only items with enough history
    keep_feats = []
    kept_items = []
    summary = []

    for item in items:
        item_df = df_all[df_all["item"] == item].copy()
        # Only count rows with a valid price
        valid_points = int(item_df["sell_median"].notna().sum())
        if valid_points < REQUIRED_HISTORY_STEPS:
            summary.append((item, valid_points, "skip_short"))
            continue

        snap = build_snapshot_join(weights_df, df_flags, supply_df, index_df, item)
        feat = add_features(item_df, snap, item)

        if feat.empty or len(feat) < MIN_TRAIN_ROWS:
            summary.append((item, len(feat), "skip_few_rows"))
            continue

        keep_feats.append(feat)
        kept_items.append(item)
        summary.append((item, len(feat), "kept"))

    kept_count = len(kept_items)
    print(f"Items kept for training: {kept_count} / {len(items)}")
    if kept_count == 0:
        print("No items meet the history window. Aborting.")
        return

    # Log a small summary (top 15)
    for it, n, why in summary[:15]:
        print(f"  {it:>24} -> rows={n} [{why}]")
    if len(summary) > 15:
        print(f"  … {len(summary)-15} more")

    # Fit encoder on kept items only
    le = LabelEncoder().fit(kept_items)

    # Assemble X, y
    X_list, y_list = [], []
    for feat in keep_feats:
        item = feat["__item__"].iloc[0]
        code = int(le.transform([item])[0])
        feat = feat.drop(columns=["__item__"])
        feat.insert(0, "item_encoded", code)
        X_list.append(feat.drop(columns=["future_price"]))
        y_list.append(feat["future_price"])

    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)

    # Define feature columns (strict order)
    FEATURE_COLUMNS_ORDER = [
        "item_encoded","hour","day_of_week","day_of_month","is_weekend",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "price","ema_6","ema_24","ema_72",
        "lag_1","lag_6","lag_24","lag_48","lag_168",
        "ret_1","ret_6","ret_24","ret_48","ret_168",
        "zscore_24",
        "volume","volume_ma_24","volume_ratio",
        "spread","spread_ma","spread_std",
        "depth","imbalance",
        "w_lith","w_meso","w_neo","w_axi","fissure_total","baro_active","supply_index",
        "index_price","index_ret_6","index_ret_24","index_zscore_24"
    ]
    X = X[FEATURE_COLUMNS_ORDER].copy()

    # Time-series split (no leakage)
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, test_size=0.2)

    # Try LightGBM; fallback to sklearn GB
    use_lgbm = False
    try:
        import lightgbm as lgb  # noqa: F401
        use_lgbm = True
    except Exception:
        pass

    t0 = time.time()
    if use_lgbm:
        from scipy.stats import randint, uniform
        import lightgbm as lgb
        base = lgb.LGBMRegressor(random_state=42, n_estimators=800, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.9)
        param_dist = {
            "num_leaves": randint(16, 96),
            "max_depth": randint(3, 10),
            "min_child_samples": randint(10, 60),
            "reg_alpha": uniform(0, 0.5),
            "reg_lambda": uniform(0, 0.8)
        }
        tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=SEARCH_ITER,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            cv=tscv,
            refit=True,
            random_state=42
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        model_name = "lgbm"
    else:
        model = GradientBoostingRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=3,
            min_samples_split=20, min_samples_leaf=10, subsample=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        best_params = {k:getattr(model, k) for k in ["n_estimators","learning_rate","max_depth","min_samples_split","min_samples_leaf","subsample"]}
        model_name = "gbr"

    train_time_s = time.time() - t0

    # Evaluation vs naive baseline
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))))
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    baseline = X_test["price"].values
    baseline_mae = mean_absolute_error(y_test, baseline)
    baseline_mape = float(np.mean(np.abs((y_test - baseline) / (y_test + 1e-9))))

    # Quantile models for intervals (best effort)
    q_low = q_high = None
    try:
        if use_lgbm:
            import lightgbm as lgb
            common = model.get_params()
            q_low = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **{k:common[k] for k in common if k in ["num_leaves","max_depth","min_child_samples","learning_rate","n_estimators","subsample","colsample_bytree","reg_alpha","reg_lambda","random_state"]})
            q_high = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **{k:common[k] for k in common if k in ["num_leaves","max_depth","min_child_samples","learning_rate","n_estimators","subsample","colsample_bytree","reg_alpha","reg_lambda","random_state"]})
            q_low.fit(X_train, y_train); q_high.fit(X_train, y_train)
        else:
            q_low = GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=42,
                                              n_estimators=400, learning_rate=0.03, max_depth=3,
                                              min_samples_split=20, min_samples_leaf=10, subsample=0.8)
            q_high = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=42,
                                               n_estimators=400, learning_rate=0.03, max_depth=3,
                                               min_samples_split=20, min_samples_leaf=10, subsample=0.8)
            q_low.fit(X_train, y_train); q_high.fit(X_train, y_train)
    except Exception as e:
        print("Quantile models failed:", e)
        q_low = q_high = None

    # Save models + metadata
    os.makedirs("models", exist_ok=True)
    payload = {
        "model": model,
        "quantile_low": q_low,
        "quantile_high": q_high,
        "item_encoder": le,
        "feature_columns": X.columns.tolist(),
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "features_version": FEATURES_VERSION,
            "train_days": DAYS,
            "horizon_steps": HORIZON_STEPS,
            "n_samples": int(len(X)),
            "n_items": int(kept_count),
            "r2_train": float(r2_train),
            "r2_test": float(r2_test),
            "mae_test": float(mae),
            "mape_test": float(mape),
            "baseline_mae": float(baseline_mae),
            "baseline_mape": float(baseline_mape),
            "train_time_seconds": round(train_time_s, 2),
            "params": best_params,
            "required_history_steps": REQUIRED_HISTORY_STEPS,
            "min_train_rows": MIN_TRAIN_ROWS
        }
    }
    joblib.dump(payload, "models/unified_model.pkl")
    with open("models/unified_metadata.json", "w") as f:
        json.dump(payload["metadata"], f, indent=2)

    # Write model_performance (if your table differs, adjust keys accordingly)
    try:
        post = {
            "model_name": model_name,
            "features_version": FEATURES_VERSION,
            "horizon_steps": HORIZON_STEPS,
            "n_samples": int(len(X)), "n_items": int(kept_count),
            "r2_train": float(r2_train), "r2_test": float(r2_test),
            "mae_test": float(mae), "mape_test": float(mape),
            "baseline_mae": float(baseline_mae), "baseline_mape": float(baseline_mape),
            "params": best_params
        }
        sb_post("/rest/v1/model_performance", post)
    except Exception as e:
        print("Failed to write model_performance:", e)

    print(f"Saved models/unified_model.pkl")
    print(f"Kept items: {kept_count} | Samples: {len(X)}")
    print(f"R2 train={r2_train:.4f} test={r2_test:.4f} | MAE={mae:.3f} vs baseline {baseline_mae:.3f}")
    print("=== Done ===")

if __name__ == "__main__":
    main()