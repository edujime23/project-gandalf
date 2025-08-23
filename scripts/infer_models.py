import os, json, glob
import pandas as pd
import numpy as np
import requests, joblib
from datetime import datetime, timedelta, timezone

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def fetch_recent(days=7, page_size=1000):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).replace(microsecond=0)
    ts = cutoff.isoformat().replace("+00:00", "Z")
    base = f"{SUPABASE_URL}/rest/v1/market_data"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Prefer": "count=exact",
        "Range-Unit": "items",
    }
    params = {"select": "*", "timestamp": f"gte.{ts}", "order": "timestamp.asc"}

    rows, offset, page = [], 0, 0
    while True:
        h = {**headers, "Range": f"{offset}-{offset+page_size-1}"}
        r = requests.get(base, headers=h, params=params, timeout=30)
        if not r.ok:
            raise requests.HTTPError(f"{r.status_code} {r.text} for URL {r.url}")
        chunk = r.json()
        rows.extend(chunk)
        print(f"DEBUG: page {page} fetched {len(chunk)} rows ({r.headers.get('Content-Range')})")
        if len(chunk) < page_size:
            break
        offset += page_size
        page += 1

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    for col in ["sell_median", "buy_median", "spread", "sell_orders", "buy_orders"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_features(item_df):
    item_df = item_df.sort_values("timestamp").copy()
    if len(item_df) < 60:
        return None, None, None

    X = pd.DataFrame(index=item_df.index)
    price = item_df["sell_median"]
    vol = (item_df.get("sell_orders", 0).fillna(0) + item_df.get("buy_orders", 0).fillna(0))

    # Time
    X["hour"] = item_df["timestamp"].dt.hour
    X["day_of_week"] = item_df["timestamp"].dt.dayofweek
    X["day_of_month"] = item_df["timestamp"].dt.day

    # Price
    X["price_ma_6"] = price.rolling(6).mean()
    X["price_ma_24"] = price.rolling(24).mean()
    X["price_std_24"] = price.rolling(24).std()
    X["price_change_6h"] = price.pct_change(6)
    X["price_change_24h"] = price.pct_change(24)

    # Volume
    X["volume"] = vol
    X["volume_ma_24"] = vol.rolling(24).mean()
    X["volume_ratio"] = vol / (X["volume_ma_24"] + 1)

    # Spread
    if "spread" in item_df.columns:
        X["spread"] = item_df["spread"]
        X["spread_ma"] = X["spread"].rolling(24).mean()
        X["spread_ratio"] = X["spread"] / (price + 1)
    else:
        X["spread"] = np.nan
        X["spread_ma"] = np.nan
        X["spread_ratio"] = np.nan

    # Last valid row for inference
    X = X.dropna()
    if X.empty:
        return None, None, None

    current_price = float(price.loc[X.index[-1]])
    # Dynamic step size -> horizon minutes for +6 steps
    ts = item_df.loc[X.index, "timestamp"]
    if len(ts) >= 2:
        step_minutes = max(1, np.median(np.diff(ts.values).astype("timedelta64[m]").astype(float)))
    else:
        step_minutes = 5
    horizon_minutes = int(step_minutes * 6)
    return X.iloc[-1], current_price, horizon_minutes

def upsert_predictions(preds):
    if not preds:
        print("No predictions to upsert.")
        return
    url = f"{SUPABASE_URL}/rest/v1/predictions"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    r = requests.post(url, headers=headers, data=json.dumps(preds), timeout=30)
    if not r.ok:
        raise requests.HTTPError(f"Upsert failed: {r.status_code} {r.text}")

def main():
    print("Loading recent data...")
    df = fetch_recent(days=7)
    if df.empty:
        print("No recent data.")
        return

    # Load feature columns (keeps training/inference aligned)
    with open("models/feature_importance.json", "r") as f:
        fi = json.load(f)
    # Pick columns from any trained item (they all share the same set)
    ref_item = next(iter(fi))
    feature_cols = list(fi[ref_item].keys())

    preds = []
    for pkl in glob.glob("models/*_model.pkl"):
        item = os.path.basename(pkl).replace("_model.pkl", "")
        print(f"Predicting for {item}...")
        model = joblib.load(pkl)

        item_df = df[df["item"] == item]
        x_last, current_price, horizon_minutes = build_features(item_df)
        if x_last is None:
            print(f"Skipping {item}: not enough data for features.")
            continue

        # Align columns to training order
        x_last = x_last.reindex(feature_cols).fillna(0.0)
        y_pred = float(model.predict([x_last.values])[0])
        change = None
        conf = 0.7
        if current_price and current_price != 0:
            change = round((y_pred - current_price) / current_price * 100.0, 3)

        preds.append({
            "item": item,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "horizon_minutes": horizon_minutes,
            "current_price": round(current_price, 3) if current_price is not None else None,
            "predicted_price": round(y_pred, 3),
            "predicted_change": change,
            "confidence": conf,
            "model_name": os.path.basename(pkl),
            "metadata": None
        })

    print(f"Upserting {len(preds)} predictions...")
    upsert_predictions(preds)
    print("Done.")

if __name__ == "__main__":
    main()