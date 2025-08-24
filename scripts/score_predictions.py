#!/usr/bin/env python
import os, requests, json
from datetime import datetime, timedelta, timezone
from dateutil import parser as du_parser

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

def q(url, params=None, method="GET", body=None):
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
    r = requests.request(method, url, headers=headers, params=params, data=json.dumps(body) if body else None, timeout=30)
    if not r.ok:
        raise SystemExit(f"{method} {url} failed: {r.status_code} {r.text}")
    return r

def parse_ts(ts_str: str) -> datetime:
    dt = du_parser.isoparse(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def fetch_pending_predictions(limit=200, horizon_minutes=30):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=horizon_minutes)).isoformat()
    url = f"{SUPABASE_URL}/rest/v1/predictions"
    params = {"select": "id,item,predicted_at,predicted_price", "actual_price": "is.null", "predicted_at": f"lte.{cutoff}", "order": "predicted_at.asc", "limit": str(limit)}
    return q(url, params=params).json()

def fetch_actual_price(item, target_time_iso, search_window_minutes=60):
    url = f"{SUPABASE_URL}/rest/v1/market_data"
    params = {"select": "timestamp,sell_median", "item": f"eq.{item}", "timestamp": f"gte.{target_time_iso}", "order": "timestamp.asc", "limit": "1"}
    data = q(url, params=params).json()
    if data:
        return float(data[0]["sell_median"]) if data[0]["sell_median"] is not None else None
    after = q(url, params={"select": "timestamp,sell_median", "item": f"eq.{item}", "timestamp": f"gte.{target_time_iso}", "order": "timestamp.asc", "limit": "1"}).json()
    before = q(url, params={"select": "timestamp,sell_median", "item": f"eq.{item}", "timestamp": f"lte.{target_time_iso}", "order": "timestamp.desc", "limit": "1"}).json()
    candidates = []
    for r in after:
        candidates.append(("after", parse_ts(r["timestamp"]), r["sell_median"]))
    for r in before:
        candidates.append(("before", parse_ts(r["timestamp"]), r["sell_median"]))
    if not candidates:
        return None
    target = parse_ts(target_time_iso)
    best_val, best_dt = None, timedelta(days=999)
    for _, ts, val in candidates:
        if val is None:
            continue
        dt = abs(ts - target)
        if dt <= timedelta(minutes=search_window_minutes) and dt < best_dt:
            best_dt = dt
            best_val = float(val)
    return best_val

def update_prediction(pred_id, actual_price, predicted_price):
    if actual_price is None or predicted_price is None or actual_price == 0:
        acc = None
    else:
        ape = abs(predicted_price - actual_price) / abs(actual_price)
        acc = max(0.0, min(1.0, 1.0 - ape))
    url = f"{SUPABASE_URL}/rest/v1/predictions"
    params = { "id": f"eq.{pred_id}" }
    body = { "actual_price": actual_price, "accuracy": acc }
    q(url, params=params, method="PATCH", body=body)

def main():
    horizon = int(os.getenv("PREDICTION_HORIZON_MIN", "30"))
    batch = fetch_pending_predictions(limit=300, horizon_minutes=horizon)
    print(f"Found {len(batch)} predictions to score...")
    for row in batch:
        item = row["item"]
        predicted_at = parse_ts(row["predicted_at"])
        target_time_dt = predicted_at + timedelta(minutes=horizon)
        actual = fetch_actual_price(item, target_time_dt.isoformat())
        update_prediction(row["id"], actual, row["predicted_price"])
    print("Scoring complete.")

if __name__ == "__main__":
    main()