import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import requests
from datetime import datetime, timedelta, timezone

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Configurable minimum points per item to train (default 80 for initial runs)
MIN_TRAIN_POINTS = int(os.getenv('MIN_TRAIN_POINTS', '80'))

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}

    def fetch_training_data(self, days=30, page_size=1000):
        """
        Fetch all rows within the last `days` using server-side filtering and
        pagination via Range headers to bypass the 1000-row cap.
        """
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set")

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        # Use UTC 'Z' form to avoid '+00:00' parsing issues in URLs
        formatted_cutoff = cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")

        base_endpoint = f"{SUPABASE_URL}/rest/v1/market_data"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Prefer": "count=exact",
            "Range-Unit": "items",
        }
        params = {
            "select": "*",
            "timestamp": f"gte.{formatted_cutoff}",
            "order": "timestamp.asc",
        }

        all_rows = []
        offset = 0
        page_idx = 0

        while True:
            range_header = {"Range": f"{offset}-{offset + page_size - 1}"}
            req_headers = {**headers, **range_header}

            resp = requests.get(base_endpoint, headers=req_headers, params=params, timeout=30)
            if not resp.ok:
                # Helpful debug if we ever hit a 4xx again
                raise requests.HTTPError(
                    f"{resp.status_code} Error: {resp.text} for URL: {resp.url}"
                )
            page = resp.json()
            page_count = len(page)
            all_rows.extend(page)

            content_range = resp.headers.get("Content-Range", "unknown")
            print(f"DEBUG: Page {page_idx} fetched {page_count} rows (Content-Range: {content_range})")

            if page_count < page_size:
                break

            offset += page_size
            page_idx += 1

        if not all_rows:
            print("No data available for training in the last period.")
            return None

        df = pd.DataFrame(all_rows)
        print(f"DEBUG: Total rows fetched across pages: {len(df)}")

        # Robust timestamp parsing
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

        # Ensure numeric columns are numeric
        for col in ["sell_median", "buy_median", "spread", "sell_orders", "buy_orders"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def create_features(self, df, item):
        """
        Create features and target for a specific item.
        """
        item_df = df[df["item"] == item].copy()
        item_df = item_df.sort_values("timestamp")

        # Need a reasonable history
        if len(item_df) < 60:
            return None, None

        features = pd.DataFrame(index=item_df.index)

        # Time features
        features["hour"] = item_df["timestamp"].dt.hour
        features["day_of_week"] = item_df["timestamp"].dt.dayofweek
        features["day_of_month"] = item_df["timestamp"].dt.day

        # Price features
        price = item_df["sell_median"]
        features["price_ma_6"] = price.rolling(6).mean()
        features["price_ma_24"] = price.rolling(24).mean()
        features["price_std_24"] = price.rolling(24).std()
        features["price_change_6h"] = price.pct_change(6)
        features["price_change_24h"] = price.pct_change(24)

        # Volume/order features
        vol = (item_df.get("sell_orders", 0).fillna(0) + item_df.get("buy_orders", 0).fillna(0))
        features["volume"] = vol
        features["volume_ma_24"] = vol.rolling(24).mean()
        features["volume_ratio"] = vol / (features["volume_ma_24"] + 1)

        # Spread features
        if "spread" in item_df.columns:
            features["spread"] = item_df["spread"]
            features["spread_ma"] = features["spread"].rolling(24).mean()
            features["spread_ratio"] = features["spread"] / (price + 1)
        else:
            features["spread"] = np.nan
            features["spread_ma"] = np.nan
            features["spread_ratio"] = np.nan

        # Target: +6 steps ahead
        target = price.shift(-6)

        combined = pd.concat([features, target.rename("future_price")], axis=1).dropna()
        if len(combined) < 50:
            return None, None

        X = combined.drop(columns=["future_price"])
        y = combined["future_price"]
        return X, y

    def train_item_model(self, df, item):
        X, y = self.create_features(df, item)
        if X is None or y is None:
            print(f"Skipping {item} due to insufficient data after feature creation.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"{item}: Train R2={train_score:.3f}, Test R2={test_score:.3f}")

        self.feature_importance[item] = dict(
            zip(X.columns, model.feature_importances_)
        )
        return model

    def train_all_models(self):
        print("Fetching training data...")
        df = self.fetch_training_data(days=30)

        if df is None or df.empty:
            print("No data returned after fetch, exiting.")
            return

        item_counts = df["item"].value_counts()
        print("--- DEBUG: Item counts in fetched window ---")
        print(item_counts.head(20))
        print("--------------------------------------------")

        print(f"DEBUG: Using MIN_TRAIN_POINTS = {MIN_TRAIN_POINTS}")
        items_to_train = item_counts[item_counts >= MIN_TRAIN_POINTS].index.tolist()
        print(f"Found {len(items_to_train)} items with >= {MIN_TRAIN_POINTS} points to train on.")

        # Limit to 10 items to keep Action runtime safe
        items_to_train = items_to_train[:10]

        for item in items_to_train:
            print(f"\nTraining {item}...")
            model = self.train_item_model(df, item)
            if model is not None:
                self.models[item] = model

        if not self.models:
            print("No models were successfully trained.")
            return

        os.makedirs("models", exist_ok=True)
        for item, model in self.models.items():
            joblib.dump(model, f"models/{item}_model.pkl")

        with open("models/feature_importance.json", "w") as f:
            json.dump(self.feature_importance, f, indent=2)

        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "items": list(self.models.keys()),
            "training_days": 30,
            "model_type": "RandomForestRegressor",
            "min_points": MIN_TRAIN_POINTS
        }
        with open("models/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSaved {len(self.models)} models to /models directory.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models()