import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import requests
from datetime import datetime, timedelta

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        
    def fetch_training_data(self, days=30):
        """Fetch training data from Supabase"""
        headers = {'apikey': SUPABASE_KEY}
        
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/market_data?order=timestamp.desc&limit=20000",
            headers=headers
        )
        
        data = response.json()
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            print("No data available for training")
            return None
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter to last N days
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] > cutoff]
        
        return df
    
    def create_features(self, df, item):
        """Create features for a specific item"""
        item_df = df[df['item'] == item].copy()
        item_df = item_df.sort_values('timestamp')
        
        if len(item_df) < 100:
            return None, None
            
        # Features
        features = pd.DataFrame(index=item_df.index)
        
        # Time features
        features['hour'] = item_df['timestamp'].dt.hour
        features['day_of_week'] = item_df['timestamp'].dt.dayofweek
        features['day_of_month'] = item_df['timestamp'].dt.day
        
        # Price features
        features['price_ma_6'] = item_df['sell_median'].rolling(6).mean()
        features['price_ma_24'] = item_df['sell_median'].rolling(24).mean()
        features['price_std_24'] = item_df['sell_median'].rolling(24).std()
        features['price_change_6h'] = item_df['sell_median'].pct_change(6)
        features['price_change_24h'] = item_df['sell_median'].pct_change(24)
        
        # Volume features
        features['volume'] = item_df['sell_orders'] + item_df['buy_orders']
        features['volume_ma_24'] = features['volume'].rolling(24).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma_24']
        
        # Spread features
        features['spread'] = item_df['spread']
        features['spread_ma'] = features['spread'].rolling(24).mean()
        features['spread_ratio'] = features['spread'] / item_df['sell_median']
        
        # Target: next 6 hour price
        target = item_df['sell_median'].shift(-6)
        
        # Drop NaN
        valid_idx = features.dropna().index
        valid_idx = valid_idx.intersection(target.dropna().index)
        
        if len(valid_idx) < 50:
            return None, None
            
        return features.loc[valid_idx], target.loc[valid_idx]
    
    def train_item_model(self, df, item):
        """Train model for a specific item"""
        X, y = self.create_features(df, item)
        
        if X is None:
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"{item}: Train R2={train_score:.3f}, Test R2={test_score:.3f}")
        
        # Feature importance
        self.feature_importance[item] = dict(zip(
            X.columns, 
            model.feature_importances_
        ))
        
        return model
    
    def train_all_models(self):
        """Train models for all items"""
        print("Fetching training data...")
        df = self.fetch_training_data(days=30)
        
        if df is None:
            return
            
        # Get items with sufficient data
        item_counts = df['item'].value_counts()
        items_to_train = item_counts[item_counts > 200].index.tolist()
        
        print(f"Training models for {len(items_to_train)} items...")
        
        for item in items_to_train[:10]:  # Limit to 10 for GitHub Actions
            print(f"\nTraining {item}...")
            model = self.train_item_model(df, item)
            
            if model is not None:
                self.models[item] = model
        
        # Save models
        os.makedirs('models', exist_ok=True)
        
        for item, model in self.models.items():
            joblib.dump(model, f'models/{item}_model.pkl')
        
        # Save feature importance
        with open('models/feature_importance.json', 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        
        print(f"\nSaved {len(self.models)} models")
        
        # Create model metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'items': list(self.models.keys()),
            'training_days': 30,
            'model_type': 'RandomForestRegressor'
        }
        
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models()