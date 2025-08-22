import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import requests
import os

class MLTrainer:
    def __init__(self):
        self.models = {}
        
    def train_price_predictor(self, item_data):
        """Train ML model for price prediction"""
        # Feature engineering
        features = self.create_features(item_data)
        
        # Train model
        X = features[['hour', 'day_of_week', 'volume_ma', 'price_ma', 'volatility']]
        y = features['future_price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        score = model.score(X_test, y_test)
        print(f"Model R2 score: {score}")
        
        return model
    
    def create_features(self, data):
        """Create ML features from market data"""
        df = pd.DataFrame(data)
        
        # Time features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Price features
        df['price_ma'] = df['sell_median'].rolling(24).mean()
        df['volatility'] = df['sell_median'].rolling(24).std()
        
        # Volume features  
        df['volume_ma'] = df['volume'].rolling(24).mean()
        
        # Target: price in 24 hours
        df['future_price'] = df['sell_median'].shift(-24)
        
        return df.dropna()