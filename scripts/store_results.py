import os
import json
import requests
from datetime import datetime

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']

def store_analysis_results():
    """Store analysis results back to Supabase"""
    
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json'
    }
    
    # Load results from files
    results_stored = 0
    
    # Store arbitrage opportunities
    if os.path.exists('output/arbitrage.json'):
        with open('output/arbitrage.json', 'r') as f:
            arbitrage = json.load(f)
            
        for opp in arbitrage[:10]:  # Store top 10
            data = {
                'pattern_type': 'arbitrage',
                'pattern_data': opp,
                'confidence': 0.9,
                'discovered_at': datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{SUPABASE_URL}/rest/v1/discovered_patterns",
                headers=headers,
                json=data
            )
            
            if response.status_code == 201:
                results_stored += 1
    
    # Store momentum patterns
    if os.path.exists('output/patterns.json'):
        with open('output/patterns.json', 'r') as f:
            patterns = json.load(f)
            
        for pattern in patterns[:10]:  # Store top 10
            data = {
                'pattern_type': pattern['type'],
                'pattern_data': pattern,
                'confidence': abs(pattern.get('strength', 0.5)),
                'discovered_at': datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{SUPABASE_URL}/rest/v1/discovered_patterns",
                headers=headers,
                json=data
            )
            
            if response.status_code == 201:
                results_stored += 1
    
    print(f"Stored {results_stored} analysis results to Supabase")

if __name__ == "__main__":
    store_analysis_results()