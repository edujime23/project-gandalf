#!/usr/bin/env python
import os
import json
import requests
from datetime import datetime

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE') or os.getenv('SUPABASE_KEY')

def post_json(path, payload):
    headers = {
        'apikey': SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal'
    }
    resp = requests.post(f"{SUPABASE_URL}{path}", headers=headers, data=json.dumps(payload), timeout=30)
    return resp

def store_analysis_results():
    results_stored = 0

    if os.path.exists('output/arbitrage.json'):
        with open('output/arbitrage.json', 'r') as f:
            arbitrage = json.load(f) or []
        for opp in arbitrage[:10]:
            data = {
                'pattern_type': 'arbitrage',
                'pattern_data': opp,
                'confidence': 0.9,
                'discovered_at': datetime.utcnow().isoformat() + 'Z'
            }
            r = post_json("/rest/v1/discovered_patterns", data)
            if r.status_code in (200, 201, 204):
                results_stored += 1

    if os.path.exists('output/patterns.json'):
        with open('output/patterns.json', 'r') as f:
            patterns = json.load(f) or []
        for p in patterns[:10]:
            data = {
                'pattern_type': p.get('type', 'unknown'),
                'pattern_data': p,
                'confidence': abs(float(p.get('strength', 0.5))),
                'discovered_at': datetime.utcnow().isoformat() + 'Z'
            }
            r = post_json("/rest/v1/discovered_patterns", data)
            if r.status_code in (200, 201, 204):
                results_stored += 1

    print(f"Stored {results_stored} analysis results to Supabase")

if __name__ == "__main__":
    store_analysis_results()