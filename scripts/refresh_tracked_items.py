import os, requests, json

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def main(top_n=50):
    url = f"{SUPABASE_URL}/rest/v1/rpc/refresh_tracked_items"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    resp = requests.post(url, headers=headers, data=json.dumps({"top_n": top_n}), timeout=30)
    if not resp.ok:
        raise SystemExit(f"refresh_tracked_items failed: {resp.status_code} {resp.text}")
    print(f"Tracked items refreshed (top {top_n}).")

if __name__ == "__main__":
    main()