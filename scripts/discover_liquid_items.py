#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import asyncio
import aiohttp
import time
import json
import requests
from datetime import datetime, timezone, timedelta

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")
WF_BASE = "https://api.warframe.market/v1"

HEADERS_WF = {"User-Agent": "Gandalf/1.0", "Accept": "application/json"}
HEADERS_SB = {
    "apikey": SUPABASE_KEY,
    "Authorization": "Bearer " + str(SUPABASE_KEY),
    "Content-Type": "application/json",
    "Prefer": "return=minimal,resolution=merge-duplicates"
}

# Limits and scoring
STATS_SAMPLE = 350          # number of candidates to compute volume
INCLUDE_ORDERS_FOR = 120    # enrich top N with depth
TOP_N_TO_SAVE = 120         # upsert this many into tracked_items


class RateLimiter:
    """
    Simple global rate limiter to respect 3 rps. Ensures min_interval between calls.
    """
    def __init__(self, min_interval=0.35):
        self.min_interval = float(min_interval)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self):
        async with self._lock:
            now = time.perf_counter()
            delta = now - self._last
            if delta < self.min_interval:
                await asyncio.sleep(self.min_interval - delta)
            self._last = time.perf_counter()


rl = RateLimiter(min_interval=0.35)


def looks_like_set(url_name):
    # prioritize _prime_set and generic _set, exclude rivens/blueprints/etc
    return url_name.endswith("_prime_set") or re.search(r"_set$", url_name) is not None


async def fetch_json(session, url, params=None, retries=3):
    for i in range(int(retries)):
        try:
            await rl.wait()
            async with session.get(
                url,
                params=params,
                headers=HEADERS_WF,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as r:
                if r.status in (429, 500, 502, 503, 504):
                    await asyncio.sleep(1.2 * (i + 1))
                    continue
                r.raise_for_status()
                return await r.json()
        except Exception:
            if i == retries - 1:
                return None
            await asyncio.sleep(0.6 * (i + 1))
    return None


async def get_all_items(session):
    data = await fetch_json(session, WF_BASE + "/items")
    if not data:
        return []
    items = data.get("payload", {}).get("items", [])
    return [i.get("url_name") for i in items if i.get("url_name")]


def sum_24h_volume(stats_json):
    if not stats_json:
        return 0
    payload = stats_json.get("payload") or {}
    # prefer the 48h stats array
    stats = payload.get("statistics_48hours")
    if stats is None:
        stats = (payload.get("statistics") or {}).get("48hours")
    if not isinstance(stats, list):
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    total = 0
    for row in stats:
        ts = row.get("datetime") or row.get("date") or row.get("time")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
        except Exception:
            dt = None
        if dt and dt >= cutoff:
            vol = row.get("volume")
            try:
                total += int(vol) if vol is not None else 0
            except Exception:
                pass
    return total


def count_ingame_orders(orders_json):
    if not orders_json:
        return {"buy": 0, "sell": 0}
    orders = (orders_json.get("payload") or {}).get("orders", []) or []
    ingame = [o for o in orders if (o.get("user") or {}).get("status") == "ingame"]
    buys = sum(1 for o in ingame if o.get("order_type") == "buy")
    sells = sum(1 for o in ingame if o.get("order_type") == "sell")
    return {"buy": buys, "sell": sells}


async def main():
    print("Discovering liquid items from Warframe Market...")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE/KEY")

    async with aiohttp.ClientSession() as session:
        all_items = await get_all_items(session)
        candidates = [n for n in all_items if looks_like_set(n)]
        if not candidates:
            print("No candidates found.")
            return
        candidates = candidates[:STATS_SAMPLE]
        print("Candidates: {0}".format(len(candidates)))

        # Step 1: get 24h volume from statistics
        volumes = []
        for idx, name in enumerate(candidates, 1):
            s = await fetch_json(session, "{0}/items/{1}/statistics".format(WF_BASE, name))
            v = sum_24h_volume(s)
            volumes.append({"item": name, "volume24": v})
            if idx % 50 == 0:
                print("  processed {0}/{1}...".format(idx, len(candidates)))

        volumes.sort(key=lambda x: x["volume24"], reverse=True)
        top_for_depth = volumes[:INCLUDE_ORDERS_FOR]

        # Step 2: enrich with current order depth
        enriched = []
        for i, entry in enumerate(top_for_depth, 1):
            o = await fetch_json(session, "{0}/items/{1}/orders".format(WF_BASE, entry["item"]))
            depth = count_ingame_orders(o)
            enriched.append({
                "item": entry["item"],
                "volume24": entry["volume24"],
                "buy_orders": depth["buy"],
                "sell_orders": depth["sell"]
            })
            if i % 40 == 0:
                print("  depth {0}/{1}...".format(i, len(top_for_depth)))

        # Step 3: score = 0.7*volume_norm + 0.3*depth_norm
        max_vol = max((e["volume24"] for e in enriched), default=1)
        max_depth = max(((e["buy_orders"] + e["sell_orders"]) for e in enriched), default=1)
        scored = []
        for e in enriched:
            vol_s = (float(e["volume24"]) / float(max_vol)) if max_vol else 0.0
            depth = e["buy_orders"] + e["sell_orders"]
            depth_s = (float(depth) / float(max_depth)) if max_depth else 0.0
            score = 0.7 * vol_s + 0.3 * depth_s
            scored.append({
                "item": e["item"],
                "volume24": e["volume24"],
                "buy_orders": e["buy_orders"],
                "sell_orders": e["sell_orders"],
                "score": round(score, 6)
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:TOP_N_TO_SAVE]
        print("Upserting {0} items into tracked_items...".format(len(top)))

        payload = [{"item": e["item"], "score": e["score"]} for e in top]
        r = requests.post(
            SUPABASE_URL + "/rest/v1/tracked_items?on_conflict=item",
            headers=HEADERS_SB,
            data=json.dumps(payload),
            timeout=30
        )
        if not r.ok:
            print("Supabase upsert failed: {0} {1}".format(r.status_code, r.text))
            raise SystemExit(1)

        print("Done. Examples:")
        for e in top[:10]:
            depth_sum = e["buy_orders"] + e["sell_orders"]
            print(" - {0:>24}  score={1:.3f}  vol24={2}  depth={3}".format(
                e["item"], e["score"], e["volume24"], depth_sum
            ))


if __name__ == "__main__":
    asyncio.run(main())