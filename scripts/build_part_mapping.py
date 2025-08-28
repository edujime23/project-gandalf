#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import urllib.parse
import requests

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")

WF_BASE = "https://api.warframe.market/v1"
WFS_BASE = "https://api.warframestat.us"

HDR_WM = {"User-Agent": "Gandalf/1.0", "Accept": "application/json"}
HDR_SB = {
    "apikey": SUPABASE_KEY,
    "Authorization": "Bearer " + str(SUPABASE_KEY),
    "Content-Type": "application/json"
}

def get_tracked_sets(limit=200):
    url = "{0}/rest/v1/tracked_items?select=item,score,active&order=score.desc&limit={1}".format(
        SUPABASE_URL, int(limit)
    )
    r = requests.get(url, headers=HDR_SB, timeout=30)
    r.raise_for_status()
    rows = r.json()
    return [x["item"] for x in rows if isinstance(x.get("item"), str) and x["item"].endswith("_set")]

def get_items_in_set(url_name):
    r = requests.get("{0}/items/{1}".format(WF_BASE, url_name), headers=HDR_WM, timeout=30)
    if not r.ok:
        return []
    data = r.json()
    items = (data.get("payload") or {}).get("item", {}).get("items_in_set", []) or []
    parts = []
    for it in items:
        iname = it.get("item_name") or (it.get("en") or {}).get("item_name") or it.get("url_name")
        iurl = it.get("url_name")
        if not iname or not iurl:
            continue
        if iurl == url_name:
            continue
        parts.append({"item_name": iname, "url_name": iurl})
    return parts

def parse_relic_drop(entry):
    place = entry.get("place") or ""
    if "Relic (Intact)" not in place:
        return None
    m = re.match(r"^(Lith|Meso|Neo|Axi)\s+([A-Z0-9]+)\s+Relic\s+KATEX_INLINE_OPENIntactKATEX_INLINE_CLOSE$", place.strip())
    if not m:
        return None
    era = m.group(1)
    code = m.group(2)
    relic_name = "{0} {1}".format(era, code)
    rarity = (entry.get("rarity") or "").strip().lower()
    ch = entry.get("chance")
    try:
        if isinstance(ch, str) and ch.endswith("%"):
            drop = float(ch.replace("%", "").strip()) / 100.0
        elif isinstance(ch, (int, float)):
            drop = float(ch)
            if drop > 1.0:
                drop = drop / 100.0
        else:
            return None
    except Exception:
        return None
    if drop <= 0 or drop > 1:
        return None
    return {"era": era, "relic_name": relic_name, "rarity": rarity, "drop_chance": round(drop, 6)}

def search_drops(part_name):
    q = urllib.parse.quote(part_name)
    url = "{0}/drops/search/{1}".format(WFS_BASE, q)
    r = requests.get(url, headers={"User-Agent": "Gandalf/1.0"}, timeout=30)
    if not r.ok:
        return []
    data = r.json()
    if not isinstance(data, list):
        return []
    out = []
    for entry in data:
        parsed = parse_relic_drop(entry)
        if parsed:
            out.append(parsed)
    return out

def rpc_upsert_item_parts(rows):
    url = SUPABASE_URL + "/rest/v1/rpc/upsert_item_parts_json"
    r = requests.post(url, headers=HDR_SB, data=json.dumps({"rows": rows}), timeout=60)
    if not r.ok:
        raise RuntimeError("upsert_item_parts_json failed: {0} {1}".format(r.status_code, r.text))

def rpc_upsert_part_relic_drops(rows):
    url = SUPABASE_URL + "/rest/v1/rpc/upsert_part_relic_drops_json"
    r = requests.post(url, headers=HDR_SB, data=json.dumps({"rows": rows}), timeout=60)
    if not r.ok:
        raise RuntimeError("upsert_part_relic_drops_json failed: {0} {1}".format(r.status_code, r.text))

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE/KEY")
    sets = get_tracked_sets(limit=200)
    print("Building mapping for {0} sets...".format(len(sets)))

    ip_rows = []
    pr_rows = []

    for i, s in enumerate(sets, 1):
        parts = get_items_in_set(s)
        print("[{0}/{1}] {2}: {3} parts".format(i, len(sets), s, len(parts)))
        for p in parts:
            ip_rows.append({"set_item": s, "part_item": p["url_name"], "part_name": p["item_name"]})
            drops = search_drops(p["item_name"])
            for d in drops:
                pr_rows.append({
                    "part_item": p["url_name"],
                    "part_name": p["item_name"],
                    "relic_name": d["relic_name"],
                    "era": d["era"],
                    "rarity": d["rarity"],
                    "drop_chance": d["drop_chance"]
                })
            time.sleep(0.25)

        if i % 20 == 0:
            if ip_rows:
                rpc_upsert_item_parts(ip_rows)
                ip_rows = []
            if pr_rows:
                chunk = 400
                for j in range(0, len(pr_rows), chunk):
                    rpc_upsert_part_relic_drops(pr_rows[j:j+chunk])
                pr_rows = []

    if ip_rows:
        rpc_upsert_item_parts(ip_rows)
    if pr_rows:
        chunk = 400
        for j in range(0, len(pr_rows), chunk):
            rpc_upsert_part_relic_drops(pr_rows[j:j+chunk])

    print("Mapping build complete.")

if __name__ == "__main__":
    main()