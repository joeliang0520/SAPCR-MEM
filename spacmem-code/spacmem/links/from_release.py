"""Annotated turn-to-object links for the memory method's annotated-mapping condition (no model).

Sources, per episode:
  1. hidden/generation/session_*/session_result.json -> references.resolutions: every resolved physical-object
     mention (span, turn, selected track, fact_bearing). Turn ids there are mapped to the transcript's through
     public/turn_id_map_generation.json.
  2. hidden/evidence.jsonl memory_snapshot facts: (established turn, track) for every fact. Present everywhere.
     Used as the fallback where no log exists and as a completeness check.
Track ids are mapped to (scan, object id) through the evidence facts (only objects that carry facts have a mapping;
locator-only mentions of other objects are dropped).

Output: data/memory/gold/<shard>/<task>/<episode>.jsonl with one row per link:
  {"turn": global_turn_id, "scene": scan_id, "object": object_id, "phrase": span, "fact_bearing": bool, "source": "log"|"evidence"}
Usage: python -m spacmem.links.from_release
"""
import json, io, os, glob, collections

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = f"{BASE}/data/final_release"; OUT = f"{BASE}/data/memory/gold"


def build(ep_dir):
    tr = {t["global_turn_id"]: t for t in json.load(io.open(f"{ep_dir}/public/transcript.json", encoding="utf-8"))["turns"]}
    union = {}
    for r in map(json.loads, io.open(f"{ep_dir}/hidden/evidence.jsonl", encoding="utf-8")):
        for x in r["memory_snapshot"]["facts"]: union[(x["track_id"], x["established_global_turn_id"], x["slot"], x["value"])] = x
    tmap = {x["track_id"]: (x["clip_id"].replace("-full", ""), x["object_id"]) for x in union.values()}
    links = {}   # (turn, scene, object) -> row
    # 1. generation logs
    idmap = {}
    mp = f"{ep_dir}/public/turn_id_map_generation.json"
    if os.path.exists(mp): idmap = json.load(io.open(mp, encoding="utf-8"))["generation_to_transcript"]
    n_log = 0; unmapped_tracks = 0
    for f in sorted(glob.glob(f"{ep_dir}/hidden/generation/session_*/session_result.json")):
        r = json.load(io.open(f, encoding="utf-8")); si = r["session_index"]
        for m in r["references"]["resolutions"]:
            if m["mention_type"] != "physical_object" or m["resolution_status"] not in ("resolved", "clarified") or not m.get("selected_track_id"): continue
            gid = f"s{si:03d}:{m['turn_id']}"; gid = idmap.get(gid, gid)
            if gid not in tr: continue
            if m["selected_track_id"] not in tmap: unmapped_tracks += 1; continue
            scene, oid = tmap[m["selected_track_id"]]; key = (gid, scene, oid)
            row = links.setdefault(key, {"turn": gid, "scene": scene, "object": oid, "phrase": m["span"], "fact_bearing": False, "source": "log"})
            row["fact_bearing"] = row["fact_bearing"] or bool(m.get("fact_bearing")); n_log += 1
    # 1b. the compiler's own establishing turns (from the logs), as fact-bearing links
    for f in sorted(glob.glob(f"{ep_dir}/hidden/generation/session_*/session_result.json")):
        r = json.load(io.open(f, encoding="utf-8")); si = r["session_index"]
        for rec in r["compiled"]["records"]:
            if rec.get("status") != "established" or rec.get("resolved_track_id") not in tmap: continue
            gid = f"s{si:03d}:{rec['established_turn']}"; gid = idmap.get(gid, gid)
            if gid not in tr: continue
            scene, oid = tmap[rec["resolved_track_id"]]; key = (gid, scene, oid)
            row = links.setdefault(key, {"turn": gid, "scene": scene, "object": oid, "phrase": "", "fact_bearing": True, "source": "log"}); row["fact_bearing"] = True
    # 2. evidence facts: only where no generation log exists
    n_ev_added = 0
    if glob.glob(f"{ep_dir}/hidden/generation/session_*/session_result.json"): union = {}
    for x in union.values():
        gid = x["established_global_turn_id"]
        if gid not in tr: continue
        scene, oid = x["clip_id"].replace("-full", ""), x["object_id"]; key = (gid, scene, oid)
        if key not in links:
            links[key] = {"turn": gid, "scene": scene, "object": oid, "phrase": "", "fact_bearing": True, "source": "evidence"}; n_ev_added += 1
        else: links[key]["fact_bearing"] = True
    rows = sorted(links.values(), key=lambda r: (tr[r["turn"]]["global_turn_index"], r["scene"], r["object"]))
    return rows, dict(log_mentions=n_log, evidence_only_links=n_ev_added, unmapped_log_tracks=unmapped_tracks, links=len(rows), objects=len({(r["scene"], r["object"]) for r in rows}), turns=len({r["turn"] for r in rows}))


def main():
    tot = collections.Counter(); eps = 0
    for ep_dir in sorted(glob.glob(f"{REL}/shard_*/*/mcep_*")):
        rel = os.path.relpath(ep_dir, REL).replace(os.sep, "/"); rows, st = build(ep_dir)
        out = f"{OUT}/{rel}.jsonl"; os.makedirs(os.path.dirname(out), exist_ok=True)
        with io.open(out, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tot.update(st); eps += 1
    print(f"{eps} episodes -> {OUT}: " + ", ".join(f"{k}={v}" for k, v in tot.items()))


if __name__ == "__main__":
    main()
