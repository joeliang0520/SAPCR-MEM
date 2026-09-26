"""Learned linker: one model call per session decides which object each turn refers to.

Input per session (nothing hidden): the room's objects (id, class, centroid, size) from the scene record, the session's
turns in order (user and assistant, benchmark questions excluded), the objects in view around each turn (sightings
within one second), and the links already made in earlier sessions of the same room (none: rooms are never revisited).
Output: strict JSON {"links": [{"turn": "s000:t05", "object": 15, "phrase": "...", "role": "subject"|"locator"}]}, with turn and
object ids constrained by the schema to the ids in the prompt (per-session enums).
Written to data/memory/learned/<shard>/<task>/<episode>.jsonl in the same format as data/memory/gold.

Usage: python -m spacmem.links.predict --episodes shard_02/cleaning_and_organizing/mcep_d48a18773522728f [--all] [--workers 3]
"""
import argparse, collections, glob, io, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
if __package__ in (None, ""):      # so the file runs directly as well as with python -m
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from spacmem import answer
from spacmem.answer import call_llm, sha
from spacmem.context.memory import alt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = f"{BASE}/data/final_release"; STORE = f"{BASE}/data/gt_store"; OUT = f"{BASE}/data/memory/learned"
WINDOW = 60   # frames (two seconds) around a turn for the in-view list

def link_schema(turn_ids, obj_ids):
    """Strict schema for one session: turn and object ids are enums of that session's ids, so the model cannot emit an
    id outside the prompt (no bracketed ids, no objects that are not in the room)."""
    return {"type": "object", "properties": {"links": {"type": "array", "items": {"type": "object", "properties": {
        "turn": {"type": "string", "enum": sorted(turn_ids)}, "object": {"type": "integer", "enum": sorted(obj_ids)},
        "phrase": {"type": "string"}, "role": {"type": "string", "enum": ["subject", "locator"]}},
        "required": ["turn", "object", "phrase", "role"], "additionalProperties": False}}}, "required": ["links"], "additionalProperties": False}

PROMPTS = os.path.join(BASE, "spacmem", "prompts")


def prompt(name):
    """A prompt file's text, without its trailing newline so the cache key matches what is sent."""
    return io.open(os.path.join(PROMPTS, name), encoding="utf-8").read().rstrip("\n")


SYSTEM = prompt("linker/link.md")


def build_session_prompt(ep_dir, s):
    tr = json.load(io.open(f"{ep_dir}/public/transcript.json", encoding="utf-8"))["turns"]
    sid = s["clip_id"].replace("-full", ""); st = json.load(open(f"{STORE}/{sid}.json")); objs = st["objects"]
    qids = {json.loads(l)["global_turn_id"] for l in io.open(f"{ep_dir}/public/queries.jsonl", encoding="utf-8")}
    turns = [t for t in tr if t["session_index"] == s["session_index"] and t["global_turn_id"] not in qids]
    last = max(t["local_frame"] for t in turns) if turns else 0
    seen = {int(oid): fr for oid, fr in st["sightings"].items() if fr and fr[0] <= last + WINDOW}
    lines = [f"Room {sid} (day {s['day']}, {s['room_gloss']}). Objects:"]
    for oid in sorted(seen, key=lambda o: (objs[str(o)]["label"], o)):
        o = objs[str(oid)]; c = o["centroid"]; sz = [o["aabb_max"][i] - o["aabb_min"][i] for i in range(3)]
        lines.append(f"  {o['label']}{alt(o)} #{oid}   centroid [{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]   size [{sz[0]:.2f}, {sz[1]:.2f}, {sz[2]:.2f}]")
    lines.append(""); lines.append("Turns:")
    for t in turns:
        f = t["local_frame"]; inview = sorted((o for o, fr in seen.items() if any(abs(x - f) <= WINDOW for x in fr)), key=int)
        lines.append(f"[{t['global_turn_id']}] {t['speaker']} (in view: " + (", ".join(f"{objs[str(o)]['label']} #{o}" for o in inview) or "nothing") + f"): {t['text']}")
    return "\n".join(lines), sid, {t["global_turn_id"] for t in turns}, set(seen)


LOCATIVE = r"(beside|by|near|next to|under|underneath|on|on top of|with|behind|in front of|above|below|opposite|across from|closest to|nearest|nearest to|around|against|between|over|atop)\s+(the|that|this|a|an)\s+"


def rule_links(user_prompt_turns, objs_by_id, sid):
    """Deterministic links: a class name that occurs once in the room, written as a whole phrase in the turn.
    An occurrence that follows a locative preposition ("the box beside the book") is a locator, not a subject."""
    labels = collections.Counter(o["label"] for o in objs_by_id.values()); out = []
    for gid, text in user_prompt_turns:
        for oid, o in objs_by_id.items():
            if labels[o["label"]] != 1: continue
            occ = list(re.finditer(r"\b" + re.escape(o["label"]) + r"s?\b", text, re.I))
            if not occ: continue
            subject = any(not re.search(LOCATIVE + r"$", text[:m.start()], re.I) for m in occ)
            out.append({"turn": gid, "scene": sid, "object": int(oid), "phrase": o["label"], "fact_bearing": subject, "source": "rule"})
    return out


VERIFY = prompt("linker/verify.md")


def link_episode(ep_dir, a):
    rel = os.path.relpath(ep_dir, REL).replace(os.sep, "/"); out = f"{OUT}/{rel}.jsonl"
    sessions = [json.loads(l) for l in io.open(f"{ep_dir}/public/sessions.jsonl", encoding="utf-8")]
    rows = []; stats = dict(sessions=0, links=0, dropped=0, errors=0, prompt_tokens=0, completion_tokens=0)
    for s in sessions:
        user, sid, turn_ids, obj_ids = build_session_prompt(ep_dir, s)
        schema = link_schema(turn_ids, obj_ids)
        fmt = {"type": "json_schema", "json_schema": {"name": "links", "strict": True, "schema": schema}}   # per call: workers run sessions concurrently
        ckey = sha(f"linker|{sha(SYSTEM)}|{sha(json.dumps(schema, sort_keys=True))}|{a.model}|{a.reasoning_effort or ''}|{sha(user)}")
        text, usage, cached = call_llm(a.endpoint, a.model, a.key, SYSTEM, user, cache_key=ckey, max_tokens=a.max_tokens, response_format=fmt)
        stats["sessions"] += 1; stats["prompt_tokens"] += usage.get("prompt_tokens", 0); stats["completion_tokens"] += usage.get("completion_tokens", 0)
        try: links = json.loads(text)["links"]
        except Exception: stats["errors"] += 1; continue
        seen_keys = set()
        def add(l, source):
            if l["turn"] not in turn_ids or l["object"] not in obj_ids: stats["dropped"] += 1; return
            key = (l["turn"], l["object"]); fb = l["fact_bearing"] if "fact_bearing" in l else l.get("role", "subject") == "subject"
            if key in seen_keys:
                for r in rows:
                    if (r["turn"], r["object"]) == key: r["fact_bearing"] = r["fact_bearing"] or fb
                return
            seen_keys.add(key); rows.append({"turn": l["turn"], "scene": sid, "object": l["object"], "phrase": l.get("phrase", ""), "fact_bearing": fb, "source": source}); stats["links"] += 1
        for l in links: add(l, "llm")
        # rule links (unique class in the room) and a verification pass for object-bearing turns left unlinked
        st_ = json.load(open(f"{STORE}/{sid}.json")); objs_by_id = {oid: st_["objects"][str(oid)] for oid in obj_ids}
        turn_lines = [ln for ln in user.split("\n") if ln.startswith("[s")]
        turn_texts = [(ln[1:ln.index("]")], ln.split("): ", 1)[1] if "): " in ln else "") for ln in turn_lines]
        for l in rule_links(turn_texts, objs_by_id, sid): add(l, "rule")
        labels_re = re.compile(r"\b(" + "|".join(sorted({re.escape(o["label"]) for o in objs_by_id.values()}, key=len, reverse=True)) + r")s?\b", re.I)
        # re-ask about every turn that names a class for which it has no subject link (covers turns left unlinked and
        # turns where only a locator or another object was linked, e.g. "the microwave beside the refrigerator")
        subj_classes = collections.defaultdict(set)
        for r in rows:
            if r["fact_bearing"] and (r["turn"], r["object"]) in seen_keys and r["turn"] in turn_ids: subj_classes[r["turn"]].add(objs_by_id[r["object"]]["label"])
        def uncovered(ln):
            gid = ln[1:ln.index("]")]; text = ln.split("): ", 1)[-1]
            named = {m.group(1).lower() for m in labels_re.finditer(text)}
            return any(c not in {s.lower() for s in subj_classes.get(gid, ())} for c in named)
        unlinked = [ln for ln in turn_lines if uncovered(ln)]
        if unlinked:
            user2 = user.split("\nTurns:")[0] + "\n\nTurns:\n" + "\n".join(unlinked)
            ckey2 = sha(f"linker-verify|{sha(SYSTEM + VERIFY)}|{sha(json.dumps(schema, sort_keys=True))}|{a.model}|{a.reasoning_effort or ''}|{sha(user2)}")
            text2, usage2, _ = call_llm(a.endpoint, a.model, a.key, SYSTEM + "\n\n" + VERIFY, user2, cache_key=ckey2, max_tokens=a.max_tokens, response_format=fmt)
            stats["prompt_tokens"] += usage2.get("prompt_tokens", 0); stats["completion_tokens"] += usage2.get("completion_tokens", 0); stats["verify_turns"] = stats.get("verify_turns", 0) + len(unlinked)
            try:
                for l in json.loads(text2)["links"]: add(l, "llm-verify")
            except Exception: stats["errors"] += 1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rel, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="*", default=[]); ap.add_argument("--all", action="store_true"); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--endpoint", default=answer.ENDPOINT); ap.add_argument("--model", default="gpt-5.6-luna"); ap.add_argument("--key", default=answer.KEY)
    ap.add_argument("--out", default=None, help="output folder (default data/memory/learned)")
    ap.add_argument("--store", default=None, help="folder of scene records (default data/gt_store; point at a reconstruction to map turns onto reconstructed objects)")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--reasoning-effort", default=None)
    a = ap.parse_args()
    if not a.key: ap.error("no API key: pass --key or set OPENROUTER_API_KEY")
    global OUT, STORE
    if a.out: OUT = os.path.abspath(a.out)
    if a.store: STORE = os.path.abspath(a.store)
    if a.reasoning_effort: answer.EXTRA_BODY["reasoning"] = {"effort": a.reasoning_effort}
    eps = a.episodes or (sorted(os.path.relpath(p, REL).replace("\\", "/") for p in glob.glob(f"{REL}/shard_*/*/mcep_*")) if a.all else [])
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        for rel, st in pool.map(lambda e: link_episode(f"{REL}/{e}", a), eps):
            print(f"{rel}: {st}", flush=True)
    print(f"done {len(eps)} episodes in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
