"""Object-centred memory context (condition A: full memory dump, no retrieval).

Every object of the rooms seen so far with its geometry, and under each object the conversation turns linked to it,
verbatim, with that object's mention marked as [#id]. Turns that mention several objects appear under each of them.
Unlinked dialogue is dropped. Links come from data/memory/gold (the release's own logs and evidence) or from the
learned linker (spacmem/links/predict.py).
"""
import json, io, os, re, collections


def alt(o):
    """The reconstruction's runner-up classes, rendered after the class name. Absent for annotated scene records."""
    a = o.get("label_alternatives")
    return f" (or: {', '.join(a)})" if a else ""


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = f"{BASE}/data/gt_store"; LINKS = f"{BASE}/data/memory/gold"
_store = {}


def load_store(sid):
    if sid not in _store: _store[sid] = json.load(open(f"{STORE}/{sid}.json"))
    return _store[sid]


def load_links(ep_dir, links_dir=None):
    rel = "/".join(os.path.normpath(ep_dir).replace(os.sep, "/").split("/")[-3:])     # shard/task/episode, whatever the release folder
    p = f"{links_dir or LINKS}/{rel}.jsonl"
    return [json.loads(l) for l in io.open(p, encoding="utf-8")] if os.path.exists(p) else []


def mark(text, phrase, oid):
    """Append [#id] after the first occurrence of the phrase (case-insensitive); at the end if the phrase is not found."""
    tag = f" [#{oid}]"
    if phrase:
        m = re.search(re.escape(phrase), text, re.I)
        if m: return text[:m.end()] + tag + text[m.end():]
    return text + tag


def build_context_memory(ep_dir, turn_id, links_dir=None, subject_only=True):
    """subject_only: show a turn under an object only when the object is the turn's subject (fact_bearing link); a turn
    that merely uses the object as a locator ("the box beside the book") says nothing about it."""
    tr = json.load(io.open(f"{ep_dir}/public/transcript.json", encoding="utf-8"))["turns"]; by_id = {t["global_turn_id"]: t for t in tr}
    sessions = [json.loads(l) for l in io.open(f"{ep_dir}/public/sessions.jsonl", encoding="utf-8")]
    q = by_id[turn_id]; qidx, qsess, qframe = q["global_turn_index"], q["session_index"], q["local_frame"]
    links = [l for l in load_links(ep_dir, links_dir) if l["turn"] in by_id and by_id[l["turn"]]["global_turn_index"] < qidx and (l.get("fact_bearing", True) or not subject_only)]
    per_obj = collections.defaultdict(list)
    for l in links: per_obj[(l["scene"], l["object"])].append(l)
    lines = ["Memory of the walk so far. Day table: " + "; ".join(f"day {s['day']} = {s['room_gloss']} (room {s['clip_id'].replace('-full', '')}, session {s['session_index'] + 1})" for s in sessions if s["session_index"] <= qsess)]
    lines.append("")
    for s in sessions:
        if s["session_index"] > qsess: break
        sid = s["clip_id"].replace("-full", ""); st = load_store(sid); objs = st["objects"]
        last = qframe if s["session_index"] == qsess else max(t["local_frame"] for t in tr if t["session_index"] == s["session_index"])
        lines.append(f"=== room {sid} · day {s['day']} · {s['room_gloss']} · session {s['session_index'] + 1}" + (" · current room" if s["session_index"] == qsess else "") + " ===")
        seen = [(int(oid), fr[0]) for oid, fr in st["sightings"].items() if fr and fr[0] <= last]
        for oid, f0 in sorted(seen, key=lambda x: (objs[str(x[0])]["label"], x[0])):
            o = objs[str(oid)]; c = o["centroid"]; lo = o["aabb_min"]; hi = o["aabb_max"]; cam = st.get("camera", {}).get(str(f0))
            cam_s = f"   camera then [{cam[0]:.3f}, {cam[1]:.3f}, {cam[2]:.3f}]" if cam else ""
            lines.append(f"{o['label']}{alt(o)} #{oid}   centroid [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]   box [{lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f}] to [{hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f}]   first seen frame {f0}{cam_s}")
            for l in sorted(per_obj.get((sid, oid), []), key=lambda l: by_id[l["turn"]]["global_turn_index"]):
                t = by_id[l["turn"]]
                lines.append(f"    [{t['global_turn_id']} · day {t['day']} · {t['speaker']}] {mark(t['text'], l.get('phrase', ''), oid)}")
        if s["session_index"] == qsess:
            vis = sorted((int(oid) for oid, fr in st["sightings"].items() if qframe in fr), key=int)
            lines.append(f"visible now (frame {qframe}): " + (", ".join(f"{objs[str(o)]['label']} #{o}" for o in vis) or "(nothing)"))
            lines.append(f"question [{turn_id}]: {q['text']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(build_context_memory(sys.argv[1], sys.argv[2]))
