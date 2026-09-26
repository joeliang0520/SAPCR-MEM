"""Per-frame context: the 'no aggregation' control.

Once per second (and at every user turn and at the question) the context lists whatever is visible in that frame: the
class, centroid and box of each object from the scene record, plus the camera position. By default objects carry no
identifier and no dialogue is attached to them, so the reader has to work out for itself which observations are the same
object and which turns refer to it. Conversation turns sit at their frames. No room inventory and no first-sighting lines.

Identify answers are given as "day <D> | <class> | [x, y, z]" and mapped to an object id by position (resolve_position_answer).
"""
import io, json, os, re, sys
if __package__ in (None, ""):      # so the file runs directly as well as with python -m
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from spacmem.context.memory import alt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = f"{BASE}/data/final_release"; STORE = f"{BASE}/data/gt_store"


def _obs_line(frame, st, objs, vis_sets, ids, tag="observation"):
    cam = st.get("camera", {}).get(str(frame))
    cam_s = f"   camera [{cam[0]:.3f}, {cam[1]:.3f}, {cam[2]:.3f}]" if cam else ""
    parts = []
    for oid in sorted(vis_sets, key=lambda o: (objs[o]["label"], int(o))):
        if frame not in vis_sets[oid]: continue
        o = objs[oid]; c, lo, hi = o["centroid"], o["aabb_min"], o["aabb_max"]
        name = f"{o['label']} #{oid}" if ids else o["label"]
        parts.append(f"{name}{alt(o)} centroid [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}] box [{lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f}] to [{hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f}]")
    return f"frame {frame:5d}  {tag}{cam_s}   sees: " + (" | ".join(parts) if parts else "(nothing)")


def build_context_per_frame(ep_dir, turn_id, fps=1.0, ids=False, store=None):
    store = store or STORE
    tr = json.load(io.open(f"{ep_dir}/public/transcript.json", encoding="utf-8"))["turns"]
    sessions = [json.loads(l) for l in io.open(f"{ep_dir}/public/sessions.jsonl", encoding="utf-8")]
    qturn = next(t for t in tr if t["global_turn_id"] == turn_id)
    qidx, qsess, qframe = qturn["global_turn_index"], qturn["session_index"], qturn["local_frame"]
    lines = []
    for s in sessions:
        if s["session_index"] > qsess: break
        sid = s["clip_id"].replace("-full", ""); st = json.load(open(f"{store}/{sid}.json")); objs = st["objects"]
        last = qframe if s["session_index"] == qsess else max(t["local_frame"] for t in tr if t["session_index"] == s["session_index"])
        step = max(1, round(float(st.get("frame_rate", 30)) / fps))
        vis_sets = {oid: set(fr) for oid, fr in st["sightings"].items() if fr and fr[0] <= last}
        events = [(f, 0, _obs_line(f, st, objs, vis_sets, ids)) for f in range(0, last + 1, step)]
        sampled = {f for f, _, _ in events}
        for t in tr:
            if t["session_index"] != s["session_index"] or t["global_turn_index"] >= qidx: continue
            f = t["local_frame"]
            if t["speaker"] == "user" and f not in sampled:
                sampled.add(f); events.append((f, 0, _obs_line(f, st, objs, vis_sets, ids)))
            events.append((f, 1, f"frame {f:5d}  {t['speaker']:9s}  [{t['global_turn_id']}] {t['text']}"))
        lines.append(f"=== session {s['session_index'] + 1} · day {s['day']} · {s['room_gloss']} · one observation per second ===")
        lines += [e[2] for e in sorted(events, key=lambda e: (e[0], e[1]))]
        if s["session_index"] == qsess:
            lines.append(_obs_line(qframe, st, objs, vis_sets, ids, tag="visible now"))
            lines.append(f"frame {qframe:5d}  user       [{turn_id}] {qturn['text']}")
        lines.append("")
    return "\n".join(lines)


def resolve_position_answer(pred, ep_dir, store=None):
    """Map an identify answer 'day D | class | [x, y, z]' to '<scan>#<id>' (the form the scorer compares), by nearest centroid
    among the objects of that day's room, preferring the named class. Returns None when it cannot be parsed."""
    store = store or STORE
    m = re.search(r"day\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*\[?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)", str(pred or ""), re.I)
    if not m: return None
    day, label, xyz = int(m.group(1)), m.group(2).strip().lower(), [float(m.group(i)) for i in (3, 4, 5)]
    sessions = [json.loads(l) for l in io.open(f"{ep_dir}/public/sessions.jsonl", encoding="utf-8")]
    s = next((x for x in sessions if x["day"] == day), None)
    if not s: return None
    sid = s["clip_id"].replace("-full", ""); objs = json.load(open(f"{store}/{sid}.json"))["objects"]
    def d2(o): return sum((o["centroid"][i] - xyz[i]) ** 2 for i in range(3))
    same = {k: o for k, o in objs.items() if o["label"].lower() == label}
    pool = same or objs
    k = min(pool, key=lambda k: d2(pool[k]))
    if d2(pool[k]) > 0.05 ** 2: return None      # more than 5 cm from every object: not a real pointer
    return f"{sid.replace('scene', '')}#{k}"
