"""Convert a reconstruction (episode-level world_state.json) into our per-clip scene-record format.

The evaluation reads one JSON per clip with the same shape as data/gt_store/<scan>.json, so pointing --store at the
output folder is all that is needed to run any method on reconstructed geometry instead of annotated geometry.

Two differences from the annotated store are handled here.
  * The reconstruction processes every 6th frame (5 fps), while the context builders look a frame up exactly. Each
    processed frame is therefore widened to cover the frames either side of it, so a lookup at any frame resolves to
    the nearest processed one.
  * A camera pose arrives as a 4x4 matrix; we keep its translation, which is what the context builders use.

An object's runner-up classes are kept as label_alternatives. A delivery carrying label_aliases has curated them, so
that field is taken as it stands and an empty one means the delivery judged no runner-up credible; only a delivery
without the field falls back to the next two entries of label_counts, which are raw votes and much noisier. They are
weaker evidence than the top label, not synonyms, and a consumer that ignores the field sees what it saw before.

Object ids become the integer in "<scene>/object:NNNN". Coordinates stay in the reconstruction's own frame, which is
levelled but has its own origin and rotation per clip; every relation the benchmark asks for is relative.

Usage: python -m spacmem.scene.from_reconstruction data/episode_mcep_52939f339e62302e/world_state.json --out data/recon_store
"""
import argparse, collections, json, os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert(world_state_path, out_dir, step=None, half=None):
    w = json.load(open(world_state_path, encoding="utf-8"))
    step = step or int(w.get("conventions", {}).get("frame_step", 6))
    half = half if half is not None else step // 2
    objs = collections.defaultdict(dict)
    for o in w["objects"]:
        sid = o["clip_id"].replace("-full", "")
        rec = dict(label=o["label"], centroid=o["centroid"], aabb_min=o["box_min"], aabb_max=o["box_max"])
        alt = (o["label_aliases"] if "label_aliases" in o                      # the delivery's own judgement, [] included
               else [k for k, _ in sorted((o.get("label_counts") or {}).items(), key=lambda kv: -kv[1])[1:3]])
        alt = [x for x in alt if x != o["label"]]
        if alt: rec["label_alternatives"] = alt
        objs[sid][int(o["id"].rsplit(":", 1)[1])] = rec
    sight = collections.defaultdict(lambda: collections.defaultdict(set))
    cam = collections.defaultdict(dict)
    for s in w["sightings"]:
        sid = s["clip_id"].replace("-full", ""); f = s["frame"]
        window = range(max(0, f - half), f + (step - half))       # widen to the frames this one stands for
        m = s.get("camera_to_world")
        if m:
            t = [m[0][3], m[1][3], m[2][3]]
            for g in window: cam[sid][str(g)] = t
        for oid in s.get("visible", []):
            k = int(oid.rsplit(":", 1)[1])
            if k in objs[sid]: sight[sid][k].update(window)
    os.makedirs(out_dir, exist_ok=True); written = []
    for sid, o in objs.items():
        rec = dict(scene_id=sid, source=os.path.relpath(world_state_path, BASE).replace(os.sep, "/"),
                   frame_rate=30, frame_step=step, coordinates="reconstruction, per clip",
                   objects={str(k): v for k, v in sorted(o.items())},
                   sightings={str(k): sorted(sight[sid][k]) for k in sorted(o) if sight[sid][k]},
                   camera=cam[sid])
        json.dump(rec, open(f"{out_dir}/{sid}.json", "w"), separators=(",", ":"))
        written.append((sid, len(rec["objects"]), len(rec["sightings"]), len(rec["camera"])))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("world_state"); ap.add_argument("--out", default=f"{BASE}/data/recon_store")
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()
    for sid, n_obj, n_seen, n_cam in convert(a.world_state, a.out, a.step):
        print(f"{sid:20s} objects {n_obj:4d}  with sightings {n_seen:4d}  camera frames {n_cam:6d}")
    print("written to", a.out)
