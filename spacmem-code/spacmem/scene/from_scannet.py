"""Build the ground-truth scene record for each ScanNet scan used by the release.

Output: gt_store/<scan_id>.json with
  objects: {object_id: {label, raw_label, centroid, aabb_min, aabb_max, vertex_count}}
  sightings: {object_id: [frames where the instance mask covers >= 0.5% of the image]}
  frame_count, invalid_pose_frames
The structural classes, size thresholds and 0.5% visibility rule are the benchmark generator's own, so the records
agree with the geometry its answer keys were computed from.

Usage: SCANNET_SCANS=/path/to/scannet/scans python -m spacmem.scene.from_scannet [scene0000_00 ...]
"""
import io, json, os, struct, sys, time, zipfile, glob
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = f"{BASE}/data/final_release"
ROOT = os.environ.get("SCANNET_SCANS", f"{BASE}/data/scannet/scans")   # raw ScanNet scans
OUT = f"{BASE}/data/gt_store"
STRUCT = {"wall", "floor", "ceiling", "door", "doorframe", "window", "remove", "unannotated", "otherstructure", "floor mat"}
MIN_VERTS, MIN_EXTENT, MIN_PROM, MIN_PIX = 200, 0.05, 0.005, 50


def label_mapping():
    """ScanNet raw label -> the release's canonical class, from any episode's label_mapping.json (all are identical)."""
    lm = json.load(io.open(glob.glob(f"{REL}/shard_*/*/mcep_*/public/label_mapping.json")[0], encoding="utf-8"))
    return {m["raw_label"]: m["canonical_label"] for m in lm["raw_label_mappings"]}


def read_ply(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip(); header.append(line)
            if line == "end_header": break
        props = []; nverts = None; sec = None
        for h in header:
            if h.startswith("element vertex"): nverts = int(h.split()[-1]); sec = "v"
            elif h.startswith("element face"): sec = "f"
            elif h.startswith("property") and sec == "v": props.append(tuple(h.split()[1:]))
        dt = np.dtype([(n, {"float": "<f4", "uchar": "u1", "ushort": "<u2", "int": "<i4"}[t]) for t, n in props])
        return np.frombuffer(f.read(nverts * dt.itemsize), dtype=dt)


def sens_pose_validity(sid, align=None):
    """Frame count, invalid-pose frames, and (if align is given) the camera centre per valid frame in the aligned frame."""
    bad = []; centres = {}
    with open(f"{ROOT}/{sid}/{sid}.sens", "rb") as f:
        f.read(4); n = struct.unpack("Q", f.read(8))[0]; f.read(n); f.read(256); f.read(8); f.read(16); f.read(4)
        nframes = struct.unpack("Q", f.read(8))[0]
        for i in range(nframes):
            pose = np.frombuffer(f.read(64), dtype="f4"); f.read(16); cs, ds = struct.unpack("QQ", f.read(16)); f.seek(cs + ds, 1)
            if not np.all(np.isfinite(pose)): bad.append(i)
            elif align is not None: centres[str(i)] = (align @ pose.reshape(4, 4).astype(np.float64))[:3, 3].round(4).tolist()
    return nframes, bad, centres


def build(sid, raw2canon):
    t0 = time.time()
    v = read_ply(f"{ROOT}/{sid}/{sid}_vh_clean_2.ply"); xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(float)
    M = np.array([float(x) for x in open(f"{ROOT}/{sid}/{sid}.txt").read().split("axisAlignment = ")[1].split("\n")[0].split()]).reshape(4, 4)
    xyz = (M @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    segs = np.array(json.load(open(f"{ROOT}/{sid}/{sid}_vh_clean_2.0.010000.segs.json"))["segIndices"])
    objects = {}
    for g in json.load(open(f"{ROOT}/{sid}/{sid}.aggregation.json"))["segGroups"]:
        if g["label"] in STRUCT: continue
        m = np.isin(segs, g["segments"]); p = xyz[m]
        if m.sum() < MIN_VERTS or (p.max(0) - p.min(0)).max() < MIN_EXTENT: continue
        objects[int(g["objectId"])] = dict(label=raw2canon.get(g["label"], g["label"]), raw_label=g["label"],
                                          centroid=[round(float(x), 4) for x in p.mean(0)],
                                          aabb_min=[round(float(x), 4) for x in p.min(0)], aabb_max=[round(float(x), 4) for x in p.max(0)],
                                          vertex_count=int(m.sum()))
    nframes, bad, camera = sens_pose_validity(sid, align=M); badset = set(bad)
    z = zipfile.ZipFile(f"{ROOT}/{sid}/{sid}_2d-instance-filt.zip")
    sight = {oid: [] for oid in objects}
    area = None
    for fr in range(nframes):
        if fr in badset: continue
        try: a = np.array(Image.open(io.BytesIO(z.read(f"instance-filt/{fr}.png"))))
        except KeyError: continue
        if area is None: area = a.shape[0] * a.shape[1]
        ids, c = np.unique(a, return_counts=True)
        for i, n in zip(ids.tolist(), c.tolist()):
            oid = i - 1
            if oid in sight and n >= max(MIN_PIX, MIN_PROM * area): sight[oid].append(fr)
    json.dump(dict(scene_id=sid, source="scannet_gt", frame_rate=30, frame_count=nframes, image_size=[int(x) for x in a.shape[:2]],
                   invalid_pose_frames=bad, camera=camera, objects={str(k): v for k, v in objects.items()},
                   sightings={str(k): v for k, v in sight.items()}),
              open(f"{OUT}/{sid}.json", "w"))
    print(f"{sid}: {len(objects)} objects, {nframes} frames, {len(bad)} invalid poses, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    scans = sys.argv[1:] or sorted(os.listdir(ROOT))
    os.makedirs(OUT, exist_ok=True); raw2canon = label_mapping()
    for sid in scans:
        if os.path.exists(f"{OUT}/{sid}.json"): print(sid, "cached"); continue
        build(sid, raw2canon)
