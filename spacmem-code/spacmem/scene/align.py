"""Align a reconstruction's coordinates to the annotated ones, per clip, using the camera trajectory both stores share.

Identify answers name an object, so scoring a reconstruction run needs the reconstructed object mapped back to the
annotated one it stands for. The same mapping lets annotated turn-to-object links be replayed on reconstructed objects.

The transform is a similarity (scale, rotation, translation) fitted to the camera positions recorded at the same frames
in both stores.

The correspondence is one-to-one and does not look at labels.
"""
import functools, json, os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT = f"{BASE}/data/gt_store"


def _umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, d, Vt = np.linalg.svd(D.T @ S / len(src))
    F = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: F[2, 2] = -1
    R = U @ F @ Vt
    s = np.trace(np.diag(d) @ F) / ((S ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


@functools.lru_cache(maxsize=64)
def transform(sid, recon_store, gt_store=GT):
    """(scale, rotation, translation) mapping reconstruction coordinates into annotated ones, plus the mean residual."""
    rc = json.load(open(f"{recon_store}/{sid}.json"))["camera"]
    gt = json.load(open(f"{gt_store}/{sid}.json"))["camera"]
    seen, src, dst = set(), [], []
    for f in sorted(set(rc) & set(gt), key=int):
        key = tuple(round(v, 5) for v in rc[f])          # frames were widened on conversion; keep one per pose
        if key in seen: continue
        seen.add(key); src.append(rc[f]); dst.append(gt[f])
    if len(src) < 10: return None
    src, dst = np.array(src), np.array(dst)
    s, R, t = _umeyama(src, dst)
    res = float(np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1).mean())
    return s, R, t, res


@functools.lru_cache(maxsize=64)
def _gt_objects(sid, gt_store=GT):
    o = json.load(open(f"{gt_store}/{sid}.json"))["objects"]
    return [(int(k), v["label"], np.array(v["centroid"])) for k, v in o.items()]


@functools.lru_cache(maxsize=64)
def _recon_objects(sid, recon_store):
    o = json.load(open(f"{recon_store}/{sid}.json"))["objects"]
    return [(int(k), v["label"], np.array(v["centroid"])) for k, v in o.items()]


def _iou(a0, a1, b0, b1):
    d = np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0, None)
    inter = float(d.prod())
    if inter <= 0: return 0.0
    return inter / (float((a1 - a0).prod()) + float((b1 - b0).prod()) - inter)


@functools.lru_cache(maxsize=64)
def _assign(sid, recon_store, gt_store=GT, tol=1.0):
    """One-to-one correspondence between the annotated and reconstructed objects of one clip.

    Pairs are ranked by box overlap and, only where no boxes overlap at all, by centroid distance within tol. Overlap
    comes first because a small object sitting on a large one can be nearer the large one's centroid than its own
    reconstruction. Labels are deliberately not consulted:
    label agreement is one of the quantities this correspondence is used to measure.
    """
    T = transform(sid, recon_store, gt_store)
    if T is None: return {}, {}
    s, R, t, _ = T
    gt = json.load(open(f"{gt_store}/{sid}.json"))["objects"]
    rc = json.load(open(f"{recon_store}/{sid}.json"))["objects"]
    gk, rk = sorted(gt, key=int), sorted(rc, key=int)
    G = [(np.array(gt[k]["aabb_min"]), np.array(gt[k]["aabb_max"]), np.array(gt[k]["centroid"])) for k in gk]
    P = []
    for k in rk:
        lo = s * (R @ np.array(rc[k]["aabb_min"])) + t
        hi = s * (R @ np.array(rc[k]["aabb_max"])) + t
        P.append((np.minimum(lo, hi), np.maximum(lo, hi), s * (R @ np.array(rc[k]["centroid"])) + t))
    cand = []
    for i, (a0, a1, gc) in enumerate(G):
        for j, (b0, b1, pc) in enumerate(P):
            v = _iou(a0, a1, b0, b1); d = float(np.linalg.norm(pc - gc))
            if v > 0 or d <= tol: cand.append((0 if v > 0 else 1, -v, d, i, j))
    used_i, used_j, g2r, r2g = set(), set(), {}, {}
    for _, _, _, i, j in sorted(cand):
        if i in used_i or j in used_j: continue
        used_i.add(i); used_j.add(j)
        g2r[int(gk[i])] = int(rk[j]); r2g[int(rk[j])] = int(gk[i])
    return g2r, r2g


def recon_to_gt(sid, oid, recon_store, gt_store=GT, tol=1.0):
    """The annotated object a reconstructed one stands for, or None when nothing is assigned to it."""
    return _assign(sid, recon_store, gt_store, tol)[1].get(int(oid))


def gt_to_recon(sid, oid, recon_store, gt_store=GT, tol=1.0):
    """The reconstructed object standing for an annotated one, used to replay annotated links on a reconstruction."""
    return _assign(sid, recon_store, gt_store, tol)[0].get(int(oid))
