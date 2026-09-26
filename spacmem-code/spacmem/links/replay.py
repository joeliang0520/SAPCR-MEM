"""Replay the annotated turn-to-object links onto reconstructed objects.

Each annotated link names an annotated object; the reconstruction condition needs the reconstructed object standing for
it. A link whose object has no counterpart is dropped, and the count of those is reported, since a dropped link is a
fact the answerer never sees.

Usage: python -m spacmem.links.replay --store data/store_<name>/<episode> --out data/memory/annotated_<name> --episodes <shard/task/episode> ...
"""
import argparse, io, json, os
from spacmem.scene.align import gt_to_recon

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD = f"{BASE}/data/memory/gold"


def replay(ep, store, out, gold=GOLD):
    """Write <out>/<ep>.jsonl with every annotated link of the episode moved onto the reconstruction's objects."""
    store = os.path.abspath(store)
    dst = f"{out}/{ep}.jsonl"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    kept = dropped = 0
    with io.open(dst, "w", encoding="utf-8", newline="\n") as f:
        for line in io.open(f"{gold}/{ep}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            m = gt_to_recon(r["scene"], int(r["object"]), store)
            if m is None:
                dropped += 1
                continue
            r["object"] = m
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
    return kept, dropped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="the reconstruction's scene records for these episodes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", nargs="+", required=True, help="release episode paths, shard/task/episode")
    a = ap.parse_args()
    for ep in a.episodes:
        kept, dropped = replay(ep, a.store, a.out)
        print(f"{ep.split('/')[-1]}  kept {kept}  dropped {dropped}")
