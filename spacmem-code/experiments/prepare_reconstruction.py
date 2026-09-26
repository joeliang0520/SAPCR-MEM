"""Turn a reconstruction into scene records the benchmark runner can read, one store per episode.

The input is one world_state.json, or a folder searched for them (one per episode). Each is converted into
data/store_<name>/<episode>/, and the annotated turn-to-object links are replayed onto its objects into
data/memory/annotated_<name>/. Stores are per episode because each episode is reconstructed independently and object
ids are only unique within one reconstruction. Nothing here calls a model.

    python experiments/prepare_reconstruction.py path/to/packages --name recon
    python experiments/run_benchmark.py --scenes recon ...
"""
import argparse, glob, io, json, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, BASE)
from spacmem.links.replay import replay
from spacmem.scene.from_reconstruction import convert

REL = "data/final_release"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a world_state.json, or a folder holding one per episode")
    ap.add_argument("--name", required=True, help="short name for this reconstruction; run_benchmark takes it as --scenes")
    ap.add_argument("--force", action="store_true", help="rebuild stores and links that already exist")
    a = ap.parse_args()

    if not os.path.isdir("data/memory/gold"):
        sys.exit("data/memory/gold is missing: build it first with python -m spacmem.links.from_release")
    files = [a.path] if a.path.endswith(".json") else sorted(glob.glob(f"{a.path}/**/world_state.json", recursive=True))
    episode_dir = {os.path.basename(p): os.path.relpath(p, REL).replace(os.sep, "/") for p in glob.glob(f"{REL}/shard_*/*/mcep_*")}

    built = kept = dropped = 0; failures = []
    for i, f in enumerate(files, 1):
        eid = str(json.load(io.open(f, encoding="utf-8")).get("episode_id", "")).split(":")[0]
        ep = episode_dir.get(eid)
        if ep is None:
            failures.append((f, f"no release episode matches {eid!r}")); continue
        store = f"data/store_{a.name}/{eid}"
        if a.force or not os.path.isdir(store):
            convert(f, store); built += 1
        links = f"data/memory/annotated_{a.name}"
        if a.force or not os.path.exists(f"{links}/{ep}.jsonl"):
            k, d = replay(ep, store, links); kept += k; dropped += d
        if i % 20 == 0: print(f"  {i}/{len(files)}", flush=True)

    print(f"episodes        {len(files) - len(failures)} of {len(files)}")
    print(f"stores built    {built}  -> data/store_{a.name}/")
    if kept + dropped:
        print(f"annotated links {kept} replayed, {dropped} dropped ({100 * dropped / (kept + dropped):.1f}% have no reconstructed counterpart)")
    for f, msg in failures:
        print(f"  skipped {f}: {msg}")


if __name__ == "__main__":
    main()
