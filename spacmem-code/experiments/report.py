"""Accuracy for one or more run folders, and the paired difference between two of them.

    python experiments/report.py runs/recon_gpt-5.6-luna
    python experiments/report.py runs/recon_gpt-5.6-luna --against runs/annotated_gpt-5.6-luna

A run folder holds one subfolder per configuration; every results.jsonl below it counts, so a configuration answered
one episode at a time is reported as a whole. With --against, every configuration is compared against the same
configuration of the other folder over the questions both answered, with a paired bootstrap interval.
"""
import argparse
import glob
import json
import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

ORDER = ["per_frame", "full_context", "memory_annotated_links", "memory"]
PRETTY = {"per_frame": "per-frame geometric", "full_context": "full context",
          "memory_annotated_links": "memory, annotated mapping", "memory": "memory, learned mapping"}


def load(folder):
    """configuration -> {query_id: row}, for every configuration present."""
    out = {}
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        rows = {}
        for p in glob.glob(os.path.join(folder, name, "**", "results.jsonl"), recursive=True):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    rows[r["query_id"]] = r
        if rows:
            out[name] = rows
    return out


def bootstrap(a, b, n=20000):
    """Paired difference b - a in accuracy points, with a 95 percent interval."""
    keys = sorted(set(a) & set(b))
    d = [int(b[k]["correct"]) - int(a[k]["correct"]) for k in keys]
    if not d:
        return None
    rng = random.Random(0)
    m = len(d)
    draws = sorted(sum(rng.choice(d) for _ in range(m)) / m for _ in range(n))
    return 100 * sum(d) / m, 100 * draws[int(.025 * n)], 100 * draws[int(.975 * n)], m


def key(name):
    return (ORDER.index(name), name) if name in ORDER else (len(ORDER), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--against", default=None, help="a run folder to compare against")
    a = ap.parse_args()

    for folder in a.folders:
        runs = load(folder)
        if not runs:
            print(f"{folder}: no results found")
            continue
        print(f"\n{folder}")
        for name in sorted(runs, key=key):
            rows = runs[name]
            n = sum(r["correct"] for r in rows.values())
            print(f"  {PRETTY.get(name, name):28s} {n:4d}/{len(rows):<4d} = {100 * n / len(rows):5.1f}%")

    if a.against:
        base = load(a.against)
        if not base:
            sys.exit(f"{a.against}: no results found")
        for folder in a.folders:
            runs = load(folder)
            print(f"\n{folder} against {a.against}")
            for name in sorted(set(runs) & set(base), key=key):
                r = bootstrap(base[name], runs[name])
                if r is None:
                    continue
                d, lo, hi, m = r
                flag = "" if lo <= 0 <= hi else "   <- interval excludes zero"
                print(f"  {PRETTY.get(name, name):28s} {d:+6.1f} points  [{lo:+.1f}, {hi:+.1f}]  n={m}{flag}")


if __name__ == "__main__":
    main()
