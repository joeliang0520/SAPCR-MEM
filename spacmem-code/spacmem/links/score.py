"""Score learned links against the gold links (evaluation side only).
Precision: learned (turn, object) links that exist in gold (any role). Recall: gold fact-bearing (turn, object) links
found among learned links. Also the object-level view: gold fact-bearing objects whose every fact turn was linked.
Usage: python -m spacmem.links.score [--learned data/memory/learned] [--episodes rel ...]
"""
import argparse, glob, io, json, os, collections

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8")] if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--learned", default=f"{BASE}/data/memory/learned"); ap.add_argument("--gold", default=f"{BASE}/data/memory/gold"); ap.add_argument("--episodes", nargs="*", default=[])
    a = ap.parse_args()
    files = [f"{a.learned}/{e}.jsonl" for e in a.episodes] if a.episodes else sorted(glob.glob(f"{a.learned}/shard_*/*/*.jsonl"))
    T = collections.Counter(); print(f"{'episode':26s} {'gold fact links':>15s} {'learned':>8s} {'precision':>10s} {'recall':>7s} {'wrong-object':>13s}")
    for lf in files:
        rel = os.path.relpath(lf, a.learned).replace(os.sep, "/")[:-6]; L = load(lf); G = load(f"{a.gold}/{rel}.jsonl")
        if L is None or G is None: continue
        gold_all = {(g["turn"], g["scene"], g["object"]) for g in G}; gold_fact = {(g["turn"], g["scene"], g["object"]) for g in G if g["fact_bearing"]}
        learned = {(l["turn"], l["scene"], l["object"]) for l in L}
        tp = len(learned & gold_all); rec_hit = len(gold_fact & learned)
        # wrong-object: gold fact link on a turn where the learner linked a different object of the same class only
        gold_turns = collections.defaultdict(set)
        for g in G:
            if g["fact_bearing"]: gold_turns[g["turn"]].add((g["scene"], g["object"]))
        learned_turns = collections.defaultdict(set)
        for l in L: learned_turns[l["turn"]].add((l["scene"], l["object"]))
        wrong = sum(1 for t, objs in gold_turns.items() for o in objs if o not in learned_turns.get(t, set()) and learned_turns.get(t))
        p = tp / max(1, len(learned)); r = rec_hit / max(1, len(gold_fact))
        print(f"{rel.split('/')[-1]:26s} {len(gold_fact):15d} {len(learned):8d} {100 * p:9.1f}% {100 * r:6.1f}% {wrong:13d}")
        T.update(dict(gold_fact=len(gold_fact), learned=len(learned), tp=tp, rec=rec_hit, wrong=wrong))
    if T["learned"]:
        print(f"\nTOTAL: precision {100 * T['tp'] / T['learned']:.1f}%  recall of fact-bearing links {100 * T['rec'] / T['gold_fact']:.1f}%  ({T['rec']}/{T['gold_fact']}); gold fact links missed while the turn was linked to another object: {T['wrong']}")


if __name__ == "__main__":
    main()
