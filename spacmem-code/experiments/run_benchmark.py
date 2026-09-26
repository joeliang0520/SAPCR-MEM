"""Run the method and its ablations over the benchmark, then print the accuracy table.

Four configurations, each answering every question of every episode.

  memory                  the method, an object-centric memory with turns mapped to objects by the learned linker
  memory_annotated_links  the same memory with the annotated turn-to-object mapping (privileged)
  full_context            the uncompressed history, with objects as they first appear and the conversation in time order
  per_frame               per-frame observations with no object identity, the geometric baseline

--scenes picks the geometry. 'annotated' reads the ScanNet-derived scene records in data/gt_store; any other name
reads a reconstruction prepared with experiments/prepare_reconstruction.py. The prompts follow the geometry: the
estimated-geometry prompts add a paragraph saying the scene was estimated from video and may be wrong.

    python experiments/run_benchmark.py --scenes annotated --model gpt-5.6-luna
    python experiments/run_benchmark.py --scenes recon --model z-ai/glm-5.3-flash --reasoning-effort medium --max-tokens 64000

The endpoint and key come from --endpoint/--key or from SPACMEM_ENDPOINT and OPENROUTER_API_KEY. Every step resumes:
links already predicted are kept, answered questions are skipped and replies are cached, so a stopped run continues
where it stopped. Results go to runs/<scenes>_<model>/<configuration>/.
"""
import argparse, glob, os, subprocess, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

REL = "data/final_release"
PROMPTS = "spacmem/prompts/answer"
CONFIGS = {
    "memory": dict(method="memory_a", prompt="memory.md", links="learned"),
    "memory_annotated_links": dict(method="memory_a", prompt="memory.md", links="annotated"),
    "full_context": dict(method="full", prompt="full_context.md", links=None),
    "per_frame": dict(method="per_frame", prompt="per_frame.md", links=None),
}


def run(cmd, env, log):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "-m", *cmd], env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"{cmd[0]} failed with exit code {r.returncode}; see {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="annotated", help="'annotated', or the --name given to prepare_reconstruction.py")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--episodes", nargs="*", default=None, help="release episode paths, shard/task/episode (default: all)")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--endpoint", default=os.environ.get("SPACMEM_ENDPOINT", "https://openrouter.ai/api/v1"))
    ap.add_argument("--key", default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=None, help="answer budget per call (default 4000; reasoning models need more)")
    ap.add_argument("--reasoning-effort", default=None, help="low / medium / high, for models that accept it")
    ap.add_argument("--provider", default=None, help="OpenRouter only: pin one provider so consecutive calls share its prompt cache")
    ap.add_argument("--prompt-cache", action="store_true", help="mark the repeating part of each context for the provider's prompt cache")
    a = ap.parse_args()
    if not a.key:
        ap.error("no API key: pass --key or set OPENROUTER_API_KEY")

    annotated = a.scenes == "annotated"
    tag = f"{a.scenes}_{a.model.split('/')[-1]}"
    prompts = f"{PROMPTS}/{'annotated' if annotated else 'estimated'}_geometry"
    eps = a.episodes or sorted(os.path.relpath(p, REL).replace(os.sep, "/") for p in glob.glob(f"{REL}/shard_*/*/mcep_*"))
    if not eps:
        sys.exit(f"no episodes found under {REL}")
    # the scene records each episode is answered from, one store per episode for a reconstruction
    store = {ep: "data/gt_store" if annotated else f"data/store_{a.scenes}/{ep.split('/')[-1]}" for ep in eps}
    missing = [ep for ep in eps if not os.path.isdir(store[ep])]
    if missing:
        sys.exit(f"{len(missing)} episodes have no scene records, e.g. {store[missing[0]]}; run experiments/prepare_reconstruction.py first")

    env = {**os.environ, "SPACMEM_ENDPOINT": a.endpoint, "OPENROUTER_API_KEY": a.key}     # the key stays off the command line
    common = ["--model", a.model, "--workers", str(a.workers)] + (["--reasoning-effort", a.reasoning_effort] if a.reasoning_effort else [])
    links = {"annotated": "data/memory/gold" if annotated else f"data/memory/annotated_{a.scenes}",
             "learned": f"data/memory/learned_{tag}"}

    # 1. the annotated mapping, read from the release (a reconstruction's is replayed onto it by prepare_reconstruction.py)
    if annotated and not os.path.isdir(links["annotated"]):
        print("--- annotated links", flush=True)
        run(["spacmem.links.from_release"], env, f"runs/{tag}/logs/links_annotated.log")

    # 2. the learned mapping, predicted once per episode by the same model that answers
    if "memory" in a.configs:
        todo = [ep for ep in eps if not os.path.exists(f"{links['learned']}/{ep}.jsonl")]
        print(f"--- predicting links: {len(todo)} episodes", flush=True)
        groups = [todo] if annotated else [[ep] for ep in todo]        # a reconstruction needs its own store per episode
        for g in filter(None, groups):
            run(["spacmem.links.predict", "--episodes", *g, "--store", store[g[0]], "--out", links["learned"],
                 "--max-tokens", "8000", *common], env, f"runs/{tag}/logs/links_learned.log")

    # 3. answer
    for name in a.configs:
        c = CONFIGS[name]
        cmd = ["spacmem.answer", "--method", c["method"], "--prompt", f"{prompts}/{c['prompt']}", *common]
        if c["links"]: cmd += ["--links", links[c["links"]]]
        if c["method"] == "per_frame": cmd += ["--fps", "1", "--episode-serial"]
        if a.max_tokens: cmd += ["--max-tokens", str(a.max_tokens)]
        if a.provider: cmd += ["--provider", a.provider]
        if a.prompt_cache: cmd += ["--prompt-cache"]
        print(f"--- answering: {name}", flush=True)
        if annotated:
            run([*cmd, "--run", f"{tag}/{name}", "--episodes", *eps], env, f"runs/{tag}/{name}/logs/launch.log")
        else:
            for ep in eps:
                eid = ep.split("/")[-1]
                run([*cmd, "--run", f"{tag}/{name}/{eid}", "--episodes", ep, "--store", store[ep], "--align-store", "data/gt_store"],
                    env, f"runs/{tag}/{name}/{eid}/logs/launch.log")

    # 4. report
    subprocess.run([sys.executable, "experiments/report.py", f"runs/{tag}"])


if __name__ == "__main__":
    main()
