"""Answer every question of the given episodes under one configuration and score the answers.

The configuration is a context builder (--method), a prompt and a folder of scene records. Each question gets one
model call over the transcript up to that question and the scene as seen so far.

Every run lives in runs/<run_id>/ with config.json (model, endpoint, prompt file + hash, scene-record hash, episodes,
workers, timestamp), results.jsonl (one row per question, resumable), run.log and summary.json. Replies are cached in
runs/_cache/ keyed by (prompt hash, model, context hash), so re-running never re-bills.

Usage, from the repository root:
  python -m spacmem.answer --run full --episodes shard_06/workspace_setup/mcep_15f097b8f3ab209c
  python -m spacmem.answer --run full --all --workers 6
The endpoint and key come from --endpoint/--key or from SPACMEM_ENDPOINT and OPENROUTER_API_KEY.
"""
import argparse, glob, hashlib, io, json, os, re, sys, threading, time
if __package__ in (None, ""):      # so `python spacmem/answer.py` works as well as `python -m spacmem.answer`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spacmem.context.memory import alt
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = f"{BASE}/data/final_release"
STORE = f"{BASE}/data/gt_store"
RUNS = f"{BASE}/runs"
CACHE = f"{RUNS}/_cache"
ENDPOINT = os.environ.get("SPACMEM_ENDPOINT", "https://openrouter.ai/api/v1")    # any OpenAI-compatible endpoint
KEY = os.environ.get("OPENROUTER_API_KEY")


def sha(s):
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()[:16]


def load_store(sid):
    return json.load(open(f"{STORE}/{sid}.json"))


def store_hash():
    h = hashlib.sha256()
    h.update(os.path.abspath(STORE).encode())
    for p in sorted(glob.glob(f"{STORE}/*.json")):
        h.update(os.path.basename(p).encode()); h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()[:16]


CONTEXT_BUILDER = "v3: reveals+turns interleaved by frame, 0.5% sighting rule, 3-decimal coords, camera position on each reveal line, 'in view' line before each user turn, room inventory per session"


def build_context(ep_dir, turn_id, geometry=True, turn_visibility=True, inventory=True):
    tr = json.load(io.open(f"{ep_dir}/public/transcript.json", encoding="utf-8"))["turns"]
    sessions = [json.loads(l) for l in io.open(f"{ep_dir}/public/sessions.jsonl", encoding="utf-8")]
    qturn = next(t for t in tr if t["global_turn_id"] == turn_id)
    qidx, qsess, qframe = qturn["global_turn_index"], qturn["session_index"], qturn["local_frame"]
    lines = []
    for s in sessions:
        if s["session_index"] > qsess: break
        sid = s["clip_id"].replace("-full", "")
        last = qframe if s["session_index"] == qsess else max(t["local_frame"] for t in tr if t["session_index"] == s["session_index"])
        events = []
        if geometry:
            st = load_store(sid); objs = st["objects"]
            for oid, frames in st["sightings"].items():
                fr = [f for f in frames if f <= last]
                if not fr: continue
                o = objs[oid]; c = o["centroid"]; lo = o["aabb_min"]; hi = o["aabb_max"]
                cam = st.get("camera", {}).get(str(fr[0]))
                cam_s = f"   camera [{cam[0]:.3f}, {cam[1]:.3f}, {cam[2]:.3f}]" if cam else ""
                events.append((fr[0], 0, f"frame {fr[0]:5d}  seen   {o['label']}{alt(o)} #{oid}   centroid [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]   box [{lo[0]:.3f}, {lo[1]:.3f}, {lo[2]:.3f}] to [{hi[0]:.3f}, {hi[1]:.3f}, {hi[2]:.3f}]{cam_s}"))
        def in_view(frame):
            vis = sorted((int(oid) for oid, frames in st["sightings"].items() if frame in frames), key=int)
            return ", ".join(f"{objs[str(o)]['label']} #{o}" for o in vis) or "(nothing)"
        seen_frames = set()
        for t in tr:
            if t["session_index"] != s["session_index"] or t["global_turn_index"] >= qidx: continue
            if geometry and turn_visibility and t["speaker"] == "user" and t["local_frame"] not in seen_frames:
                seen_frames.add(t["local_frame"])
                events.append((t["local_frame"], 0.5, f"frame {t['local_frame']:5d}  in view: {in_view(t['local_frame'])}"))
            events.append((t["local_frame"], 1, f"frame {t['local_frame']:5d}  {t['speaker']:9s}  [{t['global_turn_id']}] {t['text']}"))
        events.sort(key=lambda e: (e[0], e[1]))
        lines.append(f"=== session {s['session_index'] + 1} · day {s['day']} · {s['room_gloss']} · room id {sid} · frames 0 to {last} ===")
        lines += [e[2] for e in events]
        if geometry and inventory:
            by_label = {}
            for oid, frames in st["sightings"].items():
                if frames and frames[0] <= last: by_label.setdefault(objs[oid]["label"], []).append(int(oid))
            lines.append(f"room inventory {sid} (every object seen so far in this room): " + "; ".join(f"{lab} " + " ".join(f"#{o}" for o in sorted(ids)) for lab, ids in sorted(by_label.items())))
        if s["session_index"] == qsess:
            if geometry:
                lines.append(f"frame {qframe:5d}  visible now: {in_view(qframe)}")
            lines.append(f"frame {qframe:5d}  user       [{turn_id}] {qturn['text']}")
        lines.append("")
    return "\n".join(lines)


def work_count(text):
    """Number of counts=true rows in the model's own work list, or None if there is no parsable list."""
    try:
        w = json.loads(text or "").get("work")
        return sum(1 for x in w if x.get("counts") is True) if isinstance(w, list) else None
    except Exception:
        return None


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "work": {"type": "array", "items": {"type": "object", "properties": {
            "object": {"type": "string"}, "why_candidate": {"type": "string"}, "value": {"type": "string"}, "counts": {"type": "boolean"}},
            "required": ["object", "why_candidate", "value", "counts"], "additionalProperties": False}},
        "answer": {"anyOf": [{"type": "integer"}, {"type": "boolean"}, {"type": "string"}]}},
    "required": ["work", "answer"], "additionalProperties": False}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "oracle_answer", "strict": True, "schema": ANSWER_SCHEMA}}


EXTRA_BODY = {}                      # merged into every request body, e.g. {"reasoning": {"effort": "medium"}}
_ENDPOINT_OK = threading.Event(); _ENDPOINT_OK.set()   # cleared while the endpoint is rate limited; every worker waits on it
_PROBE_LOCK = threading.Lock()


def _rate_limited(endpoint, model, key):
    """Circuit breaker. The first worker to see a 429 or a 5xx pauses every worker in this process and probes the
    endpoint with a tiny request once a minute; the others just wait. Nothing is retried and no error row is written
    while paused, so an outage leaves the run resumable instead of filling results.jsonl with unanswered questions."""
    with _PROBE_LOCK:
        i_probe = _ENDPOINT_OK.is_set()
        if i_probe: _ENDPOINT_OK.clear()
    if not i_probe:
        _ENDPOINT_OK.wait(); return
    t0 = time.time(); print(f"[{datetime.now():%H:%M}] rate limited by {endpoint} ({model}); pausing all workers", flush=True)
    while True:
        time.sleep(60)
        try:
            r = requests.post(f"{endpoint}/chat/completions", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json={"model": model, "messages": [{"role": "user", "content": "Reply with ok."}], "max_tokens": 5}, timeout=60)
            if r.status_code == 200: break
        except Exception:
            pass
    print(f"[{datetime.now():%H:%M}] endpoint answers again after {(time.time() - t0) / 60:.0f} min; resuming", flush=True)
    _ENDPOINT_OK.set()


def billed_usage(endpoint, key):
    """Dollars billed so far on this key, as reported by the provider (OpenRouter only)."""
    if "openrouter.ai" not in endpoint: return None
    try:
        r = requests.get(f"{endpoint}/auth/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        return float(r.json()["data"]["usage"])
    except Exception:
        return None


def cache_split(text, marker="=== session"):
    """Split a context into the part that repeats across an episode's questions and the part that varies.

    Every session block but the current one is bounded by that session's own last turn rather than by the question, so
    it is byte identical for every question of the episode. Cutting at the final session header therefore gives the
    longest prefix a provider cache can reuse. A short prefix is not worth a breakpoint: providers cache nothing under
    about 1024 tokens, so leave those requests as one block.
    """
    i = text.rfind("\n" + marker)
    if i < 4000:
        return None, text
    return text[:i + 1], text[i + 1:]


def call_llm(endpoint, model, key, system, user, cache_key=None, max_tokens=4000, retries=6, structured=True, seed=None, response_format=None, cache_prefix=None, cache_tag=None):
    cpath = f"{CACHE}/{cache_key}.json" if cache_key else None
    if cpath and os.path.exists(cpath):
        j = json.load(open(cpath, encoding="utf-8")); return j["text"], j["usage"], True
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0, "max_tokens": max_tokens}
    if cache_prefix:
        # Mark the repeating prefix so the provider caches it instead of guessing a breakpoint, and key the request to
        # the episode so every question of it is routed to the same endpoint and reads the same warm cache.
        body["messages"][1]["content"] = [{"type": "text", "text": cache_prefix, "prompt_cache_breakpoint": {"mode": "explicit"}},
                                          {"type": "text", "text": user}]
        body["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
        if cache_tag: body["prompt_cache_key"] = cache_tag
    if structured: body["response_format"] = response_format or RESPONSE_FORMAT
    if seed is not None: body["seed"] = seed
    body.update(EXTRA_BODY)
    attempt = 0
    while attempt < retries:
        _ENDPOINT_OK.wait()
        try:
            r = requests.post(f"{endpoint}/chat/completions", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                              json=body, timeout=600)
            if r.status_code == 429 or r.status_code >= 500:
                _rate_limited(endpoint, model, key); continue     # a rate limit or an outage pauses the run instead of consuming a retry
            r.raise_for_status(); j = r.json()
            text, usage = j["choices"][0]["message"]["content"], j.get("usage", {})
            if cpath:
                os.makedirs(CACHE, exist_ok=True)
                json.dump({"text": text, "usage": usage, "model": model, "endpoint": endpoint, "time": datetime.now().isoformat()}, open(cpath, "w", encoding="utf-8"))
            return text, usage, False
        except Exception as e:
            attempt += 1
            if attempt >= retries: return f"ERROR: {e}", {}, False
            time.sleep(5 * attempt)


def parse_answer(text):
    text = text or ""
    try:
        return json.loads(text).get("answer")          # structured output: the whole reply is the object
    except Exception:
        pass
    end = text.rfind("}"); depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}": depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                try: return json.loads(text[i:end + 1]).get("answer")
                except Exception: break
    m = re.findall(r"\"answer\"\s*:\s*(true|false|-?\d+|\"[^\"]*\")", text)
    if not m: return None
    v = m[-1]
    return {"true": True, "false": False}.get(v, v.strip('"') if v.startswith('"') else int(v))


def score(pred, ai, track_map):
    typ = ai["scoreability"]["answer_type"]; key = ai["canonical_answer"]
    if typ == "count":
        if not isinstance(pred, int) or isinstance(pred, bool):
            try: pred = int(pred)
            except Exception: return False
        lo, hi = ai.get("accepted_answers") or [key, key]
        return lo <= pred <= hi
    if typ == "bool":
        if isinstance(pred, str): pred = pred.strip().lower() in ("true", "yes")
        return bool(pred) is bool(key)
    if typ == "track":
        want = track_map.get(key); got = str(pred or "").strip().lower().replace("scene", "").replace(" ", "")
        return want is not None and got == want
    return False


def fam(pid):
    return pid.replace("multi_clip_semantic_", "").replace("evidence_recall_", "").replace("multi_clip_cross_slot_semantic_union_", "union_").replace("generic_CrossClip", "")


def summarize(rows):
    def acc(rs): return {"n": len(rs), "correct": sum(r["correct"] for r in rs), "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 3) if rs else None}
    sc = [r for r in rows if r["officially_scoreable"]]
    out = {"all_rows": acc(rows), "officially_scoreable": acc(sc), "identify_by_object_id": acc([r for r in rows if r["answer_type"] == "track"]),
           "count_questions_scored_from_work_list": {"n": sum(1 for r in sc if r["answer_type"] == "count"),
                                                    "correct": sum(1 for r in sc if r["answer_type"] == "count" and r.get("correct_from_work")),
                                                    "answer_disagreed_with_work": sum(1 for r in sc if r["answer_type"] == "count" and r.get("work_true") is not None and r["work_true"] != r["pred"])},
           "by_family": {f: acc([r for r in sc if fam(r["family"]) == f]) for f in sorted({fam(r["family"]) for r in sc})},
           "by_session": {s: acc([r for r in sc if r["session"] == s]) for s in sorted({r["session"] for r in sc})},
           "prompt_tokens": {"mean": int(sum(r["prompt_tokens"] or 0 for r in rows) / len(rows)) if rows else None, "max": max((r["prompt_tokens"] or 0) for r in rows) if rows else None},
           "cached_replies": sum(1 for r in rows if r.get("cached")), "errors": sum(1 for r in rows if str(r.get("raw", "")).startswith("ERROR"))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id; folder runs/<date>_<run> is created or resumed")
    ap.add_argument("--episodes", nargs="*", default=[]); ap.add_argument("--all", action="store_true")
    ap.add_argument("--prompt", default=f"{BASE}/spacmem/prompts/answer/annotated_geometry/full_context.md")
    ap.add_argument("--endpoint", default=ENDPOINT); ap.add_argument("--model", default="gpt-5.6-luna"); ap.add_argument("--key", default=KEY)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no-geometry", action="store_true"); ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--only-turn", default=None); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=4000, help="completion budget per call (reasoning models need more)")
    ap.add_argument("--reasoning-effort", default=None, help="sets body.reasoning.effort (low/medium/high) for models that accept it")
    ap.add_argument("--unstructured", action="store_true", help="do not send the JSON schema as response_format")
    ap.add_argument("--store", default=None, help="folder of <scan_id>.json scene records (default data/gt_store; point at a reconstruction to evaluate it)")
    ap.add_argument("--no-turn-visibility", action="store_true", help="omit the 'in view' line before each user turn")
    ap.add_argument("--no-inventory", action="store_true", help="omit the per-room inventory line")
    ap.add_argument("--seed", type=int, default=None, help="pass an OpenAI-style seed with every request (recorded in config)")
    ap.add_argument("--release", default=None, help="release folder (default data/final_release); e.g. data/final_release_combined")
    ap.add_argument("--method", default="full", choices=["full", "memory_a", "per_frame"], help="context: 'full' = interleaved log; 'memory_a' = object-centred memory dump built from turn-to-object links")
    ap.add_argument("--links", default=None, help="links folder for --method memory_a (default data/memory/gold)")
    ap.add_argument("--all-mentions", action="store_true", help="memory_a: show a turn under every object it mentions, locators included (default: subjects only)")
    ap.add_argument("--ids", action="store_true", help="per_frame: keep object ids on the observation lines (default: no ids)")
    ap.add_argument("--fps", type=float, default=1.0, help="per_frame: observations per second")
    ap.add_argument("--provider", default=None, help="OpenRouter only: pin one provider slug (e.g. deepinfra) so consecutive calls hit the same prompt cache")
    ap.add_argument("--prompt-cache", action="store_true", help="mark the repeating part of the context with an explicit cache breakpoint and key each episode, so the provider caches it (OpenAI GPT-5.6 and later)")
    ap.add_argument("--episode-serial", action="store_true", help="send each episode's questions in order on one worker so the provider's prompt cache is hit; workers parallelise across episodes")
    ap.add_argument("--align-store", default=None, help="annotated store to score identify answers against when --store holds a reconstruction; predicted objects are mapped back by aligning the camera trajectories")
    ap.add_argument("--query-ids", default=None, help="JSON file with a list of query ids to run (e.g. a stratified subset)")
    ap.add_argument("--only-c4", action="store_true", help="run only the C4 questions (query ids mc_query_c4_*)")
    ap.add_argument("--skip-c4", action="store_true", help="run only the C1-C3 questions")
    a = ap.parse_args()
    if not a.key: ap.error("no API key: pass --key or set OPENROUTER_API_KEY")
    if a.reasoning_effort: EXTRA_BODY["reasoning"] = {"effort": a.reasoning_effort}
    if a.provider: EXTRA_BODY["provider"] = {"only": [a.provider], "allow_fallbacks": False}
    global STORE, REL
    if a.store: STORE = os.path.abspath(a.store)
    from spacmem.context import memory as memory_context; memory_context.STORE = STORE      # the object-centric builder reads scene records too
    if a.release: REL = os.path.abspath(a.release)

    system = open(a.prompt, encoding="utf-8").read(); phash = sha(system)
    if "/" in a.run or os.path.isdir(f"{RUNS}/{a.run}"): run_dir = f"{RUNS}/{a.run}"      # a path under runs/, e.g. reported/glm/per_frame
    else: run_dir = next((d for d in glob.glob(f"{RUNS}/*_{a.run}") if os.path.isdir(d)), None) or f"{RUNS}/{datetime.now():%Y-%m-%d}_{a.run}"
    os.makedirs(run_dir, exist_ok=True)
    eps = a.episodes or (sorted(os.path.relpath(p, REL).replace("\\", "/") for p in glob.glob(f"{REL}/shard_*/*/mcep_*")) if a.all else [])
    config = {"run": a.run, "created": datetime.now().isoformat(), "model": a.model, "endpoint": a.endpoint, "store": STORE, "release": REL, "prompt_file": os.path.relpath(a.prompt, BASE).replace("\\", "/"),
              "prompt_sha": phash, "gt_store_sha": store_hash(), "geometry": not a.no_geometry, "structured_output": not a.unstructured, "workers": a.workers, "max_tokens": a.max_tokens, "reasoning_effort": a.reasoning_effort, "align_store": a.align_store, "provider": a.provider, "prompt_cache": a.prompt_cache, "episode_serial": a.episode_serial, "seed": a.seed, "only_c4": a.only_c4, "skip_c4": a.skip_c4, "episodes": eps, "only_turn": a.only_turn, "limit": a.limit,
              "context_builder": (CONTEXT_BUILDER + (" [no in-view lines]" if a.no_turn_visibility else "") + (" [no inventory]" if a.no_inventory else "")) if a.method == "full" else (f"per_frame: observations at {a.fps} fps, {'with' if a.ids else 'no'} object ids, no aggregation" if a.method == "per_frame" else f"memory_a: object-centred dump, links={a.links or 'data/memory/gold'}, {'all mentions' if a.all_mentions else 'subject links only'}"), "method": a.method, "links": a.links}
    cpath = f"{run_dir}/config.json"
    if os.path.exists(cpath):
        old = json.load(open(cpath))
        for k in ("model", "prompt_sha", "gt_store_sha", "geometry", "structured_output", "context_builder"):
            if old.get(k) != config[k]: sys.exit(f"refusing to resume {run_dir}: {k} changed ({old.get(k)} -> {config[k]}); use a new --run id")
        old["resumed"] = old.get("resumed", []) + [datetime.now().isoformat()]; config = {**config, **{k: old[k] for k in ("created",)}, "resumed": old["resumed"]}
    json.dump(config, open(cpath, "w"), indent=1)
    log = open(f"{run_dir}/run.log", "a", encoding="utf-8")

    resf = f"{run_dir}/results.jsonl"; done = set()
    if os.path.exists(resf):
        for l in open(resf, encoding="utf-8"): done.add(json.loads(l)["query_id"])
    only_ids = set(json.load(open(a.query_ids))) if a.query_ids else None
    jobs = []
    for ep in eps:
        d = f"{REL}/{ep}"
        qs = [json.loads(l) for l in io.open(f"{d}/public/queries.jsonl", encoding="utf-8")]
        ans = {json.loads(l)["query_id"]: json.loads(l) for l in io.open(f"{d}/hidden/answers.jsonl", encoding="utf-8")}
        ev = {json.loads(l)["query_id"]: json.loads(l) for l in io.open(f"{d}/hidden/evidence.jsonl", encoding="utf-8")}
        if os.path.exists(f"{d}/hidden/c4_witnesses.jsonl"):
            for l in io.open(f"{d}/hidden/c4_witnesses.jsonl", encoding="utf-8"):
                w = json.loads(l)
                ev.setdefault(w["query_id"], {"program": {"program_id": "c4_" + w["family"]}, "memory_snapshot": {"facts": []}, "c4": True})
        for q in sorted(qs, key=lambda q: q["turn"]):
            if q["query_id"] in done: continue
            if only_ids is not None and q["query_id"] not in only_ids: continue
            if a.only_turn and q["global_turn_id"] != a.only_turn: continue
            if a.only_c4 and not q["query_id"].startswith("mc_query_c4_"): continue
            if a.skip_c4 and q["query_id"].startswith("mc_query_c4_"): continue
            jobs.append((ep, d, q, ans[q["query_id"]]["answer_interface"], ev[q["query_id"]]))
    if a.limit: jobs = jobs[:a.limit]

    def run_one(job):
        ep, d, q, ai, e = job
        track_map = {f["track_id"]: f"{f['clip_id'].replace('-full','').replace('scene','')}#{f['object_id']}" for f in e["memory_snapshot"]["facts"]}
        if a.method == "memory_a":
            from spacmem.context.memory import build_context_memory
            ctx = build_context_memory(d, q["global_turn_id"], a.links, subject_only=not a.all_mentions)
        elif a.method == "per_frame":
            from spacmem.context.per_frame import build_context_per_frame
            ctx = build_context_per_frame(d, q["global_turn_id"], fps=a.fps, ids=a.ids, store=STORE)
        else:
            ctx = build_context(d, q["global_turn_id"], geometry=not a.no_geometry, turn_visibility=not a.no_turn_visibility, inventory=not a.no_inventory)
        ckey = None if a.no_cache else sha(f"{phash}|{a.model}|{'schema' if not a.unstructured else 'free'}|{sha(ctx)}" + (f"|seed{a.seed}" if a.seed is not None else '') + (f"|effort{a.reasoning_effort}" if a.reasoning_effort else ''))
        cpre, ctail = cache_split(ctx) if a.prompt_cache else (None, ctx)
        t0 = time.time(); text, usage, cached = call_llm(a.endpoint, a.model, a.key, system, ctail, cache_key=ckey, max_tokens=a.max_tokens, structured=not a.unstructured, seed=a.seed, cache_prefix=cpre, cache_tag=ep.split("/")[-1] if cpre else None); dt = time.time() - t0
        pred = parse_answer(text)
        if a.method == "per_frame" and not a.ids and ai["scoreability"]["answer_type"] == "track":
            from spacmem.context.per_frame import resolve_position_answer
            pred = resolve_position_answer(pred, d, store=STORE) or pred      # 'day D | class | [x,y,z]' -> '<scan>#<id>'
        # Alignment comes after that resolution: per_frame answers arrive as a position, so there is no object id to map
        # until the line above has produced one, in the reconstruction's own numbering. resolve_position_answer returns
        # the short '0479_00#3' form while the other builders answer with 'scene0479_00 #3', so accept either.
        if a.align_store and ai["scoreability"]["answer_type"] == "track":
            from spacmem.scene.align import recon_to_gt
            m = re.search(r"(?:scene)?(\d{4}_\d{2})\s*#\s*(\d+)", str(pred or ""))
            if m:
                g = recon_to_gt("scene" + m.group(1), int(m.group(2)), STORE, os.path.abspath(a.align_store))
                pred = f"scene{m.group(1)} #{g}" if g is not None else pred
        ok = score(pred, ai, track_map)
        wt = work_count(text) if ai["scoreability"]["answer_type"] == "count" else None
        ok_work = score(wt, ai, track_map) if wt is not None else ok
        return dict(episode=ep, query_id=q["query_id"], turn=q["global_turn_id"], session=q["session_index"] + 1, family=e["program"]["program_id"],
                    answer_type=ai["scoreability"]["answer_type"], officially_scoreable=ai["officially_scoreable"], key=ai["canonical_answer"],
                    accepted=ai.get("accepted_answers"), key_object=track_map.get(ai["canonical_answer"]), pred=pred, correct=ok,
                    work_true=wt, correct_from_work=ok_work,
                    prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"), cached_prompt_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"), seconds=round(dt, 1),
                    context_chars=len(ctx), cached=cached, raw=text)

    spend_at_start = billed_usage(a.endpoint, a.key)
    n = 0
    write_lock = threading.Lock()
    with open(resf, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        def emit(row):
            with write_lock:
                out.write(json.dumps(row) + "\n"); out.flush()
                hit = row.get("cached_prompt_tokens")
                line = f"{row['episode'].split('/')[-1]} {row['turn']} {fam(row['family']):32s} key={row['key']!s:14s} pred={row['pred']!s:14s} {'OK' if row['correct'] else 'x'} ({row['prompt_tokens']} tok{f', {hit} from provider cache' if hit else ''}, {row['seconds']:.0f}s{', cached' if row['cached'] else ''})"
                print(line, flush=True); log.write(line + "\n"); log.flush()
        if a.episode_serial:
            by_ep = {}
            for j in jobs: by_ep.setdefault(j[0], []).append(j)          # jobs are already in turn order within an episode
            def run_episode(js):
                for j in js: emit(run_one(j))
                return len(js)
            for k in pool.map(run_episode, list(by_ep.values())): n += k
        else:
            for row in pool.map(run_one, jobs): emit(row); n += 1
    rows = [json.loads(l) for l in open(resf, encoding="utf-8")]
    json.dump(summarize(rows), open(f"{run_dir}/summary.json", "w"), indent=1)
    spent = billed_usage(a.endpoint, a.key)
    if spent is not None and spend_at_start is not None:
        line = f"billed by the provider during this invocation: ${spent - spend_at_start:.2f} (account total ${spent:.2f}; includes any other traffic on the same key)"
        print(line, flush=True); log.write(line + "\n"); log.flush()
    print(f"done {n} new, {len(rows)} total -> {run_dir}")


if __name__ == "__main__":
    main()
