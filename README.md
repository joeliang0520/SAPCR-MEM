# SpaC-MEM

Code for **Have I Scene This Before? Spatially Grounded Conversational Memory for Complex Queries in Egocentric Assistants**.

SpaC-MEM builds persistent 3D objects from egocentric RGB video, associates dialogue with those objects, and answers questions from the resulting spatial and conversational memory. This release includes reconstruction, dialogue association, memory construction, and an evaluation harness with four context configurations.

The accompanying **Ego-SpaCR v1.0 dataset ZIP** contains 95 conversations, 620 sessions, and 3,218 questions. The official evaluation uses 3,100 questions across C1–C4. ScanNet recordings, model weights, and generated reconstructions are obtained separately.

## Contents

```text
spcr-mem/
├── README.md
├── Ego-SpaCR_v1.0.zip        # Accompanying benchmark dataset
└── spacmem-code/
    ├── spacmem/
    │   ├── answer.py          # Answering, caching, and local scoring
    │   ├── scene/            # Annotated and reconstructed scene records
    │   ├── links/            # Predicted and annotated dialogue–object links
    │   ├── context/          # Object memory and per-frame context builders
    │   └── prompts/          # Answering and association prompts
    ├── experiments/          # Preparation, benchmark runs, and reports
    └── reconstruction/       # RGB reconstruction, packaging, and tests
```

Further details: [memory and evaluation](spacmem-code/README.md) · [reconstruction](spacmem-code/reconstruction/README.md).

## 1. Install

Use Python 3.10 or later. Run the following from the extracted code directory:

```bash
cd spcr-mem/spacmem-code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install requests -e ./reconstruction
```

**All remaining commands run from `spacmem-code/`.** The setup above supports data preparation and API-based answering from existing scene records. RGB reconstruction also needs the model environments described in Section 5.

## 2. Prepare the dataset ZIP

Extract the accompanying archive, then set `SPACMEM_DATA` to the directory containing its `metadata/` and `tasks/` folders. For an archive named `Ego-SpaCR_v1.0.zip`:

```bash
mkdir -p data
unzip ../Ego-SpaCR_v1.0.zip -d data
export SPACMEM_DATA="$PWD/data/Ego-SpaCR_v1.0"
```

The released dataset has this layout:

```text
Ego-SpaCR_v1.0/
├── README.md
├── metadata/
│   ├── episodes.jsonl
│   ├── queries.jsonl
│   └── scannet_scenes.txt
└── tasks/<task>/<episode_id>/
    ├── inputs/
    └── ground_truth/
```

The current scripts read `data/final_release/shard_*/<task>/<episode_id>/{public,hidden}`. Run this one-time preparation step to create a compatible view. It links to the extracted files and writes a filtered question list for each episode. Transcripts retain every turn, including the 118 questions excluded from official evaluation.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["SPACMEM_DATA"]).resolve()
out = Path("data/final_release")

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

episodes = read_jsonl(root / "metadata/episodes.jsonl")
official = {q["query_id"] for q in read_jsonl(root / "metadata/queries.jsonl")
            if q["official_eval"]}
if out.exists() or out.is_symlink():
    raise SystemExit(f"{out} already exists; use a fresh code directory to prepare another dataset.")
prepared = set()
for episode in episodes:
    src = root / episode["path"]
    dst = out / "shard_release" / episode["task_id"] / episode["episode_id"]
    public = dst / "public"
    public.mkdir(parents=True)
    for path in (src / "inputs").iterdir():
        if path.name != "queries.jsonl":
            (public / path.name).symlink_to(path, target_is_directory=path.is_dir())
    queries = [q for q in read_jsonl(src / "inputs/queries.jsonl") if q["query_id"] in official]
    (public / "queries.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries), encoding="utf-8")
    (dst / "hidden").symlink_to(src / "ground_truth", target_is_directory=True)
    prepared.update(q["query_id"] for q in queries)
assert prepared == official, "Some official questions were not found in the episode files."
print(f"Prepared {len(episodes)} episodes and {len(prepared)} official questions in {out}")
PY
```

For v1.0, this prints **95 episodes and 3,100 official questions**. `shard_release` is a compatibility directory name; the episode and question IDs are unchanged. Keep the extracted dataset at the same location while using these symbolic links. The commands assume Linux or macOS.

## 3. Prepare ScanNet scene records

Obtain the 39 scenes listed in `$SPACMEM_DATA/metadata/scannet_scenes.txt` through the [ScanNet data access instructions](https://github.com/ScanNet/ScanNet#scannet-data). For each `<scene_id>`, the annotated scene builder requires:

```text
scans/<scene_id>/
├── <scene_id>.sens
├── <scene_id>.txt
├── <scene_id>_vh_clean_2.ply
├── <scene_id>_vh_clean_2.0.010000.segs.json
├── <scene_id>.aggregation.json
└── <scene_id>_2d-instance-filt.zip
```

Set the scans directory and build only the scenes used by the benchmark:

```bash
export SCANNET_SCANS="/path/to/scannet/scans"
while IFS= read -r scene; do
    [ -z "$scene" ] && continue
    python -m spacmem.scene.from_scannet "$scene" || break
done < "$SPACMEM_DATA/metadata/scannet_scenes.txt"

python -m spacmem.links.from_release
```

This creates `data/gt_store/<scene_id>.json` and `data/memory/gold/`. Annotated links are derived from the release evidence and are used by the annotated-link ablation. The annotated scene records also provide the reference identities needed to score object identification on reconstructed scenes.

## 4. Run answering and memory experiments

Set an API key and a model ID accepted by your endpoint. The client uses a Chat Completions endpoint with JSON-schema structured output. `OPENROUTER_API_KEY` is the key variable even when using another compatible provider.

```bash
export SPACMEM_ENDPOINT="https://openrouter.ai/api/v1"
export OPENROUTER_API_KEY="your-api-key"
export SPACMEM_MODEL="your-provider-model-id"
```

Start with one episode on annotated geometry:

```bash
python experiments/run_benchmark.py \
    --scenes annotated --model "$SPACMEM_MODEL" \
    --configs memory \
    --episodes shard_release/inventory_walkthrough/mcep_fc3e1cd13fff6f29 \
    --workers 1
```

Run all four configurations over the prepared evaluation set:

```bash
python experiments/run_benchmark.py \
    --scenes annotated --model "$SPACMEM_MODEL" --workers 4
```

| Configuration | Answer context |
| --- | --- |
| `memory` | Objects with dialogue attached by the predicted association model |
| `memory_annotated_links` | The same memory with annotated dialogue–object links |
| `full_context` | Object observations and dialogue in chronological order |
| `per_frame` | Repeated per-frame object observations without persistent IDs |

The association model is the same model passed to `--model`. Use `--configs` and `--episodes` for subsets. `--reasoning-effort` and `--max-tokens` configure answering for endpoints that support those settings. Defaults are four workers and an answer budget of 4,000 tokens; the benchmark runner gives association calls 8,000 tokens.

These commands use ScanNet annotations for geometry. To run SpaC-MEM with geometry and objects estimated from RGB, follow the next section.

## 5. Reconstruct objects from RGB

Reconstruction uses a CUDA GPU and three external model installations:

| Component | Model / checkpoint | Role |
| --- | --- | --- |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | DA3-Nested-Giant-Large through DA3-Streaming, plus SALAD weights | Approximately metric depth, camera intrinsics, and poses |
| [SegVGGT](https://github.com/IDEA-Research/SegVGGT) | `segvggt_scannet200.pt` | Object masks and labels |
| [DINOv2](https://huggingface.co/facebook/dinov2-large) | `facebook/dinov2-large` | Appearance features for object association |

Follow the upstream installation and weight-download instructions. The bundled [reconstruction README](spacmem-code/reconstruction/README.md#requirements) records the model revisions and environments used. DA3 weight paths are set in `reconstruction/src/video_world_state/adapters/da3_config.yaml`.

The interpreter running reconstruction must also have `torch`, `depth_anything_3`, and Open3D available: lifting depth to 3D calls DA3 utilities in that process. One way to satisfy this is to install the reconstruction package into the DA3 environment:

```bash
export DA3_PYTHON="/path/to/da3-env/bin/python"
export DA3_REPO="/path/to/Depth-Anything-3"
export SEGVGGT_PYTHON="/path/to/segvggt-env/bin/python"
export SEGVGGT_REPO="/path/to/SegVGGT"
export APPEARANCE_PYTHON="/path/to/dinov2-env/bin/python"
"$DA3_PYTHON" -m pip install -e './reconstruction[alignment]'
```

For each scene in `metadata/scannet_scenes.txt`, extract its RGB frames and run reconstruction. For example:

```bash
scene=scene0144_00
python reconstruction/tools/extract_frames.py \
    --sens "$SCANNET_SCANS/$scene/$scene.sens" --output "data/frames/$scene"

"$DA3_PYTHON" reconstruction/tools/reconstruct_scene.py \
    --scene "$scene" --frames "data/frames/$scene" --output "data/scenes/$scene" \
    --da3-python "$DA3_PYTHON" \
    --da3-script "$DA3_REPO/da3_streaming/da3_streaming.py" \
    segvggt --python "$SEGVGGT_PYTHON" --repository "$SEGVGGT_REPO" \
    --checkpoint "$SEGVGGT_REPO/checkpoint/segvggt_scannet200.pt" \
    --appearance-python "$APPEARANCE_PYTHON"
```

Frame extraction copies RGB from the `.sens` file; the reconstruction estimates geometry from those images. The pipeline samples at 5 fps. `reconstruct_scene.py` requires a new output directory for each run.

After reconstructing all required scenes, package them by episode, prepare memory inputs, and run:

```bash
"$DA3_PYTHON" reconstruction/tools/build_episode_package.py \
    --benchmark data/final_release --scenes data/scenes \
    --output data/packages --line-of-sight

python experiments/prepare_reconstruction.py data/packages --name recon
python experiments/run_benchmark.py \
    --scenes recon --model "$SPACMEM_MODEL" --workers 4
```

The intermediate packages contain one `world_state.json` per episode. Conversion creates `data/store_recon/<episode_id>/` and replays annotated links into `data/memory/annotated_recon/` for the privileged ablation.

## 6. Results and resuming

Results are written under `runs/<scenes>_<model-name>/<configuration>/`. The model name is the final component of the API model ID, after any `/`. Reconstructed runs have an additional episode subdirectory.

| File | Contents |
| --- | --- |
| `config.json` | Model, prompt, scene-store information, episode list, and run settings |
| `results.jsonl` | Predictions, local correctness, token usage, and timing per question |
| `summary.json` | Aggregate scores and diagnostic counts |
| `run.log` | Run progress |

Re-running the same command skips saved answers and existing predicted links. Model replies are cached in `runs/_cache/`. Use a separate run/output location when changing settings or comparing models with the same final name.

```bash
python experiments/report.py runs/recon_MODEL_NAME
python experiments/report.py runs/recon_MODEL_NAME \
    --against runs/annotated_MODEL_NAME
```

Replace `MODEL_NAME` with the model name used in the output directory. The second command reports paired accuracy differences and bootstrap intervals over shared question IDs.

**Scoring scope.** The bundled scorer checks accepted count intervals, Boolean answers, and object identity. Reconstructed object IDs are mapped to annotated identities for scoring. The dataset's free-text object-description judge and the other paper baselines are outside this code package. For official question selection, use `metadata/queries.jsonl` and its `official_eval` flag; Section 2 applies that selection before running the harness.

## Tests

The reconstruction tests use synthetic inputs and mocked model runners:

```bash
python -m pip install -e './reconstruction[dev]'
python -m pytest reconstruction/tests
```

For dataset fields, input visibility, C1–C4 definitions, and benchmark scoring, see the README inside the accompanying **Ego-SpaCR v1.0 dataset ZIP**.
