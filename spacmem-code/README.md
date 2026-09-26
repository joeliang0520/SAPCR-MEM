# Object-centric spatial memory for conversational egocentric assistants

The answering methods, the turn-to-object linker, the reconstruction adapters and the evaluation harness, and the
object reconstruction that builds the reconstructed scenes from RGB video.

## What is here

```
spacmem/                            the method and the harness
  answer.py                         answers every question of an episode under one configuration, resumable and cached
  scene/                            scene records, one per clip, whatever produced them
    from_scannet.py                 builds them from raw ScanNet
    from_reconstruction.py          builds them from a reconstruction's world_state.json
    align.py                        aligns a reconstruction to the annotated frame and matches objects across the two
  context/                          what the answerer is shown
    memory.py                       object-centric, one record per object with the turns that refer to it
    per_frame.py                    per-frame, the geometric baseline
  links/                            the turn-to-object mapping
    from_release.py                 the annotated links, read from the release's own evidence
    predict.py                      the learned linker, one model call per session, schema-constrained
    replay.py                       moves the annotated links onto a reconstruction's objects
    score.py                        precision and recall of predicted links against annotated ones
  prompts/
    answer/annotated_geometry/      answering prompts for the ScanNet-derived scenes
    answer/estimated_geometry/      the same prompts for reconstructed scenes, noting that the geometry is estimated
      full_context.md               answering from the time-ordered history
      memory.md                     answering from the object-centric memory
      per_frame.md                  answering from per-frame observations
    linker/
      link.md                       the turn-to-object linker
      verify.md                     its second pass, over turns the first pass left unlinked

experiments/                        entry points for the reported experiments
  run_benchmark.py                  the method and its ablations over the whole benchmark, then the accuracy table
  prepare_reconstruction.py         turns a reconstruction into scene records and replays the annotated links onto it
  report.py                         accuracy per configuration, and paired bootstrap intervals between two run folders

reconstruction/                     persistent 3D objects from each room's RGB frames, packaged as one world_state.json
                                    per episode; a separate package with its own README and tests
```

## Running the experiments

Every command that calls a model reads the endpoint from `SPACMEM_ENDPOINT` (default `https://openrouter.ai/api/v1`,
any OpenAI-compatible endpoint works) and the key from `OPENROUTER_API_KEY`. `--endpoint` and `--key` override both.

On the annotated scenes:

```bash
python experiments/run_benchmark.py --scenes annotated --model gpt-5.6-luna
```

On a reconstruction, first convert it (one `world_state.json` per episode), then run on it by name:

```bash
python experiments/prepare_reconstruction.py path/to/reconstruction --name recon
python experiments/run_benchmark.py --scenes recon --model gpt-5.6-luna
```

Both answer under four configurations and print the table. `--configs` runs a subset and `--episodes` a subset of
episodes. Every step resumes, so a stopped run continues where it stopped. Two run folders are compared with

```bash
python experiments/report.py runs/recon_gpt-5.6-luna --against runs/annotated_gpt-5.6-luna
```

| Configuration | Context the answerer sees |
| --- | --- |
| `memory` | one record per object with the turns that refer to it, mapped by the learned linker (the method) |
| `memory_annotated_links` | the same memory with the annotated turn-to-object mapping (privileged) |
| `full_context` | objects as they first appear, conversation in time order |
| `per_frame` | every detected object listed again at each observation, with no object identity |

The prompts follow the scenes. Reconstructed scenes use `prompts/answer/estimated_geometry/`, which adds one paragraph
telling the answerer that the geometry was estimated from video and how to treat decisions close to a threshold.

## Running one configuration directly

```bash
python -m spacmem.answer --run my_run --method full \
    --prompt spacmem/prompts/answer/annotated_geometry/full_context.md \
    --episodes shard_03/lab_or_office_setup/mcep_52939f339e62302e --model gpt-5.6-luna
```

On a reconstruction add `--store <its scene records> --align-store data/gt_store`. The second flag maps a reconstructed
object back to the annotated one it stands for, so identify answers can be scored. Counting questions do not use it.

Each run writes `runs/<run>/config.json` (model, prompt hash, scene-record hash, episode list) and
`runs/<run>/results.jsonl` (one row per question with the key, the prediction, correctness and token counts). Replies
are cached by (prompt hash, model, context hash), so re-running a finished configuration costs nothing and a crashed
run resumes where it stopped.

## Data this expects, none of which is in the repository

| Path | What it is | Where it comes from |
| --- | --- | --- |
| `data/final_release/` | the benchmark, with transcripts, queries and the hidden answer keys | the benchmark release |
| `data/gt_store/` | one scene record per ScanNet scan, built by `spacmem/scene/from_scannet.py` | needs raw ScanNet scans (`SCANNET_SCANS`) |
| `data/memory/gold/` | the annotated turn-to-object links, built by `spacmem/links/from_release.py` | derived from the release |
| `data/store_<name>/` | a reconstruction in scene-record format, one folder per episode | `experiments/prepare_reconstruction.py` |

A scene record is one JSON per clip.

```
objects    {id: {label, label_alternatives?, centroid, aabb_min, aabb_max}}
sightings  {id: [frames the object was visible in]}
camera     {frame: [x, y, z]}
```

Anything that can produce that shape can be evaluated, which is how the reconstruction and its ablations are run.

## Conventions

Distances are centroid-to-centroid in metres and only ever within one room. "Near" means at most 1.5 m. Box fit
compares sorted extents and allows axis permutation. These follow the benchmark's own definitions, and changing them
changes the task rather than the method.
