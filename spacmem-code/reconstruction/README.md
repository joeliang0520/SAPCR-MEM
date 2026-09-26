# Object reconstruction

Builds persistent 3D objects from the RGB frames of a room recording, then
packages them per benchmark episode as the `world_state.json` files the memory
method reads.

Each scene goes through three passes over the same 5 FPS selection of its
frames:

1. **Geometry.** Depth Anything 3 (DA3-Streaming) estimates depth, confidence,
   intrinsics and camera poses. The scene is then levelled from its own
   geometry: up from the floor (or walls, or a horizontal surface), yaw from
   the walls, and the floor placed near height zero.
2. **Segmentation.** SegVGGT returns instance masks, class labels and scores.
   Every backend writes the same cache format.
3. **Objects.** Masks are lifted to 3D with the geometry, associated frame by
   frame into persistent objects, and consolidated: fragments are merged,
   thinly seen objects dropped, and boxes refined.

The SegVGGT path reads RGB frames only. ScanNet's annotations are read only by
the `scannet` backend, which is the ground-truth ablation: ScanNet's 2D
instance masks, labels and instance identities in place of SegVGGT and
association, on the same geometry.

## Layout

```
src/video_world_state/
  frames.py          reading a frame folder; the shared 5 FPS selection
  geometry.py        geometry access; runs DA3 and levelling for a new scene
  alignment.py       levelling from floor, walls and horizontal surfaces
  segmentation.py    the shared segmentation cache, its writer and reader
  objects.py         lifting, association, merging, carving, box refinement
  pipeline.py        the passes in order, and the pass-3 settings per backend
  labels.py          label ranking for packages
  adapters/
    da3.py, da3_config.yaml           DA3-Streaming runner and its settings
    segvggt.py, segvggt_worker.py     SegVGGT runner and decoder
    scannet.py                        ScanNet instance masks (ground-truth ablation)
    appearance.py, appearance_worker.py   DINOv2 descriptors for merging
tools/
  extract_frames.py         colour frames out of a ScanNet .sens file
  reconstruct_scene.py      all passes for one scene
  build_episode_package.py  one world_state.json per benchmark episode
tests/                      synthetic data only; no model runs
```

All settings are fixed in code and the same for every scene: pass 3 in
`pipeline.py`, SegVGGT decoding in `adapters/segvggt.py`, and DA3-Streaming in
`adapters/da3_config.yaml`.

## Requirements

This package needs Python 3.10 or later:

```bash
pip install -e ".[alignment,dev]"
pytest
```

The models run in their own environments, each passed to the tools as an
interpreter path:

| Model | Source | Used |
| --- | --- | --- |
| Depth Anything 3 | github.com/ByteDance-Seed/Depth-Anything-3, `da3_streaming/` | commit `3fe327a`; `da3nested-giant-large` and the SALAD loop-closure checkpoint in `da3_streaming/weights/` |
| SegVGGT | github.com/IDEA-Research/SegVGGT | commit `cd5f156`; config `segvggt_scannet200`, checkpoint `segvggt_scannet200.pt` |
| DINOv2 | `facebook/dinov2-large` through `transformers` | any environment with `torch` and `transformers` |

Environments used for the reported runs: DA3 with Python 3.10 and torch 2.8.0;
SegVGGT with Python 3.10 and torch 2.3.1; DINOv2 with Python 3.12, torch
2.10.0 and transformers 5.17.0. A CUDA GPU is needed for all three.

## Data

- ScanNet scans, obtained under ScanNet's terms of use: the `.sens` file of
  each scan and, for the ground-truth ablation, `<scan>.aggregation.json` and
  the `instance-filt` folder from `<scan>_2d-instance-filt.zip`.
- The benchmark release, for each episode's `public/episode.json` and
  `public/label_mapping.json`.

## Running

For each scene of the benchmark:

```bash
python tools/extract_frames.py --sens <scan>/scene0000_00.sens --output frames/scene0000_00

python tools/reconstruct_scene.py --scene scene0000_00 --frames frames/scene0000_00 \
    --output scenes/scene0000_00 \
    --da3-python <da3 env>/bin/python \
    --da3-script <Depth-Anything-3>/da3_streaming/da3_streaming.py \
    segvggt --python <segvggt env>/bin/python --repository <SegVGGT> \
    --checkpoint <SegVGGT>/checkpoint/segvggt_scannet200.pt \
    --appearance-python <env with torch and transformers>/bin/python
```

Then, once every scene is built:

```bash
python tools/build_episode_package.py --benchmark <benchmark release> \
    --scenes scenes --output packages --line-of-sight
```

`packages/episode_<id>_segvggt/world_state.json` is the input to the memory
method's `experiments/prepare_reconstruction.py`.

For the ground-truth ablation, build each scene again on the geometry of its
first run, then package those scenes the same way:

```bash
python tools/reconstruct_scene.py --scene scene0000_00 --frames frames/scene0000_00 \
    --output scannet-scenes/scene0000_00 --geometry-from scenes/scene0000_00 \
    scannet --instances <scan>/instance-filt \
    --aggregation <scan>/scene0000_00.aggregation.json \
    --label-mapping <episode>/public/label_mapping.json
```

## Output

Each scene directory holds:

| Path | Contents |
| --- | --- |
| `geometry/`, `world_alignment.npy` | DA3 output and the levelling transform |
| `segmentation/cleaned/` | the masks, labels and scores in the shared cache format |
| `segmentation/raw/` | SegVGGT's own output, before decoding (SegVGGT only) |
| `appearance/` | DINOv2 descriptors (SegVGGT only) |
| `objects/` | `objects.json` (objects, observations, settings) and `object_points.npz` |
| `provenance.json` | inputs, backend, pass-3 settings and a hash of this code |

A package lists each object once, with an episode-wide ID, its label and
runner-up labels, centroid and axis-aligned box in its room's levelled
coordinates, and, for every processed frame, the camera pose and the objects
visible in it. `conventions` inside each package describes its frame
numbering, coordinates and label fields.
