import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from video_world_state.adapters import segvggt, segvggt_worker
from video_world_state.adapters.segvggt import (
    SegVggtBackend,
    SegVggtDecoder,
    SegVggtOutputLoader,
    SegVggtRunner,
)
from video_world_state.contracts import FrameInput, FrameSequence
from video_world_state.segmentation import SegmentationAdapter


# A real lightweight subprocess speaking the worker's CLI; no torch or GPU.
# Query 7 is a chair on the left column, query 5 a table on the right (its
# no-object probability is highest and must be ignored), query 3 scores
# below the decoder threshold. Staged position 4 has no masks at all, and
# query 7 is absent on the second frame of every chunk.
FAKE_SEGVGGT = """
import argparse, json
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path)
p.add_argument("--chunk", action="append")
args, _ = p.parse_known_args()
classes = ["chair", "table", "desk", "lamp", "box", "bag"]
chunks = []
for index, item in enumerate(args.chunk):
    start, end = (int(v) for v in item.split(":"))
    frames = end - start
    probabilities = np.array([
        [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01],
        [0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.02, 0.06, 0.01, 0.005, 0.003, 0.002, 0.9],
    ], dtype=np.float16)
    maps = np.zeros((3, frames, 2, 3), np.uint8)
    maps[0, :, :, 0], maps[0, :, :, 1] = 255, 60
    if frames > 1:
        maps[0, 1] = 0
    maps[1] = 255
    maps[2, :, :, 2] = 255
    for offset, position in enumerate(range(start, end)):
        if position == 4:
            maps[:, offset] = 0
    np.savez_compressed(args.output / f"chunk_{index:03d}.npz",
        positions=np.arange(start, end, dtype=np.int64),
        queries=np.array([7, 3, 5], dtype=np.int64),
        probabilities=probabilities,
        score=np.array([0.9, 0.02, 0.5], dtype=np.float32), maps=maps)
    chunks.append({"index": index, "start": start, "end": end})
(args.output / "results.json").write_text(json.dumps({
    "completed": True, "classes": classes, "query_count": 10,
    "output_hw": [2, 3], "chunks": chunks}))
print("synthetic SegVGGT CLI")
"""


@pytest.fixture
def selected(tmp_path):
    frames = []
    for i in range(7):
        path = tmp_path / f"source {100 - i}.png"
        Image.new("RGB", (6, 4), (i, i, i)).save(path)
        frames.append(FrameInput(f"room:{6*i+2}", 6 * i + 2, i / 5, path))
    return FrameSequence("room", frames)


@pytest.fixture
def tools(tmp_path):
    script = tmp_path / "fake_segvggt.py"
    script.write_text(FAKE_SEGVGGT)
    repository = tmp_path / "SegVGGT"
    repository.mkdir()
    checkpoint = tmp_path / "segvggt.pt"
    checkpoint.write_bytes(b"weights")
    return script, repository, checkpoint


@pytest.fixture
def runner(tools):
    script, repository, checkpoint = tools
    return SegVggtRunner(
        Path(sys.executable), repository, checkpoint, worker_script=script, chunk_frames=3
    )


@pytest.fixture
def raw(tmp_path, selected, runner):
    path = tmp_path / "raw"
    runner.run(selected, path)
    return path


def snapshot(directory):
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }


def test_runner_stages_given_frames_and_chunks_without_filename_arithmetic(
    raw, selected, runner
):
    manifest = json.loads((raw / "frame_manifest.json").read_text())
    assert [entry["frame_id"] for entry in manifest["frames"]] == [
        f.frame_id for f in selected.frames
    ]
    assert manifest["frames"][3]["rgb_path"] == str(selected.frames[3].rgb_path.resolve())
    assert manifest["chunk_ranges"] == [[0, 3], [3, 6], [6, 7]]
    assert manifest["overlap_frames"] == 0
    assert manifest["keep_rule"] == segvggt.KEEP_RULE
    assert (raw / "input_frames/00003.jpg").resolve() == selected.frames[3].rgb_path.resolve()
    command = manifest["command"]
    assert command[command.index("--chunk") + 1] == "0:3"
    assert "synthetic SegVGGT CLI" in (raw / "worker.log").read_text()
    assert json.loads((raw / "run_complete.json").read_text())["chunk_count"] == 3


def test_worker_reads_the_flags_the_runner_sends(selected, tools, tmp_path):
    _, repository, checkpoint = tools
    runner = SegVggtRunner(Path(sys.executable), repository, checkpoint, chunk_frames=3)
    command = runner._command(tmp_path / "in", tmp_path / "out", [(0, 3), (3, 5)])
    assert command[1:3] == ["-m", segvggt.WORKER_MODULE]
    arguments = segvggt_worker.parse_arguments(command[3:])
    assert arguments.chunks == [(0, 3), (3, 5)]
    assert arguments.repo == repository.resolve()
    assert arguments.min_class_probability == 0.001
    assert arguments.min_chunk_pixels == 200
    assert arguments.pixel_sigmoid == 0.3


def test_loader_round_trip_and_final_short_chunk(raw, selected):
    loader = SegVggtOutputLoader(raw, selected)
    assert loader.ranges == [(0, 3), (3, 6), (6, 7)]
    chunk = loader.load_chunk(2)
    assert chunk.frame_ids == ("room:38",)
    assert chunk.maps.shape == (3, 1, 2, 3)
    assert chunk.queries.tolist() == [7, 3, 5]
    assert loader.classes[0] == "chair"


@pytest.mark.parametrize(
    "failure",
    ["missing_marker", "incomplete", "overlap", "gap", "dtype", "shape", "repeat", "missing_chunk"],
)
def test_loader_rejects_incomplete_or_malformed_runs(raw, selected, failure):
    manifest_path = raw / "frame_manifest.json"
    chunk_path = raw / "chunk_001.npz"
    if failure == "missing_marker":
        (raw / "run_complete.json").unlink()
    elif failure == "incomplete":
        result = json.loads((raw / "results.json").read_text())
        result["completed"] = False
        (raw / "results.json").write_text(json.dumps(result))
    elif failure in ("overlap", "gap"):
        manifest = json.loads(manifest_path.read_text())
        manifest["chunk_ranges"][1][0] = 2 if failure == "overlap" else 4
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "missing_chunk":
        chunk_path.unlink()
    else:
        with np.load(chunk_path) as saved:
            arrays = dict(saved)
        if failure == "dtype":
            arrays["probabilities"] = arrays["probabilities"].astype(np.float32)
        elif failure == "shape":
            arrays["maps"] = arrays["maps"][:, :2]
        else:
            arrays["queries"] = np.array([7, 7, 5], dtype=np.int64)
        np.savez_compressed(chunk_path, **arrays)
    with pytest.raises((ValueError, FileNotFoundError)):
        loader = SegVggtOutputLoader(raw, selected)
        loader.load_chunk(1)


def test_loader_rejects_reordered_frames(raw, selected):
    frames = list(selected.frames)
    frames[1], frames[2] = frames[2], frames[1]
    for i, frame in enumerate(frames):
        frame.timestamp_seconds = i / 5
    with pytest.raises(ValueError, match="does not match selected frames"):
        SegVggtOutputLoader(raw, FrameSequence("room", frames))


def test_decoder_filters_scores_labels_candidates_and_empty_frames(raw, selected):
    loader = SegVggtOutputLoader(raw, selected)
    frames = SegVggtDecoder().decode(
        loader.load_chunk(0), loader.classes, [(4, 6)] * 3, "room"
    )
    table, chair = frames[0]  # ordered by query ID within a frame
    assert [o.track_hint for o in frames[0]] == [
        "room:segvggt:chunk:0:query:5",
        "room:segvggt:chunk:0:query:7",
    ]
    assert chair.confidence == pytest.approx(0.9)
    assert [c.label for c in chair.label_candidates] == ["table", "desk", "lamp", "box"]
    assert [c.score for c in chair.label_candidates] == pytest.approx(
        [0.6, 0.2, 0.1, 0.06], rel=2e-3
    )
    assert table.label == "table"  # no-object is never the label
    assert [c.label for c in table.label_candidates] == ["chair", "desk", "lamp", "box"]
    assert table.label_candidates[0].score == pytest.approx(0.02 / 0.06, rel=2e-3)
    # Query 7 is missing on the chunk's second frame; query 3 never passes 0.05.
    assert [o.label for o in frames[1]] == ["table"]
    assert all("query:3" not in o.track_hint for frame in frames for o in frame)


def test_decoder_resizes_bilinearly_before_thresholding(raw, selected):
    loader = SegVggtOutputLoader(raw, selected)
    frames = SegVggtDecoder().decode(
        loader.load_chunk(0), loader.classes, [(4, 6)] * 3, "room"
    )
    chair = next(o for o in frames[0] if o.label == "chair")
    # Columns 255, 60, 0 upsampled x2 give 255, 206, 109, 45, 15, 0 across a
    # row; > 102 keeps three columns. Nearest-neighbour would keep only two.
    assert chair.mask.shape == (4, 6)
    assert chair.mask.all(axis=0).tolist() == [True, True, True, False, False, False]
    stricter = SegVggtDecoder(mask_threshold=0.5).decode(
        loader.load_chunk(0), loader.classes, [(4, 6)] * 3, "room"
    )
    chair = next(o for o in stricter[0] if o.label == "chair")
    assert chair.mask.all(axis=0).tolist() == [True, True, False, False, False, False]


def test_prepare_raw_writes_unique_ids_chunk_hints_and_reopens(
    raw, selected, runner, tmp_path
):
    before = snapshot(raw)
    output = tmp_path / "cleaned"
    adapter = SegVggtBackend(runner).prepare_raw(selected, raw, output)
    assert adapter.segmentation_frame_ids == tuple(f.frame_id for f in selected.frames)
    first = adapter.load_frame("room:2")
    assert [o.observation_id for o in first] == ["room:obs:000000:000", "room:obs:000000:001"]
    assert adapter.load_frame("room:26") == []  # empty frame is written
    last = adapter.load_frame("room:38")  # final one-frame chunk
    assert {o.track_hint for o in last} == {
        "room:segvggt:chunk:2:query:5",
        "room:segvggt:chunk:2:query:7",
    }
    assert {o.track_hint for o in first}.isdisjoint(o.track_hint for o in last)
    assert snapshot(raw) == before
    metadata = json.loads((output / "manifest.json").read_text())
    assert metadata["backend"] == "segvggt"
    assert metadata["decoder_settings"] == {
        "score_threshold": 0.05,
        "mask_threshold": 0.4,
        "runner_ups": 4,
    }
    reopened = SegmentationAdapter.from_saved_run(selected, output)
    again = reopened.load_frame("room:2")
    assert [o.observation_id for o in again] == [o.observation_id for o in first]
    np.testing.assert_array_equal(again[0].mask, first[0].mask)
    assert again[1].label_candidates == first[1].label_candidates


def test_prepare_creates_raw_and_cleaned_siblings(selected, runner, tmp_path):
    root = tmp_path / "run"
    backend = SegVggtBackend(runner)
    adapter = backend.prepare(selected, root)
    assert (root / "raw/run_complete.json").is_file()
    assert (root / "cleaned/run_complete.json").is_file()
    assert len(adapter.load_frame("room:2")) == 2
    with pytest.raises(FileExistsError):
        backend.prepare(selected, root)


def test_runner_failure_retains_log_without_marker(selected, runner, tools, tmp_path):
    tools[0].write_text("raise SystemExit(1)")
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="worker failed"):
        runner.run(selected, output)
    assert (output / "worker.log").is_file()
    assert (output / "frame_manifest.json").is_file()
    assert not (output / "run_complete.json").exists()


def test_runner_rejects_mixed_rgb_sizes_before_writing(selected, runner, tmp_path):
    Image.new("RGB", (8, 4)).save(selected.frames[2].rgb_path)
    output = tmp_path / "mixed"
    with pytest.raises(ValueError, match="consistent RGB"):
        runner.run(selected, output)
    assert not output.exists()


def test_segvggt_reads_no_ground_truth():
    sources = [
        Path(segvggt.__file__).read_text(),
        Path(segvggt_worker.__file__).read_text(),
    ]
    forbidden = [
        "map_pred_inst_to_gt",
        "load_camera_pose",
        "load_camera_intrinsic",
        "superpoint",
        "instance_seg_eval",
        "scannet_utils",
        "aggregation.json",
        "predict_by_feat",
    ]
    for source in sources:
        assert not [word for word in forbidden if word in source]
