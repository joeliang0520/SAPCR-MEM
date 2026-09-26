import json
import os
from pathlib import Path
import sys

import pytest
from PIL import Image

from video_world_state.adapters.da3 import Da3Runner, Da3OutputLoader
from video_world_state.contracts import FrameInput, FrameSequence
from video_world_state.geometry import GeometryAdapter


@pytest.fixture
def runnable_sequence(tmp_path):
    frames = []
    # Source names deliberately sort differently from sequence order.
    for index, name in enumerate(("z source.PNG", "a source.JPEG", "middle.jpg")):
        source = tmp_path / name
        Image.new("RGB", (6, 4), color=(index + 1,) * 3).save(source)
        frames.append(FrameInput(f"scene:{index * 6}", index * 6, index / 5, source))
    return FrameSequence("scene", frames)


@pytest.fixture
def fake_runner(tmp_path):
    """A real subprocess speaking DA3's CLI, with tiny synthetic outputs."""
    tool_directory = tmp_path / "tool with spaces"
    tool_directory.mkdir()
    (tool_directory / "relative_weight.txt").write_text("available")
    script = tool_directory / "fake_da3.py"
    script.write_text(
        """
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
p = argparse.ArgumentParser()
p.add_argument('--image_dir')
p.add_argument('--config')
p.add_argument('--output_dir')
a = p.parse_args()
assert Path('relative_weight.txt').is_file(), 'wrong working directory'
assert sys.stdin.read() == '', 'stdin should be noninteractive'
mode = Path(a.config).read_text().strip()
print('synthetic DA3 log', flush=True)
if mode == 'fail':
    sys.exit(7)
images = sorted(list(Path(a.image_dir).glob('*.jpg')) + list(Path(a.image_dir).glob('*.png')))
out = Path(a.output_dir)
results = out / 'results_output'
results.mkdir()
poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(images), axis=0)
poses[:, 0, 3] = np.arange(len(images))
if mode != 'missing_pose':
    np.savetxt(out / 'camera_poses.txt', poses.reshape(-1, 16))
for i, image in enumerate(images):
    if mode == 'missing_depth' and i == 1:
        continue
    with Image.open(image) as rgb:
        value = rgb.getpixel((0, 0))[0]
    depth = np.zeros(3) if mode == 'bad_array' and i == 1 else np.full((2, 3), value, dtype=np.float32)
    np.savez(results / f'frame_{i}.npz', depth=depth,
             conf=np.ones((2, 3)), intrinsics=np.eye(3))
"""
    )
    config = tmp_path / "config.yaml"
    config.write_text("ok")
    return Da3Runner(Path(sys.executable), script, config), config


def test_runs_cli_in_sequence_order_and_saves_readable_results(
    tmp_path, runnable_sequence, fake_runner
):
    runner, config = fake_runner
    output = tmp_path / "run with spaces"
    runner.run(runnable_sequence, output)
    manifest = json.loads((output / "frame_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["sequence_id"] == "scene"
    assert manifest["completion_required"] is True
    assert [entry["frame_id"] for entry in manifest["frames"]] == [
        "scene:0",
        "scene:6",
        "scene:12",
    ]
    assert [entry["source_frame_index"] for entry in manifest["frames"]] == [0, 6, 12]
    assert [entry["timestamp_seconds"] for entry in manifest["frames"]] == [0, 0.2, 0.4]
    assert [entry["rgb_size_hw"] for entry in manifest["frames"]] == [[4, 6]] * 3
    assert manifest["image_preprocessing"] == {
        "method": "upper_bound_resize",
        "pixel_convention": "integer",
    }
    assert (output / "da3_config.yaml").read_text() == config.read_text()
    assert "synthetic DA3 log" in (output / "da3.log").read_text()
    assert json.loads((output / "run_complete.json").read_text())["anchor_count"] == 3
    loader = Da3OutputLoader(output, runnable_sequence)
    for i, frame in enumerate(runnable_sequence.frames):
        assert loader.load_frame(frame.frame_id).depth[0, 0] == i + 1
        assert loader.load_frame(frame.frame_id).rgb_to_geometry[0, 0] == 0.5
    assert [path.name for path in sorted((output / "input_frames").iterdir())] == [
        "000000000.png",
        "000000001.jpg",
        "000000002.jpg",
    ]


def test_refuses_existing_directory(tmp_path, runnable_sequence, fake_runner):
    runner, _ = fake_runner
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "user_data"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run(runnable_sequence, output)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize(
    "case", ["missing_rgb", "unsupported_rgb", "empty", "duplicate", "reversed", "nan"]
)
def test_invalid_inputs_create_no_output(
    tmp_path, runnable_sequence, fake_runner, case
):
    runner, _ = fake_runner
    if case == "missing_rgb":
        runnable_sequence.frames[0].rgb_path = tmp_path / "missing.jpg"
    elif case == "unsupported_rgb":
        source = tmp_path / "frame.bmp"
        source.write_text("1")
        runnable_sequence.frames[0].rgb_path = source
    elif case == "empty":
        runnable_sequence.frames.clear()
    elif case == "duplicate":
        runnable_sequence.frames[1].frame_id = runnable_sequence.frames[0].frame_id
    elif case == "reversed":
        runnable_sequence.frames.reverse()
    elif case == "nan":
        runnable_sequence.frames[0].timestamp_seconds = float("nan")
    output = tmp_path / "not_created"
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.run(runnable_sequence, output)
    assert not output.exists()


@pytest.mark.parametrize("missing", ["python", "script", "config"])
def test_missing_tool_paths_create_no_output(tmp_path, runnable_sequence, missing):
    paths = [Path(sys.executable), tmp_path / "script.py", tmp_path / "config.yaml"]
    paths[1].write_text("")
    paths[2].write_text("")
    paths[["python", "script", "config"].index(missing)] = tmp_path / "missing"
    runner = Da3Runner(*paths)
    output = tmp_path / "not_created"
    with pytest.raises(FileNotFoundError):
        runner.run(runnable_sequence, output)
    assert not output.exists()


@pytest.mark.parametrize("mode", ["fail", "missing_depth", "missing_pose", "bad_array"])
def test_execution_or_output_failure_is_not_success(
    tmp_path, runnable_sequence, fake_runner, mode
):
    runner, config = fake_runner
    config.write_text(mode)
    output = tmp_path / "failed_run"
    with pytest.raises((RuntimeError, FileNotFoundError, ValueError)):
        runner.run(runnable_sequence, output)
    assert (output / "frame_manifest.json").exists()
    assert (output / "da3.log").exists()
    assert not (output / "run_complete.json").exists()


def test_preserves_virtualenv_executable_path(tmp_path, runnable_sequence, fake_runner):
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    _, config = fake_runner
    runner = Da3Runner(executable, tmp_path / "tool with spaces/fake_da3.py", config)
    output = tmp_path / "venv_run"
    runner.run(runnable_sequence, output)
    manifest = json.loads((output / "frame_manifest.json").read_text())
    assert manifest["command"][0] == str(executable)


def test_manifest_roundtrip_preserves_6k_plus_2_source_frames(
    tmp_path, runnable_sequence, fake_runner
):
    runner, _ = fake_runner
    for frame, index in zip(runnable_sequence.frames, [2, 8, 14]):
        frame.frame_id = f"scene:{index}"
        frame.source_frame_index = index
        frame.timestamp_seconds = index / 30
    anchor_paths = {
        frame.frame_id: frame.rgb_path for frame in runnable_sequence.frames
    }
    native = FrameSequence(
        "scene",
        [
            FrameInput(
                f"scene:{index}",
                index,
                index / 30,
                anchor_paths.get(f"scene:{index}", tmp_path / f"native_{index}.jpg"),
            )
            for index in range(15)
        ],
    )
    output = tmp_path / "offset_run"
    runner.run(runnable_sequence, output)
    geometry = GeometryAdapter.from_saved_run(native, output)
    assert geometry.geometry_frame_ids == ("scene:2", "scene:8", "scene:14")
    for value, frame_id in enumerate(geometry.geometry_frame_ids, start=1):
        assert geometry.load_frame(frame_id).depth[0, 0] == value
        assert geometry.load_pose(frame_id).method == "measured"
    assert geometry.load_pose("scene:0").method == "held"
    with pytest.raises(KeyError):
        geometry.load_frame("scene:6")

    # A positional 6k index must not masquerade as the actual 6k+2 native index.
    manifest_path = output / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frames"][0]["source_frame_index"] = 0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata does not match native frame"):
        GeometryAdapter.from_saved_run(native, output)


def test_rejects_mixed_image_sizes_before_creating_output(
    tmp_path, runnable_sequence, fake_runner
):
    runner, _ = fake_runner
    Image.new("RGB", (8, 4)).save(runnable_sequence.frames[0].rgb_path)
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="Mixed RGB sizes"):
        runner.run(runnable_sequence, output)
    assert not output.exists()


def test_saved_mapping_does_not_require_original_images(
    tmp_path, runnable_sequence, fake_runner
):
    runner, _ = fake_runner
    output = tmp_path / "run"
    runner.run(runnable_sequence, output)
    for frame in runnable_sequence.frames:
        frame.rgb_path.unlink()
    geometry = Da3OutputLoader(output, runnable_sequence).load_frame("scene:0")
    assert geometry.rgb_to_geometry[0, 0] == 0.5


def test_deterministic_run_pins_alignment_backend_and_requests_seeds(
    tmp_path, runnable_sequence, fake_runner
):
    """Both halves of reproducibility are recorded in the run.

    Seeds alone leave the Triton alignment kernels free to vary, and switching
    the backend alone leaves the seeds unset; two runs only come out
    bit-identical when both are applied, so both are asserted here.
    """
    runner, config = fake_runner
    config.write_text(
        "Model:\n"
        "  align_lib: 'triton' # choose among 'triton', 'torch', 'numba'\n"
        "  chunk_size: 30\n"
    )
    output = tmp_path / "run"
    runner.run(runnable_sequence, output)

    saved = (output / "da3_config.yaml").read_text()
    assert "align_lib: 'torch'" in saved
    assert "triton" not in saved
    # Untouched settings survive the rewrite.
    assert "chunk_size: 30" in saved

    environment = runner._environment()
    assert environment["VWS_DETERMINISM"] == "1"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["PYTHONPATH"].split(os.pathsep)[0].endswith("_da3_determinism")


def test_opting_out_leaves_the_supplied_config_byte_for_byte(
    tmp_path, runnable_sequence, fake_runner
):
    """A run asked not to be deterministic must not be silently rewritten."""
    runner, config = fake_runner
    original = "Model:\n  align_lib: 'triton'\n"
    config.write_text(original)
    runner._deterministic = False

    output = tmp_path / "run"
    runner.run(runnable_sequence, output)

    assert (output / "da3_config.yaml").read_text() == original
    assert "VWS_DETERMINISM" not in runner._environment()
