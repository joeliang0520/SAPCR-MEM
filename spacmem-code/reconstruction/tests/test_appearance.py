"""Cropping and the runner's contract with its worker; no image model runs."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from video_world_state.adapters import appearance
from video_world_state.contracts import FrameInput, FrameSequence, SegmentationObservation

# Stands in for the model: one distinct unit vector per crop, in manifest order.
FAKE_WORKER = """
import argparse, json
from pathlib import Path
import numpy as np
parser = argparse.ArgumentParser()
for flag in ("--directory", "--model", "--device", "--batch-size"):
    parser.add_argument(flag)
arguments = parser.parse_args()
directory = Path(arguments.directory)
crops = json.loads((directory / "manifest.json").read_text())["crops"]
np.savez(
    directory / "embeddings.npz",
    observation_ids=np.array([crop["observation_id"] for crop in crops]),
    vectors=np.eye(len(crops), 4, dtype=np.float32),
)
"""


class FakeSegmentation:
    def __init__(self, frames: dict[str, list[SegmentationObservation]]) -> None:
        self._frames = frames

    @property
    def segmentation_frame_ids(self) -> tuple[str, ...]:
        return tuple(self._frames)

    def load_frame(self, frame_id: str) -> list[SegmentationObservation]:
        return self._frames[frame_id]


def square_mask(top: int, left: int, size: int) -> np.ndarray:
    mask = np.zeros((20, 30), dtype=bool)
    mask[top : top + size, left : left + size] = True
    return mask


@pytest.fixture
def scene(tmp_path):
    frames = []
    for i in range(2):
        path = tmp_path / f"rgb_{i}.png"
        Image.fromarray(np.full((20, 30, 3), 200, np.uint8)).save(path)
        frames.append(FrameInput(f"s:{i}", i, i / 5, path))
    segmentation = FakeSegmentation({
        "s:0": [
            SegmentationObservation("s:obs:0:0", "s:0", square_mask(2, 2, 10), "chair", 0.9),
            SegmentationObservation("s:obs:0:1", "s:0", square_mask(0, 0, 3), "cup", 0.9),
        ],
        "s:1": [
            SegmentationObservation("s:obs:1:0", "s:1", square_mask(5, 10, 8), "chair", 0.9),
        ],
    })
    return FrameSequence("s", frames), segmentation


@pytest.fixture
def runner(tmp_path):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    return appearance.AppearanceRunner(Path(sys.executable), worker_script=worker)


def test_a_crop_keeps_the_object_and_greys_out_the_rest():
    pixels = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    mask = np.zeros((20, 30), dtype=bool)
    mask[4:14, 6:16] = True
    mask[4, 6] = False

    patch = appearance.crop_observation(pixels, mask)

    assert patch.shape == (10, 10, 3)
    assert (patch[0, 0] == appearance.BACKGROUND).all()
    assert (patch[5, 5] == pixels[9, 11]).all()


def test_a_mask_too_small_to_recognise_is_not_cropped():
    assert appearance.crop_observation(
        np.zeros((20, 30, 3), np.uint8), square_mask(0, 0, 7)
    ) is None


def test_every_usable_observation_gets_a_descriptor(scene, runner, tmp_path):
    sequence, segmentation = scene

    descriptors = runner.run(sequence, segmentation, tmp_path / "appearance")

    assert set(descriptors) == {"s:obs:0:0", "s:obs:1:0"}
    assert len(list((tmp_path / "appearance" / "crops").iterdir())) == 2
    reopened = appearance.load_descriptors(tmp_path / "appearance")
    assert all(np.array_equal(reopened[key], descriptors[key]) for key in descriptors)


def test_an_existing_run_is_not_overwritten(scene, runner, tmp_path):
    (tmp_path / "appearance").mkdir()
    with pytest.raises(FileExistsError):
        runner.run(*scene, tmp_path / "appearance")


def test_a_failed_worker_keeps_its_log(scene, tmp_path):
    worker = tmp_path / "broken_worker.py"
    worker.write_text("raise SystemExit('no model here')")
    runner = appearance.AppearanceRunner(Path(sys.executable), worker_script=worker)

    with pytest.raises(RuntimeError, match="worker.log"):
        runner.run(*scene, tmp_path / "appearance")
    assert "no model here" in (tmp_path / "appearance" / "worker.log").read_text()
