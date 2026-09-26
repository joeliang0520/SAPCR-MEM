"""Small synthetic native sequences and DA3 files; no real images or model runs."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from video_world_state.contracts import CameraPoseEstimate, FrameInput, FrameSequence


@pytest.fixture
def native_sequence() -> FrameSequence:
    return FrameSequence(
        "example",
        [FrameInput(f"example:{i}", i, i / 10, Path(f"rgb_{i}.png")) for i in range(7)],
    )


@pytest.fixture
def anchor_poses() -> list[CameraPoseEstimate]:
    poses = []
    for frame_index, angle in [(2, 0), (4, 90)]:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
        matrix[0, 3] = frame_index
        poses.append(CameraPoseEstimate(f"example:{frame_index}", matrix, "measured"))
    return poses


@pytest.fixture
def saved_run(
    tmp_path: Path,
    native_sequence: FrameSequence,
    anchor_poses: list[CameraPoseEstimate],
) -> Path:
    results_directory = tmp_path / "results_output"
    results_directory.mkdir()
    matrices = np.stack([pose.camera_to_world for pose in anchor_poses])
    np.savetxt(tmp_path / "camera_poses.txt", matrices.reshape(-1, 16))
    anchors = [native_sequence.frames[2], native_sequence.frames[4]]
    for i, frame in enumerate(anchors):
        np.savez(
            results_directory / f"frame_{i}.npz",
            depth=np.full((2, 3), frame.source_frame_index, dtype=np.float32),
            conf=np.full((2, 3), 0.75, dtype=np.float32),
            intrinsics=np.eye(3, dtype=np.float32),
        )
    manifest = {
        "sequence_id": native_sequence.sequence_id,
        "image_preprocessing": {
            "method": "upper_bound_resize",
            "pixel_convention": "integer",
        },
        "frames": [
            {
                "frame_id": frame.frame_id,
                "source_frame_index": frame.source_frame_index,
                "timestamp_seconds": frame.timestamp_seconds,
                "rgb_path": str(frame.rgb_path),
                "rgb_size_hw": [4, 6],
            }
            for frame in anchors
        ],
    }
    (tmp_path / "frame_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "run_complete.json").write_text(
        json.dumps({"anchor_count": len(anchors)})
    )
    return tmp_path
