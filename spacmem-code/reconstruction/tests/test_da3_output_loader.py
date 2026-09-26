import json
from pathlib import Path

import numpy as np
import pytest

from video_world_state.adapters.da3 import Da3OutputLoader
from video_world_state.contracts import FrameInput, FrameSequence


def _frame_sequence() -> FrameSequence:
    return FrameSequence(
        sequence_id="example",
        frames=[
            FrameInput("example:0", 0, 0.0, Path("frame_0.png")),
            FrameInput("example:6", 6, 0.2, Path("frame_6.png")),
        ],
    )


def _write_da3_output(output_directory: Path, pose_count: int = 2) -> None:
    results_directory = output_directory / "results_output"
    results_directory.mkdir()

    poses = np.repeat(np.eye(4, dtype=np.float32)[None], pose_count, axis=0)
    poses[:, 0, 3] = np.arange(pose_count)
    np.savetxt(output_directory / "camera_poses.txt", poses.reshape(-1, 16))

    for index in range(2):
        np.savez(
            results_directory / f"frame_{index}.npz",
            depth=np.full((2, 3), index + 1, dtype=np.float32),
            conf=np.full((2, 3), 0.75, dtype=np.float32),
            intrinsics=np.eye(3, dtype=np.float32),
        )
    (output_directory / "frame_manifest.json").write_text(
        json.dumps(
            {
                "sequence_id": "example",
                "image_preprocessing": {
                    "method": "upper_bound_resize",
                    "pixel_convention": "integer",
                },
                "frames": [
                    {"frame_id": frame.frame_id, "rgb_size_hw": [4, 6]}
                    for frame in _frame_sequence().frames
                ],
            }
        )
    )


def test_loads_geometry_by_stable_frame_id(tmp_path: Path) -> None:
    _write_da3_output(tmp_path)
    loader = Da3OutputLoader(tmp_path, _frame_sequence())

    geometry = loader.load_frame("example:6")

    assert geometry.frame_id == "example:6"
    assert geometry.depth.dtype == np.float32
    assert geometry.confidence.dtype == np.float32
    assert geometry.intrinsics.dtype == np.float32
    assert geometry.camera_to_world.dtype == np.float32
    assert geometry.rgb_to_geometry.dtype == np.float32
    np.testing.assert_array_equal(geometry.rgb_to_geometry, np.diag([0.5, 0.5, 1]))
    np.testing.assert_array_equal(geometry.depth, np.full((2, 3), 2))
    np.testing.assert_array_equal(geometry.confidence, np.full((2, 3), 0.75))
    np.testing.assert_array_equal(geometry.intrinsics, np.eye(3))
    np.testing.assert_array_equal(
        geometry.camera_to_world,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
    )


def test_rejects_unknown_frame_id(tmp_path: Path) -> None:
    _write_da3_output(tmp_path)
    loader = Da3OutputLoader(tmp_path, _frame_sequence())

    with pytest.raises(KeyError, match="Unknown frame ID"):
        loader.load_frame("example:12")


def test_rejects_pose_count_mismatch(tmp_path: Path) -> None:
    _write_da3_output(tmp_path, pose_count=1)

    with pytest.raises(ValueError, match="1 poses for 2 frames"):
        Da3OutputLoader(tmp_path, _frame_sequence())


def test_rejects_duplicate_anchor_ids(tmp_path: Path) -> None:
    _write_da3_output(tmp_path)
    sequence = _frame_sequence()
    sequence.frames[1].frame_id = sequence.frames[0].frame_id
    with pytest.raises(ValueError, match="Duplicate frame ID"):
        Da3OutputLoader(tmp_path, sequence)


def test_pose_only_lookup_does_not_open_depth(tmp_path: Path, monkeypatch) -> None:
    _write_da3_output(tmp_path)
    loader = Da3OutputLoader(tmp_path, _frame_sequence())

    def unexpected_depth_load(*args, **kwargs):
        raise AssertionError("Pose-only lookup must not open an NPZ")

    monkeypatch.setattr(np, "load", unexpected_depth_load)
    pose = loader.load_pose("example:6")
    assert pose.frame_id == "example:6"
    assert pose.method == "measured"
    assert pose.camera_to_world.dtype == np.float32
    assert pose.camera_to_world[0, 3] == 1


def test_pose_matches_geometry_and_is_an_independent_copy(tmp_path: Path) -> None:
    _write_da3_output(tmp_path)
    loader = Da3OutputLoader(tmp_path, _frame_sequence())
    pose = loader.load_pose("example:6")
    np.testing.assert_array_equal(
        pose.camera_to_world, loader.load_frame("example:6").camera_to_world
    )
    pose.camera_to_world[0, 3] = 999
    assert loader.load_pose("example:6").camera_to_world[0, 3] == 1
    with pytest.raises(KeyError, match="Unknown frame ID"):
        loader.load_pose("example:3")


def test_rejects_non_finite_saved_pose(tmp_path: Path) -> None:
    _write_da3_output(tmp_path)
    rows = np.loadtxt(tmp_path / "camera_poses.txt")
    rows[0, 0] = np.nan
    np.savetxt(tmp_path / "camera_poses.txt", rows)
    with pytest.raises(ValueError, match="non-finite"):
        Da3OutputLoader(tmp_path, _frame_sequence())


def test_mapping_uses_actual_grid_and_matches_intrinsics_scaling(tmp_path):
    _write_da3_output(tmp_path)
    manifest_path = tmp_path / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["frames"]:
        entry["rgb_size_hw"] = [968, 1296]
    manifest_path.write_text(json.dumps(manifest))
    original_k = np.array([[900, 0, 648], [0, 910, 484], [0, 0, 1]], dtype=np.float32)
    expected_map = np.diag([504 / 1296, 378 / 968, 1]).astype(np.float32)
    np.savez(
        tmp_path / "results_output/frame_0.npz",
        depth=np.ones((378, 504)),
        conf=np.ones((378, 504)),
        intrinsics=expected_map @ original_k,
    )
    geometry = Da3OutputLoader(tmp_path, _frame_sequence()).load_frame("example:0")
    np.testing.assert_array_equal(geometry.rgb_to_geometry, expected_map)
    # A projected camera ray has the same direction on the original and depth grids.
    rgb_pixel = np.array([800, 600, 1])
    geometry_pixel = geometry.rgb_to_geometry @ rgb_pixel
    np.testing.assert_allclose(
        np.linalg.inv(geometry.intrinsics) @ geometry_pixel,
        np.linalg.inv(original_k) @ rgb_pixel,
        atol=1e-6,
    )
    geometry.rgb_to_geometry[0, 0] = 123
    assert (
        Da3OutputLoader(tmp_path, _frame_sequence())
        .load_frame("example:0")
        .rgb_to_geometry[0, 0]
        != 123
    )


@pytest.mark.parametrize("shape", [(2, 3), (4, 6), (8, 12)])
def test_mapping_supports_identity_downscale_and_upscale(tmp_path, shape):
    _write_da3_output(tmp_path)
    np.savez(
        tmp_path / "results_output/frame_0.npz",
        depth=np.ones(shape),
        conf=np.ones(shape),
        intrinsics=np.eye(3),
    )
    geometry = Da3OutputLoader(tmp_path, _frame_sequence()).load_frame("example:0")
    np.testing.assert_allclose(
        geometry.rgb_to_geometry, np.diag([shape[1] / 6, shape[0] / 4, 1])
    )


@pytest.mark.parametrize("missing", ["manifest", "preprocessing"])
def test_legacy_runs_allow_poses_but_do_not_guess_pixel_mapping(tmp_path, missing):
    _write_da3_output(tmp_path)
    manifest_path = tmp_path / "frame_manifest.json"
    if missing == "manifest":
        manifest_path.unlink()
    else:
        manifest = json.loads(manifest_path.read_text())
        del manifest["image_preprocessing"]
        manifest_path.write_text(json.dumps(manifest))
    loader = Da3OutputLoader(tmp_path, _frame_sequence())
    assert loader.load_pose("example:0").method == "measured"
    with pytest.raises(ValueError, match="Missing DA3 pixel mapping metadata"):
        loader.load_frame("example:0")


@pytest.mark.parametrize("method", ["upper_bound_crop", "lower_bound_crop", "unknown"])
def test_rejects_unsupported_preprocessing(tmp_path, method):
    _write_da3_output(tmp_path)
    manifest_path = tmp_path / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["image_preprocessing"]["method"] = method
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported DA3 image preprocessing"):
        Da3OutputLoader(tmp_path, _frame_sequence())


@pytest.mark.parametrize(
    "size", [None, [], [0, 6], [4, -6], [True, 6], [4, 6.0], [4, 6, 3]]
)
def test_rejects_invalid_original_rgb_size(tmp_path, size):
    _write_da3_output(tmp_path)
    manifest_path = tmp_path / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frames"][0]["rgb_size_hw"] = size
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rgb_size_hw"):
        Da3OutputLoader(tmp_path, _frame_sequence())


def test_rejects_mixed_original_sizes_in_saved_mapping(tmp_path):
    _write_da3_output(tmp_path)
    manifest_path = tmp_path / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frames"][1]["rgb_size_hw"] = [5, 6]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Mixed RGB sizes"):
        Da3OutputLoader(tmp_path, _frame_sequence())


def test_rejects_mapping_for_a_different_anchor_order(tmp_path):
    _write_da3_output(tmp_path)
    sequence = _frame_sequence()
    sequence.frames.reverse()
    with pytest.raises(ValueError, match="mapping manifest does not match"):
        Da3OutputLoader(tmp_path, sequence)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_rejects_empty_depth_grids_instead_of_returning_singular_mapping(
    tmp_path, shape
):
    _write_da3_output(tmp_path)
    np.savez(
        tmp_path / "results_output/frame_0.npz",
        depth=np.empty(shape),
        conf=np.empty(shape),
        intrinsics=np.eye(3),
    )
    with pytest.raises(ValueError, match="non-empty 2D depth"):
        Da3OutputLoader(tmp_path, _frame_sequence()).load_frame("example:0")


def test_saved_aligned_geometry_ignores_chunk_pose_and_scale(tmp_path):
    _write_da3_output(tmp_path)
    raw_chunk_pose = np.eye(4, dtype=np.float32)[:3]
    raw_chunk_pose[:, 3] = [99, 100, 101]
    saved_depth = np.full((2, 3), 2.22, dtype=np.float32)
    np.savez(
        tmp_path / "results_output/frame_1.npz",
        depth=saved_depth,
        conf=np.ones((2, 3)),
        intrinsics=np.eye(3),
        extrinsics=raw_chunk_pose,
        s=np.float32(1.11),
        R=np.eye(3),
        T=np.ones(3) * 99,
    )
    loader = Da3OutputLoader(tmp_path, _frame_sequence())
    geometry = loader.load_frame("example:6")
    # Depth has already been scaled by DA3; stored s must not be applied again.
    np.testing.assert_array_equal(geometry.depth, saved_depth)
    # camera_poses.txt is final c2w; NPZ extrinsics describe a different raw chunk.
    expected_pose = np.eye(4, dtype=np.float32)
    expected_pose[0, 3] = 1
    np.testing.assert_array_equal(geometry.camera_to_world, expected_pose)
    np.testing.assert_array_equal(
        loader.load_pose("example:6").camera_to_world, expected_pose
    )
    geometry.depth[:] = -1
    np.testing.assert_array_equal(loader.load_frame("example:6").depth, saved_depth)
