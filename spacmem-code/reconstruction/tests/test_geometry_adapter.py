import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from video_world_state.geometry import GeometryAdapter
from video_world_state.frames import select_frames


def test_saved_run_joins_native_ids_and_outputs(native_sequence, saved_run):
    geometry = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    assert geometry.geometry_frame_ids == ("example:2", "example:4")
    for frame_id in geometry.geometry_frame_ids:
        frame = geometry.load_frame(frame_id)
        pose = geometry.load_pose(frame_id)
        np.testing.assert_array_equal(frame.camera_to_world, pose.camera_to_world)
        assert frame.depth[0, 0] == int(frame_id.split(":")[1])
        assert pose.method == "measured"
        np.testing.assert_array_equal(frame.rgb_to_geometry, np.diag([0.5, 0.5, 1]))
    assert geometry.load_pose("example:3").method == "interpolated"
    assert geometry.load_pose("example:3").camera_to_world[0, 3] == 3
    assert geometry.load_pose("example:0").method == "held"
    with pytest.raises(KeyError):
        geometry.load_frame("example:3")
    with pytest.raises(KeyError):
        geometry.load_pose("unknown")
    with pytest.raises(AttributeError):
        geometry.geometry_frame_ids = ("different",)


def test_pose_setup_and_lookup_do_not_read_depth(
    native_sequence, saved_run, monkeypatch
):
    def unexpected_depth_load(*args, **kwargs):
        raise AssertionError("Pose setup must not open an NPZ")

    monkeypatch.setattr(np, "load", unexpected_depth_load)
    geometry = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    assert geometry.load_pose("example:3").camera_to_world[0, 3] == 3


def test_saved_run_setup_and_reads_leave_cache_unchanged(native_sequence, saved_run):
    def snapshot():
        return {
            str(path.relative_to(saved_run)): path.read_bytes()
            for path in saved_run.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    geometry = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    geometry.load_frame("example:2")
    geometry.load_pose("example:3")
    assert snapshot() == before


def test_missing_manifest_is_not_guessed(native_sequence, tmp_path):
    with pytest.raises(FileNotFoundError, match="frame_manifest.json"):
        GeometryAdapter.from_saved_run(native_sequence, tmp_path)


def test_direct_construction_cannot_bypass_manifest(native_sequence, tmp_path):
    with pytest.raises(FileNotFoundError, match="frame_manifest.json"):
        GeometryAdapter(native_sequence, tmp_path)


def test_missing_completion_marker_is_rejected_by_default(native_sequence, saved_run):
    (saved_run / "run_complete.json").unlink()
    with pytest.raises(ValueError, match="not marked complete"):
        GeometryAdapter.from_saved_run(native_sequence, saved_run)


def test_verified_legacy_cache_requires_explicit_opt_in(native_sequence, saved_run):
    (saved_run / "run_complete.json").unlink()
    geometry = GeometryAdapter.from_saved_run(
        native_sequence, saved_run, allow_legacy=True
    )
    assert geometry.load_pose("example:3").method == "interpolated"
    assert geometry.load_frame("example:2").depth[0, 0] == 2


@pytest.mark.parametrize("allow_legacy", [False, True])
@pytest.mark.parametrize(
    "constructor", [GeometryAdapter, GeometryAdapter.from_saved_run]
)
def test_new_failed_run_cannot_use_legacy_exception(
    native_sequence, saved_run, allow_legacy, constructor
):
    manifest_path = saved_run / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["completion_required"] = True
    manifest_path.write_text(json.dumps(manifest))
    (saved_run / "run_complete.json").unlink()
    # Reproduce the reported failure: valid poses and only some frame NPZs.
    (saved_run / "results_output/frame_1.npz").unlink()
    with pytest.raises(ValueError, match="not marked complete"):
        constructor(native_sequence, saved_run, allow_legacy=allow_legacy)


@pytest.mark.parametrize("allow_legacy", [False, True])
@pytest.mark.parametrize(
    "completion",
    [None, [], {}, {"anchor_count": 1}, {"anchor_count": True}, {"anchor_count": "2"}],
)
def test_bad_completion_marker_is_not_ignored(
    native_sequence, saved_run, completion, allow_legacy
):
    (saved_run / "run_complete.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="run_complete.json"):
        GeometryAdapter.from_saved_run(
            native_sequence, saved_run, allow_legacy=allow_legacy
        )


@pytest.mark.parametrize("allow_legacy", [False, True])
def test_truncated_completion_marker_is_rejected(
    native_sequence, saved_run, allow_legacy
):
    (saved_run / "run_complete.json").write_text('{"anchor_count":')
    with pytest.raises(ValueError, match="Invalid DA3 run_complete.json"):
        GeometryAdapter.from_saved_run(
            native_sequence, saved_run, allow_legacy=allow_legacy
        )


@pytest.mark.parametrize("rgb_path", [None, 123, "resized.png"])
def test_manifest_rgb_path_is_not_needed(native_sequence, saved_run, rgb_path):
    manifest_path = saved_run / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["frames"]:
        if rgb_path is None:
            del entry["rgb_path"]
        else:
            entry["rgb_path"] = rgb_path
    manifest_path.write_text(json.dumps(manifest))
    geometry = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    assert geometry.load_pose("example:3").camera_to_world[0, 3] == 3


def test_adapter_rejects_duplicate_native_ids(native_sequence, saved_run):
    native_sequence.frames[1].frame_id = native_sequence.frames[0].frame_id
    with pytest.raises(ValueError, match="Duplicate native frame ID"):
        GeometryAdapter.from_saved_run(native_sequence, saved_run)


@pytest.mark.parametrize("field", ["source_frame_index", "timestamp_seconds"])
def test_boolean_metadata_cannot_match_numeric_one(native_sequence, saved_run, field):
    manifest_path = saved_run / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if field == "source_frame_index":
        native_sequence.frames[2].source_frame_index = 1
    else:
        for frame in native_sequence.frames:
            frame.timestamp_seconds += 0.8
        for entry in manifest["frames"]:
            entry["timestamp_seconds"] += 0.8
    manifest["frames"][0][field] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata does not match"):
        GeometryAdapter.from_saved_run(native_sequence, saved_run)


@pytest.mark.parametrize(
    "case",
    [
        "sequence_id",
        "unknown_id",
        "duplicate_id",
        "timestamp",
        "index",
        "missing_field",
        "empty",
        "reversed",
    ],
)
def test_rejects_manifest_mismatch(native_sequence, saved_run, case):
    manifest_path = saved_run / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if case == "sequence_id":
        manifest["sequence_id"] = "other-scene"
    elif case == "unknown_id":
        manifest["frames"][0]["frame_id"] = "not-a-native-frame"
    elif case == "duplicate_id":
        manifest["frames"][1] = manifest["frames"][0]
    elif case == "timestamp":
        manifest["frames"][0]["timestamp_seconds"] = 0.21
    elif case == "index":
        manifest["frames"][0]["source_frame_index"] = 999
    elif case == "missing_field":
        del manifest["frames"][0]["source_frame_index"]
    elif case == "empty":
        manifest["frames"] = []
    elif case == "reversed":
        manifest["frames"].reverse()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        GeometryAdapter.from_saved_run(native_sequence, saved_run)


def test_geometry_reports_missing_depth_but_pose_still_works(
    native_sequence, saved_run
):
    geometry = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    # The copied directory has pose/manifest files but no depth NPZs.
    pose_only_directory = saved_run / "pose_only"
    pose_only_directory.mkdir()
    for filename in ("camera_poses.txt", "frame_manifest.json"):
        (pose_only_directory / filename).write_bytes(
            (saved_run / filename).read_bytes()
        )
    pose_only = GeometryAdapter.from_saved_run(
        native_sequence, pose_only_directory, allow_legacy=True
    )
    assert pose_only.geometry_frame_ids == geometry.geometry_frame_ids
    assert pose_only.load_pose("example:3").method == "interpolated"
    with pytest.raises(FileNotFoundError, match="Missing DA3 frame result"):
        pose_only.load_frame("example:2")


def test_pose_count_mismatch_is_rejected(native_sequence, saved_run):
    np.savetxt(saved_run / "camera_poses.txt", np.eye(4).reshape(1, 16))
    with pytest.raises(ValueError, match="pose count"):
        GeometryAdapter.from_saved_run(native_sequence, saved_run)


@pytest.mark.parametrize(
    "fps,indices", [(5, [0, 2, 4, 6]), (3, [0, 4]), (30, list(range(7)))]
)
def test_new_run_uses_supplied_anchors_then_opens_loader(
    native_sequence, tmp_path, fps, indices, monkeypatch
):
    from unittest.mock import Mock
    import video_world_state.geometry as geometry_module

    aligner = Mock()
    aligner.estimate.return_value = np.eye(4, dtype=np.float32)
    monkeypatch.setattr(geometry_module, "SceneAligner", Mock(return_value=aligner))

    class SyntheticRunner:
        def run(self, anchors, output):
            assert anchors.sequence_id == native_sequence.sequence_id
            assert [frame.source_frame_index for frame in anchors.frames] == indices
            output.mkdir()
            (output / "results_output").mkdir()
            poses = np.repeat(np.eye(4)[None], len(anchors.frames), axis=0)
            poses[:, 0, 3] = indices
            np.savetxt(output / "camera_poses.txt", poses.reshape(-1, 16))
            for i in range(len(anchors.frames)):
                np.savez(
                    output / "results_output" / f"frame_{i}.npz",
                    depth=np.ones((2, 3)),
                    conf=np.ones((2, 3)),
                    intrinsics=np.eye(3),
                )
            (output / "frame_manifest.json").write_text(
                json.dumps(
                    {
                        "sequence_id": anchors.sequence_id,
                        "image_preprocessing": {
                            "method": "upper_bound_resize",
                            "pixel_convention": "integer",
                        },
                        "frames": [
                            {
                                "frame_id": frame.frame_id,
                                "source_frame_index": frame.source_frame_index,
                                "timestamp_seconds": frame.timestamp_seconds,
                                "rgb_size_hw": [4, 6],
                            }
                            for frame in anchors.frames
                        ],
                    }
                )
            )
            (output / "run_complete.json").write_text(
                json.dumps({"anchor_count": len(anchors.frames)})
            )

    geometry = GeometryAdapter.run(
        native_sequence,
        select_frames(native_sequence, fps),
        SyntheticRunner(),
        tmp_path / "run",
    )
    assert geometry.geometry_frame_ids == tuple(f"example:{i}" for i in indices)
    assert geometry.alignment_method == "leveled"
    np.testing.assert_array_equal(
        geometry.load_frame(geometry.geometry_frame_ids[0]).rgb_to_geometry,
        np.diag([0.5, 0.5, 1]),
    )
    for frame in native_sequence.frames:
        assert geometry.load_pose(frame.frame_id).method in {
            "measured",
            "interpolated",
            "held",
        }


@pytest.mark.parametrize("fps", [0, -1, float("inf"), float("nan")])
def test_invalid_sampling_rate_is_rejected(native_sequence, fps):
    with pytest.raises(ValueError, match="fps"):
        select_frames(native_sequence, fps)


@pytest.mark.parametrize("case", ["empty", "duplicate", "reversed", "nan"])
def test_sampling_rejects_bad_native_sequence(native_sequence, case):
    if case == "empty":
        native_sequence.frames.clear()
    elif case == "duplicate":
        native_sequence.frames[1].frame_id = native_sequence.frames[0].frame_id
    elif case == "reversed":
        native_sequence.frames.reverse()
    else:
        native_sequence.frames[0].timestamp_seconds = float("nan")
    with pytest.raises(ValueError):
        select_frames(native_sequence, 5)


def test_sampling_uses_timestamps_not_ids_or_frame_positions(native_sequence):
    for frame, timestamp in zip(
        native_sequence.frames, [10, 10.11, 10.19, 10.24, 10.39, 10.44, 11.2]
    ):
        frame.timestamp_seconds = timestamp
        frame.source_frame_index *= 13

    anchors = select_frames(native_sequence, 5)
    assert [frame.frame_id for frame in anchors.frames] == [
        "example:0",
        "example:3",
        "example:5",
        "example:6",
    ]
    assert [frame.source_frame_index for frame in anchors.frames] == [0, 39, 65, 78]


def test_world_points_need_the_da3_library(native_sequence, saved_run, monkeypatch):
    """Reading saved geometry must not require Torch or DA3; unprojecting does."""
    monkeypatch.setitem(sys.modules, "depth_anything_3", None)
    geometry = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    frame = geometry.load_frame("example:2")

    with pytest.raises(ImportError, match="depth_anything_3"):
        geometry.world_points(frame)


def test_world_points_use_da3s_own_unprojection(native_sequence, saved_run):
    """Known intrinsics and pose: optical-axis depth on integer pixels."""
    pytest.importorskip("torch")
    pytest.importorskip("depth_anything_3")
    geometry = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    frame = geometry.load_frame("example:2")
    frame.intrinsics[:] = [[2, 0, 1], [0, 2, 0.5], [0, 0, 1]]
    frame.camera_to_world[:] = np.eye(4)
    frame.camera_to_world[:3, 3] = [10, 20, 30]

    points = geometry.world_points(frame)

    assert points.shape == (*frame.depth.shape, 3)
    depth = frame.depth[0, 0]
    rows, columns = np.indices(frame.depth.shape)
    expected_x = (columns - 1) / 2 * depth + 10
    expected_y = (rows - 0.5) / 2 * depth + 20
    np.testing.assert_allclose(points[..., 0], expected_x, atol=1e-5)
    np.testing.assert_allclose(points[..., 1], expected_y, atol=1e-5)
    np.testing.assert_allclose(points[..., 2], depth + 30, atol=1e-5)


def test_leveling_rotates_world_points_exactly_once(native_sequence, saved_run):
    pytest.importorskip("torch")
    pytest.importorskip("depth_anything_3")
    leveling = np.eye(4, dtype=np.float32)
    leveling[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()

    raw = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    leveled = GeometryAdapter.from_saved_run(
        native_sequence, saved_run, world_alignment=leveling
    )
    raw_points = raw.world_points(raw.load_frame("example:2"))
    leveled_points = leveled.world_points(leveled.load_frame("example:2"))

    assert raw.alignment_method == "raw" and leveled.alignment_method == "leveled"
    np.testing.assert_allclose(
        leveled_points.reshape(-1, 3) @ leveling[:3, :3],
        raw_points.reshape(-1, 3),
        atol=1e-5,
    )
