from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from video_world_state.contracts import CameraPoseEstimate, FrameInput, FrameSequence
from video_world_state.poses import PoseInterpolator


@pytest.mark.parametrize("anchor_position", [0, 1])
def test_preserves_anchor_exactly(native_sequence, anchor_poses, anchor_position):
    interpolator = PoseInterpolator(native_sequence, anchor_poses)
    anchor = anchor_poses[anchor_position]
    pose = interpolator.load_pose(anchor.frame_id)
    assert pose.method == "measured"
    assert pose.camera_to_world.dtype == np.float32
    np.testing.assert_array_equal(pose.camera_to_world, anchor.camera_to_world)


def test_interpolates_position_and_rotation(native_sequence, anchor_poses):
    pose = PoseInterpolator(native_sequence, anchor_poses).load_pose("example:3")
    assert pose.frame_id == "example:3"
    assert pose.method == "interpolated"
    assert pose.camera_to_world.dtype == np.float32
    np.testing.assert_allclose(pose.camera_to_world[:3, 3], [3, 0, 0])
    expected = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    np.testing.assert_allclose(pose.camera_to_world[:3, :3], expected, atol=1e-7)
    np.testing.assert_array_equal(pose.camera_to_world[3], [0, 0, 0, 1])


def test_uses_timestamps_not_frame_indices():
    sequence = FrameSequence(
        "irregular",
        [
            FrameInput("left", 0, 0.0, Path("left.png")),
            FrameInput("middle", 100, 0.1, Path("middle.png")),
            FrameInput("right", 200, 1.0, Path("right.png")),
        ],
    )
    left = np.eye(4, dtype=np.float32)
    right = left.copy()
    right[0, 3] = 10
    right[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    interpolator = PoseInterpolator(
        sequence,
        [
            CameraPoseEstimate("left", left, "measured"),
            CameraPoseEstimate("right", right, "measured"),
        ],
    )
    pose = interpolator.load_pose("middle")
    assert pose.camera_to_world[0, 3] == pytest.approx(1)
    expected = Rotation.from_euler("z", 9, degrees=True).as_matrix()
    np.testing.assert_allclose(pose.camera_to_world[:3, :3], expected, atol=1e-7)


def test_rotation_takes_shortest_path(native_sequence, anchor_poses):
    for anchor, angle in zip(anchor_poses, [170, -170]):
        anchor.camera_to_world[:3, :3] = Rotation.from_euler(
            "z", angle, degrees=True
        ).as_matrix()
    pose = PoseInterpolator(native_sequence, anchor_poses).load_pose("example:3")
    np.testing.assert_allclose(
        pose.camera_to_world[:3, :3], np.diag([-1, -1, 1]), atol=1e-7
    )


@pytest.mark.parametrize(
    "frame_id,anchor_position", [("example:0", 0), ("example:6", 1)]
)
def test_holds_nearest_pose_outside_anchor_range(
    native_sequence, anchor_poses, frame_id, anchor_position
):
    pose = PoseInterpolator(native_sequence, anchor_poses).load_pose(frame_id)
    assert pose.method == "held"
    np.testing.assert_array_equal(
        pose.camera_to_world, anchor_poses[anchor_position].camera_to_world
    )


def test_one_anchor_holds_all_other_frames(native_sequence, anchor_poses):
    interpolator = PoseInterpolator(native_sequence, anchor_poses[:1])
    for frame in native_sequence.frames:
        pose = interpolator.load_pose(frame.frame_id)
        expected_method = "measured" if frame.frame_id == "example:2" else "held"
        assert pose.method == expected_method
        np.testing.assert_array_equal(
            pose.camera_to_world, anchor_poses[0].camera_to_world
        )


def test_unknown_id_and_mutation_isolation(native_sequence, anchor_poses):
    interpolator = PoseInterpolator(native_sequence, anchor_poses)
    anchor_poses[0].camera_to_world[0, 3] = 999
    native_sequence.frames[3].timestamp_seconds = 999
    assert interpolator.load_pose("example:2").camera_to_world[0, 3] == 2
    returned = interpolator.load_pose("example:3")
    returned.camera_to_world[0, 3] = 999
    assert interpolator.load_pose("example:3").camera_to_world[0, 3] == 3
    with pytest.raises(KeyError, match="Unknown native frame ID"):
        interpolator.load_pose("not-a-frame")


def test_timestamp_origin_can_be_negative(native_sequence, anchor_poses):
    for frame in native_sequence.frames:
        frame.timestamp_seconds -= 10
    pose = PoseInterpolator(native_sequence, anchor_poses).load_pose("example:3")
    assert pose.method == "interpolated"
    assert pose.camera_to_world[0, 3] == pytest.approx(3)


@pytest.mark.parametrize(
    "case",
    [
        "empty_native",
        "duplicate_native",
        "nan_time",
        "equal_times",
        "unordered_times",
        "no_anchors",
        "unknown_anchor",
        "duplicate_anchor",
        "unordered_anchors",
        "non_measured",
        "bad_shape",
        "nan_pose",
        "scale",
        "reflection",
        "bad_last_row",
    ],
)
def test_rejects_invalid_inputs(native_sequence, anchor_poses, case):
    if case == "empty_native":
        native_sequence.frames = []
    elif case == "duplicate_native":
        native_sequence.frames[1].frame_id = native_sequence.frames[0].frame_id
    elif case in ("nan_time", "equal_times", "unordered_times"):
        native_sequence.frames[1].timestamp_seconds = {
            "nan_time": np.nan,
            "equal_times": 0,
            "unordered_times": 1,
        }[case]
    elif case == "no_anchors":
        anchor_poses = []
    elif case == "unknown_anchor":
        anchor_poses[0].frame_id = "unknown"
    elif case == "duplicate_anchor":
        anchor_poses = [anchor_poses[0], anchor_poses[0]]
    elif case == "unordered_anchors":
        anchor_poses.reverse()
    elif case == "non_measured":
        anchor_poses[0].method = "interpolated"
    elif case == "bad_shape":
        anchor_poses[0].camera_to_world = np.eye(3)
    elif case == "nan_pose":
        anchor_poses[0].camera_to_world[0, 3] = np.nan
    elif case == "scale":
        anchor_poses[0].camera_to_world[0, 0] = 2
    elif case == "reflection":
        anchor_poses[0].camera_to_world[0, 0] = -1
    elif case == "bad_last_row":
        anchor_poses[0].camera_to_world[3, 3] = 2
    with pytest.raises(ValueError):
        PoseInterpolator(native_sequence, anchor_poses)
