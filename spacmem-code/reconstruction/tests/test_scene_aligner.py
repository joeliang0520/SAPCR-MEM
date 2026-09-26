from types import SimpleNamespace
from unittest.mock import Mock

import inspect

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from video_world_state.alignment import FLOOR_STEP, SceneAligner, floor_height
from video_world_state.contracts import CameraPoseEstimate, FrameGeometry
from video_world_state.geometry import GeometryAdapter
from video_world_state.objects import ObjectAssociator


def test_new_run_orchestrates_leveling_without_changing_raw_files(
    native_sequence, saved_run, tmp_path, monkeypatch
):
    from video_world_state.contracts import FrameSequence
    import video_world_state.geometry as geometry_module

    anchors = FrameSequence(native_sequence.sequence_id, native_sequence.frames[2:5:2])
    before = {p.name: p.read_bytes() for p in saved_run.iterdir() if p.is_file()}
    raw = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    aligner = Mock()
    aligner.estimate.return_value = transform
    monkeypatch.setattr(geometry_module, "SceneAligner", Mock(return_value=aligner))
    runner = Mock()
    alignment_path = tmp_path / "prepared" / "rotation.npy"
    alignment_path.parent.mkdir()
    leveled = GeometryAdapter.run(
        native_sequence, anchors, runner, saved_run, alignment_path=alignment_path
    )
    runner.run.assert_called_once_with(anchors, saved_run)
    assert aligner.estimate.call_args.args[0].alignment_method == "raw"
    assert aligner.estimate.call_count == 1
    assert leveled.alignment_method == "leveled"
    np.testing.assert_array_equal(np.load(alignment_path), transform)
    np.testing.assert_allclose(
        leveled.load_frame("example:2").camera_to_world,
        transform @ raw.load_frame("example:2").camera_to_world,
    )
    assert before == {
        p.name: p.read_bytes() for p in saved_run.iterdir() if p.is_file()
    }
    with pytest.raises(FileExistsError):
        GeometryAdapter.run(
            native_sequence, anchors, runner, saved_run, alignment_path=alignment_path
        )
    assert runner.run.call_count == 1


def test_new_run_leveling_failure_does_not_save_alignment(
    native_sequence, saved_run, tmp_path, monkeypatch
):
    from video_world_state.contracts import FrameSequence
    import video_world_state.geometry as geometry_module

    aligner = Mock()
    aligner.estimate.side_effect = ValueError("No convincing floor plane")
    monkeypatch.setattr(geometry_module, "SceneAligner", Mock(return_value=aligner))
    path = tmp_path / "rotation.npy"
    anchors = FrameSequence(native_sequence.sequence_id, native_sequence.frames[2:5:2])
    with pytest.raises(ValueError, match="No convincing floor"):
        GeometryAdapter.run(
            native_sequence, anchors, Mock(), saved_run, alignment_path=path
        )
    assert not path.exists()


def plane_geometry(heights, *, ceiling=False, tilt=None, slopes=None):
    """Ray-intersect known horizontal planes, then rotate the entire scene."""
    world = np.eye(4, dtype=np.float32)
    if tilt is not None:
        world[:3, :3] = tilt
    camera = np.eye(4, dtype=np.float32)
    camera[:3, :3] = Rotation.from_euler(
        "x", -70 if ceiling else -110, degrees=True
    ).as_matrix()
    camera[:3, 3] = [0, -2, 1.5]
    intrinsics = np.array([[110, 0, 60], [0, 110, 60], [0, 0, 1]], dtype=np.float32)
    v, u = np.mgrid[:120, :120]
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(intrinsics).T
    world_rays = rays @ camera[:3, :3].T
    frames = {}
    for index, height in enumerate(heights):
        slope = slopes[index] if slopes is not None else 0
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (height - camera[2, 3]) / (
                world_rays[:, :, 2] - slope * world_rays[:, :, 0]
            )
        depth[depth <= 0] = np.nan
        if slope:
            # A finite tabletop, not an infinite sloped plane extending below
            # the floor. Keep background floor evidence in the other frame.
            surface = world_rays * depth[:, :, None] + camera[:3, 3]
            depth[(np.abs(surface[:, :, 0]) > 2) | (surface[:, :, 2] < 0.3)] = np.nan
        frame_id = f"test:{index}"
        frames[frame_id] = FrameGeometry(
            frame_id,
            depth.astype(np.float32),
            np.ones((120, 120), dtype=np.float32),
            intrinsics.copy(),
            world @ camera,
            np.eye(3, dtype=np.float32),
        )
    return SimpleNamespace(
        alignment_method="raw",
        geometry_frame_ids=tuple(frames),
        load_frame=lambda frame_id: frames[frame_id],
        load_pose=lambda frame_id: CameraPoseEstimate(
            frame_id, frames[frame_id].camera_to_world.copy(), "measured"
        ),
    )


def test_floor_leveling_recovers_tilt_without_other_changes():
    pytest.importorskip("open3d")
    tilt = Rotation.from_euler("xy", [20, 25], degrees=True).as_matrix()
    geometry = plane_geometry([0, 0], tilt=tilt)
    before = geometry.load_frame("test:0").camera_to_world.copy()
    transform = SceneAligner(pixel_stride=3).estimate(geometry)
    np.testing.assert_allclose(transform[:3, :3] @ tilt[:, 2], [0, 0, 1], atol=1e-5)
    np.testing.assert_allclose(transform[:3, 3], 0, atol=1e-5)  # this floor is at zero
    np.testing.assert_allclose(
        transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6
    )
    np.testing.assert_array_equal(geometry.load_frame("test:0").camera_to_world, before)


def test_already_level_floor_returns_identity():
    pytest.importorskip("open3d")
    np.testing.assert_allclose(
        SceneAligner(pixel_stride=3).estimate(plane_geometry([0])), np.eye(4), atol=1e-5
    )


def test_the_origin_lands_on_the_floor():
    """A height in the output is a height above the floor, not above the camera."""
    pytest.importorskip("open3d")
    transform = SceneAligner(pixel_stride=3).estimate(plane_geometry([0.4]))
    np.testing.assert_allclose(transform[2, 3], -0.4, atol=FLOOR_STEP / 2)
    np.testing.assert_allclose(transform[:2, 3], 0, atol=1e-6)


def test_the_floor_moves_by_whole_association_voxels():
    """Part of a voxel would change which points share one, and so which match."""
    pytest.importorskip("open3d")
    assert FLOOR_STEP == inspect.signature(ObjectAssociator).parameters["voxel_size"].default
    # Not multiples of a voxel, and low enough to keep the camera above them.
    for floor in (0.4, 0.63, 0.88):
        offset = SceneAligner(pixel_stride=3).estimate(plane_geometry([floor]))[2, 3]
        assert offset / FLOOR_STEP == pytest.approx(round(offset / FLOOR_STEP))
        assert abs(offset + floor) <= FLOOR_STEP / 2 + 1e-3


def test_a_tabletop_puts_the_floor_under_the_room_not_on_the_table():
    """Levelling from a tabletop must not leave the room's contents below zero.

    The tilted scene here has no floor plane at all: its lowest geometry is the
    clutter standing on the floor, which is what the floor has to be read from.
    """
    tilt = Rotation.from_euler("xy", [12, -8], degrees=True).as_matrix()
    points, _, _ = tabletop_without_floor(tilt)

    # tabletop_without_floor tilts the scene by `tilt`, so `tilt.T` levels it.
    height = floor_height(points, tilt.T)

    levelled = points @ tilt.T[2] - height
    # Slightly above the true floor: the percentile counts the tabletop's points
    # too, and here the only things standing on the floor are sparse clutter.
    assert height == pytest.approx(0.0, abs=0.06)
    assert np.mean(levelled > 0) >= 0.98
    assert levelled.max() > 0.6  # the tabletop, still well above the floor


def test_ceiling_is_not_a_floor_or_camera_up_fallback():
    pytest.importorskip("open3d")
    with pytest.raises(ValueError, match="No convincing floor"):
        SceneAligner(pixel_stride=3).estimate(plane_geometry([3], ceiling=True))


def test_scene_points_below_table_preserve_floor_selection():
    pytest.importorskip("open3d")
    tilt = Rotation.from_euler("xy", [10, 15], degrees=True).as_matrix()
    # The tabletop occupies more input frames, so largest-plane alone is wrong.
    geometry = plane_geometry([0, 0.7, 0.7, 0.7], tilt=tilt, slopes=[0, 0.2, 0.2, 0.2])
    transform = SceneAligner(pixel_stride=3).estimate(geometry)
    np.testing.assert_allclose(transform[:3, :3] @ tilt[:, 2], [0, 0, 1], atol=1e-5)


def tabletop_without_floor(tilt):
    """A level tabletop with clutter under it and no floor plane, then tilted.

    The clutter is sparse and volumetric, so no plane can be fitted to it, but
    it puts 11% of the scene below the tabletop - more than any floor allows.
    """
    rng = np.random.default_rng(3)
    grid = np.mgrid[-0.8:0.8:0.02, -0.8:0.8:0.02].reshape(2, -1).T
    table = np.column_stack((grid, np.full(len(grid), 0.7)))
    clutter = rng.uniform([-0.8, -0.8, 0.0], [0.8, 0.8, 0.6], size=(800, 3))
    cameras = np.column_stack((rng.uniform(-1, 1, (20, 2)), np.full(20, 1.5)))
    points = np.vstack((table, clutter))
    return points @ tilt.T, cameras @ tilt.T, tilt @ np.array([0.0, 0.0, 1.0])


def test_a_tabletop_levels_a_scene_whose_floor_is_not_found():
    pytest.importorskip("open3d")
    tilt = Rotation.from_euler("xy", [12, -8], degrees=True).as_matrix()
    points, cameras, up = tabletop_without_floor(tilt)
    normal = SceneAligner()._floor_normal(points, cameras, up + [0.1, 0.0, 0.0])
    np.testing.assert_allclose(normal, up, atol=2e-3)


def floorless_room(tilt, walls, slope=15.0, lean=0.0):
    """A counter against walls, no floor plane, then tilted.

    By default the counter falls 15 degrees across its depth, as the one on a
    real clip did, and the walls stand upright. walls holds the directions they
    face, as headings in degrees; lean tips each one back by that many degrees.
    """
    rng = np.random.default_rng(5)
    grid = np.mgrid[-0.8:0.8:0.02, -0.8:0.8:0.02].reshape(2, -1).T
    rise_per_metre = np.tan(np.radians(slope))
    counter = np.column_stack((grid, 0.7 + rise_per_metre * grid[:, 0]))
    clutter = rng.uniform([-0.8, -0.8, 0.0], [0.8, 0.8, 0.6], size=(800, 3))
    along, rise = np.mgrid[-1.2:1.2:0.02, 0.0:2.2:0.02].reshape(2, -1)
    planes = []
    for heading in walls:
        facing = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading)), 0])
        side = np.array([-facing[1], facing[0], 0.0])
        tip = np.radians(lean)
        upward = np.cos(tip) * np.array([0, 0, 1.0]) - np.sin(tip) * facing
        planes.append(-1.4 * facing + along[:, None] * side + rise[:, None] * upward)
    cameras = np.column_stack((rng.uniform(-1, 1, (20, 2)), np.full(20, 1.5)))
    points = np.vstack((counter, clutter, *planes))
    return points @ tilt.T, cameras @ tilt.T, tilt @ np.array([0.0, 0.0, 1.0])


def test_walls_level_a_scene_whose_counter_slopes():
    pytest.importorskip("open3d")
    tilt = Rotation.from_euler("xy", [12, -8], degrees=True).as_matrix()
    points, cameras, up = floorless_room(tilt, walls=[0, 90])
    normal = SceneAligner()._floor_normal(points, cameras, up + [0.1, 0.0, 0.0])
    assert np.degrees(np.arccos(normal @ up)) < 1.0


def test_a_level_counter_levels_a_scene_whose_walls_lean():
    """Walls leaning 25 degrees put up 33 out; camera-up sides with the counter."""
    pytest.importorskip("open3d")
    tilt = Rotation.from_euler("xy", [12, -8], degrees=True).as_matrix()
    points, cameras, up = floorless_room(tilt, walls=[0, 90], slope=0.0, lean=25.0)
    normal = SceneAligner()._floor_normal(points, cameras, up + [0.1, 0.0, 0.0])
    assert np.degrees(np.arccos(normal @ up)) < 1.0


def test_walls_facing_one_way_leave_the_tabletop_to_level():
    pytest.importorskip("open3d")
    tilt = Rotation.from_euler("xy", [12, -8], degrees=True).as_matrix()
    points, cameras, up = floorless_room(tilt, walls=[0, 180])
    normal = SceneAligner()._floor_normal(points, cameras, up + [0.2, 0.0, 0.0])
    # The counter's own normal, 15 degrees off: there is nothing better to use.
    assert np.degrees(np.arccos(normal @ up)) == pytest.approx(15.0, abs=1.0)


def test_no_horizontal_surface_still_refuses_rather_than_guessing():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(4)
    clutter = rng.uniform([-1.0, -1.0, 0.0], [1.0, 1.0, 0.6], size=(900, 3))
    cameras = np.column_stack((rng.uniform(-1, 1, (20, 2)), np.full(20, 1.5)))
    with pytest.raises(ValueError, match="No convincing floor"):
        SceneAligner()._floor_normal(clutter, cameras, np.array([0.0, 0.0, 1.0]))


def room_geometry(yaw_degrees=0.0, walls=True):
    """A corner of a room: a floor and two perpendicular walls, turned by a yaw.

    Rays are intersected against bounded planes and the nearest hit kept, so the
    depth map is what a camera in that corner would actually see. The whole
    scene, camera included, is then turned about the vertical, which is the only
    thing the squaring is allowed to recover.
    """
    surfaces = [(np.array([0.0, 0.0, 1.0]), 0.0)]
    if walls:
        surfaces += [
            (np.array([1.0, 0.0, 0.0]), 2.0),
            (np.array([0.0, 1.0, 0.0]), 2.0),
        ]
    turn = Rotation.from_euler("z", yaw_degrees, degrees=True).as_matrix()

    forward = np.array([1.0, 1.0, -0.35])
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera = np.eye(4, dtype=np.float32)
    camera[:3, :3] = turn @ np.stack([right, down, forward], axis=1)
    camera[:3, 3] = turn @ np.array([-1.5, -1.5, 1.4])

    intrinsics = np.array([[110, 0, 60], [0, 110, 60], [0, 0, 1]], dtype=np.float32)
    v, u = np.mgrid[:120, :120]
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(intrinsics).T
    world_rays = rays @ camera[:3, :3].T
    origin = camera[:3, 3]

    best = np.full(world_rays.shape[:2], np.inf)
    for normal, offset in surfaces:
        normal = turn @ normal
        denominator = world_rays @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (offset - origin @ normal) / denominator
        point = world_rays * distance[:, :, None] + origin
        local = point @ turn  # back into the room's own axes
        inside = (distance > 0.2) & (np.abs(local[:, :, 0]) < 2.5)
        inside &= (np.abs(local[:, :, 1]) < 2.5)
        inside &= (local[:, :, 2] > -0.05) & (local[:, :, 2] < 2.2)
        best = np.where(inside & (distance < best), distance, best)
    depth = np.where(np.isfinite(best), best, np.nan).astype(np.float32)

    frame = FrameGeometry(
        "test:0",
        depth,
        np.ones((120, 120), dtype=np.float32),
        intrinsics.copy(),
        camera,
        np.eye(3, dtype=np.float32),
    )
    return SimpleNamespace(
        alignment_method="raw",
        geometry_frame_ids=("test:0",),
        load_frame=lambda frame_id: frame,
        load_pose=lambda frame_id: CameraPoseEstimate(
            frame_id, frame.camera_to_world.copy(), "measured"
        ),
    )


def recovered_yaw(transform):
    rotation = transform[:3, :3]
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])) % 90.0)


@pytest.mark.parametrize("yaw", [0.0, 12.0, 37.0, 61.0])
def test_walls_square_the_scene_up(yaw):
    pytest.importorskip("open3d")
    transform = SceneAligner(pixel_stride=2).estimate(room_geometry(yaw))

    # Turning the room by y should call for a turn of -y to undo it.
    expected = (-yaw) % 90.0
    error = abs(recovered_yaw(transform) - expected) % 90.0
    assert min(error, 90.0 - error) < 2.0


def test_squaring_turns_about_the_vertical_only():
    pytest.importorskip("open3d")
    geometry = room_geometry(30.0)
    squared = SceneAligner(pixel_stride=2).estimate(geometry)[:3, :3]
    levelled = SceneAligner(pixel_stride=2, square_to_walls=False).estimate(
        geometry
    )[:3, :3]

    added = squared @ levelled.T
    np.testing.assert_allclose(added[:, 2], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(added[2, :], [0, 0, 1], atol=1e-6)


def test_squaring_can_be_switched_off():
    pytest.importorskip("open3d")
    geometry = room_geometry(30.0)

    levelled = SceneAligner(pixel_stride=2, square_to_walls=False).estimate(geometry)

    # A plane fitted by RANSAC is not exact; this only has to be un-turned, and
    # its floor is already at zero give or take a millimetre of that same fit.
    np.testing.assert_allclose(levelled[:3, :3], np.eye(3), atol=1e-3)
    np.testing.assert_allclose(levelled[:3, 3], 0, atol=5e-3)


def test_a_scene_without_walls_is_levelled_but_not_turned():
    pytest.importorskip("open3d")
    transform = SceneAligner(pixel_stride=2).estimate(room_geometry(30.0, walls=False))

    np.testing.assert_allclose(transform, np.eye(4), atol=1e-3)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"wall_distance_threshold": 0}, "Wall plane tolerance"),
        ({"max_wall_tilt_degrees": 50}, "Wall tilt tolerance"),
        ({"min_wall_extent": -1}, "Minimum wall extent"),
        ({"min_corroborating_walls": 0}, "At least one wall"),
        ({"min_wall_agreement": 2.0}, "Wall agreement"),
    ],
)
def test_wall_settings_are_checked(settings, message):
    with pytest.raises(ValueError, match=message):
        SceneAligner(**settings)


def test_estimation_rejects_already_leveled_geometry():
    geometry = plane_geometry([0])
    geometry.alignment_method = "leveled"
    with pytest.raises(ValueError, match="raw geometry"):
        SceneAligner().estimate(geometry)


def test_insufficient_points_do_not_use_camera_up():
    geometry = plane_geometry([0])
    geometry.load_frame("test:0").depth[:] = np.nan
    with pytest.raises(ValueError, match="Not enough valid"):
        SceneAligner().estimate(geometry)


def test_alignment_is_applied_once_to_all_pose_methods(native_sequence, saved_run):
    raw = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    leveled = GeometryAdapter.from_saved_run(
        native_sequence, saved_run, world_alignment=transform
    )
    assert leveled.alignment_method == "leveled"
    assert raw.alignment_method == "raw"
    for frame_id in ("example:0", "example:2", "example:3", "example:4"):
        original = raw.load_pose(frame_id)
        aligned = leveled.load_pose(frame_id)
        np.testing.assert_allclose(
            aligned.camera_to_world, transform @ original.camera_to_world
        )
        assert aligned.method == original.method
    for frame_id in raw.geometry_frame_ids:
        original, aligned = raw.load_frame(frame_id), leveled.load_frame(frame_id)
        np.testing.assert_allclose(
            aligned.camera_to_world, leveled.load_pose(frame_id).camera_to_world
        )
        for field in ("depth", "confidence", "intrinsics", "rgb_to_geometry"):
            np.testing.assert_array_equal(
                getattr(original, field), getattr(aligned, field)
            )
        # Unproject through the returned pose, not a second leveling rotation.
        pixel = np.array([1, 1, 1], dtype=np.float32)
        camera = np.linalg.inv(aligned.intrinsics) @ pixel * aligned.depth[1, 1]
        original_world = original.camera_to_world @ np.r_[camera, 1]
        aligned_world = aligned.camera_to_world @ np.r_[camera, 1]
        np.testing.assert_allclose(aligned_world, transform @ original_world, atol=1e-6)
    expected = leveled.load_pose("example:2").camera_to_world.copy()
    transform[:] = 0
    np.testing.assert_array_equal(
        leveled.load_pose("example:2").camera_to_world, expected
    )


def test_leveled_reads_leave_cache_unchanged(native_sequence, saved_run):
    before = {
        str(p.relative_to(saved_run)): p.read_bytes()
        for p in saved_run.rglob("*")
        if p.is_file()
    }
    geometry = GeometryAdapter.from_saved_run(
        native_sequence, saved_run, world_alignment=np.eye(4)
    )
    geometry.load_frame("example:2")
    geometry.load_pose("example:3")
    after = {
        str(p.relative_to(saved_run)): p.read_bytes()
        for p in saved_run.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_a_floor_offset_is_accepted_and_applied(native_sequence, saved_run):
    """The leveling may drop the origin to the floor; that is not a stray translation."""
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.from_euler("x", 15, degrees=True).as_matrix()
    transform[2, 3] = -1.4

    leveled = GeometryAdapter.from_saved_run(
        native_sequence, saved_run, world_alignment=transform
    )

    raw = GeometryAdapter.from_saved_run(native_sequence, saved_run)
    for frame_id in raw.geometry_frame_ids:
        np.testing.assert_allclose(
            leveled.load_pose(frame_id).camera_to_world,
            transform @ raw.load_pose(frame_id).camera_to_world,
        )


@pytest.mark.parametrize("kind", ["translation", "scale", "reflection", "nan", "shape"])
def test_invalid_alignment_is_rejected(native_sequence, saved_run, kind):
    matrix = np.eye(4, dtype=np.float32)
    if kind == "translation":
        matrix[0, 3] = 1  # sideways: the leveling does not place the origin in the room
    elif kind == "scale":
        matrix[0, 0] = 2
    elif kind == "reflection":
        matrix[0, 0] = -1
    elif kind == "nan":
        matrix[0, 0] = np.nan
    else:
        matrix = np.eye(3)
    with pytest.raises(ValueError, match="Leveling"):
        GeometryAdapter.from_saved_run(
            native_sequence, saved_run, world_alignment=matrix
        )
