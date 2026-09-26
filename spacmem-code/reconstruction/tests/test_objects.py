"""Synthetic arrays and fake adapters only; no model or GPU runs."""

import json
from dataclasses import replace

import numpy as np
import pytest

from video_world_state.contracts import (
    FrameGeometry,
    LabelCandidate,
    ObjectObservation3D,
    PersistentObject,
    SegmentationObservation,
)
from video_world_state.objects import (
    IdentityAssociator,
    FragmentMerger,
    LiftedObservation,
    MaskCarver,
    ObjectAssociator,
    ObjectConstructionResult,
    ObjectConstructionWriter,
    ObjectConstructor,
    ObservationLifter,
    SkipDiagnostic,
    drop_thinly_seen_objects,
    read_object_construction,
)


def geometry(
    shape=(2, 3), rgb_shape=(4, 6), depth=1.0, confidence=1.0, frame_id="example:0"
):
    height, width = shape
    return FrameGeometry(
        frame_id=frame_id,
        depth=np.full(shape, depth, dtype=np.float32),
        confidence=np.full(shape, confidence, dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        rgb_to_geometry=np.diag(
            [width / rgb_shape[1], height / rgb_shape[0], 1]
        ).astype(np.float32),
    )


def observation(mask, label="chair", confidence=0.8, frame_id="example:0", hint=None):
    return SegmentationObservation(
        f"{frame_id}:obs", frame_id, mask, label, confidence, hint
    )


def world_grid(shape=(2, 3)):
    """Distinct, easily recognised world coordinates per geometry pixel."""
    rows, columns = np.indices(shape)
    return np.stack(
        [columns, rows, np.zeros(shape)], axis=2
    ).astype(np.float32)


def lifter(threshold=0.0, **settings):
    return ObservationLifter(threshold, **settings)


# --- mask resampling -------------------------------------------------------


def test_coverage_above_below_and_exactly_at_the_threshold():
    """Halving the width centres geometry pixel t on RGB pixel 2t.

    The mapping has no half-pixel offset, so a footprint is two RGB pixels wide
    and straddles its neighbours instead of averaging a 2x2 block. Geometry
    column 0's footprint also hangs half off the image, and coverage is
    measured over the part that has pixels.
    """
    mask = np.zeros((1, 6), bool)
    mask[0, 0] = True  # column 0 covers all 1.0 of its 1.5 in-image width
    mask[0, 2] = True  # column 1 covers 1.0 of 2.0: exactly the threshold
    mask[0, 5] = True  # column 2 covers 0.5 of 2.0

    lifted = lifter().lift(
        observation(mask), geometry(shape=(1, 3), rgb_shape=(1, 6)), world_grid((1, 3))
    )

    assert lifted is not None
    assert lifted.points_xyz.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_non_integer_scale_splits_rgb_pixels_by_area():
    """Three geometry columns over four RGB columns: footprints are 4/3 wide."""
    mask = np.zeros((1, 4), bool)
    mask[0, 0] = True
    frame = geometry(shape=(1, 3), rgb_shape=(1, 4))

    lifted = lifter().lift(observation(mask), frame, world_grid((1, 3)))

    # Footprints are 4/3 RGB columns wide: column 0 takes all of RGB column 0
    # and 1/6 of column 1, so 0.857 of its area is masked. Columns 1 and 2
    # reach RGB column 1 at the earliest and take nothing from column 0.
    assert lifted is not None
    assert lifted.points_xyz.tolist() == [[0.0, 0.0, 0.0]]


def test_mask_orientation_is_preserved_on_a_different_grid():
    mask = np.zeros((4, 6), bool)
    mask[2:4, 0:2] = True  # bottom left in RGB

    lifted = lifter().lift(observation(mask), geometry(), world_grid())

    assert lifted is not None
    # world_grid stores (column, row): bottom left is row 1, column 0.
    assert lifted.points_xyz.tolist() == [[0.0, 1.0, 0.0]]


def test_a_mask_on_the_wrong_grid_is_refused():
    with pytest.raises(ValueError, match="original RGB grid"):
        lifter().lift(observation(np.ones((4, 4), bool)), geometry(), world_grid())


def test_transforms_that_are_not_a_resize_are_refused():
    frame = geometry()
    frame.rgb_to_geometry[0, 2] = 3.0
    with pytest.raises(ValueError, match="resize-only"):
        lifter().lift(observation(np.ones((4, 6), bool)), frame, world_grid())


def test_world_points_must_match_the_depth_grid():
    with pytest.raises(ValueError, match="height x width x 3"):
        lifter().lift(observation(np.ones((4, 6), bool)), geometry(), world_grid((3, 3)))


# --- point selection -------------------------------------------------------


def test_invalid_depth_coordinates_and_confidence_are_dropped():
    frame = geometry(shape=(1, 4), rgb_shape=(1, 4))
    frame.depth[0] = [1.0, 0.0, np.nan, 1.0]
    frame.confidence[0] = [1.0, 1.0, 1.0, np.nan]
    points = world_grid((1, 4))

    assert lifter().lift(observation(np.ones((1, 4), bool)), frame, points) is not None
    kept = lifter().lift(observation(np.ones((1, 4), bool)), frame, points)
    assert kept.points_xyz.tolist() == [[0.0, 0.0, 0.0]]

    points[0, 0, 2] = np.inf
    assert lifter().lift(observation(np.ones((1, 4), bool)), frame, points) is None


def test_the_confidence_threshold_is_the_supplied_one():
    frame = geometry(shape=(1, 2), rgb_shape=(1, 2))
    frame.confidence[0] = [0.4, 0.6]
    mask = np.ones((1, 2), bool)

    assert lifter(0.6).lift(observation(mask), frame, world_grid((1, 2))).observation.point_count == 1
    assert lifter(0.7).lift(observation(mask), frame, world_grid((1, 2))) is None
    assert lifter(0.0).lift(observation(mask), frame, world_grid((1, 2))).observation.point_count == 2


def test_segmentation_confidence_never_filters():
    frame = geometry(shape=(1, 2), rgb_shape=(1, 2))
    low = observation(np.ones((1, 2), bool), confidence=0.01)

    assert lifter(0.5).lift(low, frame, world_grid((1, 2))).observation.point_count == 2


def test_a_lifter_needs_an_explicit_finite_threshold():
    with pytest.raises(TypeError):
        ObservationLifter()
    with pytest.raises(ValueError, match="finite"):
        ObservationLifter(np.nan)
    with pytest.raises(ValueError, match="coverage"):
        ObservationLifter(0.5, mask_coverage_threshold=0)
    with pytest.raises(ValueError, match="percentiles"):
        ObservationLifter(0.5, lower_box_percentile=60, upper_box_percentile=40)


# --- summaries -------------------------------------------------------------


def test_an_outlier_moves_the_bounds_but_not_the_points():
    frame = geometry(shape=(1, 5), rgb_shape=(1, 5))
    points = np.zeros((1, 5, 3), dtype=np.float32)
    points[0, :, 0] = [0, 1, 2, 3, 100]

    lifted = lifter(0.0, lower_box_percentile=0, upper_box_percentile=75).lift(
        observation(np.ones((1, 5), bool)), frame, points
    )

    assert lifted.observation.point_count == 5
    assert lifted.points_xyz[:, 0].max() == 100  # kept for association
    assert lifted.observation.bounds_max_xyz[0] == 3  # robust bound excludes it
    assert lifted.observation.centroid_xyz[0] == pytest.approx(21.2)


def test_percentiles_interpolate_between_small_point_sets():
    frame = geometry(shape=(1, 2), rgb_shape=(1, 2))
    points = np.zeros((1, 2, 3), dtype=np.float32)
    points[0, :, 0] = [0, 10]

    lifted = lifter(0.0, lower_box_percentile=25, upper_box_percentile=75).lift(
        observation(np.ones((1, 2), bool)), frame, points
    )

    assert lifted.observation.bounds_min_xyz[0] == pytest.approx(2.5)
    assert lifted.observation.bounds_max_xyz[0] == pytest.approx(7.5)


def test_the_record_copies_segmentation_identity_and_adds_no_object_id():
    lifted = lifter().lift(
        observation(np.ones((4, 6), bool), label="table", hint="track:1"),
        geometry(),
        world_grid(),
    )

    assert lifted.observation.observation_id == "example:0:obs"
    assert lifted.observation.frame_id == "example:0"
    assert lifted.observation.label == "table"
    assert lifted.observation.track_hint == "track:1"
    assert lifted.observation.segmentation_confidence == 0.8
    assert not hasattr(lifted.observation, "object_id")


# --- the frame loop --------------------------------------------------------


class FakeGeometry:
    """Serves prepared frames and counts how often world points are derived."""

    def __init__(self, frames, points):
        self._frames = frames
        self._points = points
        self.alignment_method = "leveled"
        self.world_point_calls = []

    @property
    def geometry_frame_ids(self):
        return tuple(self._frames)

    def load_frame(self, frame_id):
        return self._frames[frame_id]

    def world_points(self, frame_geometry):
        self.world_point_calls.append(frame_geometry.frame_id)
        return self._points[frame_geometry.frame_id]


class FakeSegmentation:
    def __init__(self, observations):
        self._observations = observations

    @property
    def segmentation_frame_ids(self):
        return tuple(self._observations)

    def load_frame(self, frame_id):
        return self._observations[frame_id]


class CountingAssociator:
    """Test-only stand-in: collects every observation into one growing object."""

    def __init__(self):
        self.frames = []

    def associate_frame(self, lifted_observations, objects):
        self.frames.append([item.observation.observation_id for item in lifted_observations])
        if not lifted_observations:
            return
        if not objects:
            objects.append(
                PersistentObject(
                    "object:0",
                    np.zeros((0, 3), dtype=np.float32),
                    {},
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    [],
                )
            )
        held = objects[0]
        for item in lifted_observations:
            held.points_xyz = np.vstack([held.points_xyz, item.points_xyz])
            label = item.observation.label
            held.label_counts[label] = held.label_counts.get(label, 0) + 1
            held.observation_ids.append(item.observation.observation_id)


def two_frame_adapters(second_frame_observations=None):
    frame_ids = ["example:0", "example:5"]
    frames = {frame_id: geometry(frame_id=frame_id) for frame_id in frame_ids}
    points = {frame_id: world_grid() for frame_id in frame_ids}
    masks = {
        "example:0": [observation(np.ones((4, 6), bool))],
        "example:5": (
            [observation(np.ones((4, 6), bool), frame_id="example:5")]
            if second_frame_observations is None
            else second_frame_observations
        ),
    }
    return FakeGeometry(frames, points), FakeSegmentation(masks)


def test_the_frame_loop_derives_one_world_map_per_frame_and_carries_objects(tmp_path):
    ready_geometry, ready_segmentation = two_frame_adapters()
    associator = CountingAssociator()

    result = ObjectConstructor.run(
        ready_geometry,
        ready_segmentation,
        lifter(),
        associator,
        tmp_path / "objects",
    )

    assert ready_geometry.world_point_calls == ["example:0", "example:5"]
    assert associator.frames == [["example:0:obs"], ["example:5:obs"]]
    assert [item.observation_id for item in result.observations] == [
        "example:0:obs",
        "example:5:obs",
    ]
    assert len(result.objects) == 1
    assert result.objects[0].label_counts == {"chair": 2}
    assert len(result.objects[0].points_xyz) == 12

    metadata, restored = read_object_construction(tmp_path / "objects")
    assert metadata["frame_ids"] == ["example:0", "example:5"]
    assert metadata["world_coordinates"] == "leveled"
    assert metadata["lifter"]["mask_coverage_threshold"] == 0.5
    assert metadata["associator"]["class"] == "CountingAssociator"
    assert "frames" not in metadata["associator"], "state is not a setting"
    assert metadata["merger"] is None and metadata["min_observations"] == 1
    assert [item.object_id for item in restored.objects] == ["object:0"]


def test_empty_frames_still_reach_the_associator(tmp_path):
    ready_geometry, ready_segmentation = two_frame_adapters(second_frame_observations=[])
    associator = CountingAssociator()

    result = ObjectConstructor.run(
        ready_geometry, ready_segmentation, lifter(), associator, tmp_path / "objects"
    )

    assert associator.frames == [["example:0:obs"], []]
    assert len(result.observations) == 1


def test_observations_with_no_valid_points_are_skipped_not_recorded(tmp_path):
    ready_geometry, ready_segmentation = two_frame_adapters()
    empty = np.zeros((4, 6), bool)
    ready_segmentation._observations["example:5"] = [
        observation(empty, frame_id="example:5")
    ]

    result = ObjectConstructor.run(
        ready_geometry,
        ready_segmentation,
        lifter(),
        CountingAssociator(),
        tmp_path / "objects",
    )

    assert [item.observation_id for item in result.observations] == ["example:0:obs"]
    assert len(result.skipped_observations) == 1
    assert result.skipped_observations[0].frame_id == "example:5"
    assert result.skipped_observations[0].reason


def test_differing_frame_ids_are_refused(tmp_path):
    ready_geometry, ready_segmentation = two_frame_adapters()
    del ready_segmentation._observations["example:5"]

    with pytest.raises(ValueError, match="same ordered frame IDs"):
        ObjectConstructor.run(
            ready_geometry,
            ready_segmentation,
            lifter(),
            CountingAssociator(),
            tmp_path / "objects",
        )


def lifted(points, observation_id, frame_id="example:0", label="chair"):
    """A lifted observation carrying exactly the points a test cares about."""
    points = np.asarray(points, dtype=np.float32)
    return LiftedObservation(
        ObjectObservation3D(
            observation_id=observation_id,
            frame_id=frame_id,
            label=label,
            centroid_xyz=points.mean(axis=0).astype(np.float32),
            bounds_min_xyz=points.min(axis=0).astype(np.float32),
            bounds_max_xyz=points.max(axis=0).astype(np.float32),
            point_count=len(points),
            segmentation_confidence=0.8,
            track_hint=None,
        ),
        points,
    )


def blob(centre, count=40, spread=0.15, seed=0):
    """A small cloud of points around one place, distinct voxel by voxel."""
    generator = np.random.default_rng(seed)
    return np.array(centre, dtype=float) + generator.uniform(
        -spread, spread, size=(count, 3)
    )


def associator(**settings):
    defaults = {"min_points": 4, "min_new_object_points": 4}
    return ObjectAssociator(**{**defaults, **settings})


def with_hint(item, hint):
    """The same lifted observation, carrying a segmenter track identity."""
    return LiftedObservation(replace(item.observation, track_hint=hint), item.points_xyz)


def test_the_same_object_seen_twice_becomes_one_object():
    objects = []
    associator().associate_frame([lifted(blob([0, 0, 0]), "a")], objects)
    associator().associate_frame(
        [lifted(blob([0.03, 0.02, 0], seed=1), "b", frame_id="example:1")], objects
    )

    assert len(objects) == 1
    assert objects[0].observation_ids == ["a", "b"]
    assert objects[0].label_counts == {"chair": 2}


def test_two_distant_observations_stay_separate():
    objects = []
    associator().associate_frame(
        [lifted(blob([0, 0, 0]), "a"), lifted(blob([5, 5, 0], seed=1), "b")], objects
    )

    assert len(objects) == 2
    assert [held.observation_ids for held in objects] == [["a"], ["b"]]


def test_one_object_absorbs_at_most_one_observation_per_frame():
    objects = []
    associator().associate_frame([lifted(blob([0, 0, 0]), "first")], objects)
    overlapping = [
        lifted(blob([0.02, 0, 0], seed=2), "same_frame_a", frame_id="example:1"),
        lifted(blob([0.04, 0, 0], seed=3), "same_frame_b", frame_id="example:1"),
    ]
    associator().associate_frame(overlapping, objects)

    absorbed = [item for held in objects for item in held.observation_ids]
    assert absorbed.count("same_frame_a") + absorbed.count("same_frame_b") <= 2
    assert len(objects[0].observation_ids) == 2, "one object took one observation"


def test_an_observation_between_two_objects_is_left_unassigned():
    objects = []
    associator().associate_frame(
        [lifted(blob([0, 0, 0]), "left"), lifted(blob([1, 0, 0], seed=1), "right")],
        objects,
    )
    straddling = np.vstack([blob([0, 0, 0], seed=4), blob([1, 0, 0], seed=5)])

    # Ordinary creation threshold: an observation rejected as ambiguous must be
    # dropped, not turned into a third object. Raising the threshold here would
    # hide exactly the behaviour under test.
    associator().associate_frame(
        [lifted(straddling, "ambiguous", frame_id="example:1")], objects
    )

    assert len(objects) == 2, "an undecidable observation must not start an object"
    for held in objects:
        assert "ambiguous" not in held.observation_ids


def test_a_small_unmatched_observation_does_not_start_an_object():
    objects = []
    associator(min_points=4, min_new_object_points=500).associate_frame(
        [lifted(blob([0, 0, 0], count=40), "tiny")], objects
    )

    assert objects == []


def test_object_points_are_voxel_reduced_not_accumulated():
    objects = []
    dense = blob([0, 0, 0], count=400, spread=0.02)
    associator().associate_frame([lifted(dense, "a")], objects)

    assert len(objects[0].points_xyz) < len(dense)


def test_a_matching_observation_must_clear_the_overlap_floor():
    objects = []
    associator().associate_frame([lifted(blob([0, 0, 0]), "a")], objects)
    barely = np.vstack([blob([0, 0, 0], seed=6)[:4], blob([0.6, 0, 0], seed=7)])

    associator(min_overlap=0.9, min_new_object_points=10_000).associate_frame(
        [lifted(barely, "b", frame_id="example:1")], objects
    )

    assert objects[0].observation_ids == ["a"]


def test_a_tracked_observation_rejoins_its_object_without_geometry():
    """With trust_track_hints, a tracked observation joins its track's object."""
    objects = []
    associator_with_hints = associator(trust_track_hints=True)
    associator_with_hints.associate_frame(
        [with_hint(lifted(blob([0, 0, 0]), "a"), "chunk_000:7")], objects
    )
    # The same track, later, somewhere geometry would never have matched it.
    associator_with_hints.associate_frame(
        [with_hint(
            lifted(blob([9, 9, 9], seed=1), "b", frame_id="example:1"), "chunk_000:7"
        )],
        objects,
    )

    assert len(objects) == 1
    assert objects[0].observation_ids == ["a", "b"]


def test_a_different_track_is_not_evidence_of_a_different_object():
    """A differing hint does not keep observations apart; geometry still decides."""
    objects = []
    keeper = associator(trust_track_hints=True)
    keeper.associate_frame(
        [with_hint(lifted(blob([0, 0, 0]), "a"), "chunk_000:1")], objects
    )
    keeper.associate_frame(
        [with_hint(
            lifted(blob([0.02, 0, 0], seed=1), "b", frame_id="example:1"),
            "chunk_001:9",
        )],
        objects,
    )

    assert len(objects) == 1
    assert objects[0].observation_ids == ["a", "b"]


def test_a_hinted_match_still_takes_only_one_observation_per_frame():
    objects = []
    keeper = associator(trust_track_hints=True)
    keeper.associate_frame(
        [with_hint(lifted(blob([0, 0, 0]), "a"), "chunk_000:5")], objects
    )
    rival = [
        with_hint(lifted(blob([0.02, 0, 0], seed=1), "tracked", frame_id="example:1"),
                  "chunk_000:5"),
        lifted(blob([0.04, 0, 0], seed=2), "untracked", frame_id="example:1"),
    ]
    keeper.associate_frame(rival, objects)

    assert objects[0].observation_ids == ["a", "tracked"], "the tracked one wins"
    assert "untracked" not in objects[0].observation_ids


def test_geometry_decides_when_hints_are_not_trusted():
    objects = []
    keeper = associator()
    keeper.associate_frame(
        [with_hint(lifted(blob([0, 0, 0]), "a"), "chunk_000:7")], objects
    )
    keeper.associate_frame(
        [with_hint(
            lifted(blob([9, 9, 9], seed=1), "b", frame_id="example:1"), "chunk_000:7"
        )],
        objects,
    )

    assert len(objects) == 2, "without hints, distance decides and these are far apart"


def test_associator_settings_are_checked():
    with pytest.raises(ValueError, match="at least as wide"):
        ObjectAssociator(match_radius=1.0, gate_radius=0.1)
    with pytest.raises(ValueError, match="in .0, 1."):
        ObjectAssociator(min_overlap=0)
    with pytest.raises(ValueError, match="at least as many points"):
        ObjectAssociator(min_points=100, min_new_object_points=10)


def test_the_real_associator_builds_objects_through_the_frame_loop(tmp_path):
    ready_geometry, ready_segmentation = two_frame_adapters()
    output = tmp_path / "objects"

    result = ObjectConstructor.run(
        ready_geometry,
        ready_segmentation,
        lifter(),
        ObjectAssociator(min_points=1, min_new_object_points=1),
        output,
    )

    assert output.exists()
    assert result.objects, "the frame loop reached the associator and it built objects"
    claimed = [item for held in result.objects for item in held.observation_ids]
    assert len(claimed) == len(set(claimed)), "an observation belongs to one object"


# --- mask edge cleaning ----------------------------------------------------


def test_depth_far_from_the_mask_core_is_dropped():
    shape = (5, 5)
    depth = np.full(shape, 1.0, dtype=np.float32)
    depth[0, 0] = 6.0  # a mask edge spilling onto the wall behind
    frame = FrameGeometry(
        frame_id="example:0",
        depth=depth,
        confidence=np.ones(shape, dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        rgb_to_geometry=np.eye(3, dtype=np.float32),
    )
    points = world_grid(shape)
    mask = np.ones(shape, dtype=bool)

    cleaned = lifter(erosion_pixels=1, min_core_pixels=1, depth_tolerance=0.1)
    kept = cleaned.lift(observation(mask), frame, points)
    uncleaned = lifter(erosion_pixels=0)
    everything = uncleaned.lift(observation(mask), frame, points)

    assert kept.observation.point_count == everything.observation.point_count - 1
    assert not np.any(np.all(kept.points_xyz == points[0, 0], axis=1))


def test_a_core_too_small_to_erode_still_cleans_from_the_whole_mask():
    """The fallback widens the reference; it does not switch cleaning off."""
    shape = (3, 3)
    depth = np.full(shape, 1.0, dtype=np.float32)
    depth[0, 0] = 6.0
    frame = FrameGeometry(
        frame_id="example:0",
        depth=depth,
        confidence=np.ones(shape, dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        rgb_to_geometry=np.eye(3, dtype=np.float32),
    )
    mask = np.ones(shape, dtype=bool)

    # Radius 2 on a 3 x 3 mask erodes to nothing, so the reference falls back
    # to every covered pixel. The far pixel is still an outlier against it.
    kept = lifter(erosion_pixels=2, min_core_pixels=20, depth_tolerance=0.1).lift(
        observation(mask), frame, world_grid(shape)
    )

    assert kept.observation.point_count == 8
    assert not np.any(np.all(kept.points_xyz == world_grid(shape)[0, 0], axis=1))


def test_cleaning_can_be_switched_off():
    shape = (3, 3)
    depth = np.full(shape, 1.0, dtype=np.float32)
    depth[0, 0] = 6.0
    frame = FrameGeometry(
        frame_id="example:0",
        depth=depth,
        confidence=np.ones(shape, dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        rgb_to_geometry=np.eye(3, dtype=np.float32),
    )

    kept = lifter(erosion_pixels=0).lift(
        observation(np.ones(shape, dtype=bool)), frame, world_grid(shape)
    )

    assert kept.observation.point_count == 9


def test_cleaning_settings_are_checked():
    with pytest.raises(ValueError, match="Erosion radius"):
        lifter(erosion_pixels=-1)
    with pytest.raises(ValueError, match="Depth tolerance"):
        lifter(depth_tolerance=0)
    with pytest.raises(ValueError, match="Core percentiles"):
        lifter(core_percentiles=(90.0, 10.0))


# --- merging and filtering after the loop -----------------------------------


def built(object_id, points, observation_ids, label="chair"):
    points = np.asarray(points, dtype=np.float32)
    return PersistentObject(
        object_id=object_id,
        points_xyz=points,
        label_counts={label: len(observation_ids)},
        centroid_xyz=points.mean(axis=0).astype(np.float32),
        bounds_min_xyz=points.min(axis=0).astype(np.float32),
        bounds_max_xyz=points.max(axis=0).astype(np.float32),
        observation_ids=list(observation_ids),
    )


def seen_in(observation_id, frame_id, track=None):
    return ObjectObservation3D(
        observation_id, frame_id, "chair",
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32), 5, 0.9, track,
    )


def looks(**by_observation):
    """Unit appearance vectors, as the merger expects them."""
    return {
        observation_id: np.asarray(vector, dtype=float) / np.linalg.norm(vector)
        for observation_id, vector in by_observation.items()
    }


def test_fragments_never_seen_together_are_fused():
    objects = [
        built("object:0000", blob([0, 0, 0]), ["a"]),
        built("object:0001", blob([0.05, 0, 0], seed=1), ["b"]),
    ]
    observations = [seen_in("a", "f:0", "t1"), seen_in("b", "f:9", "t1")]

    merged = FragmentMerger().merge(objects, observations)

    assert len(merged) == 1
    assert sorted(merged[0].observation_ids) == ["a", "b"]
    assert merged[0].label_counts == {"chair": 2}


def test_fragments_seen_in_one_frame_together_are_never_fused():
    """Two things visible at once are two things, however close they sit."""
    objects = [
        built("object:0000", blob([0, 0, 0]), ["a"]),
        built("object:0001", blob([0.05, 0, 0], seed=1), ["b"]),
    ]
    observations = [seen_in("a", "f:0"), seen_in("b", "f:0")]

    merged = FragmentMerger().merge(objects, observations)

    assert len(merged) == 2


def test_distant_fragments_are_not_fused():
    objects = [
        built("object:0000", blob([0, 0, 0]), ["a"]),
        built("object:0001", blob([9, 9, 9], seed=1), ["b"]),
    ]
    observations = [seen_in("a", "f:0"), seen_in("b", "f:9")]

    assert len(FragmentMerger().merge(objects, observations)) == 2


def test_a_chain_of_compatible_fragments_becomes_one_object():
    """Whether a pair is examined first must not decide the outcome."""
    objects = [
        built("object:0000", blob([0, 0, 0]), ["a"]),
        built("object:0001", blob([0.05, 0, 0], seed=1), ["b"]),
        built("object:0002", blob([0.10, 0, 0], seed=2), ["c"]),
    ]
    observations = [
        seen_in("a", "f:0", "t1"), seen_in("b", "f:5", "t1"),
        seen_in("c", "f:9", "t1"),
    ]

    merged = FragmentMerger().merge(objects, observations)

    assert len(merged) == 1
    assert sorted(merged[0].observation_ids) == ["a", "b", "c"]


def test_a_bridge_cannot_merge_two_fragments_seen_together():
    """The veto must survive chaining: A and C share a frame, B overlaps both."""
    objects = [
        built("object:0000", blob([0, 0, 0]), ["a"]),
        built("object:0001", blob([0.05, 0, 0], seed=1), ["b"]),
        built("object:0002", blob([0.10, 0, 0], seed=2), ["c"]),
    ]
    observations = [
        seen_in("a", "f:0", "t1"), seen_in("b", "f:5", "t1"),
        seen_in("c", "f:0", "t1"),
    ]

    merged = FragmentMerger().merge(objects, observations)

    grouped = [set(held.observation_ids) for held in merged]
    assert not any({"a", "c"} <= group for group in grouped), (
        "a and c appear in the same frame and must never share an object"
    )


def test_merged_objects_are_renumbered_without_gaps():
    objects = [
        built("object:0007", blob([0, 0, 0]), ["a"]),
        built("object:0009", blob([9, 9, 9], seed=1), ["b"]),
    ]
    merged = FragmentMerger().merge(
        objects, [seen_in("a", "f:0"), seen_in("b", "f:9")]
    )

    assert [held.object_id for held in merged] == ["object:0000", "object:0001"]


def test_merger_settings_are_checked():
    with pytest.raises(ValueError, match="Match radius"):
        FragmentMerger(match_radius=0)
    with pytest.raises(ValueError, match="Overlap threshold"):
        FragmentMerger(min_overlap=1.5)


def test_thinly_seen_objects_are_dropped_and_renumbered():
    objects = [
        built("object:0000", blob([0, 0, 0]), ["a", "b", "c", "d"]),
        built("object:0001", blob([5, 5, 5], seed=1), ["e", "f"]),
        built("object:0002", blob([9, 9, 9], seed=2), ["g", "h", "i", "j", "k"]),
    ]

    kept = drop_thinly_seen_objects(objects, 4)

    assert [held.object_id for held in kept] == ["object:0000", "object:0001"]
    assert [len(held.observation_ids) for held in kept] == [4, 5]


def test_a_minimum_of_one_keeps_everything():
    objects = [built("object:0000", blob([0, 0, 0]), ["a"])]

    assert len(drop_thinly_seen_objects(objects, 1)) == 1
    with pytest.raises(ValueError, match="at least one"):
        drop_thinly_seen_objects(objects, 0)


def test_the_frame_loop_runs_both_refinements_before_saving(tmp_path):
    ready_geometry, ready_segmentation = two_frame_adapters()

    result = ObjectConstructor.run(
        ready_geometry,
        ready_segmentation,
        lifter(),
        ObjectAssociator(min_points=1, min_new_object_points=1),
        tmp_path / "objects",
        merger=FragmentMerger(),
        min_observations=2,
    )

    assert all(len(held.observation_ids) >= 2 for held in result.objects)
    metadata = json.loads(
        (tmp_path / "objects" / "objects.json").read_text()
    )["metadata"]
    assert metadata["merger"]["class"] == "FragmentMerger"
    assert metadata["min_observations"] == 2


def test_dropping_an_object_leaves_its_observations_unresolved(tmp_path):
    """Filtering changes which objects are saved, never the record of what was seen."""
    ready_geometry, ready_segmentation = two_frame_adapters()

    everything = ObjectConstructor.run(
        ready_geometry, ready_segmentation, lifter(),
        ObjectAssociator(min_points=1, min_new_object_points=1),
        tmp_path / "all",
    )
    filtered = ObjectConstructor.run(
        ready_geometry, ready_segmentation, lifter(),
        ObjectAssociator(min_points=1, min_new_object_points=1),
        tmp_path / "filtered", min_observations=99,
    )

    assert filtered.objects == []
    assert len(filtered.observations) == len(everything.observations)

# --- storage ---------------------------------------------------------------


def record(observation_id, frame_id="example:0", label="chair"):
    return ObjectObservation3D(
        observation_id,
        frame_id,
        label,
        np.array([1, 2, 3], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.array([2, 4, 6], dtype=np.float32),
        7,
        0.8,
        None,
    )


def persistent(object_id, observation_ids, points=None):
    return PersistentObject(
        object_id,
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        if points is None
        else points,
        {"chair": len(observation_ids)},
        np.array([2.5, 3.5, 4.5], dtype=np.float32),
        np.array([1, 2, 3], dtype=np.float32),
        np.array([4, 5, 6], dtype=np.float32),
        list(observation_ids),
    )


def test_association_accumulates_primary_and_runner_up_label_evidence():
    associator = ObjectAssociator(
        min_points=1,
        min_new_object_points=1,
        voxel_size=0.05,
    )
    points = np.array([[0, 0, 0], [0.01, 0, 0]], dtype=np.float32)
    first = replace(
        record("a", label="chair"),
        centroid_xyz=np.zeros(3, dtype=np.float32),
        label_candidates=(LabelCandidate("stool", 0.5),),
    )
    second = replace(
        record("b", frame_id="example:5", label="stool"),
        centroid_xyz=np.zeros(3, dtype=np.float32),
        label_candidates=(LabelCandidate("chair", 0.25),),
    )
    objects = []

    associator.associate_frame([LiftedObservation(first, points)], objects)
    associator.associate_frame([LiftedObservation(second, points)], objects)

    assert len(objects) == 1
    assert objects[0].label_counts == {"chair": 1, "stool": 1}
    assert objects[0].label_scores == {"chair": 1.25, "stool": 1.5}


def saved(tmp_path, result, metadata=None):
    output = tmp_path / "objects"
    ObjectConstructionWriter(metadata or {"world_coordinates": "leveled"}).save(
        result, output
    )
    return output


def test_written_records_and_points_read_back_unchanged(tmp_path):
    first_object = persistent("object:0", ["a"])
    first_object.label_scores = {"chair": 1.0, "stool": 0.4}
    first_record = replace(
        record("a"),
        label_candidates=(LabelCandidate("stool", 0.4),),
    )
    result = ObjectConstructionResult(
        [first_object, persistent("object:1", [])],
        [first_record, record("b")],
        [SkipDiagnostic("c", "example:5", "no points")],
    )

    metadata, restored = read_object_construction(saved(tmp_path, result))

    assert metadata == {"world_coordinates": "leveled"}
    assert [item.object_id for item in restored.objects] == ["object:0", "object:1"]
    assert np.array_equal(restored.objects[0].points_xyz, result.objects[0].points_xyz)
    assert restored.objects[0].label_counts == {"chair": 1}
    assert restored.objects[0].label_scores == {"chair": 1.0, "stool": 0.4}
    assert restored.objects[0].observation_ids == ["a"]
    assert [item.observation_id for item in restored.observations] == ["a", "b"]
    assert restored.observations[0].point_count == 7
    assert restored.observations[0].label_candidates == (
        LabelCandidate("stool", 0.4),
    )
    assert np.array_equal(restored.observations[0].centroid_xyz, [1, 2, 3])
    assert restored.skipped_observations == result.skipped_observations


def test_an_empty_result_is_a_valid_saved_run(tmp_path):
    metadata, restored = read_object_construction(
        saved(tmp_path, ObjectConstructionResult([], [], []))
    )

    assert restored.objects == [] and restored.observations == []
    assert metadata == {"world_coordinates": "leveled"}


def test_observation_points_are_never_saved(tmp_path):
    result = ObjectConstructionResult([persistent("object:0", ["a"])], [record("a")], [])
    output = saved(tmp_path, result)

    with np.load(output / "object_points.npz", allow_pickle=False) as arrays:
        assert list(arrays) == ["points_000000"]
    assert "points_xyz" not in (output / "objects.json").read_text()


def test_existing_output_is_refused(tmp_path):
    result = ObjectConstructionResult([], [], [])
    output = saved(tmp_path, result)
    with pytest.raises(FileExistsError):
        ObjectConstructionWriter({}).save(result, output)


def test_broken_references_are_refused_before_writing(tmp_path):
    output = tmp_path / "objects"

    with pytest.raises(ValueError, match="unknown observation"):
        ObjectConstructionWriter({}).save(
            ObjectConstructionResult([persistent("object:0", ["missing"])], [], []),
            output,
        )
    assert not output.exists()

    with pytest.raises(ValueError, match="two objects"):
        ObjectConstructionWriter({}).save(
            ObjectConstructionResult(
                [persistent("object:0", ["a"]), persistent("object:1", ["a"])],
                [record("a")],
                [],
            ),
            output,
        )

    with pytest.raises(ValueError, match="Repeated observation ID"):
        ObjectConstructionWriter({}).save(
            ObjectConstructionResult([], [record("a"), record("a")], []), output
        )

    with pytest.raises(ValueError, match="recorded and skipped"):
        ObjectConstructionWriter({}).save(
            ObjectConstructionResult(
                [], [record("a")], [SkipDiagnostic("a", "example:0", "no points")]
            ),
            output,
        )

    with pytest.raises(ValueError, match="N x 3 points"):
        ObjectConstructionWriter({}).save(
            ObjectConstructionResult(
                [persistent("object:0", [], points=np.zeros((2, 2), dtype=np.float32))],
                [],
                [],
            ),
            output,
        )


def test_an_unfinished_directory_is_not_readable_as_complete(tmp_path):
    output = saved(
        tmp_path, ObjectConstructionResult([persistent("object:0", [])], [], [])
    )
    (output / "run_complete.json").unlink()

    with pytest.raises(FileNotFoundError):
        read_object_construction(output)

    marker = output / "run_complete.json"
    marker.write_text(json.dumps({"object_count": 99}))
    with pytest.raises(ValueError, match="completion marker"):
        read_object_construction(output)


def test_overlap_alone_does_not_merge():
    """Geometric overlap alone also fuses distinct objects, so it never merges by itself."""
    objects = [
        built("object:0000", blob([0, 0, 0]), ["a"]),
        built("object:0001", blob([0.05, 0, 0], seed=1), ["b"]),
    ]
    observations = [seen_in("a", "f:0"), seen_in("b", "f:9")]

    assert len(FragmentMerger().merge(objects, observations)) == 2


def test_appearance_can_confirm_a_merge_no_track_connects():
    """A tracker that loses an object renames it; appearance still reaches it."""
    objects = [
        built("object:0000", blob([0, 0, 0]), ["a"]),
        built("object:0001", blob([0.05, 0, 0], seed=1), ["b"]),
    ]
    observations = [seen_in("a", "f:0", "t1"), seen_in("b", "f:9", "t2")]

    merged = FragmentMerger().merge(
        objects, observations, looks(a=[1.0, 0.0], b=[0.96, 0.28])
    )

    assert len(merged) == 1


def test_unalike_fragments_are_left_apart():
    objects = [
        built("object:0000", blob([0, 0, 0]), ["a"]),
        built("object:0001", blob([0.05, 0, 0], seed=1), ["b"]),
    ]
    observations = [seen_in("a", "f:0", "t1"), seen_in("b", "f:9", "t2")]

    merged = FragmentMerger().merge(
        objects, observations, looks(a=[1.0, 0.0], b=[0.0, 1.0])
    )

    assert len(merged) == 2


def test_a_shared_track_confirms_without_any_appearance():
    """The fallback with no descriptors is the stricter signal, not the looser."""
    objects = [
        built("object:0000", blob([0, 0, 0]), ["a"]),
        built("object:0001", blob([0.05, 0, 0], seed=1), ["b"]),
    ]
    observations = [seen_in("a", "f:0", "t1"), seen_in("b", "f:9", "t1")]

    assert len(FragmentMerger().merge(objects, observations, {})) == 1


def test_a_stray_point_no_longer_decides_the_box():
    """Mask edges bleed onto the background; a box is decided by its extremes."""
    from video_world_state.objects import tighten_boxes

    body = np.random.default_rng(0).normal(0, 0.05, size=(200, 3))
    halo = np.array([[3.0, 3.0, 3.0]])
    held = built("object:0000", np.vstack([body, halo]), ["a"])

    before = held.bounds_max_xyz.copy()
    after = tighten_boxes([held])[0]

    assert before.max() > 2.0, "the stray point should reach the untrimmed box"
    assert after.bounds_max_xyz.max() < 1.0
    # Identity was decided on these points and must not move.
    assert len(after.points_xyz) == len(held.points_xyz)
    assert np.array_equal(after.centroid_xyz, held.centroid_xyz)


def test_tightening_leaves_a_small_object_alone():
    """Too few points to judge a neighbourhood means no trimming, not a crash."""
    from video_world_state.objects import tighten_boxes

    held = built("object:0000", np.zeros((3, 3)), ["a"])
    assert len(tighten_boxes([held])) == 1


def surface(object_id, centre, observation_ids, label="table", spread=0.6):
    """A fragment wide enough to count as part of a large surface."""
    return built(object_id, blob(centre, spread=spread, seed=len(object_id)),
                 observation_ids, label=label)


def test_two_pieces_of_one_large_surface_are_fused_although_seen_together():
    """A table too wide for one view returns as two masks in the same frame."""
    objects = [
        surface("object:0000", [0, 0, 0], ["a"]),
        surface("object:0001", [0.1, 0, 0], ["b"]),
    ]
    observations = [seen_in("a", "f:0", "t1"), seen_in("b", "f:0", "t1")]

    merged = FragmentMerger().merge(objects, observations)

    assert len(merged) == 1
    assert sorted(merged[0].observation_ids) == ["a", "b"]


def test_small_things_of_the_same_kind_seen_together_stay_apart():
    """The exemption is for surfaces, not for everything sharing a label."""
    objects = [
        surface("object:0000", [0, 0, 0], ["a"], spread=0.15),
        surface("object:0001", [0.05, 0, 0], ["b"], spread=0.15),
    ]
    observations = [seen_in("a", "f:0", "t1"), seen_in("b", "f:0", "t1")]

    assert len(FragmentMerger().merge(objects, observations)) == 2


def test_neighbouring_surfaces_of_different_kinds_stay_apart():
    """A counter beside a cabinet is two things, and both are large."""
    objects = [
        surface("object:0000", [0, 0, 0], ["a"], label="counter"),
        surface("object:0001", [0.1, 0, 0], ["b"], label="cabinet"),
    ]
    observations = [seen_in("a", "f:0", "t1"), seen_in("b", "f:0", "t1")]

    assert len(FragmentMerger().merge(objects, observations)) == 2


def test_the_large_surface_size_must_be_positive():
    with pytest.raises(ValueError, match="large-surface size"):
        FragmentMerger(large_surface_metres=0)


# --- label-gated matching ----------------------------------------------------


def test_same_label_only_keeps_a_different_label_apart():
    for gated, expected in ((False, 1), (True, 2)):
        objects = []
        associator(same_label_only=gated).associate_frame([lifted(blob([0, 0, 0]), "a")], objects)
        associator(same_label_only=gated).associate_frame(
            [lifted(blob([0.03, 0, 0], seed=1), "b", frame_id="example:1", label="table")],
            objects,
        )
        assert len(objects) == expected


def test_same_label_only_ignores_a_rival_of_another_label():
    objects = []
    associator(same_label_only=True).associate_frame(
        [
            lifted(blob([0, 0, 0]), "bench"),
            lifted(blob([1, 0, 0], seed=1), "table", label="table"),
        ],
        objects,
    )
    straddling = np.vstack([blob([0, 0, 0], seed=4), blob([1, 0, 0], seed=5)])
    associator(same_label_only=True).associate_frame(
        [lifted(straddling, "between", frame_id="example:1")], objects
    )

    assert objects[0].observation_ids == ["bench", "between"]


# --- carving -----------------------------------------------------------------


def carving_scene(frames=("example:0", "example:5"), hidden=False):
    """Nine points seen at the mask's corner and one stray point beside it."""
    body = np.array([[0.1 + 0.05 * i, 0.2, 1.0] for i in range(9)])
    stray = np.array([[2.2, 1.2, 1.0]])
    held = PersistentObject(
        "object:0",
        np.vstack([body, stray]).astype(np.float32),
        {"chair": len(frames)},
        np.zeros(3, np.float32),
        np.zeros(3, np.float32),
        np.zeros(3, np.float32),
        [f"{frame_id}:obs" for frame_id in frames],
    )
    mask = np.zeros((4, 6), bool)
    mask[:2, :2] = True
    frame_geometry = {}
    for frame_id in frames:
        frame_geometry[frame_id] = geometry(frame_id=frame_id)
        if hidden:
            frame_geometry[frame_id].depth[1, 2] = 0.5
    ready_geometry = FakeGeometry(frame_geometry, {})
    ready_segmentation = FakeSegmentation(
        {frame_id: [observation(mask, frame_id=frame_id)] for frame_id in frames}
    )
    return held, ready_geometry, ready_segmentation


def test_a_point_outside_the_objects_masks_is_carved():
    held, ready_geometry, ready_segmentation = carving_scene()
    (carved,) = MaskCarver().carve([held], ready_geometry, ready_segmentation)

    assert len(carved.points_xyz) == 9
    np.testing.assert_allclose(carved.centroid_xyz, [0.3, 0.2, 1.0], atol=1e-6)
    assert carved.observation_ids == held.observation_ids


def test_a_hidden_point_or_a_single_vote_is_not_carved():
    for scene in (carving_scene(hidden=True), carving_scene(frames=("example:0",))):
        held, ready_geometry, ready_segmentation = scene
        (carved,) = MaskCarver().carve([held], ready_geometry, ready_segmentation)
        assert len(carved.points_xyz) == 10


def test_carving_never_leaves_fewer_than_eight_points():
    held, ready_geometry, ready_segmentation = carving_scene()
    held.points_xyz = held.points_xyz[5:]
    (carved,) = MaskCarver().carve([held], ready_geometry, ready_segmentation)

    assert len(carved.points_xyz) == 5


def test_carver_settings_are_checked():
    with pytest.raises(ValueError):
        MaskCarver(occlusion_tolerance=-1)
    with pytest.raises(ValueError):
        MaskCarver(min_votes=0)


def test_identity_joins_by_hint_alone_and_never_across_hints():
    """Two far-apart masks of one instance join; two overlapping instances stay apart."""
    rng = np.random.default_rng(0)
    cube = rng.uniform(0, 0.5, (800, 3))
    far = cube + [5.0, 0, 0]
    associator = IdentityAssociator()
    objects = []
    associator.associate_frame([with_hint(lifted(cube, "a:0"), "gt:1"),
                                with_hint(lifted(cube + 0.01, "b:0"), "gt:2")], objects)
    associator.associate_frame([with_hint(lifted(far, "a:1", frame_id="example:1"), "gt:1")], objects)

    assert [o.observation_ids for o in objects] == [["a:0", "a:1"], ["b:0"]]


def test_identity_refuses_observations_without_an_identity():
    with pytest.raises(ValueError, match="identity"):
        IdentityAssociator().associate_frame([lifted(np.zeros((300, 3)) + np.arange(300)[:, None] * 0.01, "a:0")], [])
