import json

import numpy as np

from tools.build_episode_package import (
    build_package,
    frame_number,
    in_line_of_sight,
    package_object,
)
from video_world_state.contracts import FrameGeometry


def held(label_counts, label_scores=None):
    result = {
        "object_id": "object:0007",
        "label_counts": label_counts,
        "centroid_xyz": [1, 2, 3],
        "bounds_min_xyz": [0, 1, 2],
        "bounds_max_xyz": [2, 3, 4],
        "observation_ids": [],
    }
    if label_scores is not None:
        result["label_scores"] = label_scores
    return result


def test_package_object_exposes_label_aliases_without_hiding_votes():
    packaged = package_object(
        held({"printer": 56, "copier": 10, "paper": 1, "shelf": 6}),
        "scene0136_01",
        {"clip_id": "scene0136_01-full", "day": 4, "room": "office"},
    )

    assert packaged["label"] == "printer"
    assert packaged["label_aliases"] == ["copier", "shelf"]
    assert [item["label"] for item in packaged["label_candidates"]] == [
        "copier",
        "shelf",
        "paper",
    ]
    assert packaged["label_counts"] == {
        "printer": 56,
        "copier": 10,
        "paper": 1,
        "shelf": 6,
    }


def test_scored_runner_ups_feed_both_candidates_and_legacy_aliases():
    packaged = package_object(
        held(
            {"chair": 4},
            {"chair": 4.0, "stool": 2.4, "office chair": 0.4},
        ),
        "scene",
        {},
    )

    assert packaged["label"] == "chair"
    assert packaged["label_aliases"] == ["stool"]
    assert [item["label"] for item in packaged["label_candidates"]] == [
        "stool",
        "office chair",
    ]


def test_frame_number_uses_the_source_video_index():
    assert frame_number("scene0136_01:648") == 648


def test_a_clip_not_yet_reconstructed_is_listed_as_pending(tmp_path):
    clip = {"clip_id": "scene0136_01-full", "day": 1, "session_index": 0,
            "room_gloss": "office"}
    episode = tmp_path / "episode.json"
    episode.write_text(json.dumps({"episode_id": "e", "ordered_clips": [clip]}))

    state = build_package(episode, tmp_path / "clips")

    pending = [entry["clip_id"] for entry in state["pending_clips"]]
    assert pending == ["scene0136_01-full"]
    assert state["objects"] == [] and state["sightings"] == []


def camera(depth=5.0):
    """A 40 x 40 depth grid seen by a camera at the origin looking along +z."""
    return FrameGeometry(
        frame_id="scene:0",
        depth=np.full((40, 40), depth, dtype=np.float32),
        confidence=np.ones((40, 40), dtype=np.float32),
        intrinsics=np.array([[20, 0, 20], [0, 20, 20], [0, 0, 1]], dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float32),
        rgb_to_geometry=np.eye(3, dtype=np.float32),
    )


def cube(centre, side=1.0):
    grid = np.linspace(-side / 2, side / 2, 8)
    return np.array([[x, y, z] for x in grid for y in grid for z in grid]) + centre


def test_an_object_in_front_of_the_camera_is_in_line_of_sight():
    assert in_line_of_sight(cube([0, 0, 3]), camera())


def test_hidden_outside_or_tiny_objects_are_not():
    assert not in_line_of_sight(cube([0, 0, 3]), camera(depth=1.0))
    assert not in_line_of_sight(cube([10, 0, 3]), camera())
    assert not in_line_of_sight(cube([0, 0, 3], side=0.05), camera())
