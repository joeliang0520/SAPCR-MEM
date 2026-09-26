"""Package reconstructed scenes into one world_state.json per benchmark episode.

    python tools/build_episode_package.py --benchmark final_release \
        --scenes scenes --output packages --line-of-sight

--scenes holds one tools/reconstruct_scene.py output per scene. Each episode's
public/episode.json lists its clips in order; the package gives every object
of those clips an episode-wide ID and lists, for every processed frame, the
camera pose and the objects visible in it. Nothing from the benchmark's hidden
answers or ScanNet's annotations is read here.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_world_state.frames import read_frame_folder, select_frames  # noqa: E402
from video_world_state.geometry import GeometryAdapter  # noqa: E402
from video_world_state.labels import label_with_candidates  # noqa: E402


# Line of sight: an object also counts as visible in a frame when enough of
# its stored points project into the image in front of the measured depth.
SIGHT_POINTS = 300
SIGHT_FRACTION = 0.25
SIGHT_AREA = 0.005
OCCLUSION_TOLERANCE = 0.1  # metres nearer than a point before it counts as hidden


def frame_number(frame_id: str) -> int:
    """Return the clip-local video frame number from a frame ID."""
    return int(frame_id.rsplit(":", 1)[1])


def package_object(held: dict, sequence_id: str, context: dict) -> dict:
    """Convert one working object into the compact downstream representation."""
    counts = {str(label): int(count) for label, count in held["label_counts"].items()}
    scores = {
        str(label): float(score)
        for label, score in held.get("label_scores", counts).items()
    }
    label, aliases, candidates = label_with_candidates(scores)
    return {
        "id": f"{sequence_id}/{held['object_id']}",
        **context,
        "label": label,
        "label_aliases": aliases,
        "label_candidates": candidates,
        "label_counts": counts,
        "centroid": held["centroid_xyz"],
        "box_min": held["bounds_min_xyz"],
        "box_max": held["bounds_max_xyz"],
    }


def in_line_of_sight(points: np.ndarray, frame) -> bool:
    """Whether a camera frame sees enough of an object to count it as visible.

    points are the object's stored world points and frame its FrameGeometry.
    At least SIGHT_FRACTION of an evenly spaced sample must project inside the
    image, in front of the camera and not behind the frame's measured depth,
    and the visible part must span at least SIGHT_AREA of the image.
    """
    sample = points[:: max(1, int(np.ceil(len(points) / SIGHT_POINTS)))]
    world_to_camera = np.linalg.inv(np.asarray(frame.camera_to_world, dtype=float))
    camera = sample @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera[:, 2]
    pixel = camera @ np.asarray(frame.intrinsics, dtype=float).T
    height, width = frame.depth.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = pixel[:, 0] / depth, pixel[:, 1] / depth
    seen = (depth > 0) & (u >= 0) & (u < width - 0.5) & (v >= 0) & (v < height - 0.5)
    column = np.round(np.where(seen, u, 0)).astype(int)
    row = np.round(np.where(seen, v, 0)).astype(int)
    surface = frame.depth[row, column]
    seen &= ~(np.isfinite(surface) & (surface < depth - OCCLUSION_TOLERANCE))
    if seen.mean() < SIGHT_FRACTION or seen.sum() < 2:
        return False
    return np.ptp(u[seen]) * np.ptp(v[seen]) >= SIGHT_AREA * width * height


def clip_state(
    clip: dict, run: Path, frames_directory: Path, *, line_of_sight: bool = False
) -> tuple[list, list]:
    sequence_id = clip["clip_id"].removesuffix("-full")
    native = read_frame_folder(frames_directory, sequence_id)
    geometry = GeometryAdapter.from_saved_run(
        native,
        run / "geometry",
        world_alignment=np.load(run / "world_alignment.npy"),
    )
    saved = json.loads((run / "objects/objects.json").read_text())
    frame_of = {
        observation["observation_id"]: observation["frame_id"]
        for observation in saved["observations"]
    }
    context = {
        "clip_id": clip["clip_id"],
        "day": clip["day"],
        "room": clip["room_gloss"],
    }

    objects = []
    seen_in: dict[str, list[str]] = collections.defaultdict(list)
    for held in saved["objects"]:
        packaged = package_object(held, sequence_id, context)
        objects.append(packaged)
        for observation_id in held["observation_ids"]:
            seen_in[frame_of[observation_id]].append(packaged["id"])

    points = {}
    if line_of_sight:
        stored = np.load(run / "objects/object_points.npz")
        points = {
            f"{sequence_id}/{held['object_id']}": stored[held["points"]].astype(float)
            for held in saved["objects"]
        }

    sightings = []
    for frame in select_frames(native).frames:
        pose = geometry.load_pose(frame.frame_id).camera_to_world
        visible = set(seen_in.get(frame.frame_id, []))
        if line_of_sight:
            frame_geometry = geometry.load_frame(frame.frame_id)
            visible |= {
                object_id
                for object_id, object_points in points.items()
                if in_line_of_sight(object_points, frame_geometry)
            }
        sightings.append(
            {
                "clip_id": context["clip_id"],
                "day": context["day"],
                "frame": frame_number(frame.frame_id),
                "camera_to_world": np.round(
                    np.asarray(pose, dtype=float), 5
                ).tolist(),
                "visible": sorted(visible),
            }
        )
    return objects, sightings


def frames_directory(run: Path) -> Path:
    """The folder of native RGB frames the clip's geometry was computed from."""
    manifest = json.loads((run / "geometry/frame_manifest.json").read_text())
    return Path(manifest["frames"][0]["rgb_path"]).parent


def build_package(
    episode_path: Path,
    clips_directory: Path,
    *,
    line_of_sight: bool = False,
) -> dict:
    """Build a complete or explicitly partial episode world state.

    With line_of_sight, an object is also listed as visible in frames whose
    camera could see it, not only in frames where the segmenter detected it.
    """
    episode = json.loads(Path(episode_path).read_text())
    objects, sightings, clips, pending = [], [], [], []
    for clip in episode["ordered_clips"]:
        sequence_id = clip["clip_id"].removesuffix("-full")
        run = Path(clips_directory) / sequence_id
        entry = {
            key: clip[key]
            for key in ("clip_id", "day", "session_index", "room_gloss")
        }
        if not (run / "objects/run_complete.json").is_file():
            pending.append(entry)
            continue
        clip_objects, clip_sightings = clip_state(
            clip, run, frames_directory(run), line_of_sight=line_of_sight
        )
        objects.extend(clip_objects)
        sightings.extend(clip_sightings)
        notes_path = run / "notes.json"
        notes = json.loads(notes_path.read_text()) if notes_path.is_file() else None
        clips.append(
            {
                **entry,
                "objects": len(clip_objects),
                "sightings": len(clip_sightings),
                **({"notes": notes} if notes else {}),
            }
        )

    return {
        "episode_id": episode["episode_id"],
        "conventions": {
            "frame": "the clip's own video frame number, 30 fps, counted from 0",
            "frame_step": 6,
            "frame_note": "only every 6th frame (5 fps) was processed",
            "coordinates": (
                "per clip, in metres: levelled to the floor and squared to the "
                "walls; clips never share coordinates"
            ),
            "camera_to_world": "4x4, maps camera coordinates into the clip frame",
            "visible": (
                "objects detected by the segmenter in that frame, plus objects "
                "the camera could see there by a line-of-sight test"
                if line_of_sight
                else "objects detected by the segmenter in that frame, not a "
                "line-of-sight test"
            ),
            "label_aliases": (
                "up to two credible runner-ups projected from label_candidates; "
                "weak alternatives, not asserted synonyms"
            ),
            "label_candidates": (
                "all runner-up labels ranked by accumulated relative support; "
                "support is not a calibrated probability"
            ),
            "id": "<scene>/<object id>, unique across the episode",
        },
        "clips": clips,
        "pending_clips": pending,
        "objects": objects,
        "sightings": sightings,
    }


SEGMENTATION = {
    "segvggt": "SegVGGT (ScanNet200 checkpoint, RGB only), 50-frame chunks",
    "scannet": "ScanNet 2D instance masks and labels (oracle ablation); "
    "object identity from ScanNet instances",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="the benchmark release, searched for */public/episode.json")
    parser.add_argument(
        "--scenes", type=Path, required=True, help="reconstructed scenes, one folder each"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--line-of-sight", action="store_true",
                        help="also list objects the camera could see, not only detected ones")
    parser.add_argument("--episodes", nargs="*", help="episode IDs; default: every episode")
    arguments = parser.parse_args()

    backends = {
        json.loads(path.read_text())["backend"]
        for path in arguments.scenes.glob("*/provenance.json")
    }
    if len(backends) != 1:
        raise SystemExit(f"Expected scenes from one segmentation backend, found {sorted(backends)}")
    backend = backends.pop()

    failed = []
    for path in sorted(arguments.benchmark.rglob("public/episode.json")):
        episode = json.loads(path.read_text())
        if arguments.episodes and episode["episode_id"] not in arguments.episodes:
            continue
        target = arguments.output / f"episode_{episode['episode_id']}_{backend}"
        if (target / "world_state.json").is_file():
            continue
        state = build_package(path, arguments.scenes, line_of_sight=arguments.line_of_sight)
        if state["pending_clips"]:
            failed.append(episode["episode_id"])
            print(f"{episode['episode_id']}: scenes not reconstructed: "
                  f"{[clip['clip_id'] for clip in state['pending_clips']]}")
            continue
        state["segmentation"] = SEGMENTATION[backend]
        target.mkdir(parents=True)
        (target / "world_state.json").write_text(json.dumps(state, indent=1) + "\n")
        (target / "episode.json").write_text(path.read_text())
        print(f"{episode['episode_id']}: {len(state['clips'])} clips, "
              f"{len(state['objects'])} objects -> {target}")
    if failed:
        raise SystemExit(f"{len(failed)} episodes not packaged")


if __name__ == "__main__":
    main()
