"""One reader and writer for the frame manifests every saved cache carries.

Each cache directory (DA3 geometry, raw SegVGGT chunks, cleaned masks) stores the
frames it was built from, so reopening can prove the cache belongs to the
FrameSequence in hand. Backends add their own keys beside these; only the
shared shape lives here, so outside tooling has one format to follow.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import FrameInput, FrameSequence

SCHEMA_VERSION = 1
MANIFEST_NAME = "frame_manifest.json"
_REQUIRED_FIELDS = ("frame_id", "source_frame_index", "timestamp_seconds")


def frame_entries(
    sequence: FrameSequence,
    rgb_sizes: list[tuple[int, int]] | None = None,
    rgb_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Describe a sequence's frames, optionally with the images actually read."""
    entries = []
    for position, frame in enumerate(sequence.frames):
        entry: dict[str, Any] = {
            "frame_id": frame.frame_id,
            "source_frame_index": frame.source_frame_index,
            "timestamp_seconds": frame.timestamp_seconds,
            "rgb_path": str(
                frame.rgb_path if rgb_paths is None else rgb_paths[position]
            ),
        }
        if rgb_sizes is not None:
            entry["rgb_size_hw"] = list(rgb_sizes[position])
        entries.append(entry)
    return entries


def write(
    path: Path,
    sequence_id: str,
    entries: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    """Write a manifest with the shared header, refusing to overwrite one."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sequence_id": sequence_id,
        "frames": entries,
        **extra,
    }
    with Path(path).open("x") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    return manifest


def read(
    path: Path,
    sequence_id: str,
    *,
    label: str,
    allow_missing_schema_version: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a manifest and check its header against the sequence being opened.

    Caches written before manifests carried a version are only accepted with
    allow_missing_schema_version; a present version must be the current one.
    """
    with Path(path).open() as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict) or manifest.get("sequence_id") != sequence_id:
        raise ValueError(f"{label} manifest sequence_id does not match the sequence")
    version = manifest.get("schema_version")
    if version is None:
        if not allow_missing_schema_version:
            raise ValueError(f"{label} manifest has no schema_version")
    elif version != SCHEMA_VERSION:
        raise ValueError(
            f"{label} manifest schema_version {version!r} is not {SCHEMA_VERSION}"
        )
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{label} manifest must contain a non-empty ordered frames list")
    for entry in entries:
        if not isinstance(entry, dict) or not set(_REQUIRED_FIELDS).issubset(entry):
            raise ValueError(f"{label} manifest frame is missing required fields")
    return manifest, entries


def matches_frames(entries: list[dict[str, Any]], frames: list[FrameInput]) -> bool:
    """Whether a manifest lists exactly these frames, in order."""
    if len(entries) != len(frames):
        return False
    return all(
        entry["frame_id"] == frame.frame_id
        and type(entry["source_frame_index"]) is int
        and entry["source_frame_index"] == frame.source_frame_index
        and type(entry["timestamp_seconds"]) in (int, float)
        and np.isclose(
            entry["timestamp_seconds"], frame.timestamp_seconds, atol=1e-9, rtol=0
        )
        for entry, frame in zip(entries, frames)
    )


def resolve_frames(
    entries: list[dict[str, Any]],
    sequence: FrameSequence,
    *,
    label: str,
    noun: str = "sequence",
) -> list[FrameInput]:
    """Look each manifest frame up in a sequence, keeping the manifest's order.

    Used where the manifest lists a subset, such as DA3 anchors within the
    native sequence. Metadata must agree with the sequence exactly.
    """
    known = {frame.frame_id: frame for frame in sequence.frames}
    resolved = []
    for entry in entries:
        frame_id = entry["frame_id"]
        if not isinstance(frame_id, str) or frame_id not in known:
            raise ValueError(f"{label} manifest frame is not in the {noun}: {frame_id}")
        frame = known[frame_id]
        if not matches_frames([entry], [frame]):
            raise ValueError(
                f"{label} manifest metadata does not match {noun}: {frame_id}"
            )
        resolved.append(frame)
    return resolved
