"""Select model input frames without changing native IDs or timestamps."""

from pathlib import Path

import numpy as np

from .contracts import FrameInput, FrameSequence

SCANNET_FPS = 30.0


def read_frame_folder(
    directory: Path, sequence_id: str, fps: float = SCANNET_FPS
) -> FrameSequence:
    """The native sequence of a folder of frames named 000000.jpg, 000001.jpg, ...

    Frame IDs are "<sequence_id>:<index>" and timestamps index / fps, which is
    how every cache and package refers to a frame. A gap in the numbering is
    refused rather than guessed across.
    """
    paths = sorted(Path(directory).glob("*.jpg"))
    if not paths or any(path.name != f"{i:06d}.jpg" for i, path in enumerate(paths)):
        raise ValueError(f"Expected contiguous frames 000000.jpg, ... in {directory}")
    return FrameSequence(
        sequence_id,
        [FrameInput(f"{sequence_id}:{i}", i, i / fps, path) for i, path in enumerate(paths)],
    )


def select_frames(sequence: FrameSequence, fps: float = 5) -> FrameSequence:
    """Take the first frame in each time bin, relative to the first timestamp.

    Returns a subset for both geometry and segmentation. Gaps stay empty; rates
    above the input rate select every frame once. Does not read images, change
    the native sequence, renumber frames or interpolate anything.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    timestamps = frame_timestamps(sequence)
    bins = np.floor((timestamps - timestamps[0]) * fps + 1e-9)
    selected = np.r_[True, np.diff(bins) > 0]
    return FrameSequence(
        sequence.sequence_id,
        [frame for frame, keep in zip(sequence.frames, selected) if keep],
    )


def validate_subset(native: FrameSequence, selected: FrameSequence) -> None:
    """Reject a changed, reordered or unrelated selection before inference."""
    frame_timestamps(native)
    frame_timestamps(selected)
    if native.sequence_id != selected.sequence_id:
        raise ValueError("Selected sequence_id does not match native input")
    positions = {frame.frame_id: i for i, frame in enumerate(native.frames)}
    last_position = -1
    for frame in selected.frames:
        position = positions.get(frame.frame_id, -1)
        if position <= last_position:
            raise ValueError("Selected frames must be an ordered native subset")
        original = native.frames[position]
        if (
            frame.source_frame_index != original.source_frame_index
            or frame.timestamp_seconds != original.timestamp_seconds
            or frame.rgb_path != original.rgb_path
        ):
            raise ValueError(f"Selected frame metadata changed: {frame.frame_id}")
        last_position = position


def frame_timestamps(sequence: FrameSequence) -> np.ndarray:
    """Return finite, strictly ordered times; reject empty/duplicate-ID input."""
    if not sequence.frames:
        raise ValueError("FrameSequence is empty")
    if len({frame.frame_id for frame in sequence.frames}) != len(sequence.frames):
        raise ValueError("Duplicate frame ID")
    timestamps = np.asarray(
        [frame.timestamp_seconds for frame in sequence.frames], dtype=float
    )
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Frame timestamps must be finite and strictly increasing")
    return timestamps
