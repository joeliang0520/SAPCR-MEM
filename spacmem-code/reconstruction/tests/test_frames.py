from dataclasses import replace
from pathlib import Path

import pytest

from video_world_state.contracts import FrameInput, FrameSequence
from video_world_state.frames import read_frame_folder, select_frames, validate_subset


def test_native_30fps_selection_preserves_ids_paths_and_native_input():
    native = FrameSequence(
        "room",
        [FrameInput(f"room:{i}", i, i / 30, Path(f"{i}.jpg")) for i in range(31)],
    )
    selected = select_frames(native)
    assert [f.source_frame_index for f in selected.frames] == [0, 6, 12, 18, 24, 30]
    assert len(native.frames) == 31
    assert selected.frames[1] is native.frames[6]
    validate_subset(native, selected)


@pytest.mark.parametrize(
    "change", ["id", "timestamp", "source_index", "path", "sequence_id", "order"]
)
def test_reject_changed_or_unrelated_subset(native_sequence, change):
    selected = select_frames(native_sequence)
    selected.frames = [replace(f) for f in selected.frames]
    if change == "id":
        selected.frames[0].frame_id = "unknown"
    elif change == "timestamp":
        selected.frames[0].timestamp_seconds += 0.01
    elif change == "source_index":
        selected.frames[0].source_frame_index += 1
    elif change == "path":
        selected.frames[0].rgb_path = Path("different.jpg")
    elif change == "sequence_id":
        selected.sequence_id = "different"
    else:
        selected.frames.reverse()
    with pytest.raises(ValueError):
        validate_subset(native_sequence, selected)


def test_a_frame_folder_reads_as_the_native_30fps_sequence(tmp_path):
    for index in range(3):
        (tmp_path / f"{index:06d}.jpg").write_bytes(b"")

    sequence = read_frame_folder(tmp_path, "scene0000_00")

    assert [frame.frame_id for frame in sequence.frames] == [
        "scene0000_00:0", "scene0000_00:1", "scene0000_00:2"
    ]
    assert sequence.frames[2].timestamp_seconds == 2 / 30
    assert sequence.frames[1].rgb_path == tmp_path / "000001.jpg"


def test_a_gap_in_the_frame_folder_is_refused(tmp_path):
    for index in (0, 2):
        (tmp_path / f"{index:06d}.jpg").write_bytes(b"")
    with pytest.raises(ValueError):
        read_frame_folder(tmp_path, "scene0000_00")
