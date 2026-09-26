import json

import pytest

from video_world_state import manifests
from video_world_state.contracts import FrameInput, FrameSequence


def sequence(count=3, sequence_id="scene"):
    return FrameSequence(
        sequence_id,
        [
            FrameInput(f"{sequence_id}:{i}", i, i / 30, f"/frames/{i:05d}.jpg")
            for i in range(count)
        ],
    )


def test_written_manifest_reopens_and_lists_the_same_frames(tmp_path):
    frames = sequence()
    path = tmp_path / manifests.MANIFEST_NAME
    manifests.write(path, frames.sequence_id, manifests.frame_entries(frames), extra=7)

    manifest, entries = manifests.read(path, frames.sequence_id, label="Test")

    assert manifest["schema_version"] == manifests.SCHEMA_VERSION
    assert manifest["extra"] == 7
    assert manifests.matches_frames(entries, frames.frames)
    assert manifests.resolve_frames(entries, frames, label="Test") == frames.frames


def test_refuses_to_overwrite_an_existing_manifest(tmp_path):
    frames = sequence()
    path = tmp_path / manifests.MANIFEST_NAME
    manifests.write(path, frames.sequence_id, manifests.frame_entries(frames))
    with pytest.raises(FileExistsError):
        manifests.write(path, frames.sequence_id, manifests.frame_entries(frames))


def test_rejects_another_sequence_and_an_unknown_schema_version(tmp_path):
    frames = sequence()
    path = tmp_path / manifests.MANIFEST_NAME
    manifests.write(path, frames.sequence_id, manifests.frame_entries(frames))

    with pytest.raises(ValueError, match="sequence_id does not match"):
        manifests.read(path, "other-scene", label="Test")

    manifest = json.loads(path.read_text())
    manifest["schema_version"] = manifests.SCHEMA_VERSION + 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema_version"):
        manifests.read(path, frames.sequence_id, label="Test")


def test_caches_without_a_version_need_the_legacy_exception(tmp_path):
    """DA3 runs predate the version field, so only that path may accept them."""
    frames = sequence()
    path = tmp_path / manifests.MANIFEST_NAME
    manifests.write(path, frames.sequence_id, manifests.frame_entries(frames))
    manifest = json.loads(path.read_text())
    del manifest["schema_version"]
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="no schema_version"):
        manifests.read(path, frames.sequence_id, label="Test")

    manifest, entries = manifests.read(
        path, frames.sequence_id, label="Test", allow_missing_schema_version=True
    )
    assert manifests.matches_frames(entries, frames.frames)


def test_altered_frame_metadata_stops_matching(tmp_path):
    frames = sequence()
    entries = manifests.frame_entries(frames)
    assert manifests.matches_frames(entries, frames.frames)

    entries[1]["source_frame_index"] = 99
    assert not manifests.matches_frames(entries, frames.frames)
    with pytest.raises(ValueError, match="metadata does not match"):
        manifests.resolve_frames(entries, frames, label="Test")


def test_boolean_indices_are_not_accepted_as_frame_numbers(tmp_path):
    """JSON has no integer/bool distinction, so True must not pass for 1."""
    frames = sequence()
    entries = manifests.frame_entries(frames)
    entries[1]["source_frame_index"] = True
    assert not manifests.matches_frames(entries, frames.frames)


def test_resolve_accepts_a_subset_in_manifest_order(tmp_path):
    frames = sequence(count=5)
    subset = FrameSequence(frames.sequence_id, [frames.frames[4], frames.frames[1]])
    entries = manifests.frame_entries(subset)

    assert manifests.resolve_frames(entries, frames, label="Test") == subset.frames


def test_resolve_rejects_frames_outside_the_sequence(tmp_path):
    frames = sequence()
    entries = manifests.frame_entries(frames)
    entries[0]["frame_id"] = "scene:99"
    with pytest.raises(ValueError, match="not in the sequence"):
        manifests.resolve_frames(entries, frames, label="Test")
