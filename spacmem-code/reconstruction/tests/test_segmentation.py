import numpy as np
from PIL import Image
import pytest

from video_world_state.contracts import (
    FrameInput,
    FrameSequence,
    LabelCandidate,
    SegmentationObservation,
)
from video_world_state.segmentation import (
    SavedSegmentationBackend,
    SegmentationAdapter,
    SegmentationCacheWriter,
)


@pytest.fixture
def selected(tmp_path):
    frames = []
    for i in range(4):
        path = tmp_path / f"source {100 - i}.png"
        Image.new("RGB", (6, 4), (i, i, i)).save(path)
        frames.append(FrameInput(f"room:{6*i+2}", 6 * i + 2, i / 5, path))
    return FrameSequence("room", frames)


def observation(frame_id, number, label, *, hint=None, candidates=()):
    mask = np.zeros((4, 6), dtype=bool)
    mask[number % 4, number : number + 2] = True
    return SegmentationObservation(
        f"room:obs:{frame_id}:{number}", frame_id, mask, label, 0.9, hint,
        tuple(LabelCandidate(name, score) for name, score in candidates),
    )


@pytest.fixture
def cache(selected, tmp_path):
    output = tmp_path / "cleaned"
    writer = SegmentationCacheWriter(selected, output)
    for position, frame in enumerate(selected.frames):
        items = []
        if position < 3:
            items = [
                observation(frame.frame_id, 0, "desk", hint="a", candidates=[("table", 0.8)]),
                observation(frame.frame_id, 3, "chair", hint="b"),
            ]
        writer.write_frame(frame.frame_id, items)
    writer.finish({"backend": "test"})
    return output


def snapshot(directory):
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }


def test_cache_reopens_without_rgb_and_never_writes(cache, selected, tmp_path):
    before = snapshot(cache)
    # Reads must not reopen the original frames.
    for frame in selected.frames:
        frame.rgb_path = tmp_path / "unavailable.jpg"
    adapter = SegmentationAdapter.from_saved_run(selected, cache)
    assert adapter.segmentation_frame_ids == tuple(f.frame_id for f in selected.frames)
    first = adapter.load_frame("room:2")
    assert [item.label for item in first] == ["desk", "chair"]
    assert [item.track_hint for item in first] == ["a", "b"]
    assert [(c.label, c.score) for c in first[0].label_candidates] == [
        ("table", pytest.approx(0.8))
    ]
    np.testing.assert_array_equal(first[0].mask, observation("room:2", 0, "desk").mask)
    assert adapter.load_frame("room:20") == []
    with pytest.raises(KeyError):
        adapter.load_frame("room:3")
    first[0].mask[:] = False
    assert adapter.load_frame("room:2")[0].mask.any()
    assert snapshot(cache) == before


def test_writer_refuses_existing_output(cache, selected):
    with pytest.raises(FileExistsError):
        SegmentationCacheWriter(selected, cache)


def test_writer_requires_frames_in_order_and_complete(selected, tmp_path):
    writer = SegmentationCacheWriter(selected, tmp_path / "out")
    with pytest.raises(ValueError):
        writer.write_frame("room:8", [])
    writer.write_frame("room:2", [])
    with pytest.raises(ValueError):
        writer.finish({})
    assert not (tmp_path / "out/run_complete.json").exists()


def test_writer_rejects_repeated_hint_in_one_frame(selected, tmp_path):
    writer = SegmentationCacheWriter(selected, tmp_path / "out")
    with pytest.raises(ValueError):
        writer.write_frame(
            "room:2",
            [observation("room:2", 0, "desk", hint="a"), observation("room:2", 3, "chair", hint="a")],
        )


def test_saved_backend_reuses_canonical_cache(cache, selected, tmp_path):
    output = tmp_path / "reused"
    adapter = SavedSegmentationBackend(cache).prepare(selected, output)

    assert (output / "cleaned").is_symlink()
    assert (output / "cleaned").resolve() == cache.resolve()
    assert adapter.segmentation_frame_ids == tuple(
        frame.frame_id for frame in selected.frames
    )


def test_missing_cleaned_marker_rejected(cache, selected):
    (cache / "run_complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        SegmentationAdapter.from_saved_run(selected, cache)


def test_truncated_packed_mask_is_rejected(cache, selected):
    path = cache / "masks/000000.npz"
    with np.load(path, allow_pickle=False) as result:
        arrays = {name: result[name] for name in result.files}
    arrays["packed"] = arrays["packed"][:, :-1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        SegmentationAdapter.from_saved_run(selected, cache).load_frame("room:2")
