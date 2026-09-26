import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from video_world_state.adapters.scannet import ScanNetGtBackend, ScanNetInstanceSegmentation
from video_world_state.contracts import FrameInput, FrameSequence


@pytest.fixture
def reader(tmp_path):
    instances = tmp_path / "instance-filt"
    instances.mkdir()
    values = np.zeros((4, 6), np.uint8)
    values[:, :2] = 1  # objectId 0, kitchen cabinets
    values[:, 2:4] = 3  # objectId 2, wall
    values[:, 4] = 4  # objectId 3, a label the mapping does not know
    Image.fromarray(values).save(instances / "12.png")
    Image.fromarray(np.zeros((4, 6), np.uint8)).save(instances / "18.png")
    aggregation = tmp_path / "scene.aggregation.json"
    aggregation.write_text(json.dumps({"segGroups": [
        {"objectId": 0, "label": "kitchen cabinets"},
        {"objectId": 1, "label": "chair"},
        {"objectId": 2, "label": "wall"},
        {"objectId": 3, "label": "coffee kettle"},
    ]}))
    selected = FrameSequence("room", [
        FrameInput("room:12", 12, 0.4, Path("unused")),
        FrameInput("room:18", 18, 0.6, Path("unused")),
    ])
    return ScanNetInstanceSegmentation(
        selected, instances, aggregation, {"kitchen cabinets": "cabinet", "chair": "chair"}
    )


def test_objects_are_read_with_mapped_labels_and_no_identity(reader):
    observations = reader.load_frame("room:12")

    assert [o.label for o in observations] == ["cabinet", "coffee kettle"]
    assert [o.observation_id for o in observations] == ["room:obs:000000:000", "room:obs:000000:001"]
    assert all(o.track_hint is None and o.confidence == 1.0 for o in observations)
    assert observations[0].mask[:, :2].all() and not observations[0].mask[:, 2:].any()
    assert reader.unmapped == ["coffee kettle"]


def test_structure_and_unannotated_pixels_are_left_out(reader):
    assert reader.load_frame("room:18") == []
    assert all(not o.mask[:, 2:4].any() for o in reader.load_frame("room:12"))
    assert reader.segmentation_frame_ids == ("room:12", "room:18")
    with pytest.raises(KeyError):
        reader.load_frame("room:24")


def test_an_instance_missing_from_the_aggregation_is_refused(reader):
    values = np.full((4, 6), 9, np.uint8)
    Image.fromarray(values).save(reader.instance_directory / "18.png")
    with pytest.raises(ValueError, match="not in the aggregation"):
        reader.load_frame("room:18")


def test_identity_hints_carry_the_scannet_instance_when_asked(reader):
    oracle = ScanNetInstanceSegmentation.__new__(ScanNetInstanceSegmentation)
    oracle.__dict__.update(reader.__dict__, identity_hints=True)

    assert [o.track_hint for o in oracle.load_frame("room:12")] == ["room:scannet:0", "room:scannet:3"]


def test_the_backend_saves_the_reader_output_as_a_canonical_cache(reader, tmp_path):
    frames = []
    for index, frame in enumerate(("room:12", "room:18")):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (6, 4)).save(path)
        frames.append(FrameInput(frame, 12 + 6 * index, 0.4 + 0.2 * index, path))
    selected = FrameSequence("room", frames)
    backend = ScanNetGtBackend(
        reader.instance_directory,
        tmp_path / "scene.aggregation.json",
        {"kitchen cabinets": "cabinet", "chair": "chair"},
        identity_hints=True,
    )

    adapter = backend.prepare(selected, tmp_path / "segmentation")

    saved = adapter.load_frame("room:12")
    assert [o.label for o in saved] == ["cabinet", "coffee kettle"]
    assert [o.track_hint for o in saved] == ["room:scannet:0", "room:scannet:3"]
    assert [o.observation_id for o in saved] == ["room:obs:000000:000", "room:obs:000000:001"]
    assert saved[0].mask[:, :2].all() and not saved[0].mask[:, 2:].any()
    assert adapter.load_frame("room:18") == []
    manifest = json.loads((tmp_path / "segmentation/cleaned/manifest.json").read_text())
    assert manifest["backend"] == "scannet_gt" and manifest["unmapped_labels"] == ["coffee kettle"]
