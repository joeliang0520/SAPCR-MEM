"""Coordinator wiring only; model adapters and leveling have their own tests."""

from unittest.mock import Mock

import pytest

from video_world_state import pipeline


@pytest.fixture
def stages(monkeypatch, native_sequence):
    geometry_run, backend = Mock(), Mock()
    covered = tuple(frame.frame_id for frame in native_sequence.frames[::2])
    geometry_run.return_value.geometry_frame_ids = covered
    backend.prepare.return_value.segmentation_frame_ids = covered
    monkeypatch.setattr(pipeline.GeometryAdapter, "run", geometry_run)
    return geometry_run, backend


def test_passes_covering_different_frames_are_refused(
    native_sequence, tmp_path, stages
):
    """Masks and geometry that do not line up cannot be combined in pass 3."""
    geometry_run, backend = stages
    geometry_run.return_value.geometry_frame_ids = ("example:0",)
    with pytest.raises(ValueError, match="cover the selected frames exactly"):
        pipeline.run_pipeline(native_sequence, Mock(), backend, tmp_path / "mismatch")


def test_one_selection_and_only_backend_calls(native_sequence, tmp_path, stages):
    geometry_run, backend = stages
    output = tmp_path / "pipeline"
    geometry_runner = Mock()
    result = pipeline.run_pipeline(
        native_sequence, geometry_runner, backend, output
    )
    geometry_run.assert_called_once()
    native, selected, runner, directory = geometry_run.call_args.args
    assert native is native_sequence
    assert selected.frames == native_sequence.frames[::2]
    assert runner is geometry_runner
    assert directory == output / "geometry"
    assert geometry_run.call_args.kwargs == {
        "alignment_path": output / "world_alignment.npy"
    }
    backend.prepare.assert_called_once_with(selected, output / "segmentation")
    assert result == (geometry_run.return_value, backend.prepare.return_value)


def test_existing_directory_is_not_touched(native_sequence, tmp_path, stages):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing work")
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(native_sequence, Mock(), stages[1], output)
    assert sentinel.read_text() == "existing work"
    stages[0].assert_not_called()
    stages[1].prepare.assert_not_called()


@pytest.mark.parametrize("error", [RuntimeError("DA3 failed"), ValueError("No floor")])
def test_failed_geometry_preparation_stops_segmentation(
    native_sequence, tmp_path, stages, error
):
    stages[0].side_effect = error
    output = tmp_path / "failed"
    with pytest.raises(type(error), match=str(error)):
        pipeline.run_pipeline(native_sequence, Mock(), stages[1], output)
    assert output.is_dir()
    assert not (output / "world_alignment.npy").exists()
    stages[1].prepare.assert_not_called()


def test_pass_three_runs_at_the_chosen_settings(
    native_sequence, tmp_path, stages, monkeypatch
):
    """The chosen configuration is what the pipeline actually applies."""
    construct = Mock()
    monkeypatch.setattr(pipeline.ObjectConstructor, "run", construct)

    pipeline.run_all_passes(
        native_sequence, Mock(), stages[1], Mock(), tmp_path / "full"
    )

    (_geometry, _segmentation, lifter, associator, objects_dir), keywords = (
        construct.call_args
    )
    assert lifter.min_geometry_confidence == 0.5
    assert lifter.erosion_pixels == pipeline.ACTIVE["erosion_pixels"]
    assert isinstance(associator, pipeline.ObjectAssociator)
    assert associator.same_label_only == pipeline.ACTIVE["same_label_only"]
    # Fragments are fused, but only where a track or appearance confirms it.
    assert isinstance(keywords["merger"], pipeline.FragmentMerger)
    assert keywords["min_observations"] == 4
    assert isinstance(keywords["carver"], pipeline.MaskCarver) == pipeline.ACTIVE["carve"]
    assert objects_dir == tmp_path / "full" / "objects"


def test_the_active_settings_are_segvggts():
    """The default run uses SegVGGT and its pass-3 settings."""
    assert pipeline.ACTIVE is pipeline.SEGVGGT
    assert pipeline.SEGVGGT == {
        "min_segmentation_confidence": 0.10,
        "erosion_pixels": 4,
        "same_label_only": True,
        "carve": True,
        "identity": False,
    }


def test_scannet_identity_takes_identity_from_the_hints_and_merges_nothing(
    native_sequence, tmp_path, monkeypatch
):
    """The ground-truth ablation swaps association for identity and keeps lifting and carving."""
    construct = Mock()
    monkeypatch.setattr(pipeline.ObjectConstructor, "run", construct)

    pipeline.build_world_state(
        Mock(), Mock(), tmp_path / "objects", settings=pipeline.SCANNET_IDENTITY
    )

    (_geometry, _segmentation, lifter, associator, _output), keywords = construct.call_args
    assert isinstance(associator, pipeline.IdentityAssociator)
    assert keywords["merger"] is None
    assert isinstance(keywords["carver"], pipeline.MaskCarver)
    assert lifter.erosion_pixels == 2


def test_low_confidence_observations_never_reach_pass_three():
    """The confidence floor hides observations as if the backend never emitted them."""
    frame = Mock(segmentation_frame_ids=("example:0",))
    frame.load_frame.return_value = [Mock(confidence=0.05), Mock(confidence=0.10), Mock(confidence=0.4)]
    view = pipeline._ConfidentObservations(frame, 0.10)

    assert view.segmentation_frame_ids == ("example:0",)
    assert [item.confidence for item in view.load_frame("example:0")] == [0.10, 0.4]


def test_objects_are_not_built_when_the_passes_do_not_line_up(
    native_sequence, tmp_path, stages, monkeypatch
):
    """A mismatch must stop before pass 3, not produce objects from bad input."""
    construct = Mock()
    monkeypatch.setattr(pipeline.ObjectConstructor, "run", construct)
    geometry_run, backend = stages
    geometry_run.return_value.geometry_frame_ids = ("example:0",)

    with pytest.raises(ValueError, match="cover the selected frames exactly"):
        pipeline.run_all_passes(
            native_sequence, Mock(), backend, Mock(), tmp_path / "bad"
        )
    construct.assert_not_called()


def test_appearance_reaches_the_merger_when_it_is_supplied(
    native_sequence, tmp_path, stages, monkeypatch
):
    """Descriptors are the caller's to provide, and must not be dropped."""
    construct = Mock()
    monkeypatch.setattr(pipeline.ObjectConstructor, "run", construct)
    descriptors = {"example:0": [1.0, 0.0]}

    pipeline.build_world_state(Mock(), Mock(), tmp_path / "objects", descriptors)

    assert construct.call_args.kwargs["descriptors"] is descriptors


def test_the_full_pipeline_describes_what_it_segmented(
    native_sequence, tmp_path, stages, monkeypatch
):
    """Merging is only as good as its second opinion, so it must be supplied."""
    construct = Mock()
    monkeypatch.setattr(pipeline.ObjectConstructor, "run", construct)
    _, backend = stages
    appearance = Mock()

    pipeline.run_all_passes(
        native_sequence, Mock(), backend, appearance, tmp_path / "full"
    )

    appearance.run.assert_called_once_with(
        native_sequence, backend.prepare.return_value, tmp_path / "full" / "appearance"
    )
    assert construct.call_args.kwargs["descriptors"] is appearance.run.return_value
