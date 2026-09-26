"""Coordinate the three passes using one shared frame selection."""

from pathlib import Path

from .adapters.appearance import AppearanceRunner
from .adapters.da3 import Da3Runner
from .contracts import FloatArray, FrameSequence
from .frames import select_frames
from .geometry import GeometryAdapter
from .objects import (
    FragmentMerger,
    IdentityAssociator,
    MaskCarver,
    ObjectAssociator,
    ObjectConstructionResult,
    ObjectConstructor,
    ObservationLifter,
)
from .segmentation import SegmentationAdapter, SegmentationBackend


def run_pipeline(
    native_sequence: FrameSequence,
    geometry_runner: Da3Runner,
    segmentation_backend: SegmentationBackend,
    output_directory: Path,
) -> tuple[GeometryAdapter, SegmentationAdapter]:
    """Run geometry and the selected segmentation backend.

    Input is an already prepared native RGB sequence; no extraction/downloads
    happen here. Select 5 FPS once, using the same defaults for every scene.
    Runners supply tool/environment configuration, not per-scene adjustments.

    Creates geometry/, segmentation/ (raw/ and cleaned/), and
    world_alignment.npy in a fresh output directory. The alignment file is the
    rotation needed to reopen geometry in the returned aligned coordinates;
    raw geometry files remain untouched. Requires Open3D for scene alignment.
    The segmentation backend may run a model, read annotations for the
    ground-truth ablation, or reopen a completed canonical cache. The
    coordinator does not know which tool it represents.

    Both passes must end up covering the selected frames exactly; a mismatch
    raises rather than returning caches that cannot be combined later.
    Existing output raises FileExistsError. Any stage failure propagates and
    preserves partial outputs/logs; there is no automatic retry, unlevelled
    fallback or resume. No pair of ready adapters is returned on failure.
    """
    output = Path(output_directory)
    selected = select_frames(native_sequence)
    output.mkdir(parents=True, exist_ok=False)

    geometry = GeometryAdapter.run(
        native_sequence,
        selected,
        geometry_runner,
        output / "geometry",
        alignment_path=output / "world_alignment.npy",
    )

    segmentation = segmentation_backend.prepare(selected, output / "segmentation")
    expected = tuple(frame.frame_id for frame in selected.frames)
    if (
        geometry.geometry_frame_ids != expected
        or segmentation.segmentation_frame_ids != expected
    ):
        raise ValueError(
            "Geometry and segmentation must cover the selected frames exactly"
        )
    return geometry, segmentation


MIN_GEOMETRY_CONFIDENCE = 0.5
MIN_OBSERVATIONS = 4

# Pass-3 settings that differ by segmentation backend. build_world_state takes
# one of these blocks and uses ACTIVE when none is given. Each run records the
# lifter, associator and carver settings it used in objects.json.
#
# min_segmentation_confidence: observations below it are ignored by pass 3, as
#   if the backend had never emitted them (None keeps everything). For SegVGGT
#   the confidence is the query's score.
# erosion_pixels: ObservationLifter's mask erosion before the depth core.
# same_label_only: ObjectAssociator matches only objects of the mask's label.
# carve: MaskCarver trims object points against their own masks.
# identity: the segmentation's hints are true identities (the ground-truth
#   ablation); IdentityAssociator replaces matching and nothing is merged.
SEGVGGT = {
    "min_segmentation_confidence": 0.10,
    "erosion_pixels": 4,
    "same_label_only": True,
    "carve": True,
    "identity": False,
}
SCANNET_IDENTITY = {  # ScanNet masks and instances on our own geometry
    "min_segmentation_confidence": None,
    "erosion_pixels": 2,
    "same_label_only": False,  # no matching happens
    "carve": True,
    "identity": True,
}
ACTIVE = SEGVGGT


class _ConfidentObservations:
    """A segmentation adapter showing only observations at or above a confidence."""

    def __init__(self, segmentation: SegmentationAdapter, minimum: float) -> None:
        self._segmentation = segmentation
        self._minimum = float(minimum)
        self.segmentation_frame_ids = segmentation.segmentation_frame_ids

    def load_frame(self, frame_id: str):
        return [
            item
            for item in self._segmentation.load_frame(frame_id)
            if item.confidence >= self._minimum
        ]


def build_world_state(
    geometry: GeometryAdapter,
    segmentation: SegmentationAdapter,
    output_directory: Path,
    descriptors: dict[str, FloatArray] | None = None,
    settings: dict | None = None,
) -> ObjectConstructionResult:
    """Lift every cleaned mask and associate it into persistent objects.

    This is pass three at its chosen settings: the shared ones above and the
    backend-specific ones in settings, one of the blocks above (ACTIVE when
    not given).

    `min_observations` decides how thin an object may be and still be listed.
    Objects seen once or twice are usually a sliver caught at an odd angle and
    never matched again, and the reader of this output never sees a frame, so
    it cannot tell such a sliver from a real thing.

    descriptors are unit appearance vectors per observation ID, from whatever
    embedder the caller prefers; the merger uses them as one of two ways to
    confirm a fusion. Without them it confirms on a shared track alone, which
    is the stricter signal.
    """
    settings = ACTIVE if settings is None else settings
    if settings["min_segmentation_confidence"] is not None:
        segmentation = _ConfidentObservations(
            segmentation, settings["min_segmentation_confidence"]
        )
    return ObjectConstructor.run(
        geometry,
        segmentation,
        ObservationLifter(
            MIN_GEOMETRY_CONFIDENCE, erosion_pixels=settings["erosion_pixels"]
        ),
        IdentityAssociator()
        if settings["identity"]
        else ObjectAssociator(same_label_only=settings["same_label_only"]),
        Path(output_directory),
        merger=None if settings["identity"] else FragmentMerger(),
        descriptors=descriptors,
        min_observations=MIN_OBSERVATIONS,
        carver=MaskCarver() if settings["carve"] else None,
    )


def run_all_passes(
    native_sequence: FrameSequence,
    geometry_runner: Da3Runner,
    segmentation_backend: SegmentationBackend,
    appearance_runner: AppearanceRunner,
    output_directory: Path,
) -> ObjectConstructionResult:
    """Reconstruct, segment, describe and build objects.

    Writes appearance/ and objects/ beside the first two passes' output. The
    same guarantees as run_pipeline apply, and pass three adds no new failure
    mode of its own: it reads only what the earlier stages wrote.

    Each stage reads the recording in time order: geometry and segmentation
    work through consecutive chunks of frames, association builds objects frame
    by frame from what earlier frames produced, and the steps after it read
    only the objects accumulated so far.
    """
    output = Path(output_directory)
    geometry, segmentation = run_pipeline(
        native_sequence, geometry_runner, segmentation_backend, output
    )
    descriptors = appearance_runner.run(
        native_sequence, segmentation, output / "appearance"
    )
    return build_world_state(geometry, segmentation, output / "objects", descriptors)
