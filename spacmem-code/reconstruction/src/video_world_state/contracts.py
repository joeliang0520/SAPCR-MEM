"""Data passed between the first spatial-pipeline stages.

FrameSequence describes ordered RGB frames, either native input or a selected
subset shared by geometry and segmentation. Geometry produces FrameGeometry for
selected anchor frames and
CameraPoseEstimate for every native frame. Segmentation produces per-frame
masks; combining an anchor's geometry and mask produces ObjectObservation3D,
and association groups those records into PersistentObject.
Sightings use native-frame poses and associated persistent object IDs without
requiring depth for every frame.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass
class FrameInput:
    """One RGB frame belonging to a FrameSequence."""

    frame_id: str
    source_frame_index: int
    timestamp_seconds: float
    rgb_path: Path


@dataclass
class FrameSequence:
    """Ordered RGB frames; input preparation can select an unchanged subset."""

    sequence_id: str
    frames: list[FrameInput]


@dataclass
class FrameGeometry:
    """Measured anchor geometry on its own pixel grid.

    Depth, confidence and intrinsics share this grid. Pixel coordinates are
    integer (u, v), with u right and v down; depth is estimated metres along the
    optical axis, not ray distance, and an adapter must convert arbitrary-scale
    output before using this contract. Camera axes are right/down/forward.
    Confidence is backend-specific reliability, not a calibrated probability.
    Our DA3 adapter preserves its unnormalized scores; no shared threshold is
    implied and scores from different backends are not directly comparable.
    rgb_to_geometry maps original RGB pixels to this grid:
    [u_geometry, v_geometry, 1] = rgb_to_geometry @ [u_rgb, v_rgb, 1].
    The map follows the geometry tool's intrinsics convention; mask resampling
    is a separate operation, not integer rounding of mapped coordinates.
    """

    frame_id: str
    depth: FloatArray  # height x width, estimated metres
    confidence: FloatArray  # height x width, backend-specific reliability scores
    intrinsics: FloatArray  # 3 x 3
    camera_to_world: FloatArray  # 4 x 4
    rgb_to_geometry: FloatArray  # 3 x 3, original RGB -> geometry grid


@dataclass
class CameraPoseEstimate:
    """Native-frame pose in the same world coordinates as FrameGeometry.

    Measured means estimated by the geometry tool, not sensor ground truth.
    Interpolated poses blend surrounding anchors using timestamps; held poses
    copy the nearest anchor outside the anchor time range. At a measured anchor,
    camera_to_world must equal that frame's FrameGeometry pose.
    """

    frame_id: str
    camera_to_world: FloatArray  # 4 x 4
    method: Literal["measured", "interpolated", "held"]


@dataclass(frozen=True)
class LabelCandidate:
    """A runner-up label scored relative to an observation's primary label.

    The primary label has implicit score 1.0. Candidate scores therefore say
    how strongly the same backend supported an alternative relative to its
    winner; they are not calibrated probabilities and are not comparable
    between segmentation backends.
    """

    label: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("Candidate labels must be non-empty strings")
        if not np.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("Candidate scores must be finite and in [0, 1]")


@dataclass
class SegmentationObservation:
    """One visible object mask on the original RGB frame's pixel grid.

    Match frame_id, then use FrameGeometry.rgb_to_geometry to map the mask to
    the depth grid before lifting. Matching IDs alone does not align pixels.
    """

    observation_id: str
    frame_id: str
    mask: BoolArray  # same height x width as the original RGB frame
    label: str
    confidence: float
    track_hint: str | None = None
    label_candidates: tuple[LabelCandidate, ...] = ()


@dataclass
class ObjectObservation3D:
    """Spatial summary produced from matching geometry and segmentation."""

    observation_id: str  # copied from SegmentationObservation
    frame_id: str
    label: str
    centroid_xyz: FloatArray  # length 3
    bounds_min_xyz: FloatArray  # length 3
    bounds_max_xyz: FloatArray  # length 3
    point_count: int
    segmentation_confidence: float
    track_hint: str | None = None
    label_candidates: tuple[LabelCandidate, ...] = ()


@dataclass
class PersistentObject:
    """One object's working state, accumulated across observed frames.

    points_xyz is compact accumulated evidence chosen by association, not every
    point its observations contributed. label_counts records primary-label
    votes for diagnostics. label_scores adds relative runner-up evidence and is
    the single ranking used for the final label, candidates and legacy aliases.
    Summaries describe the retained points. observation_ids reference records
    by ID: an observation belongs to at most one object, and the records
    themselves carry no object ID.
    """

    object_id: str
    points_xyz: FloatArray  # N x 3 in the same world coordinates as geometry
    label_counts: dict[str, int]
    centroid_xyz: FloatArray  # length 3
    bounds_min_xyz: FloatArray  # length 3
    bounds_max_xyz: FloatArray  # length 3
    observation_ids: list[str]
    label_scores: dict[str, float] = field(default_factory=dict)
