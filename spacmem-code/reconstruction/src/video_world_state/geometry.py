"""Coordinate saved DA3 geometry and a native-frame pose timeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .adapters.da3 import (
    Da3OutputLoader,
    Da3Runner,
    load_anchor_sequence,
    unproject_depth,
)
from .alignment import SceneAligner
from .contracts import CameraPoseEstimate, FloatArray, FrameGeometry, FrameSequence
from .frames import validate_subset
from .poses import PoseInterpolator


class GeometryAdapter:
    """Public geometry access to a new or manifest-validated saved run.

    Full geometry exists only for anchors, listed by geometry_frame_ids. Pose
    lookup supports every native frame. No method fabricates intermediate depth.
    """

    def __init__(
        self,
        native_sequence: FrameSequence,
        output_directory: Path,
        *,
        allow_legacy: bool = False,
        world_alignment: FloatArray | None = None,
    ) -> None:
        anchor_sequence = load_anchor_sequence(
            native_sequence, Path(output_directory), allow_legacy=allow_legacy
        )
        self._loader = Da3OutputLoader(output_directory, anchor_sequence)
        self._geometry_frame_ids = tuple(
            frame.frame_id for frame in anchor_sequence.frames
        )
        measured_poses = [
            self._loader.load_pose(frame_id) for frame_id in self._geometry_frame_ids
        ]
        self._poses = PoseInterpolator(native_sequence, measured_poses)
        self._world_alignment = None
        if world_alignment is not None:
            matrix = np.array(world_alignment, dtype=np.float32, copy=True)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("Leveling must be a finite 4 x 4 transform")
            rotation = matrix[:3, :3]
            if (
                not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                or not np.allclose(matrix[:2, 3], 0, atol=1e-6, rtol=0)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)
            ):
                raise ValueError(
                    "Leveling must rotate and raise the floor to z = 0, without "
                    "horizontal translation, scale or reflection"
                )
            self._world_alignment = matrix

    @classmethod
    def from_saved_run(
        cls,
        native_sequence: FrameSequence,
        output_directory: Path,
        *,
        allow_legacy: bool = False,
        world_alignment: FloatArray | None = None,
    ) -> GeometryAdapter:
        """Open DA3 files and frame_manifest.json; never run inference.

        The manifest is the saved anchor FrameSequence: sequence_id and an
        ordered frames list with frame_id, source_frame_index,
        timestamp_seconds. IDs, indices and timestamps must match the supplied
        native sequence. RGB paths are ignored because DA3 can use resized
        inputs. Manifest order determines positional DA3 file order.
        A valid run_complete.json with the matching anchor count is required.
        allow_legacy=True also opens caches written before completion markers
        existed, but never a new run whose marker is missing.
        world_alignment explicitly supplies one scene-alignment transform for all
        returned poses. It is never estimated, discovered or saved here.
        """
        return cls(
            native_sequence,
            output_directory,
            allow_legacy=allow_legacy,
            world_alignment=world_alignment,
        )

    @classmethod
    def run(
        cls,
        native_sequence: FrameSequence,
        anchor_sequence: FrameSequence,
        runner: Da3Runner,
        output_directory: Path,
        *,
        alignment_path: Path | None = None,
    ) -> GeometryAdapter:
        """Run DA3 on supplied anchors, estimate scene alignment, and return aligned geometry.

        Input preparation selects anchors; segmentation can use that same
        FrameSequence without depending on geometry. The native sequence is
        retained for the existing pose API. No sampling occurs in this method.
        SceneAligner owns the estimation algorithm. Failure propagates without
        a camera-up fallback; raw inference remains available for diagnosis.
        Saved-run opening and ordinary lookups never estimate alignment.
        If alignment_path is supplied, explicitly save the transform there for
        later reopening with world_alignment; refuse an existing alignment file.
        Inference files are not modified. Scene alignment requires Open3D.
        """
        validate_subset(native_sequence, anchor_sequence)
        if alignment_path is not None and Path(alignment_path).exists():
            raise FileExistsError(f"Refusing to overwrite alignment: {alignment_path}")
        runner.run(anchor_sequence, output_directory)
        raw = cls.from_saved_run(native_sequence, output_directory)
        alignment = SceneAligner().estimate(raw)
        leveled = cls.from_saved_run(
            native_sequence, output_directory, world_alignment=alignment
        )
        if alignment_path is not None:
            with Path(alignment_path).open("xb") as alignment_file:
                np.save(alignment_file, alignment, allow_pickle=False)
        return leveled

    @property
    def geometry_frame_ids(self) -> tuple[str, ...]:
        """Ordered, immutable IDs with expected full geometry in the saved run."""
        return self._geometry_frame_ids

    @property
    def alignment_method(self) -> str:
        """Whether output uses raw coordinates or an explicitly supplied alignment."""
        return "raw" if self._world_alignment is None else "leveled"

    def load_frame(self, frame_id: str) -> FrameGeometry:
        """Load anchor geometry and its original-RGB pixel mapping.

        Non-anchor IDs raise KeyError. Legacy caches without verified resize
        metadata raise ValueError rather than guessing pixel correspondence.
        """
        frame = self._loader.load_frame(frame_id)
        if self._world_alignment is None:
            return frame
        return replace(
            frame, camera_to_world=self._world_alignment @ frame.camera_to_world
        )

    def world_points(self, frame_geometry: FrameGeometry) -> FloatArray:
        """Unproject one already loaded anchor into world points.

        Takes the FrameGeometry rather than a frame ID so depth is not read
        twice. Pass geometry this adapter returned: its pose already carries
        any scene alignment, so the points come back in the coordinates named by
        alignment_method, rotated exactly once. Saved depth already includes
        its chunk scale and is not scaled again here.
        Returns height x width x 3 covering the whole grid, including pixels
        whose depth or confidence a caller should reject; selecting valid
        points is the caller's job. Requires Torch and the DA3 library, which
        ordinary saved-run loading does not.
        """
        return unproject_depth(
            frame_geometry.depth,
            frame_geometry.intrinsics,
            frame_geometry.camera_to_world,
        )

    def load_pose(self, frame_id: str) -> CameraPoseEstimate:
        """Look up any native frame's measured, interpolated or held pose."""
        pose = self._poses.load_pose(frame_id)
        if self._world_alignment is None:
            return pose
        return replace(
            pose, camera_to_world=self._world_alignment @ pose.camera_to_world
        )
