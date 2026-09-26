"""Derive native-frame camera poses from measured geometry anchors."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .contracts import CameraPoseEstimate, FrameSequence


class PoseInterpolator:
    """Look up poses by native frame ID, using timestamps rather than indices.

    Supply ordered native frames and ordered measured anchors in one world
    coordinate system. Anchors are preserved exactly. Between anchors, position
    is linear and rotation uses SLERP. Outside them, the nearest pose is held.
    A single anchor is allowed; all other frames then receive held poses.
    Inputs are copied so later edits cannot change this pose timeline.
    """

    def __init__(
        self,
        native_sequence: FrameSequence,
        anchor_poses: list[CameraPoseEstimate],
    ) -> None:
        frames = native_sequence.frames
        if not frames:
            raise ValueError("The native FrameSequence is empty")
        times = np.asarray([frame.timestamp_seconds for frame in frames], dtype=float)
        if not np.isfinite(times).all():
            raise ValueError("Native timestamps must be finite")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Native timestamps must be strictly increasing")
        self._frame_times = {frame.frame_id: time for frame, time in zip(frames, times)}
        if len(self._frame_times) != len(frames):
            raise ValueError("Duplicate native frame ID")
        if not anchor_poses:
            raise ValueError("At least one measured anchor pose is required")

        self._anchor_positions: dict[str, int] = {}
        matrices = []
        anchor_times = []
        for position, anchor in enumerate(anchor_poses):
            if anchor.frame_id not in self._frame_times:
                raise ValueError(f"Anchor is not a native frame: {anchor.frame_id}")
            if anchor.frame_id in self._anchor_positions:
                raise ValueError(f"Duplicate anchor frame ID: {anchor.frame_id}")
            if anchor.method != "measured":
                raise ValueError("Interpolation anchors must be measured poses")
            matrix = np.array(anchor.camera_to_world, dtype=np.float32, copy=True)
            self._validate_pose(matrix)
            self._anchor_positions[anchor.frame_id] = position
            matrices.append(matrix)
            anchor_times.append(self._frame_times[anchor.frame_id])

        self._anchor_times = np.asarray(anchor_times)
        if np.any(np.diff(self._anchor_times) <= 0):
            raise ValueError("Anchor poses must be in strictly increasing time order")
        self._matrices = np.stack(matrices)
        self._rotations = (
            Slerp(self._anchor_times, Rotation.from_matrix(self._matrices[:, :3, :3]))
            if len(matrices) > 1
            else None
        )

    def load_pose(self, frame_id: str) -> CameraPoseEstimate:
        """Return an independent float32 pose; reject unknown native frame IDs."""
        try:
            time = self._frame_times[frame_id]
        except KeyError as error:
            raise KeyError(f"Unknown native frame ID: {frame_id}") from error

        if frame_id in self._anchor_positions:
            matrix = self._matrices[self._anchor_positions[frame_id]].copy()
            return CameraPoseEstimate(frame_id, matrix, "measured")

        if time < self._anchor_times[0] or time > self._anchor_times[-1]:
            position = 0 if time < self._anchor_times[0] else -1
            return CameraPoseEstimate(frame_id, self._matrices[position].copy(), "held")

        right = int(np.searchsorted(self._anchor_times, time))
        left = right - 1
        fraction = (time - self._anchor_times[left]) / (
            self._anchor_times[right] - self._anchor_times[left]
        )
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = (1 - fraction) * self._matrices[
            left, :3, 3
        ] + fraction * self._matrices[right, :3, 3]
        matrix[:3, :3] = self._rotations([time]).as_matrix()[0]
        return CameraPoseEstimate(frame_id, matrix, "interpolated")

    @staticmethod
    def _validate_pose(matrix: np.ndarray) -> None:
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Anchor pose must be a finite 4 x 4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
            raise ValueError("Anchor pose must have homogeneous last row [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        # Allow float32 roundoff, but reject scale/shear/reflections before
        # SciPy can silently turn an invalid matrix into a proper rotation.
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0):
            raise ValueError("Anchor rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0):
            raise ValueError("Anchor rotation must have determinant +1")
