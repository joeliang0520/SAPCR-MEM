"""Estimate scene leveling from floor evidence without changing saved geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import FloatArray

if TYPE_CHECKING:
    from .geometry import GeometryAdapter


# Where the floor sits in a levelled scene: low enough to be under the room's
# contents, high enough to ignore points that drift below it.
FLOOR_PERCENTILE = 1.0

# The floor is dropped by a whole number of these. Association voxelises on a
# grid anchored at the origin, so moving a scene by part of a voxel changes
# which points share one and, through that, which observations match.
# Levelling should not decide that, so it moves the scene in whole voxels
# and leaves the floor within half a voxel of zero. Must stay equal to
# ObjectAssociator's voxel size, which a test checks.
FLOOR_STEP = 0.05

# Walls that all face one way leave up free to turn about their shared normal,
# so the walls that level a scene must include two facing this far apart.
MIN_WALL_SPREAD_DEGREES = 30.0


def floor_height(points: FloatArray, rotation: FloatArray) -> float:
    """How high the floor is, in the scene this rotation levels.

    The accepted plane is not used: whenever no floor qualified there is none,
    or only a tabletop, and a table's height is not the floor's. The scene's
    own low percentile works either way, and reads the floor off the clutter
    standing on it when the floor itself was never seen cleanly.
    """
    return float(np.percentile(points @ rotation[2], FLOOR_PERCENTILE))


class SceneAligner:
    """Explicit, bounded floor and wall estimation in raw geometry coordinates.

    Call estimate(raw_geometry), then pass its 4 x 4 transform as world_alignment
    when opening a GeometryAdapter. Estimation needs the optional Open3D
    dependency; ordinary geometry access does not. No files are written.
    Camera-up only guides plane direction/sign, and is never used in place of
    a surface: with no floor, walls or horizontal surface, estimate raises
    ValueError.

    Leveling needs only the up direction, so a scene whose floor never appears
    as a clean plane has two other witnesses: its walls, whose up is the
    direction they are all perpendicular to, and its largest plane passing
    every floor test but "almost nothing below it" - a table or counter top.
    Both are taken, and the one nearer camera-up is used. The floor is
    preferred whenever one qualifies.

    The floor fixes which way is up; the walls fix which way the room faces.
    Without the second, every box is drawn on arbitrary horizontal axes and
    comes out larger than the object inside it. Walls are used rather than the
    objects themselves because a room's furniture need not agree with its
    room, as in an office of independently angled desks.

    Defaults: 24 uniformly sampled anchors, every sixth pixel, 4 cm voxels and
    a 2.5 cm plane tolerance, for metre-scale indoor geometry.
    """

    def __init__(
        self,
        *,
        max_frames: int = 24,
        pixel_stride: int = 6,
        distance_threshold: float = 0.025,
        seed: int = 7,
        square_to_walls: bool = True,
        wall_distance_threshold: float = 0.03,
        max_wall_tilt_degrees: float = 15.0,
        min_wall_extent: float = 0.8,
        min_corroborating_walls: int = 2,
        min_wall_agreement: float = 0.35,
    ) -> None:
        if max_frames < 1 or pixel_stride < 1:
            raise ValueError("max_frames and pixel_stride must be positive")
        if not np.isfinite(distance_threshold) or distance_threshold <= 0:
            raise ValueError("distance_threshold must be finite and positive")
        if not np.isfinite(wall_distance_threshold) or wall_distance_threshold <= 0:
            raise ValueError("Wall plane tolerance must be finite and positive")
        if not 0 < max_wall_tilt_degrees < 45:
            raise ValueError("Wall tilt tolerance must be in (0, 45) degrees")
        if not np.isfinite(min_wall_extent) or min_wall_extent <= 0:
            raise ValueError("Minimum wall extent must be finite and positive")
        if min_corroborating_walls < 1:
            raise ValueError("At least one wall must be required to square up")
        if not 0 <= min_wall_agreement <= 1:
            raise ValueError("Wall agreement must be a fraction in [0, 1]")
        self._max_frames = max_frames
        self._pixel_stride = pixel_stride
        self._distance_threshold = distance_threshold
        self._seed = seed
        self._square_to_walls = square_to_walls
        self._wall_distance_threshold = float(wall_distance_threshold)
        self._max_wall_tilt = float(max_wall_tilt_degrees)
        self._min_wall_extent = float(min_wall_extent)
        self._min_corroborating_walls = int(min_corroborating_walls)
        self._min_wall_agreement = float(min_wall_agreement)

    def estimate(self, geometry: GeometryAdapter) -> FloatArray:
        """Level the scene to +Z and put its floor at z = 0.

        The rotation is the shortest one taking the accepted surface normal to
        +Z. The origin then drops to the floor, so a height is a height above
        the floor and heights mean the same thing from one scene to the next.
        The floor is taken as the first percentile of the levelled scene rather
        than from the accepted plane, because there is none, or only a
        tabletop, whenever no floor qualified; a percentile rather than the
        lowest point, which a few strays below the floor would otherwise set.

        geometry must be raw, with original depth-grid intrinsics and final
        camera-to-world poses. Floor acceptance requires broad support, cameras
        above the plane and few scene points below it. Ceiling and tabletop
        rejection use these geometric checks, not camera-up alone.
        """
        if geometry.alignment_method != "raw":
            raise ValueError("Estimate leveling from raw geometry, not leveled output")
        ids = geometry.geometry_frame_ids
        if not ids:
            raise ValueError("At least one geometry anchor is required")
        poses = np.stack(
            [geometry.load_pose(frame_id).camera_to_world for frame_id in ids]
        )
        hint = -poses[:, :3, 1].mean(axis=0)
        if np.linalg.norm(hint) < 0.1:
            raise ValueError("Camera-up directions do not provide a usable hint")
        hint /= np.linalg.norm(hint)
        selected = np.unique(
            np.linspace(0, len(ids) - 1, min(len(ids), self._max_frames)).astype(int)
        )
        points = []
        for index in selected:
            frame = geometry.load_frame(ids[index])
            v, u = np.mgrid[
                2 : frame.depth.shape[0] - 2 : self._pixel_stride,
                2 : frame.depth.shape[1] - 2 : self._pixel_stride,
            ]
            depth = frame.depth[v, u]
            valid = np.isfinite(depth) & (depth > 0) & (depth < 10)
            pixels = np.stack((u[valid], v[valid], np.ones(valid.sum())), axis=1)
            camera = (pixels @ np.linalg.inv(frame.intrinsics).T) * depth[valid, None]
            pose = frame.camera_to_world
            points.append(camera @ pose[:3, :3].T + pose[:3, 3])
        points = np.concatenate(points)
        if len(points) < 300:
            raise ValueError("Not enough valid scene points to find a floor")
        normal = self._floor_normal(points, poses[:, :3, 3], hint)
        vertical = np.array([0.0, 0.0, 1.0])
        axis = np.cross(normal, vertical)
        sine = np.linalg.norm(axis)
        cosine = float(normal @ vertical)
        if sine < 1e-8:
            # Opposite vertical has no unique shortest rotation axis.
            rotation = (
                np.eye(3)
                if cosine > 0
                else Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
            )
        else:
            rotation = Rotation.from_rotvec(
                axis / sine * np.arctan2(sine, cosine)
            ).as_matrix()
        if self._square_to_walls:
            rotation = self._square_up(points @ rotation.T) @ rotation
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation
        height = floor_height(points, rotation)
        transform[2, 3] = -round(height / FLOOR_STEP) * FLOOR_STEP
        return transform

    def _square_up(self, leveled: np.ndarray) -> np.ndarray:
        """Turn about the vertical until the walls lie on the horizontal axes.

        Works on already-levelled points, so a wall is a surface whose normal is
        horizontal. Headings are averaged on the quadruple angle because a box
        is unchanged by a quarter turn, which makes zero and ninety degrees the
        same direction.

        Large flat walls give the sharpest answer. Below min_corroborating_walls
        the heading is taken from every patch of vertical surface instead. With
        no vertical surface at all the scene is left levelled but not squared
        up.
        """
        headings, weights = self._wall_headings(leveled)
        if headings.size < self._min_corroborating_walls:
            headings, weights = self._vertical_surface_headings(leveled)
            # Loose patches only carry a heading if they agree on one. A scene
            # with no walls still has a few near-vertical normals at the edge of
            # what was sampled, and left ungated they invent an angle: a
            # floor-only scene came out turned by fifteen degrees.
            if self._agreement(headings, weights) < self._min_wall_agreement:
                return np.eye(3)
        if not headings.size:
            return np.eye(3)
        angles = 4 * headings
        mean = np.arctan2(
            (weights * np.sin(angles)).sum(), (weights * np.cos(angles)).sum()
        )
        return Rotation.from_rotvec([0, 0, -mean / 4]).as_matrix()

    @staticmethod
    def _agreement(headings: np.ndarray, weights: np.ndarray) -> float:
        """How concentrated the headings are, from 0 (scattered) to 1 (one angle)."""
        if not headings.size:
            return 0.0
        angles = 4 * headings
        total = weights.sum()
        if total <= 0:
            return 0.0
        return float(
            abs(
                complex(
                    (weights * np.cos(angles)).sum() / total,
                    (weights * np.sin(angles)).sum() / total,
                )
            )
        )

    def _wall_headings(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Heading and supporting area of each broad vertical plane.

        Walls take a looser plane tolerance than the floor. A floor is flat; a
        wall carries skirting, doorframes, radiators and whatever is pinned to
        it, so it is flat to a couple of centimetres rather than a few
        millimetres. Fitting it at the floor's tolerance picks out a sub-patch
        and reads its heading instead of the wall's.
        """
        import open3d as o3d

        o3d.utility.random.seed(self._seed)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud = cloud.voxel_down_sample(0.04)
        upright = np.cos(np.radians(90.0 - self._max_wall_tilt))
        headings, areas = [], []
        for _ in range(12):
            if len(cloud.points) < 300:
                break
            model, indices = cloud.segment_plane(
                self._wall_distance_threshold, 3, 700
            )
            if len(indices) < 300:
                break
            inliers = np.asarray(cloud.points)[indices]
            cloud = cloud.select_by_index(indices, invert=True)
            normal = np.array(model[:3], dtype=float)
            normal /= np.linalg.norm(normal)
            if abs(normal[2]) > upright:
                continue  # floor, ceiling or tabletop: no heading to read
            centre = inliers.mean(axis=0)
            _, _, axes = np.linalg.svd(inliers - centre, full_matrices=False)
            spread = np.percentile((inliers - centre) @ axes[:2].T, [5, 95], axis=0)
            width, height = spread[1] - spread[0]
            if min(width, height) < self._min_wall_extent:
                continue  # a sliver, not a wall
            headings.append(np.arctan2(normal[1], normal[0]))
            areas.append(width * height)
        return np.array(headings), np.array(areas)

    def _vertical_surface_headings(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Heading of every vertical surface patch, each weighted equally.

        A wall broken up by furniture is never segmented as one plane but still
        votes here.
        """
        import open3d as o3d

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud = cloud.voxel_down_sample(0.04)
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        normals = np.asarray(cloud.normals)
        if not len(normals):
            return np.array([]), np.array([])
        vertical = normals[np.abs(normals[:, 2]) <= np.sin(np.radians(self._max_wall_tilt))]
        if len(vertical) < 300:
            return np.array([]), np.array([])
        return (
            np.arctan2(vertical[:, 1], vertical[:, 0]),
            np.ones(len(vertical)),
        )

    def _floor_normal(
        self, points: np.ndarray, cameras: np.ndarray, hint: np.ndarray
    ) -> np.ndarray:
        try:
            import open3d as o3d
        except ImportError as error:
            raise ImportError(
                "Floor estimation needs Open3D; install requirements-alignment.txt"
            ) from error

        o3d.utility.random.seed(self._seed)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud = cloud.voxel_down_sample(0.04)
        floors, surfaces = [], []
        for _ in range(12):
            if len(cloud.points) < 300:
                break
            _, indices = cloud.segment_plane(self._distance_threshold, 3, 700)
            if len(indices) < 150:
                break
            inliers = np.asarray(cloud.points)[indices]
            centre = inliers.mean(axis=0)
            _, _, axes = np.linalg.svd(inliers - centre, full_matrices=False)
            normal = axes[-1]
            if normal @ hint < 0:
                normal = -normal
            extent = np.percentile((inliers - centre) @ axes[:2].T, [5, 95], axis=0)
            area = np.prod(extent[1] - extent[0])
            height = (cameras - centre) @ normal
            below = np.mean((points - centre) @ normal < -2 * self._distance_threshold)
            if (
                area >= 1
                and normal @ hint >= np.cos(np.pi / 4)
                and np.mean(height > 0.5) >= 0.9
                and np.median(height) < 2.8
            ):
                (floors if below < 0.08 else surfaces).append((len(indices), normal))
            cloud = cloud.select_by_index(indices, invert=True)
        if floors:
            return max(floors, key=lambda candidate: candidate[0])[1]
        witnesses = []
        walls = self._up_from_walls(points, hint)
        if walls is not None:
            witnesses.append(walls)
        if surfaces:
            witnesses.append(max(surfaces, key=lambda candidate: candidate[0])[1])
        if not witnesses:
            raise ValueError(
                "No convincing floor plane, walls or horizontal surface; "
                "leveling was not applied"
            )
        return max(witnesses, key=lambda up: up @ hint)

    def _up_from_walls(
        self, points: np.ndarray, hint: np.ndarray
    ) -> np.ndarray | None:
        """The direction every broad vertical plane is perpendicular to, or None.

        Runs on raw geometry, where the only sign of which planes are walls is
        camera-up, and that leans with the camera. So walls are first taken
        loosely, as any plane
        nearer vertical than horizontal, then again within max_wall_tilt of the
        up they gave. Planes are found as _wall_headings finds them, but twice
        as many are searched, since in a cluttered room the walls come after
        the larger surfaces. None unless two
        of the walls face MIN_WALL_SPREAD_DEGREES apart.
        """
        import open3d as o3d

        o3d.utility.random.seed(self._seed)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud = cloud.voxel_down_sample(0.04)
        normals, areas = [], []
        for _ in range(24):
            if len(cloud.points) < 300:
                break
            model, indices = cloud.segment_plane(
                self._wall_distance_threshold, 3, 700
            )
            if len(indices) < 300:
                break
            inliers = np.asarray(cloud.points)[indices]
            cloud = cloud.select_by_index(indices, invert=True)
            centre = inliers.mean(axis=0)
            _, _, axes = np.linalg.svd(inliers - centre, full_matrices=False)
            spread = np.percentile((inliers - centre) @ axes[:2].T, [5, 95], axis=0)
            width, height = spread[1] - spread[0]
            if min(width, height) < self._min_wall_extent:
                continue  # a sliver, not a wall
            normals.append(np.asarray(model[:3], dtype=float))
            areas.append(width * height)
        if not normals:
            return None
        normals = np.array(normals) / np.linalg.norm(normals, axis=1, keepdims=True)
        up = hint
        for tilt in (45.0, self._max_wall_tilt):
            walls = np.abs(normals @ up) <= np.sin(np.radians(tilt))
            facing = normals[walls]
            if not len(facing) or np.abs(facing @ facing.T).min() > np.cos(
                np.radians(MIN_WALL_SPREAD_DEGREES)
            ):
                return None
            weighted = facing * np.sqrt(np.array(areas)[walls])[:, None]
            up = np.linalg.svd(weighted)[2][-1]
            if up @ hint < 0:
                up = -up
        return up
