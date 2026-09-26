"""Build persistent objects from one pass over matching geometry and masks.

The constructor owns the frame loop: it lifts each cleaned mask into world
points through the geometry boundary and hands one frame's lifted observations
to an associator, which decides identity. No model runs here; geometry and
segmentation must already be complete.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .contracts import (
    BoolArray,
    FloatArray,
    FrameGeometry,
    LabelCandidate,
    ObjectObservation3D,
    PersistentObject,
    SegmentationObservation,
)
from .geometry import GeometryAdapter
from .labels import label_with_candidates
from .segmentation import SegmentationAdapter


SCHEMA_VERSION = 1

_NO_POINTS = "no geometry pixels passed coverage, depth, confidence and finiteness"


@dataclass
class LiftedObservation:
    """One observation's record plus the points association may read.

    Temporary: the points are released when the frame ends and are never
    saved. The record's ID comes from segmentation and carries no object ID.
    """

    observation: ObjectObservation3D
    points_xyz: FloatArray  # N x 3 world points, not trimmed to the record's bounds


@dataclass
class SkipDiagnostic:
    """Why one mask produced no record; a note, not object evidence."""

    observation_id: str
    frame_id: str
    reason: str


@dataclass
class ObjectConstructionResult:
    """Objects and the lightweight records association had available.

    A record missing from every object's observation_ids is unresolved, which
    is a real outcome rather than an error. Skipped observations have no record
    at all.
    """

    objects: list[PersistentObject]
    observations: list[ObjectObservation3D]
    skipped_observations: list[SkipDiagnostic]


@lru_cache(maxsize=8)
def _overlap_weights(source_length: int, target_length: int, scale: float) -> FloatArray:
    """Exact 1-D overlap between each target pixel's footprint and source pixels.

    Target pixel t is centred on source coordinate t / scale and covers half a
    target pixel either side, so the footprint is 1 / scale source pixels wide;
    source pixel s covers [s - 0.5, s + 0.5]. This is the integer-pixel
    convention rgb_to_geometry records, with no half-pixel offset. Footprint
    area outside the image is simply absent, so callers normalising by the row
    sums measure coverage of the part that has pixels.
    """
    targets = np.arange(target_length)
    sources = np.arange(source_length)
    low = (targets[:, None] - 0.5) / scale
    high = (targets[:, None] + 0.5) / scale
    overlap = np.minimum(high, sources + 0.5) - np.maximum(low, sources - 0.5)
    weights = np.clip(overlap, 0, None).astype(np.float32)
    if not weights.sum(axis=1).all():
        raise ValueError("A geometry pixel maps outside the RGB image")
    return weights


def _resize_scales(rgb_to_geometry: FloatArray) -> tuple[float, float]:
    """Read the horizontal and vertical scale of a resize-only pixel mapping."""
    matrix = np.asarray(rgb_to_geometry, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Pixel mapping must be 3 x 3")
    scale_u, scale_v = matrix[0, 0], matrix[1, 1]
    if not np.isfinite([scale_u, scale_v]).all() or min(scale_u, scale_v) <= 0:
        raise ValueError("Pixel mapping must scale both axes positively")
    if not np.allclose(
        matrix, np.diag([scale_u, scale_v, 1.0]), atol=1e-6, rtol=0
    ):
        raise ValueError(
            "Only resize-only RGB-to-geometry mappings are supported; this one "
            "also translates, shears or flips"
        )
    return float(scale_u), float(scale_v)


class ObservationLifter:
    """Turn one RGB mask into world points and a small spatial record.

    Resampling follows FrameGeometry.rgb_to_geometry, which today describes a
    resize only; anything else is refused rather than approximated. A geometry
    pixel belongs to the mask when at least mask_coverage_threshold of the RGB
    area it covers is masked, measured over the part of that area inside the
    image. Non-integer scales are handled by area, not by rounding coordinates.

    min_geometry_confidence filters DA3's unnormalized confidence, not the
    segmenter's, and has no default, so the caller must supply it. Centroid and
    point_count describe every retained point; bounds are per-axis percentiles,
    which are robust to stray points but are not guaranteed enclosing boxes.
    The point array is never trimmed to them.
    This class decides validity, never identity.
    """

    def __init__(
        self,
        min_geometry_confidence: float,
        *,
        mask_coverage_threshold: float = 0.5,
        lower_box_percentile: float = 0.5,
        upper_box_percentile: float = 99.5,
        erosion_pixels: int = 2,
        depth_tolerance: float = 0.1,
        core_percentiles: tuple[float, float] = (5.0, 95.0),
        min_core_pixels: int = 20,
    ) -> None:
        if not np.isfinite(min_geometry_confidence):
            raise ValueError("Geometry confidence threshold must be finite")
        if not 0 < mask_coverage_threshold <= 1:
            raise ValueError("Mask coverage threshold must be in (0, 1]")
        if not 0 <= lower_box_percentile < upper_box_percentile <= 100:
            raise ValueError("Box percentiles must satisfy 0 <= lower < upper <= 100")
        if erosion_pixels < 0:
            raise ValueError("Erosion radius cannot be negative")
        if not np.isfinite(depth_tolerance) or depth_tolerance <= 0:
            raise ValueError("Depth tolerance must be finite and positive")
        low, high = core_percentiles
        if not 0 <= low < high <= 100:
            raise ValueError("Core percentiles must satisfy 0 <= low < high <= 100")
        if min_core_pixels < 1:
            raise ValueError("A depth core needs at least one pixel")
        self.min_geometry_confidence = float(min_geometry_confidence)
        self.mask_coverage_threshold = float(mask_coverage_threshold)
        self.lower_box_percentile = float(lower_box_percentile)
        self.upper_box_percentile = float(upper_box_percentile)
        self.erosion_pixels = int(erosion_pixels)
        self.depth_tolerance = float(depth_tolerance)
        self.core_percentiles = (float(low), float(high))
        self.min_core_pixels = int(min_core_pixels)

    def lift(
        self,
        segmentation_observation: SegmentationObservation,
        frame_geometry: FrameGeometry,
        world_points: FloatArray,
    ) -> LiftedObservation | None:
        """Select this mask's valid world points and summarise them.

        All three arguments must describe the same frame; the caller joins them
        by frame ID and supplies the map the geometry adapter produced once for
        the frame. Returns None when nothing survives filtering, which the
        caller records as a skip rather than an empty observation.
        """
        depth = frame_geometry.depth
        points = np.asarray(world_points, dtype=np.float32)
        if points.shape != (*depth.shape, 3):
            raise ValueError(
                "World points must be height x width x 3 on the depth grid"
            )
        confidence = frame_geometry.confidence
        covered = (
            self._coverage(segmentation_observation.mask, frame_geometry)
            >= self.mask_coverage_threshold
        )
        selected = self._within_depth_core(
            covered & np.isfinite(depth) & (depth > 0), depth
        )
        selected &= np.isfinite(confidence) & (confidence >= self.min_geometry_confidence)
        selected &= np.isfinite(points).all(axis=2)

        kept = points[selected]
        if not len(kept):
            return None
        lower, upper = np.percentile(
            kept, [self.lower_box_percentile, self.upper_box_percentile], axis=0
        )
        record = ObjectObservation3D(
            observation_id=segmentation_observation.observation_id,
            frame_id=segmentation_observation.frame_id,
            label=segmentation_observation.label,
            centroid_xyz=kept.mean(axis=0).astype(np.float32),
            bounds_min_xyz=lower.astype(np.float32),
            bounds_max_xyz=upper.astype(np.float32),
            point_count=len(kept),
            segmentation_confidence=segmentation_observation.confidence,
            track_hint=segmentation_observation.track_hint,
            label_candidates=segmentation_observation.label_candidates,
        )
        return LiftedObservation(record, kept)

    def _within_depth_core(self, usable: BoolArray, depth: FloatArray) -> BoolArray:
        """Drop pixels whose depth puts them off the object's own surface.

        A mask overshoots its object by a pixel or two onto whatever lies
        behind it, and those pixels unproject to points metres away that no
        later stage can identify as wrong. Confidence does not separate them:
        the background is usually real and confidently measured. Depth does.

        Eroding the covered pixels gives an interior whose depths describe the
        object rather than its edge, and the percentile range of that interior,
        widened by depth_tolerance, decides which pixels belong. The interior is
        a *reference*, not a decision, so a pixel outside it is kept whenever
        its depth agrees; thin parts are not deleted for being thin. When
        erosion leaves too little to describe anything, the reference is taken
        from every covered pixel instead, which still removes a mask that has
        spilled onto a distant wall.

        The core is taken before the confidence filter, so a low-confidence
        interior still defines the surface its own edges are judged against.
        """
        if self.erosion_pixels < 1 or not usable.any():
            return usable
        from scipy.ndimage import binary_erosion

        size = 2 * self.erosion_pixels + 1
        core = binary_erosion(usable, np.ones((size, size), dtype=bool))
        if core.sum() < self.min_core_pixels:
            core = usable
        low, high = np.percentile(depth[core], self.core_percentiles)
        return (
            usable
            & (depth >= low - self.depth_tolerance)
            & (depth <= high + self.depth_tolerance)
        )

    def _coverage(self, mask: BoolArray, frame_geometry: FrameGeometry) -> FloatArray:
        """Masked fraction of each geometry pixel's RGB footprint."""
        scale_u, scale_v = _resize_scales(frame_geometry.rgb_to_geometry)
        height, width = frame_geometry.depth.shape
        expected = (round(height / scale_v), round(width / scale_u))
        if mask.shape != expected:
            raise ValueError(
                f"Mask is {mask.shape}; this frame's pixel mapping expects the "
                f"original RGB grid {expected}"
            )
        rows = _overlap_weights(expected[0], height, scale_v)
        columns = _overlap_weights(expected[1], width, scale_u)
        area = np.outer(rows.sum(axis=1), columns.sum(axis=1))
        return (rows @ np.asarray(mask, dtype=np.float32) @ columns.T) / area


class ObjectAssociator:
    """Decide which object each lifted observation belongs to, from geometry.

    One observation is matched to at most one object and one object absorbs at
    most one observation per frame. That rule is what keeps two chairs standing
    side by side from collapsing into one: without it a mask straddling both
    can extend either.

    Scoring is overlap, not distance. An observation's points are reduced to a
    voxel set and scored by the fraction of those voxels lying within
    match_radius of an object's points, so a small observation seen close up is
    comparable with a large one seen from across the room. A match must clear
    min_overlap and beat the runner-up by ambiguity_margin; when two objects
    are equally plausible the observation is left unassigned rather than
    guessed.

    Object points are stored voxel-reduced, so an object's memory is bounded by
    the volume it occupies rather than by how many times it was seen.

    Labels are recorded as evidence and by default never used to match. A
    segmenter whose labels are consistent across frames can use them:
    same_label_only lets an observation match only objects whose voted label
    is its own, so objects of different classes are never joined.

    Matching is decided by geometry alone. The segmenter's track hints are
    kept on each record; FragmentMerger uses a shared hint to confirm that two
    geometrically overlapping objects are one.
    """

    def __init__(
        self,
        *,
        match_radius: float = 0.10,
        gate_radius: float = 1.0,
        min_overlap: float = 0.15,
        ambiguity_margin: float = 0.10,
        min_points: int = 200,
        min_new_object_points: int = 500,
        voxel_size: float = 0.05,
        trust_track_hints: bool = False,
        same_label_only: bool = False,
    ) -> None:
        settings = (
            match_radius,
            gate_radius,
            min_overlap,
            ambiguity_margin,
            voxel_size,
        )
        if not np.isfinite(settings).all() or min(settings[:2] + settings[4:]) <= 0:
            raise ValueError("Radii and voxel size must be finite and positive")
        if not 0 < min_overlap <= 1 or not 0 <= ambiguity_margin <= 1:
            raise ValueError("min_overlap must be in (0, 1] and the margin in [0, 1]")
        if gate_radius < match_radius:
            raise ValueError("The gate must be at least as wide as the match radius")
        if min_points < 1 or min_new_object_points < min_points:
            raise ValueError(
                "Starting an object must need at least as many points as matching one"
            )
        self.match_radius = float(match_radius)
        self.gate_radius = float(gate_radius)
        self.min_overlap = float(min_overlap)
        self.ambiguity_margin = float(ambiguity_margin)
        self.min_points = int(min_points)
        self.min_new_object_points = int(min_new_object_points)
        self.voxel_size = float(voxel_size)
        self.trust_track_hints = bool(trust_track_hints)
        self.same_label_only = bool(same_label_only)
        self._object_of_hint: dict[str, str] = {}

    def associate_frame(
        self,
        lifted_observations: list[LiftedObservation],
        objects: list[PersistentObject],
    ) -> None:
        """Extend or create objects from one frame's observations, in place.

        Observations are considered together rather than one at a time, so the
        one-to-one rule can be enforced across the whole frame instead of
        depending on the order masks happen to arrive in.
        """
        usable = [
            item
            for item in lifted_observations
            if len(item.points_xyz) >= self.min_points
        ]

        # An observation the segmenter has already tracked to a known object
        # goes straight there, and that object is then spoken for this frame.
        by_id = {held.object_id: index for index, held in enumerate(objects)}
        settled: set[int] = set()
        spoken_for: set[int] = set()
        if self.trust_track_hints:
            for row, item in enumerate(usable):
                hint = item.observation.track_hint
                known = self._object_of_hint.get(hint) if hint is not None else None
                column = by_id.get(known) if known is not None else None
                if column is None or column in spoken_for:
                    continue
                self._extend(objects[column], item)
                settled.add(row)
                spoken_for.add(column)
        remaining = [item for row, item in enumerate(usable) if row not in settled]

        # Every object within the gate scores, including those scoring too low
        # to be matched: a weak rival still means the evidence is divided, and
        # hiding it would let a barely-better object claim a mask outright.
        scores = self._score(remaining, objects)
        ranked: dict[int, list[tuple[float, int]]] = {}
        for overlap, row, column in scores:
            ranked.setdefault(row, []).append((overlap, column))

        # None marks an observation that was considered and rejected as
        # ambiguous. It must not go on to start an object of its own: doing so
        # turns every undecidable mask into a new fragment, which makes the next
        # frame's masks ambiguous in turn.
        decided: dict[int, int | None] = {}
        taken_objects: set[int] = set(spoken_for)
        for overlap, row, column in sorted(scores, key=lambda entry: -entry[0]):
            if overlap < self.min_overlap or row in decided or column in taken_objects:
                continue
            runner_up = max(
                (rival for rival, other in ranked[row] if other != column), default=0.0
            )
            if overlap - runner_up < self.ambiguity_margin:
                decided[row] = None
                continue
            self._extend(objects[column], remaining[row])
            self._remember(remaining[row], objects[column])
            decided[row] = column
            taken_objects.add(column)

        for index, item in enumerate(remaining):
            if index in decided:
                continue
            if len(item.points_xyz) < self.min_new_object_points:
                continue
            created = self._create(item, len(objects))
            objects.append(created)
            self._remember(item, created)

    def _remember(self, item: LiftedObservation, held: PersistentObject) -> None:
        """Tie this observation's track to the object it turned out to be."""
        hint = item.observation.track_hint
        if self.trust_track_hints and hint is not None:
            self._object_of_hint.setdefault(hint, held.object_id)

    def _score(
        self, observations: list[LiftedObservation], objects: list[PersistentObject]
    ) -> list[tuple[float, int, int]]:
        """Overlap of each observation with each object it could plausibly touch."""
        if not observations or not objects:
            return []
        from scipy.spatial import cKDTree

        trees = [cKDTree(np.asarray(item.points_xyz)) for item in objects]
        centres = np.array([item.centroid_xyz for item in objects], dtype=float)
        labels = np.array([_voted_label(held) for held in objects])
        scores = []
        for row, item in enumerate(observations):
            voxels = self._voxels(item.points_xyz)
            centroid = np.asarray(item.observation.centroid_xyz, dtype=float)
            near = np.linalg.norm(centres - centroid, axis=1) <= self.gate_radius
            if self.same_label_only:
                near &= labels == item.observation.label
            for column in np.flatnonzero(near):
                distance, _ = trees[column].query(voxels, k=1)
                overlap = float(np.mean(distance <= self.match_radius))
                scores.append((overlap, row, int(column)))
        return scores

    def _voxels(self, points: FloatArray) -> FloatArray:
        """One representative point per occupied voxel, in world coordinates."""
        keys = np.floor(np.asarray(points, dtype=float) / self.voxel_size)
        _, first = np.unique(keys, axis=0, return_index=True)
        return np.asarray(points, dtype=float)[np.sort(first)]

    def _create(self, item: LiftedObservation, index: int) -> PersistentObject:
        points = self._voxels(item.points_xyz)
        lower, upper = _box(points)
        return PersistentObject(
            object_id=f"object:{index:04d}",
            points_xyz=points.astype(np.float32),
            label_counts={item.observation.label: 1},
            centroid_xyz=points.mean(axis=0).astype(np.float32),
            bounds_min_xyz=lower,
            bounds_max_xyz=upper,
            observation_ids=[item.observation.observation_id],
            label_scores=_observation_label_scores(item.observation),
        )

    def _extend(self, held: PersistentObject, item: LiftedObservation) -> None:
        points = self._voxels(
            np.vstack([np.asarray(held.points_xyz, dtype=float), item.points_xyz])
        )
        held.points_xyz = points.astype(np.float32)
        held.centroid_xyz = points.mean(axis=0).astype(np.float32)
        held.bounds_min_xyz, held.bounds_max_xyz = _box(points)
        scores = _persistent_label_scores(held)
        for candidate, score in _observation_label_scores(item.observation).items():
            scores[candidate] = scores.get(candidate, 0.0) + score
        held.label_scores = scores
        label = item.observation.label
        held.label_counts[label] = held.label_counts.get(label, 0) + 1
        held.observation_ids.append(item.observation.observation_id)


class IdentityAssociator(ObjectAssociator):
    """Take each observation's hint as its object's identity: an oracle, not matching.

    For the ground-truth ablation, whose segmentation carries the annotated
    instance in the hint (ScanNetGtBackend with identity_hints=True). Every
    observation with the same hint joins the same object and different hints
    never join, so no overlap, ambiguity or label rule is applied. The point
    thresholds still hold: an observation under min_points is ignored, and an
    object starts only from one with at least min_new_object_points.
    """

    def __init__(self, **settings) -> None:
        super().__init__(**settings)
        self._index_of_hint: dict[str, int] = {}

    def associate_frame(
        self,
        lifted_observations: list[LiftedObservation],
        objects: list[PersistentObject],
    ) -> None:
        for item in lifted_observations:
            if len(item.points_xyz) < self.min_points:
                continue
            hint = item.observation.track_hint
            if hint is None:
                raise ValueError(
                    "IdentityAssociator needs every observation to carry its identity"
                )
            existing = self._index_of_hint.get(hint)
            if existing is not None:
                self._extend(objects[existing], item)
            elif len(item.points_xyz) >= self.min_new_object_points:
                objects.append(self._create(item, len(objects)))
                self._index_of_hint[hint] = len(objects) - 1


def _mean_direction(
    observation_ids: list[str], descriptors: dict[str, FloatArray] | None
) -> FloatArray | None:
    """One unit vector describing how an object looks, or None if unknown."""
    if not descriptors:
        return None
    vectors = [descriptors[o] for o in observation_ids if o in descriptors]
    if not vectors:
        return None
    mean = np.mean(np.asarray(vectors, dtype=float), axis=0)
    return mean / (np.linalg.norm(mean) + 1e-9)


# Labels naming something a room is furnished with rather than something set
# down in it. These are the objects a single view cannot hold, and the only
# ones for which co-visibility says nothing; see FragmentMerger.
LARGE_SURFACES = frozenset({
    "table", "desk", "counter", "cabinet", "couch", "bed", "shelf", "bookshelf",
    "dresser", "nightstand", "tv stand", "bench", "ottoman", "wardrobe", "curtain",
})


def _voted_label(held: PersistentObject) -> str:
    """The label with the most accumulated evidence, or ''."""
    scores = _persistent_label_scores(held)
    if not scores:
        return ""
    return label_with_candidates(scores, max_aliases=0)[0]


def _observation_label_scores(item: ObjectObservation3D) -> dict[str, float]:
    """Primary vote plus relative runner-up evidence from one observation."""
    scores = {item.label: 1.0}
    for candidate in item.label_candidates:
        if candidate.label == item.label:
            continue
        scores[candidate.label] = max(
            scores.get(candidate.label, 0.0), float(candidate.score)
        )
    return scores


def _persistent_label_scores(held: PersistentObject) -> dict[str, float]:
    """Return stored evidence, falling back to legacy primary-label counts."""
    if held.label_scores:
        return dict(held.label_scores)
    return {label: float(count) for label, count in held.label_counts.items()}


def _box(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """An object's box: the 0.5th and 99.5th percentile of its points per axis."""
    return (
        np.percentile(points, 0.5, axis=0).astype(np.float32),
        np.percentile(points, 99.5, axis=0).astype(np.float32),
    )


def _longest_side(held: PersistentObject) -> float:
    return float(np.max(np.asarray(held.bounds_max_xyz) - np.asarray(held.bounds_min_xyz)))


class FragmentMerger:
    """Fuse object fragments once the frame loop has finished.

    During the loop identity is decided at frame t, comparing one surface shell
    against a partially built object. Comparing two built objects instead gives
    both sides many frames, and with them a signal no single frame carries: two
    fragments of the same object are never visible in the same frame, whereas
    two distinct objects standing near each other are visible together
    constantly.

    Co-occurrence is therefore a veto, not a score. Fragments sharing any frame
    are never fused however close they sit, which is what makes this safer than
    loosening the frame-by-frame matcher.

    One class of object breaks that premise, and only one. A surface too large
    to sit in a single view comes back from segmentation as several instances
    in the same frame, so its pieces are co-visible constantly and the veto
    keeps them apart for good. The exemption is therefore narrow: both sides
    must agree on a label naming a large surface, and one of them must already
    be at least large_surface_metres across.

    Overlap alone is not enough to act on: left to itself it also fuses
    distinct objects. So a merge also needs one independent signal to agree:

        overlap >= min_overlap  AND  (shared track  OR  similar appearance)

    The two are complementary rather than redundant. A shared track is the more
    precise signal but fires rarely, because a tracker that loses an object
    renames it; appearance reaches pairs the tracker never connected. Each
    catches merges the other misses, so either suffices.
    """

    def __init__(
        self,
        *,
        match_radius: float = 0.2,
        min_overlap: float = 0.5,
        min_similarity: float = 0.5,
        large_surface_metres: float = 1.0,
    ) -> None:
        if not np.isfinite([match_radius, min_overlap]).all() or match_radius <= 0:
            raise ValueError("Match radius must be finite and positive")
        if not 0 < min_overlap <= 1:
            raise ValueError("Overlap threshold must be in (0, 1]")
        if not -1 <= min_similarity <= 1:
            raise ValueError("Similarity is a cosine and must be in [-1, 1]")
        if not np.isfinite(large_surface_metres) or large_surface_metres <= 0:
            raise ValueError("The large-surface size must be finite and positive")
        self.match_radius = float(match_radius)
        self.min_overlap = float(min_overlap)
        self.min_similarity = float(min_similarity)
        self.large_surface_metres = float(large_surface_metres)

    def _one_large_surface(self, left: PersistentObject, right: PersistentObject) -> bool:
        """May these two be pieces of a surface too big to see all at once?

        Only then is co-visibility uninformative, because segmentation returns
        such a surface as several instances in the same frame.
        """
        label = _voted_label(left)
        if label != _voted_label(right) or label not in LARGE_SURFACES:
            return False
        return max(_longest_side(left), _longest_side(right)) >= self.large_surface_metres

    def _confirmed(
        self,
        left: set[str],
        right: set[str],
        left_look: FloatArray | None,
        right_look: FloatArray | None,
    ) -> bool:
        """Does anything other than geometry agree these are one object?"""
        if left & right:
            return True
        if left_look is None or right_look is None:
            return False
        return float(left_look @ right_look) >= self.min_similarity

    def merge(
        self,
        objects: list[PersistentObject],
        observations: list[ObjectObservation3D],
        descriptors: dict[str, FloatArray] | None = None,
    ) -> list[PersistentObject]:
        """Return objects with mutually compatible fragments fused.

        descriptors maps an observation ID to a unit appearance vector; with
        none supplied only a shared track can confirm a merge, which is the
        stricter of the two signals rather than the looser.
        """
        if len(objects) < 2:
            return list(objects)
        from scipy.spatial import cKDTree

        frames = {item.observation_id: item.frame_id for item in observations}
        tracks = {item.observation_id: item.track_hint for item in observations}
        seen = [
            {frames[observation_id] for observation_id in held.observation_ids
             if observation_id in frames}
            for held in objects
        ]
        tracked = [
            {tracks.get(observation_id) for observation_id in held.observation_ids}
            - {None}
            for held in objects
        ]
        looks = [_mean_direction(held.observation_ids, descriptors) for held in objects]
        trees = [cKDTree(np.asarray(held.points_xyz, dtype=float)) for held in objects]

        # Fragments accumulate into groups, and the veto is checked against the
        # whole group rather than the pair. Checking only the pair lets the veto
        # be broken by a chain: if A and C appear in one frame but B overlaps
        # both, A-B and B-C each pass and all three end up merged, which is the
        # false merge the veto exists to prevent.
        parent = list(range(len(objects)))
        frames_of_group = [set(item) for item in seen]

        def root(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left in range(len(objects)):
            for right in range(left + 1, len(objects)):
                one_surface = self._one_large_surface(objects[left], objects[right])
                if not one_surface and (seen[left] & seen[right]):
                    continue  # visible together: cannot be one object
                first, second = root(left), root(right)
                if first == second:
                    continue
                if not one_surface and (
                    frames_of_group[first] & frames_of_group[second]
                ):
                    continue  # merging these groups would put two co-visible
                    # fragments into one object by way of a third
                smaller, larger = sorted((left, right), key=lambda i: len(trees[i].data))
                distance, _ = trees[larger].query(trees[smaller].data, k=1)
                if float(np.mean(distance <= self.match_radius)) < self.min_overlap:
                    continue
                if not self._confirmed(
                    tracked[left], tracked[right], looks[left], looks[right]
                ):
                    continue
                keep, absorbed = sorted((first, second))
                parent[absorbed] = keep
                frames_of_group[keep] |= frames_of_group[absorbed]

        groups: dict[int, list[int]] = {}
        for index in range(len(objects)):
            groups.setdefault(root(index), []).append(index)

        merged = []
        for position, members in enumerate(groups.values()):
            if len(members) == 1:
                held = objects[members[0]]
                merged.append(
                    replace(
                        held,
                        object_id=f"object:{position:04d}",
                        label_counts=dict(held.label_counts),
                        observation_ids=list(held.observation_ids),
                        label_scores=_persistent_label_scores(held),
                    )
                )
                continue
            points = np.vstack(
                [np.asarray(objects[i].points_xyz, dtype=float) for i in members]
            )
            labels: dict[str, int] = {}
            label_scores: dict[str, float] = {}
            identifiers: list[str] = []
            for index in members:
                for label, count in objects[index].label_counts.items():
                    labels[label] = labels.get(label, 0) + count
                for label, score in _persistent_label_scores(objects[index]).items():
                    label_scores[label] = label_scores.get(label, 0.0) + score
                identifiers.extend(objects[index].observation_ids)
            lower, upper = _box(points)
            merged.append(
                PersistentObject(
                    object_id=f"object:{position:04d}",
                    points_xyz=points.astype(np.float32),
                    label_counts=labels,
                    centroid_xyz=points.mean(axis=0).astype(np.float32),
                    bounds_min_xyz=lower,
                    bounds_max_xyz=upper,
                    observation_ids=identifiers,
                    label_scores=label_scores,
                )
            )
        return merged


def tighten_boxes(
    objects: list[PersistentObject],
    *,
    neighbours: int = 12,
    sigma: float = 1.0,
) -> list[PersistentObject]:
    """Recompute each box without the halo of stray points around mask edges.

    A mask's border bleeds a little onto whatever is behind it, and those
    points land away from the object's surface. They are few, so they barely
    move the centroid, but a box is decided by its extremes and they push it
    outwards.

    Dropping points whose mean distance to their nearest neighbours is more
    than sigma above average, then taking the same percentiles of the rest,
    leaves the centroid where it was and moves only the extremes.

    This runs once, after association, and changes no identity decision: the
    points and centroids objects were matched on are left exactly as they were.
    """
    from scipy.spatial import cKDTree

    tightened = []
    for held in objects:
        points = np.asarray(held.points_xyz, dtype=float)
        kept = points
        if len(points) > neighbours + 1:
            distance, _ = cKDTree(points).query(points, k=neighbours + 1)
            mean = distance[:, 1:].mean(axis=1)
            inliers = mean <= mean.mean() + sigma * mean.std()
            if inliers.sum() >= 8:
                kept = points[inliers]
        lower, upper = _box(kept)
        tightened.append(replace(held, bounds_min_xyz=lower, bounds_max_xyz=upper))
    return tightened


def drop_thinly_seen_objects(
    objects: list[PersistentObject], minimum: int
) -> list[PersistentObject]:
    """Remove objects supported by fewer than `minimum` observations.

    An object seen once or twice is usually a sliver of a real object picked up
    at an odd angle and never matched again. Object IDs are reassigned so the
    saved set is contiguous.
    """
    if minimum < 1:
        raise ValueError("The minimum must be at least one observation")
    kept = [held for held in objects if len(held.observation_ids) >= minimum]
    return [
        replace(
            held,
            object_id=f"object:{position:04d}",
            label_scores=_persistent_label_scores(held),
        )
        for position, held in enumerate(kept)
    ]


class MaskCarver:
    """Remove object points that the object's own masks place outside it.

    Association keeps every point an accepted observation contributed, and
    mask edges and depth errors leave some of them beside the object, where
    they widen its box. Once the frame loop is over, each retained point can be
    checked against every mask the object was seen in: projected into that
    frame, a point of the object should land inside the object's mask. A point
    is removed when at least min_votes of those frames could see it and most
    of them put it outside the mask. A frame cannot see a point behind the
    camera, outside the image, or hidden by measured depth more than
    occlusion_tolerance nearer than the point. An object is never carved below
    eight points.

    It re-reads each frame's geometry and masks, runs no model, and changes
    points and centroids but no identity decision.
    """

    def __init__(self, *, occlusion_tolerance: float = 0.1, min_votes: int = 2) -> None:
        if not np.isfinite(occlusion_tolerance) or occlusion_tolerance < 0:
            raise ValueError("Occlusion tolerance must be finite and non-negative")
        if min_votes < 1:
            raise ValueError("A point needs at least one vote to be carved")
        self.occlusion_tolerance = float(occlusion_tolerance)
        self.min_votes = int(min_votes)

    def carve(
        self,
        objects: list[PersistentObject],
        ready_geometry: GeometryAdapter,
        ready_segmentation: SegmentationAdapter,
    ) -> list[PersistentObject]:
        points = [np.asarray(held.points_xyz, dtype=float) for held in objects]
        votes = [np.zeros(len(p), int) for p in points]
        inside = [np.zeros(len(p), int) for p in points]
        owner = {
            observation_id: index
            for index, held in enumerate(objects)
            for observation_id in held.observation_ids
        }
        for frame_id in ready_segmentation.segmentation_frame_ids:
            masks = [
                (owner[item.observation_id], item.mask)
                for item in ready_segmentation.load_frame(frame_id)
                if item.observation_id in owner
            ]
            if not masks:
                continue
            frame = ready_geometry.load_frame(frame_id)
            world_to_camera = np.linalg.inv(np.asarray(frame.camera_to_world, float))
            scale_u, scale_v = _resize_scales(frame.rgb_to_geometry)
            height, width = frame.depth.shape
            for index, mask in masks:
                camera = points[index] @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
                pixel = camera @ np.asarray(frame.intrinsics, float).T
                with np.errstate(divide="ignore", invalid="ignore"):
                    u, v = pixel[:, 0] / pixel[:, 2], pixel[:, 1] / pixel[:, 2]
                seen = (camera[:, 2] > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
                column = np.where(seen, u, 0).astype(int)
                row = np.where(seen, v, 0).astype(int)
                surface = frame.depth[row, column]
                seen &= ~(
                    np.isfinite(surface)
                    & (surface < camera[:, 2] - self.occlusion_tolerance)
                )
                rgb_column = np.clip((column / scale_u).astype(int), 0, mask.shape[1] - 1)
                rgb_row = np.clip((row / scale_v).astype(int), 0, mask.shape[0] - 1)
                votes[index] += seen
                inside[index] += seen & mask[rgb_row, rgb_column]
        carved = []
        for held, kept_points, voted, agreed in zip(objects, points, votes, inside):
            outside = (voted >= self.min_votes) & (agreed * 2 < voted)
            if (~outside).sum() >= 8:
                kept_points = kept_points[~outside]
            carved.append(
                replace(
                    held,
                    points_xyz=kept_points.astype(np.float32),
                    centroid_xyz=kept_points.mean(axis=0).astype(np.float32),
                )
            )
        return carved


class ObjectConstructor:
    """Run the pass-3 frame loop over two ready adapters."""

    @classmethod
    def run(
        cls,
        ready_geometry: GeometryAdapter,
        ready_segmentation: SegmentationAdapter,
        lifter: ObservationLifter,
        associator: ObjectAssociator,
        output_directory: Path,
        *,
        merger: FragmentMerger | None = None,
        descriptors: dict[str, FloatArray] | None = None,
        min_observations: int = 1,
        carver: MaskCarver | None = None,
    ) -> ObjectConstructionResult:
        """Lift every cleaned mask, associate per frame, save, and return.

        The adapters are already-open instances from run_pipeline or
        from_saved_run; no model runs here. Both must expose the same ordered
        frame IDs. The lifter and associator arrive configured so settings and
        matching can change without touching this loop.

        Two optional steps run after the loop and before saving, in this order:
        a merger fuses fragments using evidence gathered across many frames, then
        thinly seen objects are dropped. Both change which objects are saved and
        neither changes any observation record, so an object dropped here leaves
        its observations unresolved rather than removing them. Omitting both
        saves exactly what association built. A carver, if given, then trims
        each surviving object's points against its own masks. These steps read
        accumulated objects and observations, never the video, so they can run
        on the objects built up to any frame.

        One world-point map is derived per frame and released with the rest of
        that frame's data; objects and records carry over. The result is saved
        through the writer and returned only after saving succeeds, so a
        failing associator leaves no completed output behind.
        """
        frame_ids = tuple(ready_geometry.geometry_frame_ids)
        if frame_ids != tuple(ready_segmentation.segmentation_frame_ids):
            raise ValueError(
                "Geometry and segmentation must expose the same ordered frame IDs"
            )
        output = Path(output_directory)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite object output: {output}")

        objects: list[PersistentObject] = []
        observations: list[ObjectObservation3D] = []
        skipped: list[SkipDiagnostic] = []
        for frame_id in frame_ids:
            geometry = ready_geometry.load_frame(frame_id)
            world_points = ready_geometry.world_points(geometry)
            lifted = []
            for observation in ready_segmentation.load_frame(frame_id):
                result = lifter.lift(observation, geometry, world_points)
                if result is None:
                    skipped.append(
                        SkipDiagnostic(
                            observation.observation_id, frame_id, _NO_POINTS
                        )
                    )
                    continue
                lifted.append(result)
                observations.append(result.observation)
            associator.associate_frame(lifted, objects)

        if merger is not None:
            objects = merger.merge(objects, observations, descriptors)
        if min_observations > 1:
            objects = drop_thinly_seen_objects(objects, min_observations)
        if carver is not None:
            objects = carver.carve(objects, ready_geometry, ready_segmentation)
        objects = tighten_boxes(objects)

        result = ObjectConstructionResult(objects, observations, skipped)
        ObjectConstructionWriter(
            {
                "frame_ids": list(frame_ids),
                "world_coordinates": ready_geometry.alignment_method,
                "lifter": _settings(lifter),
                "associator": _settings(associator),
                "merger": None if merger is None else _settings(merger),
                "min_observations": min_observations,
                "carver": None if carver is None else _settings(carver),
            }
        ).save(result, output)
        return result


class ObjectConstructionWriter:
    """Save one result as readable records plus compressed point arrays.

    metadata describes the settings and coordinates needed to read the result
    later; the writer records it rather than deriving it, and never matches
    objects or chooses their labels. This is our working cache, not the
    downstream scene record, so its layout can change with pass 3.
    """

    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata

    def save(
        self, result: ObjectConstructionResult, output_directory: Path
    ) -> None:
        """Write the records, the object points, and finally the completion marker.

        Refuses an existing directory. References and arrays are checked before
        anything is written, so a rejected result leaves nothing behind, and
        the marker is written last after reading the files back: an interrupted
        save stays unmistakably incomplete. Temporary observation points and
        full-frame maps are not saved.
        """
        _validate(result)
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            output / "object_points.npz",
            **{
                f"points_{index:06d}": np.asarray(item.points_xyz, dtype=np.float32)
                for index, item in enumerate(result.objects)
            },
        )
        document = {
            "schema_version": SCHEMA_VERSION,
            "metadata": self.metadata,
            "objects": [
                {
                    "object_id": item.object_id,
                    "points": f"points_{index:06d}",
                    "label_counts": {
                        str(label): int(count)
                        for label, count in item.label_counts.items()
                    },
                    "label_scores": {
                        str(label): float(score)
                        for label, score in _persistent_label_scores(item).items()
                    },
                    "centroid_xyz": _numbers(item.centroid_xyz),
                    "bounds_min_xyz": _numbers(item.bounds_min_xyz),
                    "bounds_max_xyz": _numbers(item.bounds_max_xyz),
                    "observation_ids": list(item.observation_ids),
                }
                for index, item in enumerate(result.objects)
            ],
            "observations": [
                {
                    "observation_id": item.observation_id,
                    "frame_id": item.frame_id,
                    "label": item.label,
                    "centroid_xyz": _numbers(item.centroid_xyz),
                    "bounds_min_xyz": _numbers(item.bounds_min_xyz),
                    "bounds_max_xyz": _numbers(item.bounds_max_xyz),
                    "point_count": int(item.point_count),
                    "segmentation_confidence": float(item.segmentation_confidence),
                    "track_hint": item.track_hint,
                    "label_candidates": [
                        {"label": candidate.label, "score": candidate.score}
                        for candidate in item.label_candidates
                    ],
                }
                for item in result.observations
            ],
            "skipped_observations": [
                {
                    "observation_id": item.observation_id,
                    "frame_id": item.frame_id,
                    "reason": item.reason,
                }
                for item in result.skipped_observations
            ],
        }
        (output / "objects.json").write_text(json.dumps(document, indent=2) + "\n")

        restored = read_object_construction(output, require_complete=False)[1]
        if [item.object_id for item in restored.objects] != [
            item.object_id for item in result.objects
        ] or len(restored.observations) != len(result.observations):
            raise ValueError("Saved object records failed verification")
        (output / "run_complete.json").write_text(
            json.dumps(
                {
                    "object_count": len(result.objects),
                    "observation_count": len(result.observations),
                    "skipped_count": len(result.skipped_observations),
                }
            )
        )


def read_object_construction(
    output_directory: Path, *, require_complete: bool = True
) -> tuple[dict, ObjectConstructionResult]:
    """Read a saved result back, returning its metadata and collections.

    require_complete=False is for the writer's own verification; every other
    caller needs the completion marker, because an incomplete directory can
    hold records whose point arrays were never finished.
    """
    output = Path(output_directory)
    document = json.loads((output / "objects.json").read_text())
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported object cache schema_version: {document.get('schema_version')}"
        )
    if require_complete:
        marker = json.loads((output / "run_complete.json").read_text())
        if marker.get("object_count") != len(document["objects"]):
            raise ValueError("Object completion marker does not match the records")
    with np.load(output / "object_points.npz", allow_pickle=False) as arrays:
        objects = [
            PersistentObject(
                object_id=item["object_id"],
                points_xyz=np.asarray(arrays[item["points"]], dtype=np.float32),
                label_counts=dict(item["label_counts"]),
                centroid_xyz=np.asarray(item["centroid_xyz"], dtype=np.float32),
                bounds_min_xyz=np.asarray(item["bounds_min_xyz"], dtype=np.float32),
                bounds_max_xyz=np.asarray(item["bounds_max_xyz"], dtype=np.float32),
                observation_ids=list(item["observation_ids"]),
                label_scores={
                    str(label): float(score)
                    for label, score in item.get(
                        "label_scores", item["label_counts"]
                    ).items()
                },
            )
            for item in document["objects"]
        ]
    observations = [
        ObjectObservation3D(
            observation_id=item["observation_id"],
            frame_id=item["frame_id"],
            label=item["label"],
            centroid_xyz=np.asarray(item["centroid_xyz"], dtype=np.float32),
            bounds_min_xyz=np.asarray(item["bounds_min_xyz"], dtype=np.float32),
            bounds_max_xyz=np.asarray(item["bounds_max_xyz"], dtype=np.float32),
            point_count=item["point_count"],
            segmentation_confidence=item["segmentation_confidence"],
            track_hint=item["track_hint"],
            label_candidates=tuple(
                LabelCandidate(candidate["label"], candidate["score"])
                for candidate in item.get("label_candidates", [])
            ),
        )
        for item in document["observations"]
    ]
    skipped = [
        SkipDiagnostic(item["observation_id"], item["frame_id"], item["reason"])
        for item in document["skipped_observations"]
    ]
    return document["metadata"], ObjectConstructionResult(
        objects, observations, skipped
    )


def _settings(component: object) -> dict:
    """The component's class and whichever of its attributes are plain values.

    A run should record what produced it, but these are injection points and a
    caller's own associator may hold counters, caches or arrays alongside its
    settings. Only scalars and short sequences of scalars are recorded, so the
    metadata stays readable and saving never fails on something unexpected.
    """
    simple = (bool, int, float, str, type(None))
    recorded: dict = {"class": type(component).__name__}
    for name, value in sorted(vars(component).items()):
        if name.startswith("_"):
            continue
        if isinstance(value, simple):
            recorded[name] = value
        elif isinstance(value, (tuple, list)) and 0 < len(value) <= 8:
            if all(isinstance(entry, simple) for entry in value):
                recorded[name] = list(value)
    return recorded


def _numbers(values: FloatArray) -> list[float]:
    """Write a short vector as plain JSON numbers, refusing non-finite ones."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("Expected three finite coordinates")
    return [float(value) for value in array]


def _validate(result: ObjectConstructionResult) -> None:
    """Check IDs, references and point arrays before anything is written."""
    observation_ids = [item.observation_id for item in result.observations]
    known = set(observation_ids)
    if len(known) != len(observation_ids):
        raise ValueError("Repeated observation ID in the result")
    object_ids = [item.object_id for item in result.objects]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Repeated object ID in the result")

    claimed: set[str] = set()
    for item in result.objects:
        points = np.asarray(item.points_xyz)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"Object {item.object_id} needs finite N x 3 points")
        for observation_id in item.observation_ids:
            if observation_id not in known:
                raise ValueError(
                    f"Object {item.object_id} references unknown observation "
                    f"{observation_id}"
                )
            if observation_id in claimed:
                raise ValueError(
                    f"Observation {observation_id} belongs to two objects"
                )
            claimed.add(observation_id)

    for item in result.skipped_observations:
        if item.observation_id in known:
            raise ValueError(
                f"Observation {item.observation_id} is both recorded and skipped"
            )
