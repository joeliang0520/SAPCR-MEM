"""Write and read the tool-independent segmentation cache."""

from abc import ABC, abstractmethod
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import manifests
from .contracts import FrameSequence, LabelCandidate, SegmentationObservation
from .frames import frame_timestamps


class SegmentationAdapter:
    """Read a complete cleaned cache by stable frame ID.

    All methods receive the selected FrameSequence from input preparation,
    not the geometry adapter. Masks use original RGB pixels. Hints are opaque,
    sequence-scoped suggestions, not guaranteed physical identities.
    """

    def __init__(
        self, selected_sequence: FrameSequence, output_directory: Path
    ) -> None:
        frame_timestamps(selected_sequence)
        self.directory = Path(output_directory)
        manifest, entries = manifests.read(
            self.directory / "manifest.json",
            selected_sequence.sequence_id,
            label="Cleaned segmentation",
        )
        completion = json.loads((self.directory / "run_complete.json").read_text())
        if completion.get("frame_count") != len(entries) or not manifests.matches_frames(
            entries, selected_sequence.frames
        ):
            raise ValueError(
                "Cleaned segmentation cache does not match selected frames"
            )
        self._entries = {
            entry["frame_id"]: (i, entry) for i, entry in enumerate(entries)
        }
        observation_ids = []
        for entry in entries:
            records = entry["observations"]
            hints = [r["track_hint"] for r in records if r["track_hint"] is not None]
            if len(set(hints)) != len(hints):
                raise ValueError("Repeated track hint in one cleaned frame")
            observation_ids.extend(r["observation_id"] for r in records)
            if any(not r["label"] or not np.isfinite(r["confidence"]) for r in records):
                raise ValueError(
                    "Cleaned observations require labels and finite scores"
                )
            for record in records:
                candidates = record.get("label_candidates", [])
                parsed = tuple(
                    LabelCandidate(item["label"], item["score"])
                    for item in candidates
                )
                labels = [item.label for item in parsed]
                if record["label"] in labels or len(labels) != len(set(labels)):
                    raise ValueError(
                        "Runner-up labels must be unique and exclude the primary label"
                    )
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("Repeated observation ID in cleaned cache")

    @property
    def segmentation_frame_ids(self) -> tuple[str, ...]:
        """Processed frame IDs, including frames with zero detections."""
        return tuple(self._entries)

    def load_frame(self, frame_id: str) -> list[SegmentationObservation]:
        """Read original-grid masks; unknown/unprocessed IDs raise KeyError.

        A processed frame without detections returns []. Reopening/reading
        never runs a model or decoding, and never writes to either cache.
        """
        position, entry = self._entries[frame_id]
        records = entry["observations"]
        with np.load(
            self.directory / "masks" / f"{position:06d}.npz", allow_pickle=False
        ) as result:
            shape = tuple(int(v) for v in result["shape"])
            if shape != (len(records), *entry["rgb_size_hw"]):
                raise ValueError(
                    "Cleaned masks do not match manifest/native RGB dimensions"
                )
            packed = result["packed"]
            if packed.dtype != np.uint8 or packed.shape != (
                shape[0],
                (shape[1] * shape[2] + 7) // 8,
            ):
                raise ValueError("Cleaned packed masks have incorrect byte dimensions")
            masks = (
                np.unpackbits(packed, axis=1, count=shape[1] * shape[2])
                .reshape(shape)
                .astype(bool)
            )
        return [
            SegmentationObservation(
                r["observation_id"],
                frame_id,
                mask,
                r["label"],
                r["confidence"],
                r["track_hint"],
                tuple(
                    LabelCandidate(item["label"], item["score"])
                    for item in r.get("label_candidates", [])
                ),
            )
            for r, mask in zip(records, masks)
        ]

    @classmethod
    def from_saved_run(
        cls, selected_sequence: FrameSequence, output_directory: Path
    ) -> "SegmentationAdapter":
        """Open cleaned output only; its completion marker is mandatory."""
        return cls(selected_sequence, output_directory)


class SegmentationBackend(ABC):
    """Prepare one segmentation source behind the shared cache contract."""

    @abstractmethod
    def prepare(
        self, selected_sequence: FrameSequence, output_directory: Path
    ) -> SegmentationAdapter:
        """Prepare or reopen observations and return their read-only adapter."""


class SegmentationCacheWriter:
    """Write one complete canonical cache in selected-frame order."""

    _RESERVED_METADATA = {"schema_version", "sequence_id", "frames"}

    def __init__(
        self, selected_sequence: FrameSequence, output_directory: Path
    ) -> None:
        frame_timestamps(selected_sequence)
        self.sequence = selected_sequence
        self.directory = Path(output_directory)
        if self.directory.exists():
            raise FileExistsError(
                f"Refusing to overwrite cleaned results: {self.directory}"
            )
        self.directory.mkdir(parents=True)
        (self.directory / "masks").mkdir()
        self._entries: list[dict[str, object]] = []
        self._observation_ids: set[str] = set()
        self._finished = False

    def write_frame(
        self, frame_id: str, observations: list[SegmentationObservation]
    ) -> None:
        """Write the next selected frame and validate the shared contract."""
        if self._finished:
            raise RuntimeError("Segmentation cache is already finished")
        position = len(self._entries)
        if position >= len(self.sequence.frames):
            raise ValueError("Segmentation cache has more frames than its sequence")
        frame = self.sequence.frames[position]
        if frame_id != frame.frame_id:
            raise ValueError(
                f"Expected segmentation frame {frame.frame_id}, got {frame_id}"
            )
        with Image.open(frame.rgb_path) as image:
            rgb_size = (image.height, image.width)

        hints: list[str] = []
        for observation in observations:
            if observation.frame_id != frame_id:
                raise ValueError("Segmentation observation belongs to another frame")
            if (
                not observation.observation_id
                or observation.observation_id in self._observation_ids
            ):
                raise ValueError("Segmentation observation IDs must be unique")
            if (
                not observation.label
                or not np.isfinite(observation.confidence)
                or not isinstance(observation.mask, np.ndarray)
                or observation.mask.dtype != np.bool_
                or observation.mask.shape != rgb_size
            ):
                raise ValueError(
                    "Segmentation observations must have a native-grid Boolean mask, "
                    "label and finite confidence"
                )
            candidate_labels = [item.label for item in observation.label_candidates]
            if (
                observation.label in candidate_labels
                or len(candidate_labels) != len(set(candidate_labels))
            ):
                raise ValueError(
                    "Runner-up labels must be unique and exclude the primary label"
                )
            self._observation_ids.add(observation.observation_id)
            if observation.track_hint is not None:
                hints.append(observation.track_hint)
        if len(hints) != len(set(hints)):
            raise ValueError("Repeated track hint in one cleaned frame")

        height, width = rgb_size
        masks = (
            np.stack([item.mask for item in observations])
            if observations
            else np.zeros((0, height, width), dtype=bool)
        )
        np.savez_compressed(
            self.directory / "masks" / f"{position:06d}.npz",
            shape=np.asarray(masks.shape, dtype=np.int64),
            packed=np.packbits(
                masks.reshape(len(observations), height * width), axis=1
            ),
        )
        self._entries.append(
            {
                "frame_id": frame.frame_id,
                "source_frame_index": frame.source_frame_index,
                "timestamp_seconds": frame.timestamp_seconds,
                "rgb_size_hw": list(rgb_size),
                "observations": [
                    {
                        "observation_id": item.observation_id,
                        "label": item.label,
                        "confidence": item.confidence,
                        "track_hint": item.track_hint,
                        "label_candidates": [
                            {"label": candidate.label, "score": candidate.score}
                            for candidate in item.label_candidates
                        ],
                    }
                    for item in observations
                ],
            }
        )

    def finish(self, metadata: dict[str, object]) -> SegmentationAdapter:
        """Publish the completion marker last, then return the cache reader."""
        if self._finished:
            raise RuntimeError("Segmentation cache is already finished")
        if len(self._entries) != len(self.sequence.frames):
            raise ValueError("Cleaned preparation did not cover every selected frame")
        if self._RESERVED_METADATA & metadata.keys():
            raise ValueError("Backend metadata cannot replace shared manifest fields")
        manifests.write(
            self.directory / "manifest.json",
            self.sequence.sequence_id,
            self._entries,
            **metadata,
        )
        (self.directory / "run_complete.json").write_text(
            json.dumps({"frame_count": len(self._entries)})
        )
        self._finished = True
        return SegmentationAdapter.from_saved_run(self.sequence, self.directory)


class SavedSegmentationBackend(SegmentationBackend):
    """Reuse a complete canonical cache without its original model backend."""

    def __init__(self, source_cache: Path) -> None:
        self.source_cache = Path(source_cache)

    def prepare(
        self, selected_sequence: FrameSequence, output_directory: Path
    ) -> SegmentationAdapter:
        source = self.source_cache.resolve(strict=True)
        adapter = SegmentationAdapter.from_saved_run(selected_sequence, source)
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=False)
        (output / "cleaned").symlink_to(source, target_is_directory=True)
        return adapter
