"""Read ScanNet's 2D instance annotations as segmentation observations.

This is an oracle ablation, not RGB-only output: masks and labels are
ScanNet's own annotations. By default everything after segmentation stays the
pipeline's: identity is decided by association, not carried from ScanNet, so
observations get positional IDs and no track hints. identity_hints=True is the
further oracle rung: each observation's hint is its ScanNet instance, for
IdentityAssociator to use as the object's identity.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import FrameSequence, SegmentationObservation
from ..frames import frame_timestamps
from ..segmentation import SegmentationAdapter, SegmentationBackend, SegmentationCacheWriter

# ScanNet raw labels that are room structure rather than objects.
STRUCTURE_LABELS = frozenset(
    {"wall", "floor", "ceiling", "door", "window", "doorframe", "floor mat"}
)


class ScanNetInstanceSegmentation:
    """Serve ScanNet's projected instance masks through the adapter interface.

    instance_directory holds ScanNet's instance-filt PNGs, named by source frame
    index, where a pixel value is the aggregation objectId + 1 and 0 is
    unannotated. Labels are the aggregation's raw labels mapped through
    label_mapping (the benchmark's raw-to-canonical names); an unmapped label is
    kept as it is and listed in `unmapped`. Structure is left out.
    """

    def __init__(
        self,
        selected_sequence: FrameSequence,
        instance_directory: Path,
        aggregation_path: Path,
        label_mapping: dict[str, str],
        *,
        identity_hints: bool = False,
    ) -> None:
        frame_timestamps(selected_sequence)
        self.sequence_id = selected_sequence.sequence_id
        self._frames = {
            frame.frame_id: (position, frame)
            for position, frame in enumerate(selected_sequence.frames)
        }
        self.instance_directory = Path(instance_directory)
        self.identity_hints = bool(identity_hints)
        groups = json.loads(Path(aggregation_path).read_text())["segGroups"]
        self.labels = {
            int(group["objectId"]) + 1: label_mapping.get(group["label"], group["label"])
            for group in groups
            if group["label"] not in STRUCTURE_LABELS
        }
        self.structure = {
            int(group["objectId"]) + 1 for group in groups if group["label"] in STRUCTURE_LABELS
        }
        self.unmapped = sorted(
            {g["label"] for g in groups if g["label"] not in STRUCTURE_LABELS}
            - set(label_mapping)
        )

    @property
    def segmentation_frame_ids(self) -> tuple[str, ...]:
        return tuple(self._frames)

    def load_frame(self, frame_id: str) -> list[SegmentationObservation]:
        """One observation per annotated object visible in the frame."""
        position, frame = self._frames[frame_id]
        with Image.open(
            self.instance_directory / f"{frame.source_frame_index}.png"
        ) as image:
            values = np.asarray(image)
        observations = []
        for value in np.unique(values):
            if value == 0 or value in self.structure:
                continue
            if value not in self.labels:
                raise ValueError(
                    f"Instance {value} in frame {frame_id} is not in the aggregation"
                )
            observations.append(
                SegmentationObservation(
                    f"{self.sequence_id}:obs:{position:06d}:{len(observations):03d}",
                    frame_id,
                    values == value,
                    self.labels[int(value)],
                    1.0,
                    f"{self.sequence_id}:scannet:{int(value) - 1}"
                    if self.identity_hints
                    else None,
                )
            )
        return observations


class ScanNetGtBackend(SegmentationBackend):
    """Write ScanNet's instance masks into the canonical segmentation cache.

    The ground-truth ablation's segmentation backend: the reader above supplies
    the observations, and the shared writer saves them under cleaned/ exactly
    as a model backend's would, so pass three cannot tell the sources apart
    except through the manifest's provenance.
    """

    def __init__(
        self,
        instance_directory: Path,
        aggregation_path: Path,
        label_mapping: dict[str, str],
        *,
        identity_hints: bool = False,
    ) -> None:
        self.instance_directory = Path(instance_directory)
        self.aggregation_path = Path(aggregation_path)
        self.label_mapping = dict(label_mapping)
        self.identity_hints = bool(identity_hints)

    def prepare(
        self, selected_sequence: FrameSequence, output_directory: Path
    ) -> SegmentationAdapter:
        reader = ScanNetInstanceSegmentation(
            selected_sequence,
            self.instance_directory,
            self.aggregation_path,
            self.label_mapping,
            identity_hints=self.identity_hints,
        )
        writer = SegmentationCacheWriter(
            selected_sequence, Path(output_directory) / "cleaned"
        )
        for frame_id in reader.segmentation_frame_ids:
            writer.write_frame(frame_id, reader.load_frame(frame_id))
        return writer.finish(
            {
                "backend": "scannet_gt",
                "source": "ScanNet 2D instance annotations (ground-truth ablation)",
                "instances": str(self.instance_directory.resolve()),
                "aggregation": str(self.aggregation_path.resolve()),
                "identity_hints": self.identity_hints,
                "unmapped_labels": reader.unmapped,
            }
        )
