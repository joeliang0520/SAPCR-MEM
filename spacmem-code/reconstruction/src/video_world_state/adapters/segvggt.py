"""Drive the packaged SegVGGT worker and turn its raw queries into observations.

The caller supplies the SegVGGT environment, the upstream repository and the
checkpoint; the code that runs in that environment ships with this package.
No torch imports or GPU work happen when importing this module, reading raw
chunks or decoding them.

Labels come from the ScanNet200 checkpoint's fixed 198-class instance
vocabulary. Inference receives RGB frames only.
"""

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from .. import manifests
from ..contracts import FrameSequence, LabelCandidate, SegmentationObservation
from ..frames import frame_timestamps
from ..segmentation import (
    SegmentationAdapter,
    SegmentationBackend,
    SegmentationCacheWriter,
)

WORKER_MODULE = "video_world_state.adapters.segvggt_worker"

# Raw keep rule: loose enough that the decoder's thresholds can change later
# without rerunning the model.
KEEP_RULE = {
    "min_class_probability": 0.001,
    "min_chunk_pixels": 200,  # exclusive, counted over every frame of the chunk
    "pixel_sigmoid": 0.3,
}

SCORE_SEMANTICS = (
    "best object-class probability x mean mask sigmoid over the chunk's pixels "
    "with positive logit (SegVGGT's upstream instance score); one value per "
    "query per chunk, not a calibrated per-frame mask quality"
)
CANDIDATE_SEMANTICS = (
    "runner-up object-class probability / primary class probability, "
    "from the same query's softmax; no-object excluded"
)


@dataclass
class SegVggtChunk:
    """One chunk's raw queries, validated against the run's manifest.

    probabilities has one row per kept query: every object class in the run's
    class order, then no-object. maps holds sigmoid x 255 on the model's
    output grid for every frame of the chunk, in frame order.
    """

    index: int
    frame_ids: tuple[str, ...]
    queries: np.ndarray  # K, int64 local query IDs
    probabilities: np.ndarray  # K x (classes + 1), float16
    score: np.ndarray  # K, float32
    maps: np.ndarray  # K x frames x height x width, uint8


class SegVggtOutputLoader:
    """Read a raw SegVGGT run by stable frame IDs; never thresholds or writes.

    Requires the manifest to list exactly the selected frames, chunks that
    cover them in order without gaps or overlap, and a completed worker
    result. The completion marker is required unless the runner is verifying
    its own output before writing that marker.
    """

    def __init__(
        self,
        raw_directory: Path,
        selected_sequence: FrameSequence,
        *,
        require_complete: bool = True,
    ) -> None:
        self.directory = Path(raw_directory)
        frame_timestamps(selected_sequence)
        self.manifest, entries = manifests.read(
            self.directory / manifests.MANIFEST_NAME,
            selected_sequence.sequence_id,
            label="SegVGGT",
        )
        if not manifests.matches_frames(entries, selected_sequence.frames):
            raise ValueError("SegVGGT manifest does not match selected frames")
        self.frame_ids = tuple(entry["frame_id"] for entry in entries)
        self.ranges = [tuple(pair) for pair in self.manifest["chunk_ranges"]]
        if (
            not self.ranges
            or self.ranges[0][0] != 0
            or self.ranges[-1][1] != len(entries)
            or any(not 0 <= a < b <= len(entries) for a, b in self.ranges)
            or any(b != c for (_, b), (c, _) in zip(self.ranges, self.ranges[1:]))
        ):
            raise ValueError(
                "SegVGGT chunks must cover ordered frames without gaps or overlap"
            )
        result = json.loads((self.directory / "results.json").read_text())
        if result.get("completed") is not True:
            raise ValueError("SegVGGT worker did not complete")
        self.classes = tuple(result["classes"])
        if not self.classes or len(set(self.classes)) != len(self.classes) or not all(
            isinstance(name, str) and name for name in self.classes
        ):
            raise ValueError("SegVGGT classes must be unique non-empty names")
        self.query_count = int(result["query_count"])
        self.output_hw = tuple(result["output_hw"])
        if [(c["start"], c["end"]) for c in result["chunks"]] != self.ranges:
            raise ValueError("SegVGGT worker chunks do not match the manifest")
        if require_complete:
            completion = json.loads(
                (self.directory / "run_complete.json").read_text()
            )
            if completion.get("frame_count") != len(entries) or completion.get(
                "chunk_count"
            ) != len(self.ranges):
                raise ValueError("SegVGGT completion marker does not match raw manifest")

    def load_chunk(self, index: int) -> SegVggtChunk:
        """Return one chunk's raw queries after checking shapes, dtypes and IDs."""
        start, end = self.ranges[index]
        with np.load(
            self.directory / f"chunk_{index:03d}.npz", allow_pickle=False
        ) as saved:
            positions, queries = saved["positions"], saved["queries"]
            probabilities, score, maps = (
                saved["probabilities"],
                saved["score"],
                saved["maps"],
            )
        count = len(queries)
        if (
            not np.array_equal(positions, np.arange(start, end))
            or queries.dtype != np.int64
            or queries.ndim != 1
            or len(np.unique(queries)) != count
            or (count and not (0 <= queries.min() and queries.max() < self.query_count))
            or probabilities.dtype != np.float16
            or probabilities.shape != (count, len(self.classes) + 1)
            or not np.isfinite(probabilities).all()
            or score.dtype != np.float32
            or score.shape != (count,)
            or not np.isfinite(score).all()
            or maps.dtype != np.uint8
            or maps.shape != (count, end - start, *self.output_hw)
        ):
            raise ValueError(f"SegVGGT chunk {index} does not match its manifest")
        return SegVggtChunk(
            index, self.frame_ids[start:end], queries, probabilities, score, maps
        )


class SegVggtDecoder:
    """Turn one raw chunk into per-frame observations; one per query at most.

    Queries below the score threshold are dropped. Each remaining query's soft
    map is resized bilinearly to the frame's RGB size and thresholded; an empty
    mask emits nothing for that frame. The primary label is the most probable
    object class; the next runner_ups classes become candidates scored
    relative to it. Hints are scoped to one chunk: local query IDs restart in
    every chunk and are not linked across chunks.
    """

    def __init__(
        self,
        score_threshold: float = 0.05,
        mask_threshold: float = 0.4,
        runner_ups: int = 4,
    ) -> None:
        if not np.isfinite([score_threshold, mask_threshold]).all() or not (
            0 < mask_threshold < 1
        ):
            raise ValueError("SegVGGT thresholds must be finite; masks in (0, 1)")
        if runner_ups < 0:
            raise ValueError("runner_ups cannot be negative")
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self.runner_ups = int(runner_ups)

    def decode(
        self,
        chunk: SegVggtChunk,
        classes: tuple[str, ...],
        rgb_sizes: list[tuple[int, int]],
        sequence_id: str,
    ) -> list[list[SegmentationObservation]]:
        """Observations for each frame of the chunk, in frame then query order."""
        frames: list[list[SegmentationObservation]] = [[] for _ in chunk.frame_ids]
        cutoff = self.mask_threshold * 255
        order = np.argsort(chunk.queries, kind="stable")
        for row in order:
            if chunk.score[row] < self.score_threshold:
                continue
            object_probabilities = chunk.probabilities[row, :-1].astype(np.float64)
            ranked = np.argsort(-object_probabilities, kind="stable")
            primary = object_probabilities[ranked[0]]
            if not primary > 0:
                raise ValueError("SegVGGT query has no positive object-class probability")
            candidates = tuple(
                LabelCandidate(classes[k], float(object_probabilities[k] / primary))
                for k in ranked[1 : 1 + self.runner_ups]
            )
            query = int(chunk.queries[row])
            hint = f"{sequence_id}:segvggt:chunk:{chunk.index}:query:{query}"
            for position, (frame_id, soft) in enumerate(
                zip(chunk.frame_ids, chunk.maps[row])
            ):
                # Bilinear values never exceed their inputs' maximum.
                if soft.max() <= cutoff:
                    continue
                height, width = rgb_sizes[position]
                resized = Image.fromarray(soft.astype(np.float32)).resize(
                    (width, height), Image.Resampling.BILINEAR
                )
                mask = np.asarray(resized) > cutoff
                if not mask.any():
                    continue
                frames[position].append(
                    SegmentationObservation(
                        f"{hint}:{frame_id}",
                        frame_id,
                        mask,
                        classes[ranked[0]],
                        float(chunk.score[row]),
                        hint,
                        candidates,
                    )
                )
        return frames


class SegVggtRunner:
    """Run the SegVGGT worker once over a selected sequence, chunk by chunk.

    One worker process loads the model once and runs each chunk as an
    independent forward pass; chunks are consecutive frame counts with no
    overlap, so each needs only its own frames. The supplied interpreter selects the SegVGGT environment; the
    repository and checkpoint are used as given, never downloaded or changed.
    A worker script can be given instead of the packaged module for testing.
    Inputs are staged as numbered symlinks; positions never become frame IDs.
    """

    def __init__(
        self,
        python_executable: Path,
        repository: Path,
        checkpoint: Path,
        *,
        worker_script: Path | None = None,
        config: str = "segvggt_scannet200",
        chunk_frames: int = 50,
    ) -> None:
        self._python = Path(python_executable).absolute()
        self._repository = Path(repository).resolve()
        self._checkpoint = Path(checkpoint).resolve()
        self._script = None if worker_script is None else Path(worker_script).resolve()
        if chunk_frames < 1:
            raise ValueError("SegVGGT chunks need at least one frame")
        self._config = config
        self._chunk_frames = int(chunk_frames)

    def _environment(self) -> dict[str, str]:
        """Expose this package to the worker's interpreter."""
        package_root = str(Path(__file__).resolve().parents[2])
        existing = os.environ.get("PYTHONPATH")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = (
            package_root if not existing else os.pathsep.join([package_root, existing])
        )
        return environment

    def _command(
        self, inputs: Path, output: Path, ranges: list[tuple[int, int]]
    ) -> list[str]:
        """Build the worker command; saved in the manifest so a run states its settings."""
        return [
            str(self._python),
            *(["-m", WORKER_MODULE] if self._script is None else [str(self._script)]),
            "--repo",
            str(self._repository),
            "--checkpoint",
            str(self._checkpoint),
            "--config",
            self._config,
            "--frames",
            str(inputs),
            "--output",
            str(output),
            *[
                argument
                for name, value in KEEP_RULE.items()
                for argument in (f"--{name.replace('_', '-')}", str(value))
            ],
            *[
                argument
                for start, end in ranges
                for argument in ("--chunk", f"{start}:{end}")
            ],
        ]

    def run(self, selected_sequence: FrameSequence, output_directory: Path) -> None:
        """Run, verify every chunk, then mark complete; keep failures, refuse overwrite."""
        output = Path(output_directory).absolute()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite SegVGGT run: {output}")
        for path in (self._python, self._checkpoint, self._script):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        if not self._repository.is_dir():
            raise FileNotFoundError(self._repository)
        frame_timestamps(selected_sequence)
        count = len(selected_sequence.frames)
        ranges = [
            (start, min(start + self._chunk_frames, count))
            for start in range(0, count, self._chunk_frames)
        ]
        sources, sizes = [], []
        for frame in selected_sequence.frames:
            source = Path(frame.rgb_path).resolve()
            with Image.open(source) as image:
                sizes.append((image.height, image.width))
            sources.append(source)
        if len(set(sizes)) != 1:
            raise ValueError("SegVGGT requires consistent RGB dimensions")

        output.mkdir(parents=True, exist_ok=False)
        inputs = output / "input_frames"
        inputs.mkdir()
        for position, source in enumerate(sources):
            (inputs / f"{position:05d}.jpg").symlink_to(source)
        command = self._command(inputs, output, ranges)
        checkpoint = self._checkpoint.stat()
        manifests.write(
            output / manifests.MANIFEST_NAME,
            selected_sequence.sequence_id,
            manifests.frame_entries(
                selected_sequence, rgb_sizes=sizes, rgb_paths=sources
            ),
            chunk_ranges=ranges,
            chunk_frames=self._chunk_frames,
            overlap_frames=0,
            model={
                "repository": str(self._repository),
                "config": self._config,
                "checkpoint": str(self._checkpoint),
                "checkpoint_bytes": checkpoint.st_size,
                "checkpoint_mtime_ns": checkpoint.st_mtime_ns,
                "training_data": "ScanNet200",
            },
            preprocessing={
                "color": "RGB",
                "width": 518,
                "height": "round(rgb_height * 518 / rgb_width / 14) * 14",
                "interpolation": "cv2.INTER_LANCZOS4",
                "scale": "1/255",
                "autocast": "bfloat16",
            },
            keep_rule=KEEP_RULE,
            saved_arrays={
                "positions": "frames, int64, staged positions of the chunk",
                "queries": "K, int64, local query IDs",
                "probabilities": "K x (classes + 1), float16 softmax, no-object last",
                "score": "K, float32",
                "maps": "K x frames x output height x width, uint8 sigmoid x 255",
            },
            score_semantics=SCORE_SEMANTICS,
            inputs="RGB frames only",
            python=str(self._python),
            worker=WORKER_MODULE if self._script is None else str(self._script),
            command=command,
        )
        started = time.monotonic()
        with (output / "worker.log").open("x") as log:
            try:
                subprocess.run(
                    command,
                    cwd=output,
                    env=self._environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except (subprocess.CalledProcessError, OSError) as error:
                raise RuntimeError(
                    f"SegVGGT worker failed; see {output / 'worker.log'}"
                ) from error
        loader = SegVggtOutputLoader(output, selected_sequence, require_complete=False)
        for index in range(len(ranges)):
            loader.load_chunk(index)
        (output / "run_complete.json").write_text(
            json.dumps(
                {
                    "frame_count": count,
                    "chunk_count": len(ranges),
                    "wall_seconds": time.monotonic() - started,
                },
                indent=2,
            )
        )


class SegVggtBackend(SegmentationBackend):
    """Run and decode SegVGGT behind the shared segmentation-cache boundary.

    Distinct queries covering the same object stay distinct: there is no
    duplicate resolution or cross-chunk linking. prepare_raw re-decodes an
    existing raw run without rerunning the model.
    """

    def __init__(
        self, runner: SegVggtRunner, *, decoder: SegVggtDecoder | None = None
    ) -> None:
        self.runner = runner
        self.decoder = decoder if decoder is not None else SegVggtDecoder()

    def prepare(
        self, selected_sequence: FrameSequence, output_directory: Path
    ) -> SegmentationAdapter:
        """Run SegVGGT, keep its raw output, and publish a canonical cache."""
        output = Path(output_directory)
        frame_timestamps(selected_sequence)
        output.mkdir(parents=True, exist_ok=False)
        self.runner.run(selected_sequence, output / "raw")
        return self.prepare_raw(selected_sequence, output / "raw", output / "cleaned")

    def prepare_raw(
        self,
        selected_sequence: FrameSequence,
        raw_directory: Path,
        output_directory: Path,
    ) -> SegmentationAdapter:
        """Decode an existing raw run without changing or rerunning it."""
        raw_directory = Path(raw_directory)
        loader = SegVggtOutputLoader(raw_directory, selected_sequence)
        sizes = [
            tuple(entry["rgb_size_hw"]) for entry in loader.manifest["frames"]
        ]
        writer = SegmentationCacheWriter(selected_sequence, output_directory)
        for index, (start, _) in enumerate(loader.ranges):
            chunk = loader.load_chunk(index)
            decoded = self.decoder.decode(
                chunk,
                loader.classes,
                sizes[start : start + len(chunk.frame_ids)],
                selected_sequence.sequence_id,
            )
            for offset, (frame_id, observations) in enumerate(
                zip(chunk.frame_ids, decoded)
            ):
                position = start + offset
                writer.write_frame(
                    frame_id,
                    [
                        replace(
                            item,
                            observation_id=(
                                f"{selected_sequence.sequence_id}:obs:"
                                f"{position:06d}:{row:03d}"
                            ),
                        )
                        for row, item in enumerate(observations)
                    ],
                )
        return writer.finish(
            {
                "backend": "segvggt",
                "raw_directory": str(raw_directory.absolute()),
                "model": loader.manifest["model"],
                "score_semantics": SCORE_SEMANTICS,
                "candidate_semantics": CANDIDATE_SEMANTICS,
                "label_vocabulary": "scannet200_instance_classes",
                "classes": list(loader.classes),
                "decoder_settings": vars(self.decoder),
                "hint_scope": "chunk: local query IDs, not linked across chunks",
            }
        )
