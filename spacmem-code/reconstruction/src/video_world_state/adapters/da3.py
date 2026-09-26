"""Run Depth Anything 3 and translate saved results into pipeline contracts."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from .. import manifests
from ..contracts import (
    CameraPoseEstimate,
    FloatArray,
    FrameGeometry,
    FrameInput,
    FrameSequence,
)


_IMAGE_PREPROCESSING = {
    "method": "upper_bound_resize",
    "pixel_convention": "integer",
}


def unproject_depth(
    depth: FloatArray, intrinsics: FloatArray, camera_to_world: FloatArray
) -> FloatArray:
    """Turn one frame's depth into world points using DA3's own unprojection.

    Depth is height x width metres along the optical axis on the grid the
    intrinsics describe, and camera_to_world is that frame's final pose. The
    result is height x width x 3 in the pose's coordinates: supplying a leveled
    pose returns leveled points, and nothing here rescales, re-levels or
    filters. Runs on the CPU in this process, with no model inference.

    Torch and DA3 are imported here rather than at module import so that
    loading saved geometry keeps working without them. There is no handwritten
    fallback: the library that produced the depth also unprojects it.
    """
    try:
        import torch
        from depth_anything_3.utils.geometry import unproject_depth as unproject
    except ImportError as error:
        raise ImportError(
            "World points need torch and the depth_anything_3 package installed "
            "in the environment running this code"
        ) from error

    def batched(array: FloatArray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[
            None, None
        ]

    points = unproject(
        batched(depth)[..., None], batched(intrinsics), batched(camera_to_world)
    )
    return points[0, 0].numpy().astype(np.float32, copy=False)


def load_anchor_sequence(
    native_sequence: FrameSequence,
    output_directory: Path,
    *,
    allow_legacy: bool = False,
) -> FrameSequence:
    """Match a saved DA3 manifest to native frames in exact result order.

    Validate IDs, source indices and timestamps; RGB paths are not needed to
    read results. ID uniqueness is checked by the loader (anchors) and pose
    interpolator (native frames) when the geometry adapter opens the run.
    Require a valid completion marker by default. allow_legacy permits only
    explicitly verified old caches without a marker; runs whose manifests
    require completion can never use that exception.
    """
    manifest, entries = manifests.read(
        output_directory / manifests.MANIFEST_NAME,
        native_sequence.sequence_id,
        label="DA3",
        allow_missing_schema_version=True,
    )
    completion_path = output_directory / "run_complete.json"
    if completion_path.is_file():
        try:
            with completion_path.open() as completion_file:
                completion = json.load(completion_file)
        except json.JSONDecodeError as error:
            raise ValueError("Invalid DA3 run_complete.json") from error
        if (
            not isinstance(completion, dict)
            or type(completion.get("anchor_count")) is not int
            or completion["anchor_count"] != len(entries)
        ):
            raise ValueError(
                "DA3 run_complete.json anchor_count does not match the manifest"
            )
    elif not allow_legacy or manifest.get("completion_required", False):
        raise ValueError(
            "DA3 run is not marked complete: missing run_complete.json. "
            "Only verified legacy caches may be opened with allow_legacy=True."
        )
    anchors = manifests.resolve_frames(
        entries, native_sequence, label="DA3", noun="native frame"
    )
    return FrameSequence(native_sequence.sequence_id, anchors)


class Da3OutputLoader:
    """Read frame geometry from one completed DA3 output directory.

    The output directory and FrameSequence must describe the same ordered DA3
    run. Call load_frame with a stable pipeline frame ID; DA3's positional
    result indices remain internal to the loader.
    """

    def __init__(
        self,
        output_directory: Path,
        frame_sequence: FrameSequence,
    ) -> None:
        self._output_directory = Path(output_directory)
        self._frame_positions = self._index_frames(frame_sequence.frames)
        self._camera_poses = self._load_camera_poses()
        self._rgb_sizes = self._load_rgb_sizes(frame_sequence)

        pose_count = len(self._camera_poses)
        frame_count = len(self._frame_positions)
        if pose_count != frame_count:
            raise ValueError(
                "DA3 pose count does not match the FrameSequence: "
                f"{pose_count} poses for {frame_count} frames"
            )

    def load_frame(self, frame_id: str) -> FrameGeometry:
        """Load normalized geometry for one pipeline frame ID.

        The returned FrameGeometry uses the final aligned camera pose. Unknown
        frame IDs, missing DA3 files, and incompatible saved arrays are
        reported as errors rather than matched approximately.
        """

        result_index = self._result_index(frame_id)

        result_path = (
            self._output_directory / "results_output" / f"frame_{result_index}.npz"
        )
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing DA3 frame result: {result_path}")

        with np.load(result_path, allow_pickle=False) as result:
            required = {"depth", "conf", "intrinsics"}
            missing = required.difference(result.files)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"DA3 frame result is missing: {names}")

            # NPZ reads already create independent arrays; only convert dtype.
            depth = np.asarray(result["depth"], dtype=np.float32)
            confidence = np.asarray(result["conf"], dtype=np.float32)
            intrinsics = np.asarray(result["intrinsics"], dtype=np.float32)

        self._validate_frame_arrays(depth, confidence, intrinsics, result_path)

        # DA3's final pose table already includes cross-chunk alignment.
        # Raw per-frame extrinsics and saved s/R/T values are not used here.
        camera_to_world = self._camera_poses[result_index].copy()

        return FrameGeometry(
            frame_id=frame_id,
            depth=depth,
            confidence=confidence,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            rgb_to_geometry=self._rgb_to_geometry(frame_id, depth.shape),
        )

    def load_pose(self, frame_id: str) -> CameraPoseEstimate:
        """Return a measured anchor pose without opening its depth NPZ file.

        Only frames in this DA3 run are supported. The returned matrix is a copy
        of the same final aligned pose used by load_frame.
        """
        result_index = self._result_index(frame_id)
        return CameraPoseEstimate(
            frame_id=frame_id,
            camera_to_world=self._camera_poses[result_index].copy(),
            method="measured",
        )

    def _result_index(self, frame_id: str) -> int:
        try:
            return self._frame_positions[frame_id]
        except KeyError as error:
            raise KeyError(f"Unknown frame ID: {frame_id}") from error

    @staticmethod
    def _index_frames(frames: list[FrameInput]) -> dict[str, int]:
        positions: dict[str, int] = {}
        for position, frame in enumerate(frames):
            if frame.frame_id in positions:
                raise ValueError(f"Duplicate frame ID: {frame.frame_id}")
            positions[frame.frame_id] = position
        return positions

    def _load_camera_poses(self) -> FloatArray:
        pose_path = self._output_directory / "camera_poses.txt"
        if not pose_path.is_file():
            raise FileNotFoundError(f"Missing DA3 camera poses: {pose_path}")

        rows = np.loadtxt(pose_path, dtype=np.float32, ndmin=2)
        if rows.shape[1] != 16:
            raise ValueError(
                f"Expected 16 values per DA3 camera pose, got {rows.shape[1]}"
            )

        poses = rows.reshape(-1, 4, 4)
        if not np.isfinite(poses).all():
            raise ValueError("DA3 camera poses contain non-finite values")
        return poses

    def _load_rgb_sizes(
        self, frame_sequence: FrameSequence
    ) -> dict[str, tuple[int, int]]:
        """Read recorded resize provenance, never infer it from image shapes.

        Legacy runs without this metadata remain usable for pose lookup, but
        load_frame cannot promise pixel correspondence and must reject them.
        No original RGB files are needed to reopen a fully recorded run.
        """
        manifest_path = self._output_directory / "frame_manifest.json"
        if not manifest_path.is_file():
            return {}
        manifest = json.loads(manifest_path.read_text())
        preprocessing = manifest.get("image_preprocessing")
        if preprocessing is None:
            return {}
        if preprocessing != _IMAGE_PREPROCESSING:
            raise ValueError(
                "Unsupported DA3 image preprocessing; only resize-only is supported"
            )
        entries = manifest.get("frames", [])
        if manifest.get("sequence_id") != frame_sequence.sequence_id or [
            entry.get("frame_id") for entry in entries
        ] != [frame.frame_id for frame in frame_sequence.frames]:
            raise ValueError(
                "DA3 pixel mapping manifest does not match the anchor sequence"
            )
        sizes = {}
        for entry in entries:
            size = entry.get("rgb_size_hw")
            if (
                not isinstance(size, list)
                or len(size) != 2
                or any(
                    type(dimension) is not int or dimension <= 0 for dimension in size
                )
            ):
                raise ValueError("DA3 manifest requires positive integer rgb_size_hw")
            sizes[entry["frame_id"]] = tuple(size)
        if len(set(sizes.values())) != 1:
            raise ValueError("Mixed RGB sizes may trigger DA3 cropping; not supported")
        return sizes

    def _rgb_to_geometry(
        self, frame_id: str, geometry_shape: tuple[int, int]
    ) -> FloatArray:
        if frame_id not in self._rgb_sizes:
            raise ValueError(
                "Missing DA3 pixel mapping metadata: record verified image_preprocessing "
                "and original rgb_size_hw in frame_manifest.json before loading geometry"
            )
        rgb_height, rgb_width = self._rgb_sizes[frame_id]
        height, width = geometry_shape
        # DA3 scales fx/cx by width ratio and fy/cy by height ratio. Its integer
        # unprojection grid uses no half-pixel offset in this intrinsics map.
        return np.diag([width / rgb_width, height / rgb_height, 1]).astype(np.float32)

    @staticmethod
    def _validate_frame_arrays(
        depth: FloatArray,
        confidence: FloatArray,
        intrinsics: FloatArray,
        result_path: Path,
    ) -> None:
        if depth.ndim != 2 or 0 in depth.shape:
            raise ValueError(
                f"Expected non-empty 2D depth in {result_path}, got {depth.shape}"
            )
        if confidence.shape != depth.shape:
            raise ValueError(
                f"Depth and confidence shapes differ in {result_path}: "
                f"{depth.shape} and {confidence.shape}"
            )
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Expected 3x3 intrinsics in {result_path}, got {intrinsics.shape}"
            )


class Da3Runner:
    """Run installed DA3-Streaming on an ordered RGB anchor sequence.

    Tool/environment paths are shared across scenes, not inferred per video.
    Settings come from the packaged DA3 config unless one is supplied, and the
    file used is copied into the run. Inputs are staged as numbered symlinks
    because DA3 sorts filenames. The tool runs in its script directory so
    relative weight paths still work.
    No downloads, frame sampling, upright alignment or interactive prompts.
    """

    def __init__(
        self,
        python_executable: Path,
        da3_script: Path,
        config_path: Path | None = None,
        *,
        deterministic: bool = True,
    ) -> None:
        # Do not resolve the Python symlink: its venv path selects the environment.
        self._python_executable = Path(python_executable).absolute()
        self._da3_script = Path(da3_script).resolve()
        if config_path is None:
            config_path = Path(__file__).with_name("da3_config.yaml")
        self._config_path = Path(config_path).resolve()
        self._deterministic = bool(deterministic)

    def _environment(self) -> dict[str, str]:
        """Seed DA3 in its own interpreter when deterministic mode is enabled."""
        environment = dict(os.environ)
        if not self._deterministic:
            return environment
        shim = str(Path(__file__).with_name("_da3_determinism"))
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = f"{shim}{os.pathsep}{existing}" if existing else shim
        environment["VWS_DETERMINISM"] = "1"
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        return environment

    @staticmethod
    def _without_triton_alignment(config_text: str) -> str:
        """Use DA3's reproducible torch alignment backend in the saved config."""
        lines = []
        for line in config_text.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("align_lib:"):
                indent = line[: len(line) - len(stripped)]
                lines.append(f"{indent}align_lib: 'torch'  # pinned for reproducibility\n")
            else:
                lines.append(line)
        return "".join(lines)

    def run(self, anchor_sequence: FrameSequence, output_directory: Path) -> None:
        """Save a new run, or raise with partial files retained for diagnosis.

        Refuse any existing output directory. Missing inputs are rejected before
        creating output. da3.log records subprocess output; nonzero exit or
        missing/corrupt results raise instead of producing a completion marker.
        A successful return guarantees every anchor is readable by our loader.
        """
        output_directory = Path(output_directory).absolute()
        if output_directory.exists():
            raise FileExistsError(f"Refusing to overwrite DA3 run: {output_directory}")
        for path in (self._python_executable, self._da3_script, self._config_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing DA3 executable/script/config: {path}")
        frames = anchor_sequence.frames
        if not frames:
            raise ValueError("The anchor FrameSequence is empty")
        if len({frame.frame_id for frame in frames}) != len(frames):
            raise ValueError("Duplicate anchor frame ID")
        timestamps = np.asarray(
            [frame.timestamp_seconds for frame in frames], dtype=float
        )
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Anchor timestamps must be finite and strictly increasing")
        sources = [Path(frame.rgb_path).resolve() for frame in frames]
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(f"Missing RGB frame: {source}")
            if source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"DA3 requires JPEG or PNG frames: {source}")

        # The installed streaming CLI calls inference with upper_bound_resize
        # defaults. Equal input sizes avoid DA3's automatic batch center crop.
        rgb_sizes = []
        for source in sources:
            with Image.open(source) as image:
                rgb_sizes.append((image.height, image.width))
        if len(set(rgb_sizes)) != 1:
            raise ValueError("Mixed RGB sizes may trigger DA3 cropping; not supported")

        output_directory.mkdir(parents=True, exist_ok=False)
        inputs = output_directory / "input_frames"
        inputs.mkdir()
        for position, source in enumerate(sources):
            suffix = ".png" if source.suffix.lower() == ".png" else ".jpg"
            (inputs / f"{position:09d}{suffix}").symlink_to(source)
        saved_config = output_directory / "da3_config.yaml"
        if self._deterministic:
            saved_config.write_text(
                self._without_triton_alignment(self._config_path.read_text())
            )
        else:
            shutil.copy2(self._config_path, saved_config)
        command = [
            str(self._python_executable),
            str(self._da3_script),
            "--image_dir",
            str(inputs),
            "--config",
            str(saved_config),
            "--output_dir",
            str(output_directory),
        ]
        manifests.write(
            output_directory / manifests.MANIFEST_NAME,
            anchor_sequence.sequence_id,
            manifests.frame_entries(
                anchor_sequence, rgb_sizes=rgb_sizes, rgb_paths=sources
            ),
            completion_required=True,
            image_preprocessing=_IMAGE_PREPROCESSING.copy(),
            command=command,
            working_directory=str(self._da3_script.parent),
        )
        log_path = output_directory / "da3.log"
        start = time.monotonic()
        try:
            with log_path.open("x") as log:
                subprocess.run(
                    command,
                    cwd=self._da3_script.parent,
                    env=self._environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        except (subprocess.CalledProcessError, OSError) as error:
            raise RuntimeError(f"DA3 execution failed; see {log_path}") from error

        loader = Da3OutputLoader(output_directory, anchor_sequence)
        for frame in frames:
            loader.load_frame(frame.frame_id)
        (output_directory / "run_complete.json").write_text(
            json.dumps(
                {"anchor_count": len(frames), "wall_seconds": time.monotonic() - start},
                indent=2,
            )
        )
