"""Describe what each observation looks like, for the fragment merger.

Crops are cut here, where the masks are. The image model runs in a worker under
the caller's interpreter, as with SegVGGT and DA3, so importing this module never
imports torch.
"""

import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ..contracts import BoolArray, FloatArray, FrameSequence

if TYPE_CHECKING:
    from ..segmentation import SegmentationAdapter

MIN_MASK_PIXELS = 50
BACKGROUND = 128


class AppearanceRunner:
    """Embed every cleaned observation into one unit vector.

    Fragments of one object seen from different places look alike even when
    their geometry barely overlaps, which is what the merger needs as a second
    opinion. The descriptor is DINOv2's, from the large model.
    """

    def __init__(
        self,
        python_executable: Path,
        worker_script: Path | None = None,
        model: str = "facebook/dinov2-large",
        device: str = "cuda",
        batch_size: int = 16,
    ) -> None:
        self._python = Path(python_executable).absolute()
        self._script = Path(
            worker_script or Path(__file__).with_name("appearance_worker.py")
        ).resolve()
        self._model, self._device, self._batch_size = model, device, batch_size

    def run(
        self,
        sequence: FrameSequence,
        segmentation: "SegmentationAdapter",
        output_directory: Path,
    ) -> dict[str, FloatArray]:
        """Crop, embed and return descriptors; refuse to overwrite a run."""
        output = Path(output_directory).absolute()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite appearance run: {output}")
        for path in (self._python, self._script):
            if not path.is_file():
                raise FileNotFoundError(path)
        (output / "crops").mkdir(parents=True)
        rgb_of = {frame.frame_id: frame.rgb_path for frame in sequence.frames}

        crops = []
        for frame_id in segmentation.segmentation_frame_ids:
            observations = segmentation.load_frame(frame_id)
            if not observations:
                continue
            with Image.open(rgb_of[frame_id]) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            for observation in observations:
                patch = crop_observation(pixels, observation.mask)
                if patch is None:
                    continue
                name = f"{len(crops):07d}.png"
                Image.fromarray(patch).save(output / "crops" / name)
                crops.append({"observation_id": observation.observation_id, "file": name})
        (output / "manifest.json").write_text(
            json.dumps({"model": self._model, "crops": crops}, indent=1)
        )

        command = [
            str(self._python), str(self._script),
            "--directory", str(output),
            "--model", self._model,
            "--device", self._device,
            "--batch-size", str(self._batch_size),
        ]
        with (output / "worker.log").open("x") as log:
            try:
                subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except (subprocess.CalledProcessError, OSError) as error:
                raise RuntimeError("Appearance worker failed; see worker.log") from error
        return load_descriptors(output)


def crop_observation(pixels: np.ndarray, mask: BoolArray) -> np.ndarray | None:
    """The mask's bounding box, with everything outside the mask set to grey.

    The background is blanked so a descriptor describes the object and not the
    wall behind it; otherwise two fragments of different objects against the
    same wall would look alike for the wrong reason. Masks too small to show
    anything recognisable are skipped rather than embedded as noise.
    """
    rows, columns = np.nonzero(mask)
    if rows.size < MIN_MASK_PIXELS:
        return None
    top, bottom = rows.min(), rows.max() + 1
    left, right = columns.min(), columns.max() + 1
    patch = pixels[top:bottom, left:right].copy()
    patch[~mask[top:bottom, left:right]] = BACKGROUND
    return patch


def load_descriptors(directory: Path) -> dict[str, FloatArray]:
    """Reopen a finished run as {observation ID: unit vector}."""
    with np.load(Path(directory) / "embeddings.npz", allow_pickle=False) as saved:
        return dict(zip(saved["observation_ids"].tolist(), saved["vectors"]))
