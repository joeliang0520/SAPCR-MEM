"""Reconstruct one scene from its RGB frames into persistent objects.

The frames are the scene's native 30 FPS colour images (tools/extract_frames.py
writes them from a ScanNet .sens file). Every pass reads the same 5 FPS
selection of them. The output directory holds geometry/ and
world_alignment.npy (pass one), segmentation/ (pass two), appearance/ (SegVGGT
only), objects/ (pass three) and provenance.json. It must not exist yet.

SegVGGT, from RGB alone:

    python tools/reconstruct_scene.py --scene scene0000_00 --frames frames/scene0000_00 \\
        --output scenes/scene0000_00 --da3-python ... --da3-script ... \\
        segvggt --python ... --repository ... --checkpoint ... --appearance-python ...

The ground-truth ablation: ScanNet's 2D instance masks, labels and instance
identities on the geometry of an earlier run of the same scene:

    python tools/reconstruct_scene.py --scene scene0000_00 --frames frames/scene0000_00 \\
        --output scannet-scenes/scene0000_00 --geometry-from scenes/scene0000_00 \\
        scannet --instances <scan>/instance-filt \\
        --aggregation <scan>/scene0000_00.aggregation.json \\
        --label-mapping <episode>/public/label_mapping.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_world_state import pipeline  # noqa: E402
from video_world_state.adapters.appearance import AppearanceRunner  # noqa: E402
from video_world_state.adapters.da3 import Da3Runner  # noqa: E402
from video_world_state.adapters.scannet import ScanNetGtBackend  # noqa: E402
from video_world_state.adapters.segvggt import SegVggtBackend, SegVggtRunner  # noqa: E402
from video_world_state.frames import read_frame_folder, select_frames  # noqa: E402
from video_world_state.geometry import GeometryAdapter  # noqa: E402


def code_sha256() -> str:
    """Hash of the package source and these tools, recorded with every scene."""
    digest = hashlib.sha256()
    files = sorted((ROOT / "src/video_world_state").rglob("*")) + sorted(
        (ROOT / "tools").glob("*.py")
    )
    for path in files:
        if path.suffix in (".py", ".json", ".yaml"):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def label_mapping(path: Path) -> dict[str, str]:
    """The benchmark's ScanNet raw label to class name mapping."""
    return {
        item["raw_label"]: item["canonical_label"]
        for item in json.loads(path.read_text())["raw_label_mappings"]
    }


def saved_geometry(native, source: Path, output: Path) -> GeometryAdapter:
    """Open an earlier run's pass one and link it into this run."""
    alignment = np.load(source / "world_alignment.npy")
    geometry = GeometryAdapter.from_saved_run(
        native, source / "geometry", world_alignment=alignment
    )
    output.mkdir(parents=True)
    (output / "geometry").symlink_to((source / "geometry").resolve(), target_is_directory=True)
    shutil.copyfile(source / "world_alignment.npy", output / "world_alignment.npy")
    return geometry


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frames", type=Path, required=True, help="native frames 000000.jpg, ...")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene", required=True, help="the scan's name, e.g. scene0000_00; frame IDs use it"
    )
    parser.add_argument("--da3-python", type=Path, help="Depth Anything 3 environment interpreter")
    parser.add_argument("--da3-script", type=Path, help="DA3-Streaming's da3_streaming.py")
    parser.add_argument(
        "--geometry-from",
        type=Path,
        help="reuse pass one of an earlier run of this scene instead of running DA3",
    )
    backends = parser.add_subparsers(dest="backend", required=True)
    segvggt = backends.add_parser("segvggt", help="SegVGGT segmentation from RGB")
    segvggt.add_argument(
        "--python", type=Path, required=True, help="SegVGGT environment interpreter"
    )
    segvggt.add_argument("--repository", type=Path, required=True, help="SegVGGT checkout")
    segvggt.add_argument("--checkpoint", type=Path, required=True, help="ScanNet200 checkpoint")
    segvggt.add_argument(
        "--appearance-python",
        type=Path,
        required=True,
        help="interpreter with torch and transformers for DINOv2",
    )
    scannet = backends.add_parser(
        "scannet", help="ScanNet's own instance masks (ground-truth ablation)"
    )
    scannet.add_argument(
        "--instances", type=Path, required=True, help="the scan's instance-filt folder"
    )
    scannet.add_argument(
        "--aggregation", type=Path, required=True, help="the scan's aggregation.json"
    )
    scannet.add_argument(
        "--label-mapping", type=Path, required=True, help="the benchmark's label_mapping.json"
    )
    arguments = parser.parse_args()
    if arguments.geometry_from is None and not (arguments.da3_python and arguments.da3_script):
        parser.error("give --da3-python and --da3-script, or --geometry-from")

    scene = arguments.scene
    native = read_frame_folder(arguments.frames, scene)
    output = arguments.output
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")

    if arguments.backend == "segvggt":
        settings = pipeline.SEGVGGT
        backend = SegVggtBackend(
            SegVggtRunner(arguments.python, arguments.repository, arguments.checkpoint)
        )
        appearance = AppearanceRunner(arguments.appearance_python)
        inputs = {"segvggt_checkpoint": str(arguments.checkpoint.resolve())}
    else:
        settings = pipeline.SCANNET_IDENTITY
        backend = ScanNetGtBackend(
            arguments.instances,
            arguments.aggregation,
            label_mapping(arguments.label_mapping),
            identity_hints=settings["identity"],
        )
        appearance = None
        inputs = {
            "instances": str(arguments.instances.resolve()),
            "aggregation": str(arguments.aggregation.resolve()),
        }

    if arguments.geometry_from is None:
        geometry, segmentation = pipeline.run_pipeline(
            native, Da3Runner(arguments.da3_python, arguments.da3_script), backend, output
        )
    else:
        selected = select_frames(native)
        geometry = saved_geometry(native, arguments.geometry_from, output)
        segmentation = backend.prepare(selected, output / "segmentation")
        inputs["geometry_from"] = str(arguments.geometry_from.resolve())
    descriptors = (
        appearance.run(native, segmentation, output / "appearance") if appearance else None
    )
    result = pipeline.build_world_state(
        geometry, segmentation, output / "objects", descriptors, settings
    )

    (output / "provenance.json").write_text(
        json.dumps(
            {
                "scene": scene,
                "frames": str(arguments.frames.resolve()),
                "backend": arguments.backend,
                **inputs,
                "pass3_settings": settings,
                "code_sha256": code_sha256(),
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{scene}: {len(result.objects)} objects -> {output}")


if __name__ == "__main__":
    main()
