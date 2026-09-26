"""Write a ScanNet scan's colour frames out of its .sens file.

    python tools/extract_frames.py --sens <scan>/scene0000_00.sens --output frames/scene0000_00

ScanNet stores colour as JPEG inside the .sens file, so the frames are copied
out byte for byte rather than decoded and re-encoded: the pipeline sees exactly
the pixels the dataset shipped. Frames are named 000000.jpg, 000001.jpg, ...,
contiguous from zero, as tools/reconstruct_scene.py expects. Depth and poses in
the file are skipped; the pipeline estimates its own from the colour frames.
"""

import argparse
from pathlib import Path
import struct

COLOUR_JPEG = 2  # ScanNet's colour compression codes: 0 raw, 1 png, 2 jpeg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    with arguments.sens.open("rb") as stream:
        if struct.unpack("<I", stream.read(4))[0] != 4:
            raise SystemExit("Expected ScanNet .sens version 4")
        stream.read(struct.unpack("<Q", stream.read(8))[0])  # sensor name
        stream.read(4 * 64)  # colour and depth intrinsics and extrinsics
        colour_compression, _ = struct.unpack("<II", stream.read(8))
        if colour_compression != COLOUR_JPEG:
            raise SystemExit(
                f"Colour compression {colour_compression} is not JPEG; this tool "
                "only copies frames, it does not transcode them"
            )
        stream.read(4 * 4 + 4)  # colour and depth sizes, depth shift
        count = struct.unpack("<Q", stream.read(8))[0]

        written = 0
        for index in range(count):
            stream.read(64)  # camera to world
            _, _, colour_bytes, depth_bytes = struct.unpack("<QQQQ", stream.read(32))
            blob = stream.read(colour_bytes)
            if blob[:2] != b"\xff\xd8" or blob[-2:] != b"\xff\xd9":
                raise SystemExit(f"Frame {index} is not a complete JPEG")
            (arguments.output / f"{index:06d}.jpg").write_bytes(blob)
            stream.seek(depth_bytes, 1)
            written += 1

    if written != count:
        raise SystemExit(f"Wrote {written} of {count} frames; {arguments.output} is incomplete")
    print(f"{arguments.sens.name}: {written} frames -> {arguments.output}")


if __name__ == "__main__":
    main()
