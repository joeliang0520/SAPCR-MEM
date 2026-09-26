"""Run SegVGGT over staged frame chunks and save every query's raw evidence.

Started once per sequence by SegVggtRunner, in the SegVGGT environment rather
than the pipeline's own, so only this module imports torch. The model loads
once; each chunk is one independent forward pass. Per chunk the output keeps
the local query IDs, the full class distribution, the overall query score and
a soft mask per frame. Nothing here chooses labels, thresholds masks for
output, or links queries between chunks.

Frames are addressed by staged position; the runner maps positions back to
pipeline frame IDs. Only RGB images are read: the upstream evaluation's
ScanNet depth, pose, mesh and annotation paths are never called.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="segvggt_scannet200")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument(
        "--chunk",
        action="append",
        required=True,
        metavar="START:END",
        help="half-open staged positions; repeated once per chunk, in order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--min-class-probability", type=float, default=0.001)
    parser.add_argument("--min-chunk-pixels", type=int, default=200)
    parser.add_argument("--pixel-sigmoid", type=float, default=0.3)
    parser.add_argument("--query-batch", type=int, default=25)
    arguments = parser.parse_args(argv)
    arguments.chunks = [
        tuple(int(value) for value in item.split(":")) for item in arguments.chunk
    ]
    return arguments


def _save_chunk(path: Path, **arrays: np.ndarray) -> None:
    """Write beside the final name, then rename, so a crash leaves no valid-looking chunk."""
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    os.replace(partial, path)


def run(arguments: argparse.Namespace) -> int:
    """Process every chunk in order; return the process exit code."""
    started = time.perf_counter()
    output = arguments.output
    result: dict = {"completed": False, "chunks": []}
    error = None
    try:
        import cv2
        import torch
        import torch.nn.functional as F

        sys.path.insert(0, str(arguments.repo))
        from eval.eval_instance_seg import SCANNET200_CLASSES
        from eval.instance_eval_common import load_model_for_eval
        from segvggt.utils.image import read_image_cv2

        device, dtype = torch.device("cuda"), torch.bfloat16
        loaded = time.perf_counter()
        model, config, missing, unexpected = load_model_for_eval(
            arguments.config, str(arguments.checkpoint), device, dtype
        )
        classes = [str(name) for name in SCANNET200_CLASSES[2:-1]]
        head = model.semantic_head.scratch.output_instance[-1].out_features
        if head != len(classes) + 1:
            raise ValueError(
                f"Label head has {head} outputs for {len(classes)} classes + no-object"
            )
        result.update(
            classes=classes,
            query_count=int(config.model.instance_query_num),
            dtype="bfloat16",
            torch=torch.__version__,
            missing_keys=len(missing),
            unexpected_keys=len(unexpected),
            model_load_seconds=time.perf_counter() - loaded,
        )

        # Upstream preprocessing: RGB, width 518, height rounded to a multiple
        # of 14, LANCZOS4, values in [0, 1]. Landscape frames only.
        positions = range(arguments.chunks[0][0], arguments.chunks[-1][1])
        images = []
        for position in positions:
            image = read_image_cv2(str(arguments.frames / f"{position:05d}.jpg"))
            if image is None:
                raise FileNotFoundError(f"Unreadable staged frame {position}")
            height, width = image.shape[:2]
            resized_height = round(height * arguments.width / width / 14) * 14
            if resized_height > arguments.width:
                raise ValueError("Portrait frames need SegVGGT's crop; not supported")
            images.append(
                cv2.resize(
                    image,
                    (arguments.width, resized_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            )
        if len({image.shape for image in images}) != 1:
            raise ValueError("SegVGGT chunks need equal preprocessed frame sizes")
        batch_all = (
            torch.from_numpy(np.stack(images)).float().div_(255.0).permute(0, 3, 1, 2)
        )
        result["rgb_hw"] = [height, width]
        result["model_input_hw"] = list(batch_all.shape[-2:])
        del images

        for index, (start, end) in enumerate(arguments.chunks):
            first = positions.start
            batch = batch_all[start - first : end - first][None].to(device)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            forward_started = time.perf_counter()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    aggregated, patch_start, queries, _ = model.aggregator(batch)
                _, _, features = model.semantic_head(aggregated, batch, patch_start)
                projected = model.aggregator.instance_queries_proj(queries)
                logits = torch.einsum("bnd, bshwd -> bnshw", projected, features)[0]
                label_logits = model.semantic_head.scratch.output_instance(projected)[0]
                del aggregated, features
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_started

            # Full-chunk float32 maps do not fit at once; a few queries at a time.
            probabilities = F.softmax(label_logits.float(), dim=-1)
            best = probabilities[:, :-1].max(1).values
            inside_mean = torch.zeros_like(best)
            covered = torch.zeros_like(best)
            step = arguments.query_batch
            for b in range(0, len(logits), step):
                part = logits[b : b + step].float()
                sigmoid, inside = part.sigmoid(), (part > 0).float()
                inside_mean[b : b + step] = (sigmoid * inside).flatten(1).sum(1) / (
                    inside.flatten(1).sum(1) + 1e-6
                )
                covered[b : b + step] = (
                    (sigmoid > arguments.pixel_sigmoid).flatten(1).sum(1).float()
                )
                del part, sigmoid, inside
            score = best * inside_mean
            kept = torch.nonzero(
                (best >= arguments.min_class_probability)
                & (covered > arguments.min_chunk_pixels)
            ).squeeze(1)
            maps = (
                torch.cat(
                    [
                        (logits[kept[b : b + step]].float().sigmoid() * 255)
                        .round()
                        .to(torch.uint8)
                        .cpu()
                        for b in range(0, len(kept), step)
                    ]
                )
                if len(kept)
                else torch.zeros((0, *logits.shape[1:]), dtype=torch.uint8)
            )
            _save_chunk(
                output / f"chunk_{index:03d}.npz",
                positions=np.arange(start, end, dtype=np.int64),
                queries=kept.cpu().numpy().astype(np.int64),
                probabilities=probabilities[kept].half().cpu().numpy(),
                score=score[kept].cpu().numpy().astype(np.float32),
                maps=maps.numpy(),
            )
            result["output_hw"] = list(logits.shape[-2:])
            result["chunks"].append(
                {
                    "index": index,
                    "start": start,
                    "end": end,
                    "forward_seconds": round(forward_seconds, 3),
                    "peak_reserved_gib": round(
                        torch.cuda.max_memory_reserved() / 1024**3, 2
                    ),
                    "kept_queries": int(len(kept)),
                }
            )
            print(json.dumps(result["chunks"][-1]), flush=True)
            del logits, label_logits, probabilities, batch, maps
            torch.cuda.empty_cache()
        result["completed"] = True
    except Exception as caught:  # recorded for diagnosis; the exit code reports it
        error = caught
        result["error"] = "".join(traceback.format_exception(caught))
        print(result["error"], file=sys.stderr, flush=True)
    result["wall_seconds"] = time.perf_counter() - started
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if error is None else 1


def main(argv: list[str] | None = None) -> int:
    return run(parse_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
