"""Embed the crops an AppearanceRunner wrote. Runs in the model's environment.

Imports nothing from this package, so it needs only torch, transformers and
Pillow wherever it runs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    arguments = parser.parse_args()

    crops = json.loads((arguments.directory / "manifest.json").read_text())["crops"]
    processor = AutoImageProcessor.from_pretrained(arguments.model)
    model = AutoModel.from_pretrained(arguments.model).to(arguments.device).eval()

    vectors = []
    for start in range(0, len(crops), arguments.batch_size):
        batch = crops[start : start + arguments.batch_size]
        images = [
            Image.open(arguments.directory / "crops" / crop["file"]).convert("RGB")
            for crop in batch
        ]
        inputs = processor(images=images, return_tensors="pt").to(arguments.device)
        with torch.no_grad():
            vectors.append(model(**inputs).pooler_output.float().cpu().numpy())

    stacked = np.concatenate(vectors) if vectors else np.zeros((0, 0), np.float32)
    stacked /= np.maximum(np.linalg.norm(stacked, axis=1, keepdims=True), 1e-9)
    np.savez_compressed(
        arguments.directory / "embeddings.npz",
        observation_ids=np.array([crop["observation_id"] for crop in crops]),
        vectors=stacked.astype(np.float32),
    )
    print(f"embedded {len(crops)} crops with {arguments.model}")


if __name__ == "__main__":
    main()
