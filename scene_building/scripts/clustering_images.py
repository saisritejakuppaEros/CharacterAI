import csv
from pathlib import Path

import hdbscan
import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.preprocessing import normalize

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


def _load_image_paths(img_dir: Path) -> list[Path]:
    return sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _encode_images(
    paths: list[Path],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()

    embeddings: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        ).to(device)
        with torch.no_grad():
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())

    return np.vstack(embeddings)


def cluster_images(
    img_dir: str | Path,
    output_csv: str | Path,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    batch_size: int = 32,
) -> Path:
    """Cluster images with CLIP embeddings + HDBSCAN and write labels to CSV."""
    img_dir = Path(img_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    paths = _load_image_paths(img_dir)
    if not paths:
        raise ValueError(f"No images found in {img_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(paths)} images on {device}...")
    X = _encode_images(paths, device, batch_size=batch_size)
    X = normalize(X)

    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "cluster"])
        for path, label in zip(paths, labels):
            writer.writerow([str(path), label])

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(
        f"Wrote {len(paths)} rows to {output_csv} "
        f"({n_clusters} clusters, {n_noise} noise points)"
    )
    return output_csv