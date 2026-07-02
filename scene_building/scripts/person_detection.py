import csv
from pathlib import Path

import torch
from PIL import Image

PERSON_CLASS = "person"


def _load_clustering(clustering_csv: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with clustering_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["path"], int(row["cluster"])))
    return rows


def _person_stats_from_detections(
    detections_df, img_w: int, img_h: int
) -> tuple[int, float]:
    people = detections_df[detections_df["name"] == PERSON_CLASS]

    person_count = len(people)
    if person_count == 0:
        return 0, 0.0

    person_area = (
        (people["xmax"] - people["xmin"]) * (people["ymax"] - people["ymin"])
    ).sum()
    total_area = img_w * img_h
    return person_count, float(person_area / total_area)


def detect_people(
    clustering_csv: str | Path,
    output_dir: str | Path,
    model_name: str = "yolov5s",
    batch_size: int = 32,
) -> Path:
    """Run YOLOv5 person detection on clustered frames and write results to CSV."""
    clustering_csv = Path(clustering_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "person_detection.csv"

    entries = _load_clustering(clustering_csv)
    if not entries:
        raise ValueError(f"No rows found in {clustering_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLOv5 ({model_name}) on {device}...")
    model = torch.hub.load("ultralytics/yolov5", model_name, trust_repo=True)
    model.to(device)
    model.conf = 0.25

    rows: list[tuple[str, int, int, float]] = []
    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        paths = [path for path, _ in batch]

        images = [Image.open(p).convert("RGB") for p in paths]
        results = model(images)
        detections_per_image = results.pandas().xyxy

        for i, ((path, cluster), img) in enumerate(zip(batch, images)):
            person_count, area_ratio = _person_stats_from_detections(
                detections_per_image[i], img.width, img.height
            )
            rows.append((path, cluster, person_count, area_ratio))

        print(f"Processed {min(start + batch_size, len(entries))}/{len(entries)} images")

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["img_path", "clustering_id", "person_detected", "personarea/totalimagearea"]
        )
        for path, cluster, count, ratio in rows:
            writer.writerow([path, cluster, count, f"{ratio:.6f}"])

    with_people = sum(1 for _, _, count, _ in rows if count > 0)
    print(
        f"Wrote {len(rows)} rows to {output_csv} "
        f"({with_people} images with people detected)"
    )
    return output_csv
