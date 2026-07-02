import csv
from pathlib import Path

PERSON_AREA_THRESHOLD = 0.75
CLUSTER_DROP_FRACTION = 0.75
TOP_K = 5


def _load_detection_csv(detection_csv: Path) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    with detection_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "img_path": row["img_path"],
                    "clustering_id": int(row["clustering_id"]),
                    "person_detected": int(row["person_detected"]),
                    "personarea/totalimagearea": float(row["personarea/totalimagearea"]),
                }
            )
    return rows


def _clusters_to_drop(
    rows: list[dict[str, str | int | float]],
    person_area_threshold: float,
    cluster_drop_fraction: float,
) -> set[int]:
    by_cluster: dict[int, list[float]] = {}
    for row in rows:
        cluster_id = int(row["clustering_id"])
        by_cluster.setdefault(cluster_id, []).append(float(row["personarea/totalimagearea"]))

    dropped: set[int] = set()
    for cluster_id, area_ratios in by_cluster.items():
        if cluster_id == -1:
            continue
        high_person_fraction = sum(
            ratio > person_area_threshold for ratio in area_ratios
        ) / len(area_ratios)
        if high_person_fraction > cluster_drop_fraction:
            dropped.add(cluster_id)
    return dropped


def rank_images(
    detection_csv: str | Path,
    output_csv: str | Path,
    person_area_threshold: float = PERSON_AREA_THRESHOLD,
    cluster_drop_fraction: float = CLUSTER_DROP_FRACTION,
    top_k: int = TOP_K,
) -> Path:
    """Filter person-heavy clusters and keep the least-person frames per cluster."""
    detection_csv = Path(detection_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_detection_csv(detection_csv)
    if not rows:
        raise ValueError(f"No rows found in {detection_csv}")

    dropped_clusters = _clusters_to_drop(
        rows, person_area_threshold, cluster_drop_fraction
    )
    kept_rows = [
        row for row in rows if int(row["clustering_id"]) not in dropped_clusters
    ]

    ranked_rows: list[dict[str, str | int | float]] = []
    by_cluster: dict[int, list[dict[str, str | int | float]]] = {}
    for row in kept_rows:
        cluster_id = int(row["clustering_id"])
        by_cluster.setdefault(cluster_id, []).append(row)

    for cluster_id in sorted(by_cluster):
        cluster_rows = sorted(
            by_cluster[cluster_id],
            key=lambda row: float(row["personarea/totalimagearea"]),
        )
        for rank, row in enumerate(cluster_rows[:top_k], start=1):
            ranked_rows.append({**row, "rank": rank})

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "img_path",
                "clustering_id",
                "person_detected",
                "personarea/totalimagearea",
                "rank",
            ]
        )
        for row in ranked_rows:
            writer.writerow(
                [
                    row["img_path"],
                    row["clustering_id"],
                    row["person_detected"],
                    f"{float(row['personarea/totalimagearea']):.6f}",
                    row["rank"],
                ]
            )

    print(
        f"Wrote {len(ranked_rows)} rows to {output_csv} "
        f"({len(by_cluster)} clusters kept, {len(dropped_clusters)} clusters dropped)"
    )
    return output_csv
