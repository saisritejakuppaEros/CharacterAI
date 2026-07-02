from pathlib import Path

import cv2


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 3.0,
) -> list[Path]:
    """Extract frames from a video at the given rate (frames per second)."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        cap.release()
        raise ValueError(f"Could not read FPS from video: {video_path}")

    stride = max(1, round(video_fps / fps))
    stem = video_path.stem
    saved_paths: list[Path] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            out_path = output_dir / f"{stem}_{len(saved_paths):06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_paths.append(out_path)
        frame_idx += 1

    cap.release()
    return saved_paths
