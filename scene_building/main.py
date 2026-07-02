from pathlib import Path

from scripts.frame_extraction import extract_frames

VIDEO_PATH = Path("/mnt/data0/harsha/new_paper/CharacterAI/input/movie/manuhouse.mp4")
OUTPUT_DIR = Path("/mnt/data0/harsha/new_paper/CharacterAI/output/frame_extraction")
EXTRACT_FPS = 3


def main() -> None:
    saved_paths = extract_frames(VIDEO_PATH, OUTPUT_DIR, fps=EXTRACT_FPS)
    print(f"Extracted {len(saved_paths)} frames to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
