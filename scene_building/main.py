from pathlib import Path

from scripts.clustering_images import cluster_images
from scripts.frame_extraction import extract_frames
from scripts.image_ranking import rank_images
from scripts.person_detection import detect_people

VIDEO_PATH = Path("/mnt/data0/harsha/new_paper/CharacterAI/input/movie/manuhouse.mp4")
FRAMES_DIR = Path("/mnt/data0/harsha/new_paper/CharacterAI/output/frame_extraction")
CLUSTERING_CSV = Path("/mnt/data0/harsha/new_paper/CharacterAI/output/clustering.csv")
PERSON_DETECTION_DIR = Path("/mnt/data0/harsha/new_paper/CharacterAI/output/person_detection")
IMAGE_RANKING_CSV = Path("/mnt/data0/harsha/new_paper/CharacterAI/output/image_ranking.csv")
EXTRACT_FPS = 3


def main() -> None:
    # saved_paths = extract_frames(VIDEO_PATH, FRAMES_DIR, fps=EXTRACT_FPS)
    # print(f"Extracted {len(saved_paths)} frames to {FRAMES_DIR}")

    # cluster_images(FRAMES_DIR, CLUSTERING_CSV)
    # print(f"Clustering results saved to {CLUSTERING_CSV}")

    # person_detection_csv = detect_people(CLUSTERING_CSV, PERSON_DETECTION_DIR)
    # print(f"Person detection results saved to {person_detection_csv}")

    person_detection_csv = PERSON_DETECTION_DIR / "person_detection.csv"
    image_ranking_csv = rank_images(person_detection_csv, IMAGE_RANKING_CSV)
    print(f"Image ranking results saved to {image_ranking_csv}")


if __name__ == "__main__":
    main()