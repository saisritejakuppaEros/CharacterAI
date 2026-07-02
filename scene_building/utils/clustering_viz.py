"""Streamlit viewer for image clustering results."""

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "output" / "clustering.csv"


@st.cache_data
def load_clustering(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["cluster"] = df["cluster"].astype(int)
    df["filename"] = df["path"].apply(lambda p: Path(p).name)
    return df


def _cluster_label(cluster_id: int) -> str:
    return "Noise (unassigned)" if cluster_id == -1 else f"Cluster {cluster_id}"


def _render_image_grid(
    paths: list[str],
    columns: int,
    max_images: int,
    start_index: int,
) -> None:
    subset = paths[start_index : start_index + max_images]
    if not subset:
        st.info("No images in this range.")
        return

    for row_start in range(0, len(subset), columns):
        row_paths = subset[row_start : row_start + columns]
        cols = st.columns(columns)
        for col, path in zip(cols, row_paths):
            img_path = Path(path)
            with col:
                if img_path.exists():
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
                else:
                    st.warning(f"Missing: {img_path.name}")


def main() -> None:
    st.set_page_config(page_title="Clustering Viewer", layout="wide")
    st.title("Image Clustering Viewer")

    with st.sidebar:
        st.header("Settings")
        csv_path = st.text_input("Clustering CSV", value=str(DEFAULT_CSV))
        view_mode = st.radio("View", ["Single cluster", "All clusters"], index=0)
        columns = st.slider("Images per row", min_value=2, max_value=8, value=4)
        max_images = st.slider("Max images shown", min_value=4, max_value=200, value=40)

    csv_file = Path(csv_path)
    if not csv_file.exists():
        st.error(f"CSV not found: {csv_file}")
        return

    df = load_clustering(str(csv_file))
    cluster_ids = sorted(df["cluster"].unique())
    cluster_counts = df.groupby("cluster").size().to_dict()

    st.caption(
        f"{len(df)} images · {len([c for c in cluster_ids if c != -1])} clusters · "
        f"{cluster_counts.get(-1, 0)} noise"
    )

    if view_mode == "Single cluster":
        selected = st.selectbox(
            "Cluster",
            cluster_ids,
            format_func=lambda c: f"{_cluster_label(c)} ({cluster_counts[c]} images)",
        )
        paths = df.loc[df["cluster"] == selected, "path"].tolist()
        total = len(paths)

        if total > max_images:
            max_start = max(0, total - max_images)
            start_index = st.slider(
                "Start index",
                min_value=0,
                max_value=max_start,
                value=0,
                help=f"Showing a window of {max_images} images out of {total}.",
            )
        else:
            start_index = 0

        st.subheader(f"{_cluster_label(selected)} — {total} images")
        _render_image_grid(paths, columns, max_images, start_index)

    else:
        for cluster_id in cluster_ids:
            paths = df.loc[df["cluster"] == cluster_id, "path"].tolist()
            with st.expander(
                f"{_cluster_label(cluster_id)} — {len(paths)} images",
                expanded=cluster_id != -1 and len(paths) <= max_images,
            ):
                _render_image_grid(paths, columns, max_images, start_index=0)


if __name__ == "__main__":
    main()
