"""Streamlit viewer for image ranking results."""

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "output" / "image_ranking.csv"
AREA_COL = "personarea/totalimagearea"


@st.cache_data
def load_ranking(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["clustering_id"] = df["clustering_id"].astype(int)
    df["rank"] = df["rank"].astype(int)
    df["person_detected"] = df["person_detected"].astype(int)
    df[AREA_COL] = df[AREA_COL].astype(float)
    df["filename"] = df["img_path"].apply(lambda p: Path(p).name)
    return df


def _cluster_label(cluster_id: int) -> str:
    return "Noise (unassigned)" if cluster_id == -1 else f"Cluster {cluster_id}"


def _score_caption(row: pd.Series) -> str:
    area_pct = row[AREA_COL] * 100
    return (
        f"#{row['rank']} · {row['filename']}\n"
        f"People: {row['person_detected']} · Area: {area_pct:.1f}%"
    )


def _render_ranked_grid(rows: pd.DataFrame, columns: int) -> None:
    if rows.empty:
        st.info("No images in this cluster.")
        return

    ordered = rows.sort_values("rank")
    for row_start in range(0, len(ordered), columns):
        batch = ordered.iloc[row_start : row_start + columns]
        cols = st.columns(columns)
        for col, (_, row) in zip(cols, batch.iterrows()):
            img_path = Path(row["img_path"])
            with col:
                if img_path.exists():
                    st.image(
                        str(img_path),
                        caption=_score_caption(row),
                        use_container_width=True,
                    )
                else:
                    st.warning(f"Missing: {img_path.name}")


def main() -> None:
    st.set_page_config(page_title="Image Ranking Viewer", layout="wide")
    st.title("Image Ranking Viewer")

    with st.sidebar:
        st.header("Settings")
        csv_path = st.text_input("Ranking CSV", value=str(DEFAULT_CSV))
        view_mode = st.radio("View", ["Single cluster", "All clusters"], index=1)
        columns = st.slider("Images per row", min_value=1, max_value=5, value=5)

    csv_file = Path(csv_path)
    if not csv_file.exists():
        st.error(f"CSV not found: {csv_file}")
        return

    df = load_ranking(str(csv_file))
    cluster_ids = sorted(df["clustering_id"].unique())
    cluster_counts = df.groupby("clustering_id").size().to_dict()

    st.caption(
        f"{len(df)} ranked images · {len(cluster_ids)} clusters · "
        f"top {df.groupby('clustering_id').size().max()} per cluster"
    )

    if view_mode == "Single cluster":
        selected = st.selectbox(
            "Cluster",
            cluster_ids,
            format_func=lambda c: f"{_cluster_label(c)} ({cluster_counts[c]} images)",
        )
        cluster_df = df.loc[df["clustering_id"] == selected]
        st.subheader(f"{_cluster_label(selected)} — {len(cluster_df)} images")
        _render_ranked_grid(cluster_df, columns)

    else:
        for cluster_id in cluster_ids:
            cluster_df = df.loc[df["clustering_id"] == cluster_id]
            with st.expander(
                f"{_cluster_label(cluster_id)} — {len(cluster_df)} images",
                expanded=True,
            ):
                _render_ranked_grid(cluster_df, columns)


if __name__ == "__main__":
    main()
