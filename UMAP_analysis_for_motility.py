# -*- coding: utf-8 -*-
import os
import joblib
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


sns.set_context("notebook", font_scale=1.3)
plt.rcParams["font.family"] = "Arial"


BASE_DIR = ""
CONDITIONS = ["Control", "1ng", "5ng"]
SEGMENT_DEFS = [
    ("seg1_1-30", "1-30"),
    ("seg2_30-60", "30-60"),
    ("seg3_60-90", "60-90"),
]
FILE_COL = "source_file"

MOBILITY_COLS = [
    "speed",
    "euclidean_dist",
    "segment_length",
    "cumulative_length",
    "meandering_index",
    "outreach_ratio",
    "MSD",
    "max_dist",
    "arrest_coefficient",
    "turn_count",
    "neighbor_similarity",
    "loop_score",
    "angle_variation",
    "curvature_count",
    "polygon_area",
    "turning_index",
    "avg_acceleration",
]


def load_segment_tables(base_dir, conditions, segment_defs):
    """
    Load and concatenate per-condition, per-segment CSV files.
    """
    tables = []

    for condition in conditions:
        for segment_tag, segment_label in segment_defs:
            file_name = f"all_samples_color_vs_mobility_{condition}_{segment_tag}_with_shape_svm.csv"
            file_path = os.path.join(base_dir, file_name)

            if not os.path.exists(file_path):
                print(f"[WARNING] Missing file: {file_path}")
                continue

            df_seg = pd.read_csv(file_path)
            df_seg["Condition"] = condition
            df_seg["Segment"] = segment_label
            tables.append(df_seg)

            print(f"[INFO] Loaded {file_name} (n={len(df_seg)})")

    if not tables:
        raise RuntimeError("No input CSV files were loaded.")

    df = pd.concat(tables, ignore_index=True)
    df["Segment_start"] = df["Segment"].apply(lambda x: int(str(x).split("-")[0]))
    return df


def build_combined_label(df, color_col="Color_updated", shape_col="Shape_Label"):
    """
    Build a combined label from color and shape annotations.
    """
    if shape_col not in df.columns:
        raise ValueError(f"'{shape_col}' column is missing.")

    if color_col not in df.columns:
        raise ValueError(f"'{color_col}' column is missing.")

    ignore_colors = {"Unknown", "unknown", "Yellow", "yellow"}

    df = df.copy()
    n_before = len(df)
    df = df[~df[color_col].isin(ignore_colors)].copy()
    print(f"[INFO] Removed ignored colors: {n_before} -> {len(df)}")

    df[shape_col] = df[shape_col].astype(int)
    df["Combined_Label"] = df[color_col].astype(str) + "_S" + df[shape_col].astype(str)
    return df


def filter_valid_mobility_rows(df, mobility_cols):
    """
    Keep only rows with complete mobility features.
    """
    missing_cols = [col for col in mobility_cols if col not in df.columns]
    if missing_cols:
        print(f"[WARNING] Missing mobility features: {missing_cols}")

    cols_in_use = [col for col in mobility_cols if col in df.columns]
    df_valid = df.dropna(subset=cols_in_use).copy()

    print(f"[INFO] Rows after NaN filtering: {df_valid.shape}")
    return df_valid, cols_in_use


def subsample_per_file(df, file_col="source_file", n_per_file=300, sort_col=None):
    """
    Uniformly subsample up to n_per_file rows from each file group.
    """
    if file_col not in df.columns:
        print(f"[WARNING] '{file_col}' not found. Falling back to global subsampling.")
        df_sorted = df.sort_values(sort_col) if sort_col in df.columns else df.sort_index()
        n_total = min(len(df_sorted), n_per_file * 10)
        selected_idx = np.linspace(0, len(df_sorted) - 1, n_total, dtype=int)
        return df_sorted.iloc[selected_idx].copy()

    selected_indices = []

    for _, group in df.groupby(file_col):
        group = group.sort_values(sort_col) if sort_col in group.columns else group.sort_index()

        if len(group) <= n_per_file:
            selected_indices.extend(group.index.tolist())
        else:
            selected_idx = np.linspace(0, len(group) - 1, n_per_file, dtype=int)
            selected_indices.extend(group.index[selected_idx].tolist())

    return df.loc[selected_indices].copy()


def run_umap(df, feature_cols, n_neighbors=30, min_dist=0.3, random_state=42):
    """
    Fit StandardScaler and UMAP on the selected mobility features.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[feature_cols])

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    coords = reducer.fit_transform(x_scaled)

    df = df.copy()
    df["UMAP1"] = coords[:, 0]
    df["UMAP2"] = coords[:, 1]

    return df, scaler, reducer


def plot_umap_overview(df):
    """
    Plot the global UMAP scatter colored by combined label.
    """
    unique_labels = df["Combined_Label"].unique()
    palette = sns.color_palette("tab20", n_colors=len(unique_labels))

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="UMAP1",
        y="UMAP2",
        hue="Combined_Label",
        style="Condition",
        alpha=0.5,
        s=15,
        palette=palette,
    )
    plt.title("Mobility UMAP by Combined Label and Condition")
    plt.tight_layout()
    plt.show()

    return palette


def plot_umap_by_condition(df, palette):
    """
    Plot UMAP distributions split by condition.
    """
    grid = sns.FacetGrid(
        df,
        col="Condition",
        hue="Combined_Label",
        col_wrap=3,
        height=4,
        sharex=True,
        sharey=True,
        palette=palette,
    )
    grid.map_dataframe(
        sns.scatterplot,
        x="UMAP1",
        y="UMAP2",
        alpha=0.6,
        s=10,
    )
    grid.add_legend(title="Combined_Label")
    grid.fig.suptitle("Mobility UMAP by Condition", y=1.02, fontsize=16)
    plt.show()


def plot_umap_by_segment_and_condition(df, conditions, palette):
    """
    Plot KDE and sampled scatter across segment-condition panels.
    """
    segment_order = sorted(df["Segment"].unique(), key=lambda x: int(x.split("-")[0]))

    fig, axes = plt.subplots(
        len(segment_order),
        len(conditions),
        figsize=(4 * len(conditions), 4 * len(segment_order)),
        sharex=True,
        sharey=True,
    )

    if len(segment_order) == 1 and len(conditions) == 1:
        axes = np.array([[axes]])
    elif len(segment_order) == 1:
        axes = np.array([axes])

    for i, segment in enumerate(segment_order):
        for j, condition in enumerate(conditions):
            ax = axes[i, j]
            subset = df[(df["Segment"] == segment) & (df["Condition"] == condition)]

            if subset.empty:
                ax.set_title(f"{condition}, {segment}\n(no data)")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            sns.kdeplot(
                data=subset,
                x="UMAP1",
                y="UMAP2",
                fill=True,
                thresh=0.05,
                levels=10,
                alpha=0.7,
                ax=ax,
                common_norm=True,
                common_grid=True,
            )

            sns.scatterplot(
                data=subset.sample(min(len(subset), 300), random_state=0),
                x="UMAP1",
                y="UMAP2",
                hue="Combined_Label",
                palette=palette,
                alpha=0.4,
                s=8,
                ax=ax,
                legend=False,
            )

            ax.set_title(f"{condition}, {segment}")
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")

    plt.suptitle("Mobility UMAP Across Condition and Segment", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    df = load_segment_tables(BASE_DIR, CONDITIONS, SEGMENT_DEFS)
    print(f"[INFO] Full dataset shape: {df.shape}")

    df = build_combined_label(df, color_col="Color_updated", shape_col="Shape_Label")
    print(df["Combined_Label"].value_counts().head())

    df_valid, mobility_cols_in_use = filter_valid_mobility_rows(df, MOBILITY_COLS)

    df_plot = subsample_per_file(
        df_valid,
        file_col=FILE_COL,
        n_per_file=300,
        sort_col="Segment_start",
    )

    print(f"[INFO] Input rows after filtering: {len(df_valid)}")
    print(f"[INFO] Rows used for UMAP: {len(df_plot)}")

    df_plot, scaler, reducer = run_umap(df_plot, mobility_cols_in_use)

    out_csv = os.path.join(BASE_DIR, "all_segments_mobility_umap_sampled_svm.csv")
    df_plot.to_csv(out_csv, index=False)

    scaler_path = os.path.join(BASE_DIR, "mobility_umap_scaler.joblib")
    reducer_path = os.path.join(BASE_DIR, "mobility_umap_model.joblib")

    joblib.dump(scaler, scaler_path)
    joblib.dump(reducer, reducer_path)

    print(f"[INFO] Saved UMAP table to: {out_csv}")
    print(f"[INFO] Saved scaler to: {scaler_path}")
    print(f"[INFO] Saved model to: {reducer_path}")

    palette = plot_umap_overview(df_plot)
    plot_umap_by_condition(df_plot, palette)
    plot_umap_by_segment_and_condition(df_plot, CONDITIONS, palette)


if __name__ == "__main__":
    main()
