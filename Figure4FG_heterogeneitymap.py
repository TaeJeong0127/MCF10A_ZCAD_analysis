import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter

# USER SETTINGS  
# input : csv file with color_shape_motility combined files

CSV_PATH = "patch_heterogeneity_color_cluster_vs_mean_motility_time012.csv"
SAVE_DIR = "final_density_heterogeneity_maps"
os.makedirs(SAVE_DIR, exist_ok=True)

METRICS = ["speed_mean", "neighbor_similarity_mean"]

COND_COL = "Condition"
TIME_COL = "time_idx"
XCOL = "n_cells"
YCOL = "color_heterogeneity"

YBINS = 15
MIN_COUNT_PER_BIN = 3

SIGMA_X = 0.6
SIGMA_Y = 1.2

CMAP_NAME = "Blues"
EMPTY_COLOR = "#f2f2f2"
POINT_COLOR = "k"
POINT_ALPHA = 0.12
POINT_SIZE = 8

N_LEVELS = 20

COLOR_MODE = "manual"
COLOR_LIMITS = {
    "speed_mean": (20.0, 30.0),
    "neighbor_similarity_mean": (0.25, 0.75),
}

SAVE_ALL_TIME = True
SAVE_BY_TIME = True
SAVE_COLORBAR_SEPARATELY = True

# LOAD

df = pd.read_csv(CSV_PATH)

# speed unit change(µm/hr)
df["speed_mean"] = df["speed_mean"] * 1.27 * 4

conditions = sorted(df[COND_COL].dropna().unique())
time_values = sorted(df[TIME_COL].dropna().unique())

global_x_min = int(np.floor(df[XCOL].min()))
global_x_max = int(np.ceil(df[XCOL].max()))
global_x_edges = np.arange(global_x_min - 0.5, global_x_max + 1.5, 1)

global_y_min = float(df[YCOL].min())
global_y_max = float(df[YCOL].max())
if global_y_min == global_y_max:
    global_y_min -= 0.5
    global_y_max += 0.5
global_y_edges = np.linspace(global_y_min, global_y_max, YBINS + 1)

# =========================================================
# functions for help
# =========================================================
def get_color_limits(data, metric, mode="robust"):
    vals = data[metric].dropna().values
    if len(vals) == 0:
        return (0, 1)

    if mode == "manual":
        return COLOR_LIMITS[metric]
    elif mode == "full":
        return (np.nanmin(vals), np.nanmax(vals))
    else:
        vmin = np.nanpercentile(vals, 5)
        vmax = np.nanpercentile(vals, 95)
        if vmin == vmax:
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            if vmin == vmax:
                vmin -= 1
                vmax += 1
        return (vmin, vmax)


def build_mean_and_count_grid(sub, xcol, ycol, zcol, x_edges, y_edges):
    sub = sub[[xcol, ycol, zcol]].dropna().copy()
    if len(sub) == 0:
        ny = len(y_edges) - 1
        nx = len(x_edges) - 1
        return np.full((ny, nx), np.nan), np.zeros((ny, nx), dtype=int)

    x_vals = sub[xcol].astype(int).values
    y_vals = sub[ycol].values
    z_vals = sub[zcol].values

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    sum_grid = np.zeros((ny, nx), dtype=float)
    count_grid = np.zeros((ny, nx), dtype=int)

    x_idx = x_vals - int(x_edges[0] + 0.5)

    y_idx = np.digitize(y_vals, y_edges) - 1
    y_idx = np.clip(y_idx, 0, ny - 1)

    for xi, yi, zi in zip(x_idx, y_idx, z_vals):
        if 0 <= xi < nx:
            sum_grid[yi, xi] += zi
            count_grid[yi, xi] += 1

    mean_grid = np.full((ny, nx), np.nan, dtype=float)
    mask = count_grid > 0
    mean_grid[mask] = sum_grid[mask] / count_grid[mask]

    return mean_grid, count_grid


def smooth_mean_grid(mean_grid, count_grid, sigma_y=1.2, sigma_x=0.6, min_count=3):
    valid = (~np.isnan(mean_grid)) & (count_grid >= min_count)

    if not np.any(valid):
        return np.full_like(mean_grid, np.nan)

    value_filled = np.where(valid, mean_grid, 0.0)
    weight = np.where(valid, count_grid.astype(float), 0.0)

    smooth_num = gaussian_filter(value_filled * weight, sigma=(sigma_y, sigma_x), mode="nearest")
    smooth_den = gaussian_filter(weight, sigma=(sigma_y, sigma_x), mode="nearest")

    out = np.full_like(mean_grid, np.nan, dtype=float)
    good = smooth_den > 1e-8
    out[good] = smooth_num[good] / smooth_den[good]

    support = gaussian_filter(valid.astype(float), sigma=(sigma_y, sigma_x), mode="nearest")
    out[support < 0.05] = np.nan

    return out


def save_standalone_colorbar(metric, vmin, vmax, cmap, out_path):
    fig_cb = plt.figure(figsize=(1.4, 4.2))
    ax_cb = fig_cb.add_axes([0.35, 0.08, 0.3, 0.84])

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label(metric)

    fig_cb.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig_cb)
    print(f"Saved colorbar: {out_path}")


def plot_one_figure(data, metric, title_suffix, save_path):
    fig, axes = plt.subplots(
        1, len(conditions),
        figsize=(5.0 * len(conditions), 6.0),
        sharex=True,
        sharey=True
    )

    if len(conditions) == 1:
        axes = [axes]

    vmin, vmax = get_color_limits(data, metric, mode=COLOR_MODE)

    cmap = mpl.cm.get_cmap(CMAP_NAME).copy()
    cmap.set_bad(EMPTY_COLOR)

    x_centers = np.arange(global_x_min, global_x_max + 1)
    y_centers = 0.5 * (global_y_edges[:-1] + global_y_edges[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers)

    for ax, cond in zip(axes, conditions):
        sub = data[data[COND_COL] == cond].copy()

        mean_grid, count_grid = build_mean_and_count_grid(
            sub, XCOL, YCOL, metric, global_x_edges, global_y_edges
        )

        smooth_grid = smooth_mean_grid(
            mean_grid,
            count_grid,
            sigma_y=SIGMA_Y,
            sigma_x=SIGMA_X,
            min_count=MIN_COUNT_PER_BIN
        )

        zmask = np.ma.masked_invalid(smooth_grid)

        if np.ma.count(zmask) > 0:
            levels = np.linspace(vmin, vmax, N_LEVELS)
            ax.contourf(
                Xc, Yc, zmask,
                levels=levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extend="both"
            )

        ax.scatter(
            sub[XCOL],
            sub[YCOL],
            s=POINT_SIZE,
            c=POINT_COLOR,
            alpha=POINT_ALPHA,
            linewidths=0
        )

        ax.set_title(f"{cond}\n(n={len(sub)})", fontsize=11)
        ax.set_xlabel("Cell density (n_cells)")
        ax.set_ylabel("Color heterogeneity")
        ax.set_xlim(global_x_edges[0], global_x_edges[-1])
        ax.set_ylim(global_y_edges[0], global_y_edges[-1])
        ax.set_xticks(np.arange(global_x_min, global_x_max + 1, 1))

    fig.suptitle(
        f"{metric} across density and color heterogeneity\n{title_suffix}",
        y=1.02,
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")

    if SAVE_COLORBAR_SEPARATELY:
        colorbar_path = save_path.replace(".pdf", "_colorbar.pdf")
        save_standalone_colorbar(metric, vmin, vmax, cmap, colorbar_path)


# =========================================================
# Main code
# =========================================================
if SAVE_ALL_TIME:
    for metric in METRICS:
        save_path = os.path.join(SAVE_DIR, f"{metric}_all_time_contourmap.pdf")
        plot_one_figure(
            data=df,
            metric=metric,
            title_suffix="All time points",
            save_path=save_path
        )

if SAVE_BY_TIME:
    for t in time_values:
        df_t = df[df[TIME_COL] == t].copy()
        for metric in METRICS:
            save_path = os.path.join(SAVE_DIR, f"{metric}_{TIME_COL}_{t}_contourmap.pdf")
            plot_one_figure(
                data=df_t,
                metric=metric,
                title_suffix=f"{TIME_COL} = {t}",
                save_path=save_path
            )

print("Done.")
