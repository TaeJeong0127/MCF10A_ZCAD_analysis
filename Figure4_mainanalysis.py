import re
import numpy as np
import pandas as pd
from itertools import combinations

#EDIT ME
FINAL_CSV = # folder for optical flow motility 
OUT_CSV   = # name for output files 

PATCH_SIZE = 75
MIN_CELLS  = 6

COL_SAMPLE = "sample_key"      # sample column
COL_MASK   = "mask_file"       # time0/time1/time2
COL_X      = "Ellipse Centroid X"
COL_Y      = "Ellipse Centroid Y"
COL_COLOR  = "Color"
COL_CLUST  = "ShapeCluster"    # Morphological clusters

SWAP_XY_FOR_PATCH = True

# time0 mask -> t1, time1 mask -> t2, time2 mask -> t3
MOT_COLS_BY_TIME = {
    0: {"speed":"speed_t1", "directionality":"directionality_t1", "arrest":"arrest_t1", "neighbor_similarity":"neighbor_similarity_t1"},
    1: {"speed":"speed_t2", "directionality":"directionality_t2", "arrest":"arrest_t2", "neighbor_similarity":"neighbor_similarity_t2"},
    2: {"speed":"speed_t3", "directionality":"directionality_t3", "arrest":"arrest_t3", "neighbor_similarity":"neighbor_similarity_t3"},
}
# ====================================================

# ---- heterogeneity embedding ----
color_positions = {
    'Green':  ( 1.0,  1.0),
    'Red':    (-1.0, -1.0),
    'Yellow': ( 1.0, -1.0),
    'Black':  ( 0.0,  0.0),
}
cluster_positions = {
    0: ( 1.0, 0.0),
    1: (-0.5,  np.sqrt(3)/2.0),
    2: (-0.5, -np.sqrt(3)/2.0),
}

def get_group(filename):
    """
    filename examples:
    0702_merge_time1_pos6.npy
    0717_merge_time0_pos12.npy
    """

    fname = str(filename)

    date = fname.split('_')[0]

    import re
    m = re.search(r'pos(\d+)', fname)
    if m is None:
        return "Unknown"
    pos_match = m.group(1)

    if date in ['0702', '0709', '0911']:
        pos_group_1 = ['1', '2', '19', '20']
        pos_group_2 = ['3', '4', '17', '18']
        pos_group_3 = ['5', '6', '15', '16']

        if pos_match in pos_group_1:
            return "Group1"
        elif pos_match in pos_group_2:
            return "Group2"
        elif pos_match in pos_group_3:
            return "Group3"

    elif date == '0717':
        try:
            pos_num = int(pos_match)
            if 1 <= pos_num <= 8:
                return "Group1"
            elif 9 <= pos_num <= 16:
                return "Group2"
            elif 17 <= pos_num <= 24:
                return "Group3"
        except:
            return "Unknown"

    return "Unknown"


def extract_time_idx(mask_file: str):
    m = re.search(r"time(\d+)", str(mask_file), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def pairwise_mean_dist_pts(pts):
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return np.nan
    dists = [np.linalg.norm(np.array(a)-np.array(b)) for a,b in combinations(pts, 2)]
    return float(np.mean(dists)) if dists else np.nan

def heterogeneity_patch_components(cells: pd.DataFrame):
    """
    returns:
      color_het, cluster_het, combined_het(=color+cluster)
    """
    if len(cells) < MIN_CELLS:
        return np.nan, np.nan, np.nan

    color_xy = [color_positions.get(str(c), None) for c in cells[COL_COLOR].astype(str)]

    cl_xy = []
    for c in cells[COL_CLUST]:
        if pd.isna(c):
            cl_xy.append(None)
        else:
            try:
                cl_xy.append(cluster_positions.get(int(c), None))
            except Exception:
                cl_xy.append(None)

    color_het   = pairwise_mean_dist_pts(color_xy)
    cluster_het = pairwise_mean_dist_pts(cl_xy)

    if np.isnan(color_het) and np.isnan(cluster_het):
        combined = np.nan
    else:
        combined = (0.0 if np.isnan(color_het) else color_het) + (0.0 if np.isnan(cluster_het) else cluster_het)

    return color_het, cluster_het, combined

def nanmean_safe(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan

def nanmedian_safe(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan

# ===================== RUN =====================
df = pd.read_csv(FINAL_CSV)

if COL_SAMPLE not in df.columns:
    df[COL_SAMPLE] = df[COL_MASK].astype(str)


# time_idx
df["time_idx"] = df[COL_MASK].astype(str).apply(extract_time_idx)
df = df[df["time_idx"].isin([0,1,2])].copy()

# coords
x_raw = pd.to_numeric(df[COL_X], errors="coerce")
y_raw = pd.to_numeric(df[COL_Y], errors="coerce")
if SWAP_XY_FOR_PATCH:
    df["_x_img"] = y_raw
    df["_y_img"] = x_raw
else:
    df["_x_img"] = x_raw
    df["_y_img"] = y_raw

# drop required
need = [COL_SAMPLE, "time_idx", "_x_img", "_y_img", COL_COLOR, COL_CLUST, COL_MASK]
df = df.dropna(subset=need).copy()

# patch id
df["patch_x"] = (df["_x_img"] // PATCH_SIZE).astype(int)
df["patch_y"] = (df["_y_img"] // PATCH_SIZE).astype(int)
df["patch_id"] = df["patch_x"].astype(str) + "_" + df["patch_y"].astype(str)

rows = []

for (sample, t, patch_id), g in df.groupby([COL_SAMPLE, "time_idx", "patch_id"]):
    if len(g) < MIN_CELLS:
        continue

    color_het, cluster_het, combined_het = heterogeneity_patch_components(g)

    mot_map = MOT_COLS_BY_TIME[int(t)]

    out = {
        "Condition": get_group(g[COL_MASK].iloc[0]),
        "sample_key": sample,
        "time_idx": int(t),
        "patch_id": patch_id,
        "patch_x": int(g["patch_x"].iloc[0]),
        "patch_y": int(g["patch_y"].iloc[0]),
        "n_cells": int(len(g)),

        "color_heterogeneity": float(color_het) if np.isfinite(color_het) else np.nan,
        "cluster_heterogeneity": float(cluster_het) if np.isfinite(cluster_het) else np.nan,
        "combined_heterogeneity": float(combined_het) if np.isfinite(combined_het) else np.nan,
    }

    for name, col in mot_map.items():
        if col not in g.columns:
            out[f"{name}_mean"] = np.nan
            out[f"{name}_median"] = np.nan
        else:
            out[f"{name}_mean"] = nanmean_safe(g[col])
            out[f"{name}_median"] = nanmedian_safe(g[col])

    rows.append(out)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print("[DONE] saved:", OUT_CSV)
print(out_df.head())
