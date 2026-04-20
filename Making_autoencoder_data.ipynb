import os
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.measure import regionprops
from skimage.transform import resize, rotate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CSV_PATH = "all_segments_mobility_umap_sampled_svm_CombinedLabel_updated.csv"
MAX_MASKS = 20000
MASK_SIZE = (64, 64)
LATENT_DIM = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_SIZE = 15
MIN_COUNT_GRID = 10
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

np.random.seed(0)
torch.manual_seed(0)


def load_input_table(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded CSV: {csv_path}")
    print(f"[INFO] Total rows: {len(df)}")
    print(f"[INFO] Example columns: {df.columns.tolist()[:20]}")

    required_cols = {"UMAP1", "UMAP2"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError("Input CSV must contain precomputed UMAP1 and UMAP2 columns.")

    return df


mask_cache = {}


def load_cellpose_mask(mask_path):
    mask_path = os.path.expanduser(str(mask_path).strip())

    if mask_path in mask_cache:
        return mask_cache[mask_path]

    if not os.path.exists(mask_path):
        return None

    arr = np.load(mask_path, allow_pickle=True)

    try:
        obj = arr.item()
        if isinstance(obj, dict) and "masks" in obj:
            full_mask = obj["masks"]
        else:
            full_mask = arr
    except Exception:
        full_mask = arr

    mask_cache[mask_path] = full_mask
    return full_mask


def extract_single_cell_mask(row, size=MASK_SIZE, pad=5):
    if "matched_mask_file" not in row or pd.isna(row["matched_mask_file"]):
        return None

    full_mask = load_cellpose_mask(row["matched_mask_file"])
    if full_mask is None:
        return None

    if "matched_cell_label" in row and not pd.isna(row["matched_cell_label"]):
        cell_label = int(row["matched_cell_label"])
    elif "shape_Cell Label" in row and not pd.isna(row["shape_Cell Label"]):
        cell_label = int(row["shape_Cell Label"])
    else:
        return None

    mask = full_mask[0] if full_mask.ndim == 3 else full_mask
    cell_mask = (mask == cell_label).astype(np.uint8)

    if cell_mask.sum() == 0:
        return None

    props_list = regionprops(cell_mask)
    if not props_list:
        return None

    props = props_list[0]
    minr, minc, maxr, maxc = props.bbox
    h, w = cell_mask.shape

    minr = max(minr - pad, 0)
    minc = max(minc - pad, 0)
    maxr = min(maxr + pad, h)
    maxc = min(maxc + pad, w)

    cropped = cell_mask[minr:maxr, minc:maxc].astype(float)

    angle_deg = props.orientation * 180.0 / np.pi
    rotated = rotate(cropped, angle=-angle_deg, preserve_range=True)

    resized = resize(rotated, size, anti_aliasing=False).astype(np.float32)
    return resized


def collect_masks(df, max_masks):
    mask_list = []
    valid_indices = []

    print("[INFO] Collecting masks from table...")
    shuffled_df = df.sample(frac=1, random_state=0)

    for idx, row in tqdm(shuffled_df.iterrows(), total=len(shuffled_df)):
        mask = extract_single_cell_mask(row)
        if mask is not None:
            mask_list.append(mask)
            valid_indices.append(idx)

        if len(mask_list) >= max_masks:
            break

    print(f"[INFO] Collected mask count: {len(mask_list)}")

    if not mask_list:
        if "matched_mask_file" in df.columns:
            file_col = df["matched_mask_file"].astype(str).str.strip()
            exists_flags = file_col.apply(os.path.exists)
            print(f"[DEBUG] Existing files: {exists_flags.sum()} / {len(exists_flags)}")
            print("[DEBUG] Example existing paths:")
            print(file_col[exists_flags].head())
        raise RuntimeError("No valid masks were collected. Check mask paths and label columns.")

    mask_array = np.stack(mask_list)
    df_valid = df.loc[valid_indices].reset_index(drop=True)

    print(f"[INFO] mask_array shape: {mask_array.shape}")
    print(f"[INFO] df_valid shape: {df_valid.shape}")
    print(f"[INFO] Cached mask files: {len(mask_cache)}")

    return mask_array, df_valid


class MaskDataset(Dataset):
    def __init__(self, mask_array):
        self.data = torch.tensor(mask_array).unsqueeze(1).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * 16 * 16),
            nn.Unflatten(1, (16, 16, 16)),
            nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder(mask_array, latent_dim, device, epochs, batch_size, learning_rate):
    dataset = MaskDataset(mask_array)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

    return model


def extract_latent_vectors(model, mask_array, device):
    model.eval()
    with torch.no_grad():
        z_all = model.encoder(
            torch.tensor(mask_array).unsqueeze(1).float().to(device)
        ).cpu().numpy()

    print(f"[INFO] Latent array shape: {z_all.shape}")
    return z_all


def plot_reconstructed_masks_on_umap(df_valid, z_all, model, device):
    umap_xy = df_valid[["UMAP1", "UMAP2"]].values
    x_coords = umap_xy[:, 0]
    y_coords = umap_xy[:, 1]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(x_coords, y_coords, s=5, c="lightgray", alpha=0.3)

    for i in range(len(z_all)):
        gx, gy = x_coords[i], y_coords[i]
        z = torch.tensor(z_all[i:i + 1], dtype=torch.float32, device=device)

        with torch.no_grad():
            recon = model.decoder(z).cpu().numpy()[0, 0]

        extent = [gx - 0.3, gx + 0.3, gy - 0.3, gy + 0.3]
        ax.imshow(recon, cmap="gray", extent=extent, alpha=0.9)

    ax.set_title("AE-Reconstructed Masks on Precomputed Mobility UMAP")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def make_grid_spec(umap_xy, grid_size=12):
    x_min, x_max = float(umap_xy[:, 0].min()), float(umap_xy[:, 0].max())
    y_min, y_max = float(umap_xy[:, 1].min()), float(umap_xy[:, 1].max())

    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)

    return {
        "grid_size": grid_size,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max),
    }


def assign_cells_to_grid(umap_xy, x_edges, y_edges):
    ix = np.clip(np.digitize(umap_xy[:, 0], x_edges) - 1, 0, len(x_edges) - 2)
    iy = np.clip(np.digitize(umap_xy[:, 1], y_edges) - 1, 0, len(y_edges) - 2)
    return ix, iy


def build_latent_umap_profile(umap_xy_ref, z_ref, grid_size=12, min_count=10, eps=1e-8):
    latent_dim = z_ref.shape[1]
    grid = make_grid_spec(umap_xy_ref, grid_size)
    ix, iy = assign_cells_to_grid(umap_xy_ref, grid["x_edges"], grid["y_edges"])

    global_mean = z_ref.mean(axis=0)
    global_var = z_ref.var(axis=0) + eps

    cells = {}
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (ix == i) & (iy == j)
            n = int(mask.sum())

            if n >= min_count:
                z_sub = z_ref[mask]
                cells[(i, j)] = {
                    "n": n,
                    "mean": z_sub.mean(axis=0),
                    "var": z_sub.var(axis=0) + eps,
                }

    return {
        "latent_dim": latent_dim,
        "grid": grid,
        "min_count": min_count,
        "eps": eps,
        "global": {
            "n": int(len(z_ref)),
            "mean": global_mean,
            "var": global_var,
        },
        "cells": cells,
    }


def make_color_one_hot(df_valid, color_col="Color_updated"):
    color_vocab = ["Black", "Green", "Red", "Yellow"]

    if color_col in df_valid.columns:
        color_series = df_valid[color_col].astype(str).str.strip().str.title()
    else:
        color_series = pd.Series(["Unknown"] * len(df_valid))

    color_ohe = np.zeros((len(df_valid), len(color_vocab)), dtype=float)
    color_index = {color: i for i, color in enumerate(color_vocab)}

    for row_idx, color_value in enumerate(color_series):
        if color_value in color_index:
            color_ohe[row_idx, color_index[color_value]] = 1.0

    return color_ohe, color_vocab, color_series


def add_color_histograms_to_profile(profile, umap_xy, color_series):
    ix_ref, iy_ref = assign_cells_to_grid(
        umap_xy,
        profile["grid"]["x_edges"],
        profile["grid"]["y_edges"],
    )

    cell_color_hist = {}
    grid_size = profile["grid"]["grid_size"]

    for i in range(grid_size):
        for j in range(grid_size):
            mask = (ix_ref == i) & (iy_ref == j)
            if mask.sum() == 0:
                continue
            counts = Counter(color_series[mask])
            cell_color_hist[(i, j)] = dict(counts)

    profile["color_hist"] = cell_color_hist
    return profile


def save_outputs(model, latent_dim, mask_size, profile):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "input_size": (1, mask_size[0], mask_size[1]),
        },
        "ae_svm_state.pt",
    )
    joblib.dump(profile, "latent_umap_profile_svm.joblib")

    print("[INFO] Saved AE weights to: ae_svm_state.pt")
    print("[INFO] Saved latent profile to: latent_umap_profile_svm.joblib")


def main():
    df_all = load_input_table(CSV_PATH)
    mask_array, df_valid = collect_masks(df_all, MAX_MASKS)

    model = train_autoencoder(
        mask_array=mask_array,
        latent_dim=LATENT_DIM,
        device=DEVICE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    z_all = extract_latent_vectors(model, mask_array, DEVICE)
    plot_reconstructed_masks_on_umap(df_valid, z_all, model, DEVICE)

    umap_xy = df_valid[["UMAP1", "UMAP2"]].values
    color_ohe, color_vocab, color_series = make_color_one_hot(df_valid, color_col="Color_updated")

    z_ref_aug = np.concatenate([z_all, color_ohe], axis=1)

    profile = build_latent_umap_profile(
        umap_xy_ref=umap_xy,
        z_ref=z_ref_aug,
        grid_size=GRID_SIZE,
        min_count=MIN_COUNT_GRID,
    )
    profile["color_vocab"] = color_vocab
    profile["latent_aug_dim"] = z_ref_aug.shape[1]
    profile = add_color_histograms_to_profile(profile, umap_xy, color_series.to_numpy())

    save_outputs(model, LATENT_DIM, MASK_SIZE, profile)


if __name__ == "__main__":
    main()
