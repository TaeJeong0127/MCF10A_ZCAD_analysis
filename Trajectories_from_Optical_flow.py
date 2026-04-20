import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops


def compute_optical_flow(prev_frame, next_frame):
    """
    Compute dense optical flow using the red channel only.
    If the input is grayscale, it is used directly.
    """
    prev_img = prev_frame[:, :, 2] if prev_frame.ndim == 3 else prev_frame
    next_img = next_frame[:, :, 2] if next_frame.ndim == 3 else next_frame

    flow = cv2.calcOpticalFlowFarneback(
        prev_img,
        next_img,
        None,
        pyr_scale=0.7,
        levels=5,
        winsize=7,
        iterations=5,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow


def extract_centroids(mask):
    """
    Extract centroids from connected components in a mask.
    Returns centroid coordinates in (x, y) format.
    """
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)

    if not props:
        return np.empty((0, 2), dtype=float), labeled_mask

    centroids = np.array(
        [[prop.centroid[1], prop.centroid[0]] for prop in props],
        dtype=float
    )
    return centroids, labeled_mask


def load_mask_array(mask_path):
    """
    Load a Cellpose-style mask array from a .npy file.
    """
    mask_data = np.load(mask_path, allow_pickle=True).item()

    if "masks" not in mask_data or mask_data["masks"] is None:
        raise ValueError(f"'masks' key not found in: {mask_path}")

    return mask_data["masks"]


def process_folders_with_window(
    image_folder,
    mask_folder,
    output_csv,
    window_size=12,
    border_margin=10,
):
    """
    Track cell centroids across sequential frames using optical flow.

    Workflow
    --------
    1. Initialize trajectories from the centroids in the first mask.
    2. Update each trajectory using optical flow only in subsequent frames.
    3. Do not re-assign trajectories to later masks.
    4. Remove trajectories that end too close to the image border.
    5. Save the tracking table as a CSV file.

    Parameters
    ----------
    image_folder : str
        Folder containing sequential TIFF images.
    mask_folder : str
        Folder containing mask .npy files.
    output_csv : str
        Output path for the tracking results.
    window_size : int, optional
        Reserved parameter for compatibility.
    border_margin : int, optional
        Margin used to exclude trajectories that terminate near the border.
    """
    image_files = sorted(
        f for f in os.listdir(image_folder)
        if f.endswith(".tif") and not f.startswith("._")
    )
    mask_files = sorted(
        f for f in os.listdir(mask_folder)
        if f.endswith(".npy") and not f.startswith("._")
    )

    if not image_files:
        print("No image files found.")
        return

    if not mask_files:
        print("No mask files found.")
        return

    n_pairs = min(len(image_files), len(mask_files))

    tracking_rows = []
    trajectories = {}
    next_trajectory_label = 0

    first_img_path = os.path.join(image_folder, image_files[0])
    first_mask_path = os.path.join(mask_folder, mask_files[0])

    first_img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
    if first_img is None:
        print(f"Failed to read first image: {first_img_path}")
        return

    img_h, img_w = first_img.shape[:2]

    first_mask = load_mask_array(first_mask_path)
    centroids0, _ = extract_centroids(first_mask)

    if len(centroids0) == 0:
        print("No objects were found in the first mask.")
        return

    for cell_id, (cx, cy) in enumerate(centroids0):
        traj_label = next_trajectory_label
        next_trajectory_label += 1

        trajectories[traj_label] = {
            "x": cx,
            "y": cy,
            "last_cell_id": cell_id,
            "last_mask_file": first_mask_path,
        }

        tracking_rows.append({
            "frame_id": 0,
            "cell_id": cell_id,
            "trajectory_label": traj_label,
            "centroid_x": cx,
            "centroid_y": cy,
            "mask_file": first_mask_path,
            "matched": 1,
        })

    print(f"Initialized {len(centroids0)} trajectories from frame 0.")

    for i in range(n_pairs - 1):
        prev_img_path = os.path.join(image_folder, image_files[i])
        next_img_path = os.path.join(image_folder, image_files[i + 1])
        next_mask_path = os.path.join(
            mask_folder, mask_files[min(i + 1, len(mask_files) - 1)]
        )

        prev_frame = cv2.imread(prev_img_path, cv2.IMREAD_UNCHANGED)
        next_frame = cv2.imread(next_img_path, cv2.IMREAD_UNCHANGED)

        if prev_frame is None or next_frame is None:
            print(f"Skipping unreadable frame pair: {prev_img_path}, {next_img_path}")
            continue

        flow = compute_optical_flow(prev_frame, next_frame)

        for traj_label, info in trajectories.items():
            x_prev = info["x"]
            y_prev = info["y"]

            fx = min(max(int(x_prev), 0), flow.shape[1] - 1)
            fy = min(max(int(y_prev), 0), flow.shape[0] - 1)

            flow_x, flow_y = flow[fy, fx]
            x_new = x_prev + flow_x
            y_new = y_prev + flow_y

            info["x"] = x_new
            info["y"] = y_new
            info["last_cell_id"] = -1
            info["last_mask_file"] = next_mask_path

            tracking_rows.append({
                "frame_id": i + 1,
                "cell_id": -1,
                "trajectory_label": traj_label,
                "centroid_x": x_new,
                "centroid_y": y_new,
                "mask_file": next_mask_path,
                "matched": 0,
            })

        print(f"Processed frame {i + 1} / {n_pairs - 1}")

    tracking_df = pd.DataFrame(tracking_rows)

    if tracking_df.empty:
        print("No tracking data generated.")
        return

    last_positions = (
        tracking_df.sort_values("frame_id")
        .groupby("trajectory_label")
        .tail(1)
    )

    near_border = (
        (last_positions["centroid_x"] <= border_margin) |
        (last_positions["centroid_x"] >= img_w - border_margin) |
        (last_positions["centroid_y"] <= border_margin) |
        (last_positions["centroid_y"] >= img_h - border_margin)
    )

    excluded_ids = set(last_positions.loc[near_border, "trajectory_label"].values)
    print(f"Excluded border-ending trajectories: {len(excluded_ids)}")

    if excluded_ids:
        tracking_df = tracking_df[
            ~tracking_df["trajectory_label"].isin(excluded_ids)
        ].reset_index(drop=True)

    tracking_df.to_csv(output_csv, index=False)
    print(f"Tracking results saved to: {output_csv}")
    print(f"Final row count: {len(tracking_df)}")


if __name__ == "__main__":
    process_folders_with_window(
        image_folder="",
        mask_folder="",
        output_csv="",
        window_size=12,
        border_margin=25,
    )
