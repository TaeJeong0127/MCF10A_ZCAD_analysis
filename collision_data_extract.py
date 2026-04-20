import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy.ndimage import center_of_mass, binary_dilation
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import glob

def compute_mask_centroids(mask):
    labels = np.unique(mask)
    labels = labels[labels != 0]
    centroids = {}
    for label in labels:
        binary = mask == label
        if np.any(binary):
            r, c = center_of_mass(binary)
            centroids[label] = (c, r)
    return centroids


def match_mask_label_within_bbox(trajectory_point, mask_centroids, bbox, max_dist=20):
    tx, ty = trajectory_point
    min_dist = float('inf')
    matched_label = None
    min_r, min_c, max_r, max_c = bbox
    for label, (mx, my) in mask_centroids.items():
        if not (min_c <= mx <= max_c and min_r <= my <= max_r):
            continue
        dist = np.linalg.norm([tx - mx, ty - my])
        if dist < min_dist and dist <= max_dist:
            min_dist = dist
            matched_label = label
    return matched_label


def compute_bounding_box(binary_mask):
    coords = np.argwhere(binary_mask)
    if coords.size == 0:
        return None
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    return (min_row, min_col, max_row, max_col)


def bounding_boxes_intersect(box1, box2):
    if box1 is None or box2 is None:
        return False
    min_row1, min_col1, max_row1, max_col1 = box1
    min_row2, min_col2, max_row2, max_col2 = box2
    if max_row1 < min_row2 or max_row2 < min_row1:
        return False
    if max_col1 < min_col2 or max_col2 < min_col1:
        return False
    return True


def filter_approaching_trajectory_pairs(df, proximity_threshold=50, dT=6, velocity_threshold=2.5, consistent_proximity_threshold=24):
    start_time = time.time()
    pairs_to_check = []
    grouped = df.groupby('trajectory_label')
    trajectories = {k: v[['frame_id', 'centroid_x', 'centroid_y']].values for k, v in grouped}
    frame_groups = df.groupby('frame_id')

    for f, group in tqdm(frame_groups, desc="Identifying approaching pairs"):
        coords = group[['centroid_x', 'centroid_y']].values
        labels = group['trajectory_label'].values
        dist_matrix = cdist(coords, coords)
        interaction_indices = np.argwhere((dist_matrix < proximity_threshold) & (dist_matrix > 0))

        for i, j in interaction_indices:
            label_i, label_j = labels[i], labels[j]
            if label_i >= label_j:
                continue
            traj_i = trajectories[label_i]
            traj_j = trajectories[label_j]
            common_frames = np.intersect1d(traj_i[:, 0], traj_j[:, 0])
            if len(common_frames) < dT:
                continue

            dists = [
                np.linalg.norm(traj_i[traj_i[:, 0] == frame][0, 1:3] - traj_j[traj_j[:, 0] == frame][0, 1:3])
                for frame in common_frames
            ]
            min_frame = common_frames[np.argmin(dists)]

            close_frames = np.array(dists) < proximity_threshold
            if np.sum(close_frames) > consistent_proximity_threshold:
                continue
        

            #if np.mean(dists[:3]) - np.mean(dists[-3:]) < 0:
            traj_i_pre = traj_i[(traj_i[:, 0] >= min_frame - dT) & (traj_i[:, 0] <= min_frame)]
            traj_j_pre = traj_j[(traj_j[:, 0] >= min_frame - dT) & (traj_j[:, 0] <= min_frame)]

            if len(traj_i_pre) < 2 or len(traj_j_pre) < 2:
                continue

            v_i_pre = traj_i_pre[-1, 1:3] - traj_i_pre[0, 1:3]
            v_j_pre = traj_j_pre[-1, 1:3] - traj_j_pre[0, 1:3]

            if np.linalg.norm(v_i_pre) < velocity_threshold and np.linalg.norm(v_j_pre) < velocity_threshold:
                continue

            center_i_pre = np.mean(traj_i_pre[:, 1:3], axis=0)
            center_j_pre = np.mean(traj_j_pre[:, 1:3], axis=0)
            approach_axis = center_j_pre - center_i_pre
            approach_axis_unit = approach_axis / (np.linalg.norm(approach_axis) + 1e-6)
            v_i_proj = np.dot(v_i_pre, approach_axis_unit)
            v_j_proj = np.dot(v_j_pre, approach_axis_unit)
            relative_v = v_j_proj - v_i_proj

            if relative_v <= 0:
                pairs_to_check.append((label_i, label_j, int(min_frame)))

    elapsed_time = time.time() - start_time
    print(f"Time for filtering pairs: {elapsed_time:.2f} sec")
    return pairs_to_check


def classify_collision_events_with_overlap(df, trajectory_pairs, dT=6, overlap_threshold=10):
    start_time = time.time()
    mask_cache = {}
    collision_records = []

    for traj_i, traj_j, f in tqdm(trajectory_pairs, desc="Classifying collisions"):
        # 전후 시간대 trajectory 정보 추출
        pre_i = df[(df['trajectory_label'] == traj_i) & (df['frame_id'] >= f - dT) & (df['frame_id'] <= f)]
        post_i = df[(df['trajectory_label'] == traj_i) & (df['frame_id'] > f) & (df['frame_id'] <= f + dT)]
        pre_j = df[(df['trajectory_label'] == traj_j) & (df['frame_id'] >= f - dT) & (df['frame_id'] <= f)]
        post_j = df[(df['trajectory_label'] == traj_j) & (df['frame_id'] > f) & (df['frame_id'] <= f + dT)]

        if len(pre_i) < 2 or len(post_i) < 2 or len(pre_j) < 2 or len(post_j) < 2:
            continue

        # 마스크 파일 로딩
        mask_paths = df[df['frame_id'] == f]['mask_file'].unique()
        if len(mask_paths) == 0:
            continue

        mask_file = mask_paths[0]
        if not os.path.exists(mask_file):
            continue

        if mask_file not in mask_cache:
            try:
                mask = np.load(mask_file, allow_pickle=True).item()['masks']
                mask_cache[mask_file] = mask
            except:
                continue
        else:
            mask = mask_cache[mask_file]

        # 셀 라벨 추출
        label_i = pre_i[pre_i['frame_id'] == f]['cell_id'].values[0]
        label_j = pre_j[pre_j['frame_id'] == f]['cell_id'].values[0]

        binary_i = mask == label_i
        binary_j = mask == label_j

        boundary_i = binary_dilation(find_boundaries(binary_i), iterations=1)
        boundary_j = binary_dilation(find_boundaries(binary_j), iterations=1)

        if np.sum(boundary_i & boundary_j) < overlap_threshold:
            continue

        # vector
        v_i_pre = pre_i[['centroid_x', 'centroid_y']].values[-1] - pre_i[['centroid_x', 'centroid_y']].values[0]
        v_j_pre = pre_j[['centroid_x', 'centroid_y']].values[-1] - pre_j[['centroid_x', 'centroid_y']].values[0]
        v_i_post = post_i[['centroid_x', 'centroid_y']].values[-1] - post_i[['centroid_x', 'centroid_y']].values[0]
        v_j_post = post_j[['centroid_x', 'centroid_y']].values[-1] - post_j[['centroid_x', 'centroid_y']].values[0]

        # angle
        angle_i = np.arccos(np.clip(np.dot(v_i_pre, v_i_post) / (np.linalg.norm(v_i_pre) * np.linalg.norm(v_i_post) + 1e-6), -1, 1))
        angle_j = np.arccos(np.clip(np.dot(v_j_pre, v_j_post) / (np.linalg.norm(v_j_pre) * np.linalg.norm(v_j_post) + 1e-6), -1, 1))
        angle_between = np.arccos(np.clip(np.dot(v_i_post, v_j_post) / (np.linalg.norm(v_i_post) * np.linalg.norm(v_j_post) + 1e-6), -1, 1))
      
        angle_i_deg = np.degrees(angle_i)
        angle_j_deg = np.degrees(angle_j)
        angle_between_deg = np.degrees(angle_between)

        approach_axis = pre_j[['centroid_x', 'centroid_y']].values[-1] - pre_i[['centroid_x', 'centroid_y']].values[-1]
        approach_axis_unit = approach_axis / (np.linalg.norm(approach_axis) + 1e-6)

        incidence_angle = np.arccos(np.clip(np.dot(v_i_pre, approach_axis_unit) / (np.linalg.norm(v_i_pre) + 1e-6), -1, 1))
        reflection_angle = np.arccos(np.clip(np.dot(v_i_post, approach_axis_unit) / (np.linalg.norm(v_i_post) + 1e-6), -1, 1))
        angle_diff = np.abs(incidence_angle - reflection_angle)

        incidence_angle_deg = np.degrees(incidence_angle)
        reflection_angle_deg = np.degrees(reflection_angle)
        angle_diff_deg = np.degrees(angle_diff)

        # color and labeling
        color_i = pre_i['Color'].mode().values[0] if not pre_i['Color'].mode().empty else 'Unknown'
        color_j = pre_j['Color'].mode().values[0] if not pre_j['Color'].mode().empty else 'Unknown'

        pred_label_i = pre_i['Predicted_Label'].mode().values[0] if 'Predicted_Label' in pre_i and not pre_i['Predicted_Label'].mode().empty else 'Unknown'
        pred_label_j = pre_j['Predicted_Label'].mode().values[0] if 'Predicted_Label' in pre_j and not pre_j['Predicted_Label'].mode().empty else 'Unknown'

        # save the results
        collision_records.append({
            'frame': int(f),
            'cell_i': int(label_i),
            'cell_j': int(label_j),
            'traj_i': int(traj_i),
            'traj_j': int(traj_j),
            'color_i': color_i,
            'color_j': color_j,
            'predicted_label_i': pred_label_i,
            'predicted_label_j': pred_label_j,
            'angle_i_deg': angle_i_deg,
            'angle_j_deg': angle_j_deg,
            'angle_between_deg': angle_between_deg,
            'incidence_angle_deg': incidence_angle_deg,
            'reflection_angle_deg': reflection_angle_deg,
            'angle_diff_deg': angle_diff_deg,
            'traj_i_pre': pre_i[['frame_id', 'centroid_x', 'centroid_y']].to_dict('records'),
            'traj_i_post': post_i[['frame_id', 'centroid_x', 'centroid_y']].to_dict('records'),
            'traj_j_pre': pre_j[['frame_id', 'centroid_x', 'centroid_y']].to_dict('records'),
            'traj_j_post': post_j[['frame_id', 'centroid_x', 'centroid_y']].to_dict('records')
        })

    elapsed = time.time() - start_time
    print(f"Time for classifying collisions: {elapsed:.2f} sec")
    return pd.DataFrame(collision_records)


def run_collision_analysis(df, dT=6, proximity_threshold=50):
    trajectory_pairs = filter_approaching_trajectory_pairs(df, proximity_threshold=proximity_threshold, dT=dT)
    collision_df = classify_collision_events_with_overlap(df, trajectory_pairs, dT=dT)
    return collision_df


def run_batch_collision_analysis(
    csv_folder,
    output_folder,
    dT=6,
    proximity_threshold=50,
    velocity_threshold=5,
    overlap_threshold=2
):
    os.makedirs(output_folder, exist_ok=True)
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

    for csv_path in csv_files:
        print(f"\n Processing: {csv_path}")
        try:
            df = pd.read_csv(csv_path)

            # Candidate for collision
            trajectory_pairs = filter_approaching_trajectory_pairs(
                df,
                proximity_threshold=proximity_threshold,
                dT=dT,
                velocity_threshold=velocity_threshold
            )

            # Categorizing
            collision_df = classify_collision_events_with_overlap(
                df,
                trajectory_pairs,
                dT=dT,
                overlap_threshold=overlap_threshold
            )

            # save
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            save_path = os.path.join(output_folder, f"{base_name}_collision.pkl")
            collision_df.to_pickle(save_path)
            print(f"Saved: {save_path}")

        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

# Run code
csv_folder = "/Users/jeonghyeontae/data_mesen_test/afterlabel"  # folder with csv files
output_folder = "/Users/jeonghyeontae/collision_results_0724/mesen_withlabel"  # output folder

run_batch_collision_analysis(
    csv_folder=csv_folder,
    output_folder=output_folder,
    dT=6,
    proximity_threshold=50,
    velocity_threshold=2.5,
    overlap_threshold=3
)
