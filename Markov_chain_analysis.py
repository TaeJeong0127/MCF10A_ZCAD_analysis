import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['font.family'] = 'Arial'  # 사용 가능한 다른 폰트: 'Helvetica', 'DejaVu Sans', 'Times New Roman' 등
matplotlib.rcParams['font.size'] = 18         # 기본 폰트 크기
matplotlib.rcParams['axes.titlesize'] = 20    # 제목 폰트 크기
matplotlib.rcParams['axes.labelsize'] = 16   # 축 라벨 폰트 크기
matplotlib.rcParams['xtick.labelsize'] = 14   # X축 눈금 폰트 크기
matplotlib.rcParams['ytick.labelsize'] = 14   # Y축 눈금 폰트 크기
matplotlib.rcParams['legend.fontsize'] = 12   # 범례 폰트 크기

folder_path = "/Users/jeonghyeontae/data_for_mesen/color_change_output"

all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
df_list = []
trajectory_offset = 0

for file in all_files:
    df = pd.read_csv(os.path.join(folder_path, file))
    if 'trajectory_label' in df.columns:
        df['trajectory_label'] += trajectory_offset
        trajectory_offset = df['trajectory_label'].max() + 1
    df_list.append(df)

df_merged = pd.concat(df_list, ignore_index=True)
df_filtered = df_merged[df_merged['trajectory_length'] >= 12].copy()

unique_colors = sorted(df_filtered['Color'].dropna().unique())
color_to_idx = {color: i for i, color in enumerate(unique_colors)}
num_colors = len(unique_colors)
count_matrix = np.zeros((num_colors, num_colors))

for traj_label, group in df_filtered.groupby('trajectory_label'):
    group = group.sort_values(by='frame_id')
    if len(group) < 3:
        continue

    initial_color = group.iloc[:3]['Color'].mode()[0]
    stable_color = initial_color
    consecutive_count = 0
    prev_color = initial_color
    transitioned = False

    for _, row in group.iterrows():
        current_color = row['Color']
        if current_color == prev_color:
            consecutive_count += 1
        else:
            consecutive_count = 1

        if consecutive_count >= 3 and current_color != stable_color:
            from_idx = color_to_idx[stable_color]
            to_idx = color_to_idx[current_color]
            count_matrix[from_idx, to_idx] += 1
            transitioned = True
            break  # 첫 전이만 기록

        prev_color = current_color

    if not transitioned:
        # 색이 바뀌지 않은 경우: 자기 자신으로의 전이로 기록
        idx = color_to_idx[initial_color]
        count_matrix[idx, idx] += 1

row_sums = count_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
prob_matrix = count_matrix / row_sums

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(count_matrix, annot=True, fmt=".0f", cmap="Reds",
            xticklabels=unique_colors, yticklabels=unique_colors, ax=axes[0])
axes[0].set_title("Trajectory-wise Transition Count")
axes[0].set_xlabel("Next Color")
axes[0].set_ylabel("Initial Color")

sns.heatmap(prob_matrix, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=unique_colors, yticklabels=unique_colors, ax=axes[1])
axes[1].set_title("Trajectory-wise Transition Probability")
axes[1].set_xlabel("Next Color")
axes[1].set_ylabel("Initial Color")

plt.tight_layout()
plt.show()

pd.DataFrame(count_matrix, index=unique_colors, columns=unique_colors)\
  .to_csv(os.path.join(folder_path, "trajectorywise_transition_count_with_self.csv"))
pd.DataFrame(prob_matrix, index=unique_colors, columns=unique_colors)\
  .to_csv(os.path.join(folder_path, "trajectorywise_transition_probability_with_self.csv"))
