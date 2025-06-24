import os
import pandas as pd
import numpy as np
from collections import defaultdict
import json

# --- 配置路径 ---
annotations_dir = "/data/zwj/SPR/OphNet/data/phase_annotations" # 示例路径，请替换为您的实际路径
transition_data_path = "/data/zwj/SPR/OphNet/data/transition_data" # 新增：保存路径
os.makedirs(transition_data_path, exist_ok=True) # 确保保存目录存在

# --- 阶段ID映射 ---
original_phase_info = {
    17: "Non-functional Segment",
    26: "Step Interval",
    0: "Anterior Chamber Gas Injection", # 注意：与Viscoelastic Injection重复
    1: "Anterior Chamber Injection/Washing",
    2: "Anterior Chamber Washing",
    3: "Anterior Vitrectomy",
    4: "Capsular Membrane Staining",
    5: "Capsulorhexis",
    6: "Conjunctival Incision Creation",
    7: "Corneal Incision Creation",
    8: "Corneal Measurement",
    9: "Corneal-Scleral Tunnel Creation",
    10: "Cortex Aspiration", # 
    11: "Goniotomy", # 
    13: "Incision Closure", # 
    14: "Intraocular Lens Implantation", # 
    15: "Intraoperative Gonioscopy Application", # 
    16: "Iris Prolapse Management",
    18: "Nuclear Management (for cataract surgery)",
    12: "Hydrodissection/Hydrodelineation (PHACO, ECCE)",
    19: "Ocular Surface Irrigation",
    21: "Peripheral Iridectomy",
    22: "Placement of Bandage Contact Lens",
    23: "Placement of Eyelid Speculum",
    24: "Pupil Dilation",
    25: "Scleral Hemostasis",
    27: "Subconjunctival Drug Injection", # 
    28: "Surgical Marking",
    29: "Suspension Suture",
    30: "Swab Wiping",
    31: "Use of Iris Expander",
    32: "Viscoelastic Application on Cornea",
    33: "Viscoelastic Aspiration",
    34: "Viscoelastic Injection", # 假设新的唯一ID
    20: "Others"
}

# 从注释文件中提取所有出现过的阶段ID，并构建映射
unique_phase_ids_from_annotations = set()
for filename in os.listdir(annotations_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(annotations_dir, filename)
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        try:
                            phase_id = int(parts[1])
                            unique_phase_ids_from_annotations.add(phase_id)
                        except ValueError:
                            print(f"警告: 文件 {filename} 中有无效的阶段ID '{parts[1]}'。")
        except Exception as e:
            print(f"读取文件 {filepath} 时发生错误: {e}")

# 将从注释文件中提取的唯一ID排序，并创建索引映射
sorted_phase_ids = sorted(list(unique_phase_ids_from_annotations))
if not sorted_phase_ids:
    print("错误: 未在注释文件中找到任何有效的阶段ID。请检查文件格式和内容。")
    exit()

id_to_index = {pid: i for i, pid in enumerate(sorted_phase_ids)}
index_to_id = {i: pid for i, pid in enumerate(sorted_phase_ids)}
num_phases = 35

print(f"从注释文件中找到 {num_phases} 个唯一阶段ID。")
print("阶段ID到矩阵索引的映射:")
for pid in sorted_phase_ids:
    # 尝试从原始映射中获取阶段名称，如果不存在则显示“未知”
    phase_name = original_phase_info.get(pid, "未知阶段名称")
    print(f"  ID {pid} -> Index {id_to_index[pid]} ({phase_name})")

# 初始化转移计数矩阵
# transition_counts[from_phase_index][to_phase_index]
transition_counts = np.zeros((num_phases, num_phases), dtype=int)

# --- 遍历注释文件，统计转移 ---
print(f"\n开始统计 {annotations_dir} 中的阶段转移...")
annotation_files_processed = 0

for filename in os.listdir(annotations_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(annotations_dir, filename)
        try:
            with open(filepath, 'r') as f:
                current_sequence = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        try:
                            phase_id = int(parts[1])
                            current_sequence.append(phase_id)
                        except ValueError:
                            # 忽略无效行
                            continue
                    else:
                        # 忽略格式不正确的行
                        continue

                # 处理当前文件的阶段序列
                if len(current_sequence) > 1:
                    for i in range(len(current_sequence) - 1):
                        from_phase_id = current_sequence[i]
                        to_phase_id = current_sequence[i+1]

                        # 确保ID在我们的映射中
                        if from_phase_id in id_to_index and to_phase_id in id_to_index:
                            from_idx = id_to_index[from_phase_id]
                            to_idx = id_to_index[to_phase_id]
                            transition_counts[from_idx, to_idx] += 1
                annotation_files_processed += 1
        except Exception as e:
            print(f"处理文件 {filepath} 时发生错误: {e}")

print(f"\n已处理 {annotation_files_processed} 个注释文件。")

# --- 生成转移矩阵 (布尔值：是否允许转移) ---
# 这里我们创建一个布尔矩阵，表示是否有至少一次转移发生
# 如果某个转移计数大于0，则表示该转移是允许的
transition_matrix = transition_counts > 0

print("\n--- 生成的阶段转移矩阵 (布尔值) ---")
# 打印矩阵的头部（To Phase ID）
header = ["From/To"] + [str(pid) for pid in sorted_phase_ids]
print("\t".join(header))

# 打印矩阵内容
for i in range(num_phases):
    row_label = f"{sorted_phase_ids[i]} ({original_phase_info.get(sorted_phase_ids[i], '未知')})"
    row_values = [str(int(val)) for val in transition_matrix[i, :]]
    print(f"{row_label}\t{' '.join(row_values)}") # 使用空格分隔布尔值，方便查看


# --- 额外：生成转移概率矩阵 (可选) ---
# 这个矩阵可以用于更复杂的后处理，比如HMM或Viterbi算法
transition_probabilities = np.zeros_like(transition_counts, dtype=float)

for i in range(num_phases):
    row_sum = np.sum(transition_counts[i, :])
    if row_sum > 0:
        transition_probabilities[i, :] = transition_counts[i, :] / row_sum

print("\n--- 生成的阶段转移概率矩阵 (可选) ---")
# 打印矩阵的头部（To Phase ID）
print("\t".join(header))

# 打印矩阵内容
for i in range(num_phases):
    row_label = f"{sorted_phase_ids[i]} ({original_phase_info.get(sorted_phase_ids[i], '未知')})"
    row_values = [f"{val:.2f}" for val in transition_probabilities[i, :]]
    print(f"{row_label}\t{' '.join(row_values)}")


# --- 保存所有生成的数据 ---
np.save(os.path.join(transition_data_path, 'transition_matrix.npy'), transition_matrix)
np.save(os.path.join(transition_data_path, 'transition_probabilities.npy'), transition_probabilities) # 新增：保存概率矩阵
with open(os.path.join(transition_data_path, 'id_to_index.json'), 'w') as f:
    json.dump(id_to_index, f)
with open(os.path.join(transition_data_path, 'index_to_id.json'), 'w') as f:
    # `json.dump` 默认将字典键转换为字符串，所以这里没问题
    json.dump(index_to_id, f)

print(f"\n已将布尔转移矩阵保存到: {os.path.join(transition_data_path, 'transition_matrix.npy')}")
print(f"已将转移概率矩阵保存到: {os.path.join(transition_data_path, 'transition_probabilities.npy')}") # 打印保存路径
print(f"已将 ID 到索引映射保存到: {os.path.join(transition_data_path, 'id_to_index.json')}")
print(f"已将索引到 ID 映射保存到: {os.path.join(transition_data_path, 'index_to_id.json')}")


print("\n转移矩阵生成完成！")
print("您现在可以将此 `transition_matrix` (布尔型) 或 `transition_probabilities` 用于您之前的 `apply_transition_rules` 函数。")
print("记得传递 `id_to_index` 和 `index_to_id` 映射。")
