import os
import pandas as pd
import numpy as np
import json
from collections import Counter
from hmmlearn import hmm

input_base_dir = "./output/challenge_predictions_4080/phase"  
output_base_dir = "./output/challenge_sum_probabilities_predictions"

# ==============================================================================
# TODO: >>>>>>>>> 在这里定义模型权重 <<<<<<<<<
# ==============================================================================
model_weights_raw = {
    "20250419-1651_M_75_cuhk2714Split_SCOPE_convnextv2_lr0.0001_bs1_seq256_frozen": 0.738,
    "20250501-2326_CMR_2_trainingSplit_SCOPE_convnextv2_lr0.0001_bs1_seq256_frozen": 0.733,
    "20250502-0012_covnetv2_Resnet_trainingSplit_SCOPE_convnextv2_lr0.0001_bs1_seq256_frozen": 0.735,
    "20250510-1519_IUUU_trainingSplit_SCOPE_convnextv2_lr0.0001_bs1_seq256_frozen": 0.726,
    "20250510-1940_O2M_trainingSplit_SCOPE_convnextv2_lr0.0001_bs1_seq256_frozen": 0.736,
    "20250512-0504_IUUUU_trainingSplit_SCOPE_convnextv2_lr0.0001_bs1_seq256_frozen": 0.711,
    "20250513-0657_swin_transformer_trainingSplit_SCOPE_swintransformer_lr0.0001_bs1_seq256_frozen": 0.697,
    "20250515-2336_ConvNext_trainingSplit_SCOPE_convnext_lr0.0001_bs1_seq256_frozen": 0.736,
    "20250516-0145_SwinV2_trainingSplit_SCOPE_swintransformerv2_lr0.0001_bs1_seq256_frozen": 0.744,
    "20250517-0702_N_token_200_trainingSplit_SCOPE_swintransformer_lr0.0001_bs1_seq256_frozen": 0.746,
    "20250518-1608_N_swin_moe_trainingSplit_SCOPE_swin_moe_lr0.0001_bs1_seq256_frozen": 0.723,
    "20250522-1406_ConvNextv2_iuu_trainingSplit_SCOPE_convnextv2_lr0.0001_bs1_seq256_frozen": 0.728,
    "20250522-1409_ConvNextv2_iuuuu_trainingSplit_SCOPE_convnextv2_lr0.0001_bs1_seq256_frozen": 0.732,
    "20250523-1340_Swinv2_iuuu_trainingSplit_SCOPE_swintransformerv2_lr0.0001_bs1_seq256_frozen": 0.740,
    "20250525-0855_Swinv2_iuuuu_trainingSplit_SCOPE_swintransformerv2_lr0.0001_bs1_seq256_frozen": 0.734,
    "20250526-1434_vHeat_trainingSplit_SCOPE_vheat_lr0.0001_bs1_seq256_frozen": 0.741,
    "20250528-2045_vheat_swin_trainingSplit_SCOPE_vheat_lr0.0001_bs1_seq256_frozen": 0.743,
    "20250530-1220_vheat_swinv2_trainingSplit_SCOPE_vheat_lr0.0001_bs1_seq256_frozen": 0.751,
    "20250530-1234_vheat_no_tm_trainingSplit_SCOPE_vheat_lr0.0001_bs1_seq256_frozen": 0.727,
    "20250602-0456_Vheat_only_temporal_trainingSplit_SCOPE_vheat_2step": 0.758,
    "20250602-0553_Vheat_NO_Tem_fM2_trainingSplit_SCOPE_vheat_2step": 0.766,
        "20250606-0632_Swinv2_No_TM_CMR_trainingSplit_SCOPE_swintransformerv2_lr0.0001_bs1_seq256_frozen": 0.740,
    "20250606-0634_Vheat_No_TM_CMR_trainingSplit_SCOPE_vheat_lr0.0001_bs1_seq256_frozen": 0.743,
    "20250606-0641_Vheat_only_Tem_iuu_trainingSplit_SCOPE_vheat_2step": 0.767,
    "20250606-0641_Vheat_only_Tem_iuuu_trainingSplit_SCOPE_vheat_2step": 0.767,
    "20250606-1338_Vheat_only_Tem_iuuuuu_trainingSplit_SCOPE_vheat_2step": 0.761,
    "20250607-0547_vHeat_freeze01_trainingSplit_SCOPE_vheat123_lr0.0001_bs1_seq256_frozen": 0.744,
    "20250611-0721_Vheat_NO_Tem_bin8_trainingSplit_SCOPE_vheat_2step": 0.765,
    "20250611-0722_Vheat_NO_Tem_Bin12_trainingSplit_SCOPE_vheat_2step": 0.760,
    "20250611-0723_Vheat_NO_Tem_Bin20_trainingSplit_SCOPE_vheat_2step": 0.755,
    "20250611-1601_Vheat_NO_Tem_mtoken100_trainingSplit_SCOPE_vheat_2step": 0.770,
    "20250611-0447_vheat_Bins_20_trainingSplit_SCOPE_vheat_lr0.0001_bs1_seq256_frozen": 0.741,
    "20250611-1601_Vheat_NO_Tem_Mtoken50_trainingSplit_SCOPE_vheat_2step": 0.765,
    "20250612-0853_Vheat_NO_Tem_mtoken200_trainingSplit_SCOPE_vheat_2step": 0.766,
    "20250613-1011_Swinv2_only_Tem_Bin8_trainingSplit_SCOPE_swintransformerv2_2step": 0.768,
    "20250614-1036_Vheat_NO_Tem_ahead16_layer8_trainingSplit_SCOPE_vheat_2step": 0.766,
}
# ==============================================================================


# --- Post-processing Functions ---

def temporal_smoothing(predictions, window_size):
    """Applies a majority vote temporal smoothing to the predictions."""
    smoothed_predictions = []
    half_window = window_size // 2
    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        # Get the predicted phases within the window, excluding NaN values
        window_phases = [p for p in predictions[start:end] if not pd.isna(p)]
        if window_phases:
            # Find the most common phase in the window
            most_common = Counter(window_phases).most_common(1)
            smoothed_predictions.append(most_common[0][0])
        else:
            smoothed_predictions.append(np.nan) # If no valid predictions in window
    return smoothed_predictions

def apply_transition_rules(predictions, transition_matrix, id_to_index, index_to_id, raw_probabilities=None):
    """
    Applies transition rules to correct illogical phase sequences.
    
    Args:
        predictions (list): A list of predicted phase IDs (integers or np.nan).
        transition_matrix (np.ndarray): A boolean matrix where transition_matrix[i, j] is True
                                        if a transition from phase_id index i to phase_id index j is allowed.
        id_to_index (dict): Maps phase ID to its index in the transition matrix.
        index_to_id (dict): Maps index in the transition matrix back to phase ID.
        raw_probabilities (list of np.ndarray): A list where each element is an array of
                                                summed weighted probabilities for each phase
                                                at that frame. Used for intelligent correction.
    Returns:
        list: Corrected phase IDs.
    """
    corrected_predictions = list(predictions)

    for i in range(len(predictions) - 1):
        current_pred_id = corrected_predictions[i] # Use corrected_predictions for current frame
        next_pred_id = corrected_predictions[i+1] # Use corrected_predictions for next frame (might have been smoothed)

        # Skip NaN values
        if pd.isna(current_pred_id) or pd.isna(next_pred_id):
            continue

        # Convert phase IDs to matrix indices
        current_idx = id_to_index.get(int(current_pred_id))
        next_idx = id_to_index.get(int(next_pred_id))

        if current_idx is None or next_idx is None:
            # If ID not in mapping, likely an unknown phase or data anomaly.
            # Skip rule check for these specific frames.
            continue

        # Check if the transition is allowed according to the transition_matrix
        if not transition_matrix[current_idx, next_idx]:
            # If an illegal transition is found, attempt to correct next_pred_id
            print(f"  Illegal transition detected: From ID {int(current_pred_id)} to ID {int(next_pred_id)} at frame {i+1}.")

            # Correction strategy: Prioritize the most probable *allowed* phase
            if raw_probabilities is not None and i + 1 < len(raw_probabilities):
                # Get the summed weighted probability distribution for the next frame
                next_frame_probs = raw_probabilities[i+1]
                
                if next_frame_probs is None or np.all(np.isnan(next_frame_probs)):
                    # If raw probabilities for this frame are invalid, fall back to simple correction
                    corrected_predictions[i+1] = current_pred_id
                    print(f"    Raw probabilities for frame {i+1} are invalid; reverted to current phase ID {int(current_pred_id)}.")
                    continue

                # Find all allowed target phase indices from the current phase index
                allowed_next_indices = np.where(transition_matrix[current_idx, :])[0]
                
                if len(allowed_next_indices) == 0:
                    # If no transitions are allowed from the current phase (should be rare if matrix is good)
                    corrected_predictions[i+1] = current_pred_id # Revert to current phase
                    print(f"    No allowed subsequent phases from ID {int(current_pred_id)}; reverted frame {i+1} to ID {int(current_pred_id)}.")
                    continue

                # Filter probabilities to only include allowed transitions
                # Create a mask for allowed phases in the full probability array
                allowed_probs_mask = np.zeros_like(next_frame_probs, dtype=bool)
                allowed_probs_mask[allowed_next_indices] = True
                
                # Apply mask to get probabilities only for allowed phases
                filtered_probs = next_frame_probs * allowed_probs_mask
                
                # If all allowed probabilities are 0 (e.g., due to very low confidence in all options),
                # or if the predicted phase itself was not allowed, revert to current phase.
                # Find the best allowed phase by taking argmax on the filtered probabilities
                # Need to handle case where filtered_probs might be all zeros or negative if not careful with weights
                if np.sum(filtered_probs) > 0: # Check if there's any non-zero probability among allowed
                    new_next_idx = np.argmax(filtered_probs) # This will give the index of the highest allowed probability
                    corrected_predictions[i+1] = index_to_id[new_next_idx]
                    print(f"    Corrected frame {i+1} from ID {int(next_pred_id)} to allowed ID {int(corrected_predictions[i+1])} (based on probabilities).")
                else:
                    corrected_predictions[i+1] = current_pred_id # Fallback: revert to current phase
                    print(f"    No sufficiently probable allowed phases found; reverted frame {i+1} to ID {int(current_pred_id)}.")
            else:
                # If raw probability information is not available, simply revert to the current phase
                corrected_predictions[i+1] = current_pred_id
                print(f"    No raw probability info; reverted frame {i+1} to ID {int(current_pred_id)}.")
    return corrected_predictions

# --- 归一化权重 ---
# 这一步确保所有有效权重的总和为1。
# 如果没有提供任何权重，或者所有权重为0，则会给出警告并使用平均权重。
total_raw_weight = sum(model_weights_raw.values())
model_weights = {}

if total_raw_weight > 0:
    for model_name, weight in model_weights_raw.items():
        model_weights[model_name] = weight / total_raw_weight
    print("模型权重已归一化。")
else:
    print("警告: 原始模型权重总和为0或未定义任何权重。将使用默认的平均权重。")
    # 如果没有有效权重，我们会回退到默认的平均权重（每个模型权重相同）
    # 这一步会在后续模型文件夹遍历时动态处理，所以这里不直接计算平均值
    # 但是需要在获取 current_model_weight 时进行判断

# 确保输出文件夹存在，如果不存在则创建
os.makedirs(output_base_dir, exist_ok=True)

transition_data_path = "/data/zwj/SPR/OphNet/data/transition_data"
os.makedirs(transition_data_path, exist_ok=True) 
# --- Load Transition Data ---
transition_probabilities = np.load(os.path.join(transition_data_path, 'transition_probabilities.npy'))
transition_matrix = np.load(os.path.join(transition_data_path, 'transition_matrix.npy'))
with open(os.path.join(transition_data_path, 'id_to_index.json'), 'r') as f:
    id_to_index = {int(k): v for k, v in json.load(f).items()} # Keys are string, convert to int
with open(os.path.join(transition_data_path, 'index_to_id.json'), 'r') as f:
    index_to_id = {int(k): v for k, v in json.load(f).items()}
    
print(f"\nSuccessfully loaded transition matrix and phase mappings from {transition_data_path}.")
print(f"Transition matrix shape: {transition_matrix.shape}")
print(f"Number of phases in mapping: {len(id_to_index)}")

# Get the number of hidden states (phases) from the loaded data
n_components = len(id_to_index) # This is the number of unique phases

# --- HMM Model Setup ---
# Initialize the HMM model
# We use GaussianHMM because our observations (summed_weighted_probabilities_per_phase) are continuous.
# The 'full' covariance type allows for correlation between different phase probabilities.
# 'diag' (diagonal) is simpler and assumes independence. 'spherical' is even simpler.
# 'full' is generally more expressive but requires more data. Let's start with 'diag' for robustness.
# model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=200)

# Set initial probabilities (startprob_)
# A simple approach: assume each phase has an equal chance to start, or estimate from annotations.
# For a more robust estimate, you'd calculate the frequency of each phase appearing at frame 0 across all videos.
# For now, let's assume uniform or based on initial frequencies if available from annotations.
# Let's just use uniform for now if no specific startprob_ is pre-calculated.
model.startprob_ = np.full(n_components, 1.0 / n_components) 
# OR: If you have startprob_ calculated from annotations:
# model.startprob_ = np.load(os.path.join(transition_data_path, 'start_probabilities.npy')) 
# (You'd need to modify your transition matrix generation script to save start_probabilities too)

# Set transition probabilities (transmat_)
# Ensure the loaded transition_probabilities has compatible dimensions
if transition_probabilities.shape != (n_components, n_components):
    print(f"Warning: Loaded transition_probabilities shape {transition_probabilities.shape} does not match expected {n_components, n_components}. Recalculating uniform.")
    model.transmat_ = np.full((n_components, n_components), 1.0 / n_components) # Fallback to uniform
else:
    model.transmat_ = transition_probabilities # Use the pre-computed transition probabilities

# Set observation means (means_) and covariances (covars_) for GaussianHMM
# This is the tricky part without explicit training data for HMM.
# We assume that when the true phase is 'k' (index k), the ideal observation
# (summed_weighted_probabilities_per_phase) would be a vector where the k-th element is high and others are low.
# A simple approximation for means: an identity matrix-like structure.
# For phase k (index k), the mean observation is a one-hot vector with 1 at index k.
# This assumes that the *predicted* probabilities perfectly reflect the true phase when averaged.
model.means_ = np.eye(n_components) # Identity matrix for means (ideal observation is one-hot)

# Set observation covariances (covars_)
n_features = n_components # 你的观测维度是阶段的数量

# 这通常是 hmmlearn 在其内部验证比预期更严格时的常见解决方法。
new_covars = np.zeros((n_components, n_features, n_features))
for i in range(n_components):
    np.fill_diagonal(new_covars[i], 0.05)

model.covars_ = new_covars

# 获取所有模型文件夹的名称
model_folders = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
if not model_folders:
    print(f"错误: 在 {input_base_dir} 中未找到任何模型子文件夹。请检查路径。")
    exit()

# 获取第一个模型文件夹中的所有txt文件名，假设所有模型文件夹中的文件名是相同的
first_model_folder_path = os.path.join(input_base_dir, model_folders[0], "pred_pre_ave2")
txt_filenames = []
if os.path.exists(first_model_folder_path):
    txt_filenames = [f for f in os.listdir(first_model_folder_path) if f.endswith("-phase.txt")]
    if not txt_filenames:
        print(f"警告: 在 {first_model_folder_path} 中未找到任何 '-phase.txt' 文件。")
        exit()
else:
    print(f"错误: 路径 {first_model_folder_path} 不存在。请检查您的输入路径。")
    exit()

print(f"找到 {len(model_folders)} 个模型和 {len(txt_filenames)} 个预测文件待处理。")

for filename in txt_filenames:
    print(f"\nProcessing {filename}...")
    frame_numbers = []
    num_phases_predicted_from_file = 0 # Number of phase columns in model output
    
    all_frame_probabilities = [] 

    # Read the first model's predictions to get frame numbers and phase column info
    first_model_filepath = os.path.join(input_base_dir, model_folders[0], "pred_pre_ave2", filename)
    try:
        first_model_df = pd.read_csv(first_model_filepath, sep='\t')
        if 'Frame' not in first_model_df.columns:
            print(f"Warning: 'Frame' column not found in {first_model_filepath}, skipping.")
            continue
        frame_numbers = first_model_df['Frame'].tolist()
        phase_cols = [col for col in first_model_df.columns if col.startswith("Phase_")]
        if not phase_cols:
            print(f"Warning: 'Phase_' columns not found in {first_model_filepath}, skipping.")
            continue
        num_phases_predicted_from_file = len(phase_cols)
        
        # Verify that the number of phases from the file matches the HMM's n_components
        if num_phases_predicted_from_file != n_components:
            print(f"ERROR: Number of phases in {filename} ({num_phases_predicted_from_file}) does not match HMM's n_components ({n_components}). Skipping this file.")
            continue

        for _ in range(len(frame_numbers)):
            all_frame_probabilities.append([])

    except FileNotFoundError:
        print(f"Error: First model file {first_model_filepath} not found, skipping {filename}.")
        continue
    except pd.errors.EmptyDataError:
        print(f"Warning: {first_model_filepath} is empty, skipping {filename}.")
        continue
    except pd.errors.ParserError as e:
        print(f"Error: Could not parse {first_model_filepath}: {e}, skipping {filename}.")
        continue
    except Exception as e:
        print(f"Unexpected error reading {first_model_filepath}: {e}, skipping {filename}.")
        continue

    # Collect probability distributions from all models and apply weights
    successful_models_count = 0
    temp_summed_weighted_probabilities_for_HMM = [None] * len(frame_numbers) # Used for HMM observations

    for model_folder in model_folders:
        filepath = os.path.join(input_base_dir, model_folder, "pred_pre_ave2", filename)
        
        current_model_weight = model_weights.get(model_folder, 1.0 / len(model_folders)) 
        
        if current_model_weight == 0:
            continue

        try:
            df = pd.read_csv(filepath, sep='\t')
            if not df['Frame'].equals(pd.Series(frame_numbers)):
                print(f"Warning: Frame numbers in {filepath} do not match the first model. Skipping this model for {filename}.")
                continue

            probability_cols = [col for col in df.columns if col.startswith("Phase_")]
            if not probability_cols or len(probability_cols) != num_phases_predicted_from_file:
                print(f"Warning: 'Phase_' columns in {filepath} are incomplete or missing. Skipping this model for {filename}.")
                continue
            
            for i, row in df.iterrows():
                if i < len(all_frame_probabilities):
                    weighted_probs = row[probability_cols].values.astype(float) * current_model_weight
                    all_frame_probabilities[i].append(weighted_probs)
                else:
                    print(f"Warning: Number of rows in {filepath} exceeds expectations, possibly corrupted. Stopping processing this model file.")
                    break
            successful_models_count += 1

        except FileNotFoundError:
            print(f"Warning: File {filepath} not found, skipping this model for {filename}.")
            continue
        except pd.errors.EmptyDataError:
            print(f"Warning: {filepath} is empty, skipping this model for {filename}.")
            continue
        except pd.errors.ParserError as e:
            print(f"Error: Could not parse {filepath}: {e}. Skipping this model for {filename}.")
            continue
        except Exception as e:
            print(f"Unexpected error reading {filepath}: {e}. Skipping this model for {filename}.")
            continue
    
    if successful_models_count == 0:
        print(f"Warning: No model prediction data successfully loaded for {filename}. Skipping this file.")
        continue

    # Calculate summed weighted probabilities per phase for HMM
    # This will be our observation sequence for the HMM
    for i, frame_probs_list_for_current_frame in enumerate(all_frame_probabilities):
        if not frame_probs_list_for_current_frame:
            # If no data for this frame, fill with NaNs or zeros. HMM might struggle with NaNs.
            # It's better to ensure a valid observation vector for HMM.
            # A common strategy is to fill with uniform probabilities or a default 'unknown' observation.
            # For GaussianHMM, it needs numerical values.
            # Let's fill with a small uniform probability vector, summing to 1.0/n_components
            temp_summed_weighted_probabilities_for_HMM[i] = np.full(num_phases_predicted_from_file, 1.0 / num_phases_predicted_from_file)
            print(f"Warning: Frame {frame_numbers[i]} has no (weighted) probability data collected, using uniform probabilities for HMM.")
            continue

        stacked_weighted_probs_for_current_frame = np.vstack(frame_probs_list_for_current_frame)
        summed_weighted_probabilities_per_phase = np.sum(stacked_weighted_probs_for_current_frame, axis=0)
        
        # Ensure probabilities sum to 1 (or close to 1) for better HMM behavior if not already
        # if np.sum(summed_weighted_probabilities_per_phase) > 0:
        #     summed_weighted_probabilities_per_phase /= np.sum(summed_weighted_probabilities_per_phase)
        # else:
        #     # Fallback for empty or zero-sum probabilities
        #     summed_weighted_probabilities_per_phase = np.full(num_phases_predicted_from_file, 1.0 / num_phases_predicted_from_file)

        temp_summed_weighted_probabilities_for_HMM[i] = summed_weighted_probabilities_per_phase
    
    # Convert list of arrays to a single 2D numpy array for HMM input
    observations = np.array(temp_summed_weighted_probabilities_for_HMM)

    # --- Apply HMM Viterbi Decoding ---
    print(f"  Applying HMM Viterbi decoding...")
    try:
        # HMM requires observations to be float64
        logprob, hidden_states_indices = model.decode(observations.astype(np.float64), algorithm="viterbi")
        print(f"  HMM decoding successful. Log probability: {logprob:.2f}")

        # Convert hidden state indices back to original phase IDs
        hmm_predictions = [index_to_id[idx] for idx in hidden_states_indices]
    except Exception as e:
        print(f"  ERROR during HMM decoding for {filename}: {e}. Falling back to simple argmax predictions.")
        # Fallback: if HMM fails, revert to raw argmax predictions after smoothing
        final_predictions_raw = []
        for probs_array in temp_summed_weighted_probabilities_for_HMM:
            if probs_array is not None and not np.all(np.isnan(probs_array)):
                predicted_phase_idx = np.argmax(probs_array)
                if predicted_phase_idx < len(index_to_id):
                    final_predictions_raw.append(index_to_id[predicted_phase_idx])
                else:
                    final_predictions_raw.append(np.nan)
            else:
                final_predictions_raw.append(np.nan)
        hmm_predictions = temporal_smoothing(final_predictions_raw, window_size=7)
        hmm_predictions = apply_transition_rules(hmm_predictions, transition_matrix, id_to_index, index_to_id, temp_summed_weighted_probabilities_for_HMM)


    # --- Save Results ---
    # The HMM predictions are the final corrected sequence
    output_df = pd.DataFrame({'Frame': frame_numbers, 'Phase': hmm_predictions})
    output_filepath = os.path.join(output_base_dir, filename)
    output_df.to_csv(output_filepath, sep='\t', index=False)
    print(f"Successfully processed and saved (HMM post-processed): {output_filepath}")

print("\n--- Weighted Averaging and HMM Post-processing Complete! ---")