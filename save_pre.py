import pandas as pd
import os
import re

def fill_predictions_to_csv(prediction_dir, csv_file):
    """
    将预测的手术阶段结果填充到 APTOS_val2.csv 文件中。

    Args:
        prediction_dir (str): 包含预测结果 .txt 文件的文件夹路径。
        csv_file (str): APTOS_val2.csv 文件的路径。
    """
    try:
        df_csv = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"错误：找不到 CSV 文件 {csv_file}")
        return

    df_csv['Phase_predict'] = None  # 创建一个新的列来存储预测的阶段

    prediction_files = [f for f in os.listdir(prediction_dir) if f.endswith('-phase.txt')]
    for pred_file in prediction_files:
        video_name_match = re.match(r'videocase_(\d+)-phase\.txt', pred_file)
        if video_name_match:
            video_case = f"case_{int(video_name_match.group(1))}"
            pred_file_path = os.path.join(prediction_dir, pred_file)

            try:
                df_pred = pd.read_csv(pred_file_path, sep='\t')
                if 'Phase' in df_pred.columns and 'Phase' in df_pred.columns:
                    predicted_phases = df_pred['Phase'].tolist()
                    csv_case_rows = df_csv[df_csv['Video_name'] == video_case].index.tolist()
                    num_csv_frames = len(csv_case_rows)
                    num_pred_frames = len(predicted_phases)

                    if num_pred_frames >= num_csv_frames:
                        # 预测帧数多于或等于 CSV 帧数，取前 num_csv_frames 个
                        phases_to_fill = predicted_phases[:num_csv_frames]
                        df_csv.loc[csv_case_rows, 'Phase_predict'] = phases_to_fill
                    else:
                        # 预测帧数少于 CSV 帧数，用最后一个预测阶段补齐
                        phases_to_fill = predicted_phases + [predicted_phases[-1]] * (num_csv_frames - num_pred_frames)
                        df_csv.loc[csv_case_rows, 'Phase_predict'] = phases_to_fill
                else:
                    print(f"警告：预测文件 {pred_file} 缺少 'Frame' 或 'Phase' 列。")

            except FileNotFoundError:
                print(f"警告：找不到预测文件 {pred_file_path}")
            except Exception as e:
                print(f"读取预测文件 {pred_file_path} 时发生错误：{e}")
        else:
            print(f"警告：预测文件名 {pred_file} 格式不正确。")

    try:
        df_csv.to_csv('APTOS_val2_with_predictions.csv', index=False)
        print("预测结果已成功填充到 APTOS_val2_with_predictions.csv")
    except Exception as e:
        print(f"保存结果 CSV 文件时发生错误：{e}")

if __name__ == "__main__":
    prediction_directory = '/data/zwj/SPR/OphNet/output/challenge_sum_probabilities_predictions'  # 替换为你的预测结果文件所在的文件夹路径
    aptos_csv_file = '/data/zwj/SPR/OphNet/data/APTOS_val2.csv' 

    # 确保预测结果文件夹和 CSV 文件存在
    if not os.path.exists(prediction_directory) or not os.path.isdir(prediction_directory):
        print(f"错误：找不到预测结果文件夹 {prediction_directory}")
    elif not os.path.exists(aptos_csv_file):
        print(f"错误：找不到 CSV 文件 {aptos_csv_file}")
    else:
        fill_predictions_to_csv(prediction_directory, aptos_csv_file)