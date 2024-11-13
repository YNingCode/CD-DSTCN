import os
from scipy import signal
import numpy as np
import scipy.io
import torch
from torch.utils.data import TensorDataset

def normalize(data):
    min_vals = np.min(data, axis=(1, 2), keepdims=True)
    max_vals = np.max(data, axis=(1, 2), keepdims=True)
    normalized_data = (data - min_vals) / (max_vals - min_vals +1e5)
    return normalized_data

def getdata():
    # 定义数据文件夹路径
    data_folder = "E:\Dataset\shu_dataset\mat"

    # 初始化数据和标签列表
    all_data = []
    all_label = []
    # 遍历所有受试者和采集文件
    for subj in range(1, 26):
        subject_data = []
        subject_label = []
        for sess in range(1, 6):
            # 构建文件名
            filename = f"sub-{subj:03d}_ses-{sess:02d}_task_motorimagery_eeg.mat"
            filepath = os.path.join(data_folder, filename)

            # 检查文件是否存在
            if os.path.isfile(filepath):
                # 读取MAT文件
                mat_data = scipy.io.loadmat(filepath)
                # 假设MAT文件中的数据存储在名为'data'的变量中
                if 'data' in mat_data:
                    data = mat_data['data']
                    label_class = mat_data['labels'].flatten()-1
                    # 将数据添加到列表中
                    subject_data.append(data)
                    subject_label.append(label_class)
                else:
                    print(f"Warning: 'data' variable not found in {filename}")
            else:
                print(f"Warning: File not found: {filepath}")

        subject_data = np.concatenate(subject_data, axis=0)
        all_data.append(subject_data)
        # 将 0 替换为 -1
        subject_label = np.concatenate(subject_label, axis=0)
        subject_label = np.where(subject_label == 0, -1, subject_label)
        all_label.append(subject_label)

    # 定义保存数据的根目录
    save_dir = 'D:\\Work_1\\EEG_to_txt\\'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历每个受试者
    for subj_idx in range(len(all_data)):
        subject_data = all_data[subj_idx]  # 形状为 (x, 32, 1000)
        subject_labels = all_label[subj_idx]  # 形状为 (x,)

        # 为每个受试者创建一个单独的文件夹
        subject_dir = os.path.join(save_dir, f'subject_{subj_idx + 1}')
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)

        # 保存每条数据为单独的txt文件
        for sample_idx in range(subject_data.shape[0]):
            sample_data = subject_data[sample_idx].T  # 形状为 (32, 1000)
            # 使用 zfill 来格式化文件名为三位数
            sample_filename = os.path.join(subject_dir, f'{str(sample_idx + 1).zfill(3)}.txt')
            np.savetxt(sample_filename, sample_data, fmt='%.6f')  # 保存数据为 txt 文件

        # 将标签保存到一个 txt 文件中
        label_filename = os.path.join(subject_dir, 'labels.txt')
        np.savetxt(label_filename, subject_labels, fmt='%d')  # 保存标签为 txt 文件

    print("数据和标签已成功保存。")


getdata()