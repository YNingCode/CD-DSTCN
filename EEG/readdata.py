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

def getdataset(subj_idx):
    # 定义数据文件夹路径
    data_folder = "/lab/2023/yn/Dataset/shu_dataset/"

    # 初始化数据和标签列表
    all_data = []
    all_labels = []
    # 遍历所有受试者和采集文件
    for sess in range(1, 6):
        # 构建文件名
        filename = f"sub-{subj_idx:03d}_ses-{sess:02d}_task_motorimagery_eeg.mat"
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
                all_data.append(data)
                all_labels.append(label_class)
            else:
                print(f"Warning: 'data' variable not found in {filename}")
        else:
            print(f"Warning: File not found: {filepath}")

    # 将数据和标签转换为NumPy数组
    all_data = np.concatenate(all_data, axis=0)
    all_data = normalize(all_data)
    all_labels = np.concatenate(all_labels, axis=0)

    # 将数据和标签转换为PyTorch张量
    tensor_data = torch.tensor(all_data, dtype=torch.float32)              # 11988,32,1000
    tensor_labels = torch.tensor(all_labels, dtype=torch.long) # 11988,

    # 创建TensorDataset
    dataset = TensorDataset(tensor_data, tensor_labels)

    return dataset

if __name__ == "__main__":
    data = getdataset(subj_idx=1)