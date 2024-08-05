import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from PIL import ImageFile
from scipy.io import loadmat
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler

ImageFile.LOAD_TRUNCATED_IMAGES = True

def normalize(data):
    min_vals = np.min(data, axis=(1, 2), keepdims=True)
    max_vals = np.max(data, axis=(1, 2), keepdims=True)
    normalized_data = (data - min_vals) / (max_vals - min_vals +1e5)
    return normalized_data

def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x

def dianxiandataloader(args):
    m = loadmat('../STGCN_brain/dataset/Fmri/X_data_gnd.mat')
    data = m['data']  # (306,90,240)这是306个受试者的90个脑区在240时间点的血氧水平含量
    labels = m['gnd'][0]  # 有0、1、2三种
    # n = loadmat('dataset/Fmri/G_all.mat')  # DTI
    # ddata = n['G']
    # ddata = ddata.transpose(2, 0, 1)  # (306,90,90)

    # for i in range(labels.shape[0]):
    #     if labels[i] == 2:
    #         labels[i] = 1

    # 只取标签0，1
    # bool_idx = (labels == 0) | (labels == 1)
    # data = data[bool_idx]
    # labels = labels[bool_idx]

    # 只取标签0，2
    # bool_idx = (labels == 0) | (labels == 2)
    # data = data[bool_idx]
    # labels = labels[bool_idx]
    # for i in range(labels.shape[0]):
    #     if labels[i] == 2:
    #         labels[i] = 1

    # 只取标签1，2
    bool_idx = (labels == 1) | (labels == 2)
    data = data[bool_idx]
    labels = labels[bool_idx]
    for i in range(labels.shape[0]):
        if labels[i] == 1:
            labels[i] = 0
        if labels[i] == 2:
            labels[i] = 1


    data = normalize(data)
    index = [i for i in range(data.shape[0])]
    np.random.shuffle(index)
    data = data[index]
    labels = labels[index]

    data_tensor = torch.from_numpy(data).float()

    # dti_tensor = torch.from_numpy(ddata).float()
    labels_tensor = torch.from_numpy(labels)
    num_nodes = data_tensor.size(1)
    seq_length = data_tensor.size(2)
    num_classes = torch.unique(labels_tensor).size(0)

    dataset = TensorDataset(data_tensor, labels_tensor)
    # dataset = MultiModalDataset(data_tensor, dti_tensor, labels_tensor)
    dataset_size = len(dataset)
    # train_size = int(0.8 * dataset_size)  # 这里我们保留 60% 的数据作为训练集
    #
    # test_size = dataset_size - train_size  # 剩下的 40% 数据作为测试集
    #
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # return train_dataloader,test_dataloader,num_nodes,seq_length,num_classes,test_size,train_size
    return dataset,num_nodes,seq_length,num_classes

def ADNI(args):
    m = loadmat('../STGCN_brain/dataset/ADNI/ADNI.mat')
    data = m['feas']  # (203,90,197)这是306个受试者的90个脑区在240时间点的血氧水平含量
    labels = m['label'][0]  # 有0、1、2三种
    # n = loadmat('dataset/Fmri/G_all.mat')  # DTI
    # ddata = n['G']
    # ddata = ddata.transpose(2, 0, 1)  # (306,90,90)

    start = (197 - 195) // 2  # 中间的195个数据点的起始索引
    end = start + 195  # 结束索引
    data = data[:,:,start:end]

    # for i in range(labels.shape[0]):
    #     if labels[i] == 2:
    #         labels[i] = 1

    # 只取标签0，1
    bool_idx = (labels == 0) | (labels == 1)
    data = data[bool_idx]
    labels = labels[bool_idx]

    # 只取标签0，2
    # bool_idx = (labels == 0) | (labels == 2)
    # data = data[bool_idx]
    # labels = labels[bool_idx]
    # for i in range(labels.shape[0]):
    #     if labels[i] == 2:
    #         labels[i] = 1

    # 只取标签1，2
    # bool_idx = (labels == 1) | (labels == 2)
    # data = data[bool_idx]
    # labels = labels[bool_idx]
    # for i in range(labels.shape[0]):
    #     if labels[i] == 1:
    #         labels[i] = 0
    #     if labels[i] == 2:
    #         labels[i] = 1


    data = normalize(data)
    index = [i for i in range(data.shape[0])]
    np.random.shuffle(index)
    data = data[index]
    labels = labels[index]

    data_tensor = torch.from_numpy(data).float()

    # dti_tensor = torch.from_numpy(ddata).float()
    labels_tensor = torch.from_numpy(labels)
    num_nodes = data_tensor.size(1)
    seq_length = data_tensor.size(2)
    num_classes = torch.unique(labels_tensor).size(0)

    dataset = TensorDataset(data_tensor, labels_tensor)
    # dataset = MultiModalDataset(data_tensor, dti_tensor, labels_tensor)
    dataset_size = len(dataset)
    # train_size = int(0.8 * dataset_size)  # 这里我们保留 60% 的数据作为训练集
    #
    # test_size = dataset_size - train_size  # 剩下的 40% 数据作为测试集
    #
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # return train_dataloader,test_dataloader,num_nodes,seq_length,num_classes,test_size,train_size
    return dataset,num_nodes,seq_length,num_classes

def PD(args):
    m = loadmat('../STGCN_brain/dataset/PD/PD_dataset.mat')
    data = m['feas']  # (162,116,220)
    labels = m['label'][0]  # 有0、1、2三种(int)

    # for i in range(labels.shape[0]):
    #     if labels[i] == 2:
    #         labels[i] = 1

    # 只取标签0，1
    # bool_idx = (labels == 0) | (labels == 1)
    # data = data[bool_idx]
    # labels = labels[bool_idx]

    # 只取标签0，2
    # bool_idx = (labels == 0) | (labels == 2)
    # data = data[bool_idx]
    # labels = labels[bool_idx]
    # for i in range(labels.shape[0]):
    #     if labels[i] == 2:
    #         labels[i] = 1

    # 只取标签1，2
    bool_idx = (labels == 1) | (labels == 2)
    data = data[bool_idx]
    labels = labels[bool_idx]
    for i in range(labels.shape[0]):
        if labels[i] == 1:
            labels[i] = 0
        if labels[i] == 2:
            labels[i] = 1


    data = normalize(data)
    index = [i for i in range(data.shape[0])]
    np.random.shuffle(index)
    data = data[index]
    labels = labels[index]

    data_tensor = torch.from_numpy(data).float()
    labels_tensor = torch.from_numpy(labels)
    num_nodes = data_tensor.size(1)
    seq_length = data_tensor.size(2)
    num_classes = torch.unique(labels_tensor).size(0)

    dataset = TensorDataset(data_tensor, labels_tensor)

    return dataset,num_nodes,seq_length,num_classes

def read_eeg(args):
    # get dataset from .pt
    data_train = torch.load(f'{args.data_root}/{args.dataset}/SEED_train.pt')
    data_test = torch.load(f'{args.data_root}/{args.dataset}/SEED_test.pt')
    label_train = torch.load(f'{args.data_root}/{args.dataset}/SEED_train_label.pt')
    label_test = torch.load(f'{args.data_root}/{args.dataset}/SEED_test_label.pt')

    # init [num_variables, seq_length, num_classes]
    num_nodes = data_test.size(-2)
    num_train = data_train.size(0)
    num_test = data_test.size(0)
    seq_length = data_test.size(-1)
    num_classes = torch.unique(label_test).size(0)

    data_train = data_train.reshape(num_train, -1, num_nodes, args.patch_size).float()
    data_test = data_test.reshape(num_test, -1, num_nodes, args.patch_size).float()
    # convert data & labels to TensorDataset
    train_dataset = TensorDataset(data_train, label_train.float())
    test_dataset = TensorDataset(data_test, label_test.float())

    # data_loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,)

    return train_loader, test_loader, num_nodes, seq_length, num_classes, len(test_dataset), len(train_dataset)


# npz = np.load(r'G:\SEED\SEED_Multimodal\Chinese\02-EEG-DE-feature\eeg_used_4s\1_1.npz')
# print(npz.size)