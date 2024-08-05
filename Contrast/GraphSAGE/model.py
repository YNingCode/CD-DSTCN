import math
import numpy as np
import torch.nn as nn
import torch
from torch.nn import init
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


# 定义GraphSAGE模型
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.graphsage = GraphSAGE(195, 128, 64)

        self.fc1 = nn.Linear(64 * 90, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.d1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 2)


    def forward(self, input):
        batch_size = input.shape[0]
        data_list = []

        for i in range(batch_size):
            x = input[i]  # (116, 220)
            x = x.clone().detach()  # 节点特征

            # 计算皮尔逊相关系数并生成边列表
            adj = np.corrcoef(x.cpu().numpy())
            edge_index = np.array(np.where(adj > 0.5))
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            # 创建图数据
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
        data_list = [data.to(input.device) for data in data_list]
        batch = Batch.from_data_list(data_list)
        x = self.graphsage(batch)
        x = x.view(batch_size, -1)
        x = self.d1(self.bn1(F.relu(self.fc1(x))))
        out = self.d2(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(out)

        return F.softmax(x, dim=1), out

# dim = (20, 116, 220)
#
# test = torch.randn(dim)
# print(test.shape)
# model = Model()
# out = model(test)
# print(out.shape)