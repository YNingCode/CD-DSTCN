import math
import numpy as np
import torch.nn as nn
import torch
from torch.nn import init
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(nfeat, nhid, heads=nheads, concat=True, dropout=0.6)
        self.conv2 = GATConv(nhid * nheads, nhid, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gat = GAT(195, 20, 8)
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(1800, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.d1 = nn.Dropout(p=0.3)

        self.l2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.3)

        self.l3 = nn.Linear(256, 2)


    def forward(self, input):
        batch_size = input.size(0)
        device = input.device
        # 计算邻接矩阵
        edge_index_list = []
        for i in range(batch_size):
            net = np.corrcoef(input[i].cpu().detach().numpy())
            adj = torch.from_numpy(net).float().to(device)
            edge_index = (adj > 0.5).nonzero(as_tuple=False).t().contiguous()
            edge_index_list.append(edge_index)

        out_list = []
        for i in range(batch_size):
            out = self.gat(input[i], edge_index_list[i])
            out_list.append(out)

        out = torch.stack(out_list, dim=0)
        out_flatten = self.f1(out)
        block_outs = self.d1(self.bn1(self.l1(out_flatten)))
        block_outs = self.d2(self.bn2(self.l2(block_outs)))
        out_logits = self.l3(block_outs)

        return F.softmax(out_logits, dim=1), block_outs

# dim = (20, 116, 220)
#
# test = torch.randn(dim)
# print(test.shape)
# model = Model()
# out = model(test)
# print(out.shape)