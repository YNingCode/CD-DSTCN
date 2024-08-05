import math
import numpy as np
import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        init.xavier_uniform_(self.weight, gain=math.sqrt(2.0))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.gc1 = GraphConvolution(self.nfeat, self.nhid)
        self.gc2 = GraphConvolution(self.nhid, self.nhid)


    def forward(self, x, adj):
        out = self.gc1(x, adj)
        out = self.gc2(out, adj)

        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gcn = GCN(195, 20)
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(1800, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.d1 = nn.Dropout(p=0.3)

        self.l2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.3)

        self.l3 = nn.Linear(256, 2)


    def forward(self, input):
        # 计算邻接矩阵
        adj = []
        for i in range(input.shape[0]):
            net = np.corrcoef(input[i].cpu().detach().numpy())
            adj.append(net)
        adj = np.array(adj)
        adj = torch.from_numpy(adj).float().to(input.device)

        out = self.gcn(input, adj)
        out_flatten = self.f1(out)
        block_outs = self.d1(self.bn1(self.l1(out_flatten)))
        block_outs = self.d2(self.bn2(self.l2(block_outs)))
        out_logits = self.l3(block_outs)

        return F.softmax(out_logits, dim=1)

# dim = (20, 90, 240)
#
# test = torch.randn(dim)
# print(test.shape)
# model = Model()
# out = model(test)
# print(out.shape)