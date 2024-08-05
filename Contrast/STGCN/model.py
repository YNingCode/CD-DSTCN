import numpy as np
import torch
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from Model.STGCN_2 import *


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = torch.diag(torch.sum(W, dim=1))

    L = D - W

    eigenvalues = torch.linalg.eig(L)[0]  # 获取特征值
    magnitude = torch.abs(eigenvalues)
    lambda_max = torch.max(magnitude).item()  # 获取最大特征值
    # lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - torch.eye(W.shape[0]).to(L.device)

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [torch.eye(N).to(L_tilde.device), L_tilde.clone()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def my_corrcoef(x):
    x = x - x.mean(dim=1, keepdim=True)
    y = x / (x.norm(dim=1, keepdim=True) + 1e-6)
    return y.mm(y.t())

def pearson_adj(node_features):
    bs, N, dimen = node_features.size()

    Adj_matrices = []
    for b in range(bs):
        corr_matrix = my_corrcoef(node_features[b])
        corr_matrix = (corr_matrix + 1) / 2
        L_tilde = scaled_Laplacian(corr_matrix)
        cheb_polynomials = cheb_polynomial(L_tilde, K=3)
        # cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor) for i in cheb_polynomial(L_tilde, K=3)]
        Adj_matrices.append(torch.stack(cheb_polynomials))
    Adj = torch.stack(Adj_matrices)

    return Adj

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.st1 = san_stgcn(num_of_timesteps=1, num_of_vertices=90, num_of_features=195,
                             num_of_time_filters=10, num_of_chev_filters=45, time_conv_kernel=3,
                             time_conv_strides=1, k=3)
        # self.st2 = san_stgcn(num_of_timesteps=3, num_of_vertices=88, num_of_features=43,
        #                      num_of_time_filters=10, num_of_chev_filters=20, time_conv_kernel=3,
        #                      time_conv_strides=1, k=3)
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(3784, 1024)  # 24300 3784
        self.bn1 = nn.BatchNorm1d(1024)
        self.d1 = nn.Dropout(p=0.3)
        self.l2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.3)
        self.l3 = nn.Linear(256, 2)

    def forward(self, fdata):
        fdata = fdata.unsqueeze(1)
        # X：（20，32，62，40）
        bs, tlen, num_nodes, seq = fdata.size()
        fdata = fdata.permute(1, 0, 2, 3)

        # 皮尔逊系数计算邻接矩阵
        A_input = tr.reshape(fdata, [bs * tlen, num_nodes, seq])
        adj = pearson_adj(A_input)
        # print(adj.shape)
        adj = tr.reshape(adj, [tlen, bs, adj.shape[1], adj.shape[2], adj.shape[3]]) # 3,bs,3,90,90
        # adj = torch.mean(adj, dim=0)

        # 通过STGCN提取特征
        fdata = fdata.permute(1, 0, 2, 3)
        out = []
        for i in range(adj.size(0)):
            adj_t = adj[i]
            out_t = self.st1(fdata, adj_t)
            out.append(out_t)
        block_out = torch.stack(out)
        block_out = block_out.squeeze(2)
        block_out = block_out.squeeze(0)
        # block_out = block_out.permute(1, 0, 2, 3)



        block_outs = self.f1(block_out)
        block_outs = self.d1(self.bn1(self.l1(block_outs)))
        block_outs = self.d2(self.bn2(self.l2(block_outs)))
        out_logits = self.l3(block_outs)

        return F.softmax(out_logits, dim=1), block_outs


# dim = (20, 116, 220)
# dim = (20, 90, 195)
# test = torch.randn(dim)
# print(test.shape)
# model = Model()
# out = model(test)
# print(out.shape)