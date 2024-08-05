import math
import numpy as np
import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F



class LinearLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        # self.clf = nn.Sequential(OrderedDict([
        #     ('fc1', nn.Linear(in_dim, 128)),
        #     ('relu1', nn.ReLU(inplace=True)),
        #     ('dropout1', nn.Dropout(0.5)),
        #     ('fc2', nn.Linear(128, 32)),
        #     ('relu2', nn.ReLU(inplace=True)),
        #     ('dropout2', nn.Dropout(0.5)),
        #     ('fc3', nn.Linear(32, out_dim)),
        # ]))

    def forward(self, x):
        x = self.clf(x)
        return x

class Uncertainty(nn.Module):
    def __init__(self, mlp_dim):
        super(Uncertainty, self).__init__()

        self.layers = nn.ModuleList([LinearLayer(mlp_dim, 1)])

    def forward(self, x):
        tlen, bs, num_nodes, dim = x.size()
        x = torch.reshape(x, [bs*tlen, num_nodes*dim])
        for layer in self.layers:
            x = layer(x)
        x = x.view(bs, tlen, 1)
        x = torch.sigmoid(x)
        return x

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


def reshape_dot(x, TATT):
    outs = torch.matmul((x.permute(0, 2, 3, 1))
                        .reshape(x.shape[0], -1, x.shape[1]), TATT).reshape(-1, x.shape[1],
                                                                            x.shape[2],
                                                                            x.shape[3])
    return outs

class TemporalAttention(nn.Module):
    '''
       compute temporal attention scores
       --------
       Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)  (1,5,26,9)
       Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''

    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features):
        super(TemporalAttention, self).__init__()

        self.num_of_timesteps = num_of_timesteps
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.U_1 = nn.Parameter(torch.zeros(self.num_of_vertices, 1))
        nn.init.xavier_uniform_(self.U_1.data, gain=1.414)
        self.U_2 = nn.Parameter(torch.zeros(self.num_of_features, self.num_of_vertices))
        nn.init.xavier_uniform_(self.U_2.data, gain=1.414)
        self.U_3 = nn.Parameter(torch.zeros(self.num_of_features, 1))
        nn.init.xavier_uniform_(self.U_3.data, gain=1.414)
        self.b_e = nn.Parameter(torch.zeros(1, self.num_of_timesteps, self.num_of_timesteps))
        nn.init.xavier_uniform_(self.b_e.data, gain=1.414)
        self.v_e = nn.Parameter(torch.zeros(self.num_of_timesteps, self.num_of_timesteps))
        nn.init.xavier_uniform_(self.v_e.data, gain=1.414)

    def forward(self, x):
        # shape of lhs is (batch_size, T, V)
        a = x.permute(0, 1, 3, 2)  # 1,5,90,90
        # print(a.shape)
        # print(self.U_1.shape)
        lhs = torch.matmul(a, self.U_1)
        # print(lhs.shape)

        lhs = lhs.reshape(x.shape[0], self.num_of_timesteps, self.num_of_features)
        lhs = torch.matmul(lhs, self.U_2)  # torch.Size([1, 5, 26])

        # shape of rhs is (batch_size, V, T)
        b = x.permute(2, 0, 3, 1)  # torch.Size([26, 1, 9, 5])
        zj = torch.squeeze(self.U_3)
        rhs = torch.matmul(zj, b)  # torch.Size([26, 1, 5])
        rhs = rhs.permute(1, 0, 2)  # torch.Size([1, 26, 5])

        # shape of product is (batch_size, T, T)
        product = torch.matmul(lhs, rhs)
        product = F.sigmoid(product + self.b_e).permute(1, 2, 0)  # torch.Size([5, 5, 1])
        product = torch.matmul(self.v_e, product)
        s = product.permute(2, 0, 1)

        # normalization
        s = s - torch.max(s, dim=1, keepdim=True)[0]
        exp = torch.exp(s)
        S_normalized = exp / torch.sum(exp, dim=1, keepdim=True)

        return S_normalized  # torch.Size([1, 5, 5])
        # return S_normalized,(x.shape[0], x.shape[1], x.shape[1])

class SpatialAttention(nn.Module):
    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features):
        super(SpatialAttention, self).__init__()

        self.num_of_timesteps = num_of_timesteps
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.W_1 = nn.Parameter(torch.zeros(num_of_timesteps, 1))
        nn.init.xavier_uniform_(self.W_1.data, gain=1.414)
        self.W_2 = nn.Parameter(torch.zeros(self.num_of_features, self.num_of_timesteps))
        nn.init.xavier_uniform_(self.W_2.data, gain=1.414)
        self.W_3 = nn.Parameter(torch.zeros(self.num_of_features, 1))
        nn.init.xavier_uniform_(self.W_3.data, gain=1.414)
        self.b_s = nn.Parameter(torch.zeros(1, self.num_of_vertices, self.num_of_vertices))
        nn.init.xavier_uniform_(self.b_s.data, gain=1.414)
        self.v_s = nn.Parameter(torch.zeros(self.num_of_vertices, self.num_of_vertices))
        nn.init.xavier_uniform_(self.v_s.data, gain=1.414)

    def forward(self, x):
        # shape of lhs is (batch_size, V, T)
        lhs = torch.matmul(x.permute(0, 2, 3, 1), self.W_1)
        lhs = lhs.reshape(x.shape[0], self.num_of_vertices, self.num_of_features)
        lhs = torch.matmul(lhs, self.W_2)  # torch.Size([1, 26, 5])

        # shape of rhs is (batch_size, T, V)
        zj = torch.squeeze(self.W_3)
        rhs = torch.matmul(zj, x.permute(1, 0, 3, 2))
        rhs = rhs.permute(1, 0, 2)  # torch.Size([1, 5, 26])

        # shape of product is (batch_size, V, V)
        product = torch.matmul(lhs, rhs)  # torch.Size([1, 26, 26])
        product = F.sigmoid(product + self.b_s).permute(1, 2, 0)
        product = torch.matmul(self.v_s, product)
        s = product.permute(2, 0, 1)

        # normalization
        s = s - torch.max(s, dim=1, keepdim=True)[0]
        exp = torch.exp(s)
        S_normalized = exp / torch.sum(exp, dim=1, keepdim=True)

        return S_normalized  # torch.Size([2, 26, 26])



class STBlock(nn.Module):
    def __init__(self, gcn_input, gcn_out):
        super(STBlock, self).__init__()
        # self.lstm = nn.LSTM(input_size=90, hidden_size=90, num_layers=2, batch_first=True)
        # self.gcn = GCN(240, 30)
        self.gcn = GCN(gcn_input, gcn_out)
        # self.lstm_2 = nn.LSTM(input_size=90, hidden_size=90, num_layers=2, batch_first=True)
        self.cnn_1 = nn.Conv2d(in_channels=90, out_channels=90, kernel_size=(1, 3), padding=(0, 1))
        self.cnn_2 = nn.Conv2d(in_channels=90, out_channels=90, kernel_size=(1, 3), padding=(0, 1))
        self.linear = nn.Linear(30, 27)

    def forward(self, fdata, g_before):
        # 邻接矩阵
        adj = []
        for i in range(fdata.shape[0]):
            net = np.corrcoef(fdata[i].cpu().detach().numpy())
            adj.append(net)
        adj = np.array(adj)
        adj = torch.from_numpy(adj).float().to(fdata.device)

        # CNN
        fdata = fdata.unsqueeze(2)
        l_out = self.cnn_1(fdata)
        l_out = l_out.squeeze(2)

        # GCN
        # l_out = l_out.permute(0,2,1)
        g_before = g_before.reshape(-1,30)
        g_before_mean = torch.mean(g_before, dim=0)
        g_before = self.linear(g_before_mean)
        l_out = l_out + g_before
        g_out = self.gcn(l_out, adj)  # out: 20,90,128

        # CNN
        input = g_out.unsqueeze(2)
        l_out = self.cnn_1(input)
        out = l_out.squeeze(2)

        return out, g_out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.un = Uncertainty(2430)   # adni
        # self.un = Uncertainty(3132)     # PD
        self.st_1 = STBlock(27, 30)
        # self.st_2 = STBlock(20, 20)

        self.f1 = nn.Flatten()

        self.l1 = nn.Linear(2700, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.d1 = nn.Dropout(p=0.3)

        self.l2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(p=0.3)

        self.l3 = nn.Linear(256, 2)

        self.temporal_Att = TemporalAttention(num_of_timesteps=7, num_of_vertices=90, num_of_features=27)
        # self.spatial_At = SpatialAttention(num_of_timesteps=4, num_of_vertices=116, num_of_features=55)

    def forward(self, fdata):
        # fdata：（20，3, 90，80） bs, tlen, num_nodes, time_seq
        fdata = fdata.permute(1,0,2,3)

        # 每个时间片获得一个不确定性分数
        Confidence_score = self.un(fdata)

        # 时间注意力
        input = fdata.permute(1,0,2,3)
        temporal_Att = self.temporal_Att(input)
        x_TAt = reshape_dot(input, temporal_Att)

        # 空间注意力
        # spatial_Att = self.spatial_At(x_TAt)

        fdata = x_TAt.permute(1,0,2,3)
        all_out= []
        shape = (fdata[0].size(0),90,30)
        g_before = torch.zeros(shape).to(fdata.device)
        for t in range(fdata.size(0)):
            out, g_before = self.st_1(fdata[t], g_before)
            # out = self.st_2(out)
            all_out.append(out)
        all_out = torch.stack(all_out)

        # 不确定性分数加权融合
        all_out = all_out.permute(1,0,2,3)
        final_out = []
        for n in range(Confidence_score.size(0)):
            ten = Confidence_score[n].view(Confidence_score[n].size(0), Confidence_score[n].size(1), 1)
            result = torch.sum(ten * all_out[n], dim=0)
            final_out.append(result)
        final = torch.stack(final_out)

        # final = all_out

        # out = F.relu(g_out)
        block_outs = self.f1(final)
        block_outs = self.d1(self.bn1(self.l1(block_outs)))
        block_outs = self.d2(self.bn2(self.l2(block_outs)))
        # block_outs = self.d1(self.l1(block_outs))
        # block_outs = self.d2(self.l2(block_outs))
        out_logits = self.l3(block_outs)

        return F.softmax(out_logits, dim=1)


# class NModel(nn.Module):
#     def __init__(self):
#         super(NModel, self).__init__()
#         self.gcn = GCN(240, 30)
#
#         self.st1 = STBlock(197,30)
#         self.st2 = STBlock(30, 30)
#
#         self.f1 = nn.Flatten()
#         self.l1 = nn.Linear(2700, 1024)  # 24300 3784
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.d1 = nn.Dropout(p=0.3)
#         self.l2 = nn.Linear(1024, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.d2 = nn.Dropout(p=0.3)
#         self.l3 = nn.Linear(256, 2)
#
#     def forward(self, fdata):
#         # fdata：（20，90，240） bs, num_nodes, time_seq
#
#         out = self.st1(fdata)
#         out = self.st2(out)
#
#         block_outs = self.f1(out)
#         block_outs = self.d1(self.bn1(self.l1(block_outs)))
#         block_outs = self.d2(self.bn2(self.l2(block_outs)))
#         out_logits = self.l3(block_outs)
#
#         return F.softmax(out_logits, dim=1)

# dim = (20, 90, 240)
#
# test = torch.randn(dim)
# print(test.shape)
# model = NModel()
# out = model(test)
# print(out.shape)