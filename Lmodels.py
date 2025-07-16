""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import time
import random
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from utils import *
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool as gap
from torch.nn import LayerNorm, Parameter
from torch.nn import init, Parameter
import torch.optim.lr_scheduler as lr_scheduler
import math

def xavier_init(m):  # 定义一个函数 xavier_init，参数为 m
    if type(m) == nn.Linear:  # 检查 m 的类型是否为 nn.Linear（全连接层）
        nn.init.xavier_normal_(m.weight)  # 如果是，则使用 Xavier 正态分布初始化 m 的权重
        if m.bias is not None:  # 如果 m 有偏置项
            m.bias.data.fill_(0.0)  # 将偏置项的值填充为 0.0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def KL(alpha, c):  # 定义一个名为 KL 的函数，参数为 alpha 和 c
    beta = torch.ones((1, c)).to(device)  # 创建一个形状为 (1, c) 的张量 beta，并将其初始化为 1，然后将其移动到 GPU 上
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)  # 计算 alpha 在第 1 维的和，并保持其维度不变
    S_beta = torch.sum(beta, dim=1, keepdim=True)  # 计算 beta 在第 1 维的和，并保持其维度不变
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)  # 计算 ln(B(alpha))，其中 B 是 Beta 函数
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)  # 计算 ln(B(beta))，其中 beta 是一个统一的 Beta 分布
    dg0 = torch.digamma(S_alpha)  # 计算 S_alpha 的 digamma 函数值
    dg1 = torch.digamma(alpha)  # 计算 alpha 的 digamma 函数值
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni  # 计算 KL 散度
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):

    S = torch.sum(alpha, dim=1, keepdim=True)


    E = alpha - 1

    label = F.one_hot(p, num_classes=c)


    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)


    annealing_coef = min(1, global_step / annealing_step)


    alp = E * (1 - label) + 1


    B = annealing_coef * KL(alp, c)

    return (A + B)


def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

class GAT1(nn.Module):
    def __init__(self, dropout, alpha, dim):
        super(GAT1, self).__init__()


        self.dropout = dropout
        self.act =define_act_layer(act_type='LeakyReLU')
        self.dim = dim
        self.nhids = [40, 50, 50]
        self.nheads = [5, 4, 8]
        self.fc_dim = [self.dim+self.nhids[0] * self.nheads[0]+self.nhids[1] * self.nheads[1],512,self.nhids[0] * self.nheads[0],128,64,32,32]  # 全连接层的维度j
        self.fk=[ self.dim,self.nhids[0] * self.nheads[0]]
        self.device= torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.attentions1 = [GraphAttentionLayer1(
            dim, self.nhids[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(self.nheads[0])]
        for i, attention1 in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention1)

        self.attentions2 = [GraphAttentionLayer1(
            self.nhids[0] * self.nheads[0], self.nhids[1], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(self.nheads[1])]
        for i, attention2 in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention2)

        self.attentions3 = [GraphAttentionLayer1(
            self.nhids[1] * self.nheads[1], self.nhids[2], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(self.nheads[2])]
        for i, attention3 in enumerate(self.attentions2):
            self.add_module('attention3_{}'.format(i), attention3)


        self.embed_weight = nn.Parameter(torch.Tensor(self.nhids[0] * self.nheads[0], 1))
        g = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.embed_weight, gain=g)
        self.att_dropout = nn.Dropout(0.1)

        self.dropout_layer = nn.Dropout(p=self.dropout)

        lin_input_dim =self.dim+self.nhids[0] * self.nheads[0]+self.nhids[1] * self.nheads[1]
        self.fc1 = nn.Sequential(
            nn.Linear(lin_input_dim, self.fc_dim[0]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc1.apply(xavier_init)  # 初始化全连接层参数

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_dim[0], self.fc_dim[1]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc2.apply(xavier_init)

        self.fc3 = nn.Sequential(
            nn.Linear(self.fc_dim[1], self.fc_dim[2]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc3.apply(xavier_init)

        self.fc4 = nn.Sequential(
            nn.Linear(self.fc_dim[2], self.fc_dim[3]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc4.apply(xavier_init)

        self.fc5 = nn.Sequential(
            nn.Linear(self.fc_dim[3], self.fc_dim[4]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc5.apply(xavier_init)

        self.fc6 = nn.Sequential(
            nn.Linear(self.fc_dim[4], self.fc_dim[5]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc6.apply(xavier_init)
        self.fc7 = nn.Sequential(
            nn.Linear(self.fc_dim[5], self.fc_dim[6]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc7.apply(xavier_init)

        self.fk1 = nn.Sequential(
            nn.Linear(self.fk[0], self.fk[1]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fk1.apply(xavier_init)
        self.to(self.device)

    def forward(self, x, adj):

        x0 = self.fk1(x)

        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions1], dim=-1)

        x1 = x
        x = self.dropout_layer(x)

        x = torch.cat([att(x, adj) for att in self.attentions2], dim=-1)

        x2 = x

        x = self.dropout_layer(x)

        x_collect = [x0,x1,x2]
        h_collect_new = torch.stack(x_collec,t, dim=1)


        global_vector = torch.softmax(torch.sigmoid(torch.matmul(h_collect_new, self.embed_weight).squeeze(2)), dim=-1)  # (32, 3)
        output_final = torch.zeros_like(x0)

        for i, hidden in enumerate(x_collect):
            output_final += hidden.mul(self.att_dropout(global_vector[:, i].unsqueeze(1)))


        y = self.fc4(output_final)


        output = y

        return output

class GAT2(nn.Module):
    def __init__(self, dropout, alpha, dim):
        super(GAT2, self).__init__()


        self.dropout = dropout
        self.act = define_act_layer(act_type='LeakyReLU')
        self.dim = dim

        self.nhids = [250, 500, 80]
        self.nheads = [8, 4, 5]
        self.fc_dim = [self.dim+self.nhids[0] * self.nheads[0]+self.nhids[1] * self.nheads[1], 1024, 512, 256,128,64]  # 全连接层的维度,
        self.fk=[ self.dim,self.nhids[0] * self.nheads[0]]
        self.device= torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.attentions1 = [GraphAttentionLayer1(
            dim, self.nhids[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(self.nheads[0])]
        for i, attention1 in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention1)


        self.attentions2 = [GraphAttentionLayer1(
            self.nhids[0] * self.nheads[0], self.nhids[1], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(self.nheads[1])]
        for i, attention2 in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention2)



        self.attentions3 = [GraphAttentionLayer1(
            self.nhids[1] * self.nheads[1], self.nhids[2], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(self.nheads[2])]
        for i, attention2 in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention2)



        self.embed_weight = nn.Parameter(torch.Tensor(self.nhids[0] * self.nheads[0], 1))
        g = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.embed_weight, gain=g)
        self.att_dropout = nn.Dropout(0.1)

        self.dropout_layer = nn.Dropout(p=self.dropout)


        lin_input_dim =self.dim+self.nhids[0] * self.nheads[0]+self.nhids[1] * self.nheads[1]
        input_1=self.nhids[0] * self.nheads[0]
        self.fc1 = nn.Sequential(
            nn.Linear(lin_input_dim, self.fc_dim[0]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc1.apply(xavier_init)

        self.fc2 = nn.Sequential(
            nn.Linear(input_1, self.fc_dim[1]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc2.apply(xavier_init)

        self.fc3 = nn.Sequential(
            nn.Linear(self.fc_dim[1], self.fc_dim[2]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc3.apply(xavier_init)

        self.fc4 = nn.Sequential(
            nn.Linear(self.fc_dim[2], self.fc_dim[3]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc4.apply(xavier_init)

        self.fc5 = nn.Sequential(
            nn.Linear(self.fc_dim[3], self.fc_dim[4]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc5.apply(xavier_init)
        self.fc6 = nn.Sequential(
            nn.Linear(self.fc_dim[4], self.fc_dim[5]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc6.apply(xavier_init)

        self.fk1 = nn.Sequential(
            nn.Linear(self.fk[0], self.fk[1]),
            self.act,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fk1.apply(xavier_init)


        self.to(self.device)

    def forward(self, x, adj):

        x0 =x


        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions1], dim=-1)

        x1 = x

        x = self.dropout_layer(x)  # (N, nhids[0] * nheads[0])

        x = torch.cat([att(x, adj) for att in self.attentions2], dim=-1)  # (N, nhids[1] * nheads[1])

        x2 = x

        x = self.dropout_layer(x)  # (N, nhids[0] * nheads[0])

        x = torch.cat([att(x, adj) for att in self.attentions3], dim=-1)  # (N, nhids[1] * nheads[1])
        x3=x

        x_collect = [x0,x1]
        h_collect_new = torch.stack(x_collect, dim=1)  # (32, 3, self.nhids[0] * self.nheads[0])


        global_vector = torch.softmax(torch.sigmoid(torch.matmul(h_collect_new, self.embed_weight).squeeze(2)), dim=-1)  # (32, 3)
        output_final = torch.zeros_like(x0)

        for i, hidden in enumerate(x_collect):
            output_final += hidden.mul(self.att_dropout(global_vector[:, i].unsqueeze(1)))  heads[0])


        x = self.fc2(output_final)

        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        output = x

        return output
class GraphAttentionLayer1(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.device= torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)).to(self.device))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)).to(self.device))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化


        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inp, adj):

        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))


        zero_vec = -1e12 * torch.ones_like(e)
        adj_dense = adj.to_dense()
        attention = torch.where(adj_dense > 0, e, zero_vec)
        attention = self.dropout_layer(F.softmax(attention, dim=-1))
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#CA
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim,h2):
        super(CrossAttention, self).__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_k = nn.Linear(h2, hidden_dim, bias=True)
        self.linear_v = nn.Linear(h2, hidden_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.scale_factor = math.sqrt(hidden_dim)
        self.device= torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x1, x2):
        q = self.linear_q(x1)  # query
        k = self.linear_k(x2)  # key
        v = self.linear_v(x2)  # value

        attn_weights = torch.matmul(q, k.transpose(0, 1)) / self.scale_factor
        attn_weights = self.softmax(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        output = self.linear_out(attn_output)
        return output
#多头的
class CrossAttention2(nn.Module):
    def __init__(self, hidden_dim, h2, num_heads=4, dropout=0.1):
        super(CrossAttention2, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须被 num_heads 整除"


        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads


        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(h2, hidden_dim)
        self.linear_v = nn.Linear(h2, hidden_dim)


        self.linear_out = nn.Linear(hidden_dim, hidden_dim)


        self.softmax = nn.Softmax(dim=-1)


        self.dropout = nn.Dropout(dropout)


        self.scale_factor = math.sqrt(self.head_dim)


        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        q = self.linear_q(x1).view(batch_size, self.num_heads, self.head_dim)
        k = self.linear_k(x2).view(batch_size, self.num_heads, self.head_dim)
        v = self.linear_v(x2).view(batch_size, self.num_heads, self.head_dim)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale_factor

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.view(batch_size, self.num_heads * self.head_dim)

        output = self.linear_out(attn_output)

        return output


class Fusion(nn.Module):
    def __init__(self,data_name, classes, views, dim_list,lambda_epochs=1):
        super().__init__()
        self.dropout=0.3
        if data_name == 'ROSMAP':
            self.gat1 = GAT1(dropout=0.3, alpha=0.7, dim=dim_list[0])
            self.gat2 = GAT1(dropout=0.3, alpha=0.7, dim=dim_list[1])
            self.gat3 = GAT1(dropout=0.3, alpha=0.7, dim=dim_list[2])
            a=32
        #BRCA
        else:
            self.gat1 = GAT2(dropout=0.3, alpha=0.7, dim=dim_list[0])
            self.gat2 = GAT2(dropout=0.3, alpha=0.7, dim=dim_list[1])
            self.gat3 = GAT1(dropout=0.3, alpha=0.7, dim=dim_list[2])
            a=128

        self.sigmoid = nn.Softplus()
        self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.act = define_act_layer(act_type='LeakyReLU')
        self.device= torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        b=dim_list[0]
        c=dim_list[1]
        d=dim_list[2]
        e=b+c+d

        self.att_b=CrossAttention(hidden_dim=b,h2=e)
        self.att_e_b=CrossAttention(hidden_dim=e,h2=b)
        self.att_c=CrossAttention(hidden_dim=c,h2=e)
        self.att_d=CrossAttention(hidden_dim=d,h2=e)
        self.att_e_c=CrossAttention(hidden_dim=e,h2=c)
        self.att_e_d=CrossAttention(hidden_dim=e,h2=d)


        self.fc_dim = [d, 1024,512, a,a]
        self.fb_dim=[b,1024,512,256,128,a]
        self.fd_dim=[d,512,256,128,a,a]

        self.fo1=nn.Sequential(
            nn.Linear(3*e, e),
            self.act ,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fo1.apply(xavier_init)
        self.fb1=nn.Sequential(
            nn.Linear(self.fb_dim[0], self.fb_dim[1]),
            self.act ,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fb1.apply(xavier_init)
        self.fb2=nn.Sequential(
            nn.Linear(self.fb_dim[1], self.fb_dim[2]),
            self.act ,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fb2.apply(xavier_init)
        self.fb3=nn.Sequential(
            nn.Linear(self.fb_dim[2], self.fb_dim[3]),
            self.act ,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fb3.apply(xavier_init)
        self.fb4=nn.Sequential(
            nn.Linear(self.fb_dim[3], self.fb_dim[4]),
            self.act ,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fb4.apply(xavier_init)

        self.fd1=nn.Sequential(
            nn.Linear(self.fd_dim[0], self.fd_dim[1]),
            self.act ,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fd1.apply(xavier_init)
        self.fd2=nn.Sequential(
            nn.Linear(self.fd_dim[1], self.fd_dim[2]),
            self.act ,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fd2.apply(xavier_init)
        self.fd3=nn.Sequential(
            nn.Linear(self.fd_dim[2], self.fd_dim[3]),
            self.act ,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fd3.apply(xavier_init)

        self.fc3 = nn.Sequential(
            nn.Linear(self.fc_dim[0], self.fc_dim[1]),
            self.act ,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc3.apply(xavier_init)

        self.fc4 = nn.Sequential(
            nn.Linear(self.fc_dim[1], self.fc_dim[2]),
            self.act ,
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc4.apply(xavier_init)

        self.fc5 = nn.Sequential(
            nn.Linear(a, self.classes))
        self.fc5.apply(xavier_init)
        self.fc6 = nn.Sequential(
            nn.Linear(32, self.classes))
        self.fc6.apply(xavier_init)
        #注意力
        self.embed_weight1 = nn.Parameter(torch.Tensor(a, 1))
        self.embed_weight2 = nn.Parameter(torch.Tensor(a, 1))
        self.embed_weight3 = nn.Parameter(torch.Tensor(a, 1))
        g = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.embed_weight1, gain=g)
        nn.init.xavier_uniform_(self.embed_weight2, gain=g)
        nn.init.xavier_uniform_(self.embed_weight3, gain=g)
        self.att_dropout = nn.Dropout(0.1)


        self.to(self.device)

    def DS_Combin(self, alpha):
    # 定义一个函数用于组合多个alpha值
        def DS_Combin_two(alpha1, alpha2):
        # 组合两个alpha值的具体操作
            alpha = dict()  # 创建一个字典用于存储alpha值
            alpha[0], alpha[1] = alpha1, alpha2  # 将传入的alpha1和alpha2存储到字典中
            b, S, E, u = dict(), dict(), dict(), dict()  # 创建四个字典用于存储中间计算结果
            for v in range(2):  # 对两个alpha值进行循环处理
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)  # 计算每个alpha值的和，保持维度
                E[v] = alpha[v]-1  # 对alpha值进行减1操作
                b[v] = E[v]/(S[v].expand(E[v].shape))  # 计算比例，扩展维度使其与E[v]相同
                u[v] = self.classes/S[v]  # 计算u值，self.classes除以S[v]

             # 下面是一系列的矩阵操作和计算
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))  # 矩阵乘法，得到bb
            uv1_expand = u[1].expand(b[0].shape)  # 扩展u[1]的维度使其与b[0]相同
            bu = torch.mul(b[0], uv1_expand)  # 对b[0]和uv1_expand进行逐元素相乘
            uv_expand = u[0].expand(b[0].shape)  # 扩展u[0]的维度使其与b[0]相同
            ub = torch.mul(b[1], uv_expand)  # 对b[1]和uv_expand进行逐元素相乘
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)  # 对bb进行按行列求和
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)  # 对bb的对角线元素求和
            C = bb_sum - bb_diag  # 计算C值

            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))  # 计算b_a值
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))  # 计算u_a值

            S_a = self.classes / u_a  # 计算S_a值
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))  # 对b_a和S_a进行逐元素相乘
            alpha_a = e_a + 1  # 计算alpha_a值
            return alpha_a  # 返回计算结果alpha_a

        # 循环处理所有alpha值
        for v in range(len(alpha)-1):
            # 调用DS_Combin_two函数进行组合
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])  # 初始时调用DS_Combin_two处理前两个alpha值
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])  # 后续调用DS_Combin_two处理后续alpha值
        return alpha_a  # 返回最终组合后的alpha_a值


    def forward(self, omic1, omic2, omic3, adj1, adj2, adj3, y, global_step=1):

        output1 = self.gat1(omic1, adj1)
        output2 = self.gat2(omic2, adj2)
        output3 = self.gat3(omic3, adj3)


        outs=torch.cat([omic1,omic2,omic3],dim=1)

        o1=self.att_b(omic1,outs)
        o2=self.att_c(omic2,outs)
        o3=self.att_d(omic3,outs)



        output1_1=self.fb1(o1)
        output1_1=self.fb2(output1_1)
        output1_1=self.fb3(output1_1)
        output1_1=self.fb4(output1_1)


        output2_2=self.fb1(o2)
        output2_2=self.fb2(output2_2)
        output2_2=self.fb3(output2_2)
        output2_2=self.fb4(output2_2)

        output3_3=self.fd1(o3)
        output3_3=self.fd2(output3_3)
        output3_3=self.fd3(output3_3)

        a=0.3
        mrna=a*output1+(1-a)*output1_1
        dna=a*output2+(1-a)*output2_2
        mirna=a*output3+(1-a)*output3_3


        mrna = self.fc5(mrna)  # (N, 2)

        dna = self.fc5(dna)  # (N, 2)

        mirna = self.fc5(mirna)  # (N, 2)'''


        x_prob = self.sigmoid(mrna)
        y_prob = self.sigmoid(dna)
        z_prob = self.sigmoid(mirna)
        evidence = dict()
        evidence[0], evidence[1], evidence[2] = x_prob, y_prob, z_prob
        loss = 0
        alpha = dict()
        for v_num in range(len(evidence)):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs)
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)

        loss = torch.mean(loss)

        return evidence, evidence_a, loss, mrna,dna,mirna