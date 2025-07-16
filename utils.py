import os
import torch
import torch.nn.functional as F

import numpy as np
import math
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset



from sklearn.metrics import auc, f1_score, roc_curve, precision_score, recall_score, cohen_kappa_score
from sklearn.preprocessing import LabelBinarizer

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)

    count = np.zeros(num_class)

    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)

    return sample_weight


def one_hot_tensor(y, num_dim):

    y_onehot = torch.zeros(y.shape[0], num_dim)

    y_onehot.scatter_(1, y.view(-1, 1), 1)

    return y_onehot



def cosine_distance_torch(x1, x2=None, eps=1e-8):

    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)

    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)

    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]

    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)



    if len(indices.shape) == 0:
        return sparse_tensortype(*x.shape)

    indices = indices.t()

    values = x[tuple(indices[i] for i in range(indices.shape[0]))]

    return sparse_tensortype(indices, values, x.size())




def cal_adj_mat_parameter1(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"


    dist = cosine_distance_torch(data, data)


    sorted_distances, _ = torch.sort(dist.reshape(-1, ))


    k = min(edge_per_node * data.shape[0], sorted_distances.numel())


    parameter = sorted_distances[k - 1]
    return np.asscalar(parameter.data.cpu().numpy())


def graph_from_dist_tensor(dist, parameter, self_dist=True):

    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"

    g = (dist <= parameter).float()

    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0

    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):

    assert metric == "cosine", "Only cosine distance implemented"


    dist = cosine_distance_torch(data, data)

    g = graph_from_dist_tensor(dist, parameter, self_dist=True)


    if metric == "cosine":
        adj = 1 - dist
    else:
        raise NotImplementedError


    adj = adj * g

    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.to(device)
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)

    adj = to_sparse(adj)

    return adj


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == 'ELU':
        act_layer = nn.ELU()
    elif act_type == 'LeakyReLU':
        act_layer = nn.LeakyReLU(0.5)
    elif act_type == 'GELU':
        act_layer = nn.GELU()
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer


