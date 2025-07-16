import os
import copy
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from utils import load_model_dict
from models1 import init_model_dict
from train_test import *

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def cal_feat_imp(data_folder, view_list, num_class):

    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_he_list = [200, 200, 200, 100]
        a = 200
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [1000, 1000, 503, 200]
        a = 1000
    if data_folder == 'KIPAN':
        adj_parameter = 10
        dim_he_list = [2000, 2000, 445, 200]
        a = 2000
    if data_folder == 'LGG':
        adj_parameter = 10
        dim_he_list = [2000, 2000, 548, 200]
        a = 2000

    data_tr_list, data_te_list, data_trte_list, trte_idx, labels_tr, labels_te, labels = prepare_trte_data1(data_folder,

    te_data = torch.cat([data_te_list[0], data_te_list[1], data_te_list[2]], dim=1)
    print(te_data.shape)
    print(
        f"data_tr_list0{data_tr_list[0].shape}data_tr_list1{data_tr_list[1].shape}data_tr_list2{data_tr_list[2].shape}")
    print(
        f"data_trte_list0{data_te_list[0].shape}data_trte_list1{data_te_list[1].shape}data_trte_list2{data_te_list[2].shape}")
    te_dataset = torch.utils.data.TensorDataset(te_data, labels_te)
    te_data_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=64, shuffle=False)


    network = Fusion(data_name=data_folder, classes=num_class, views=num_view, dim_list=dim_he_list, lambda_epochs=50)
    model_path = os.path.join('./Lmodels.pth')
    checkpoint = torch.load(model_path)
    network.load_state_dict(checkpoint['net'])
    network.eval()
    network.to(device)

    all_targets = torch.tensor([], dtype=torch.long, device=device)
    all_predictions = torch.tensor([], dtype=torch.long, device=device)
    with torch.no_grad():
        for k, data in enumerate(te_data_loader, 0):
            batch_x, targets = data
            batch_x1 = batch_x[:, 0:a]
            batch_x2 = batch_x[:, a:a * 2]
            batch_x3 = batch_x[:, a * 2:]
            batch_te_list = [batch_x1, batch_x2, batch_x3]
            # print(f"batch_tr_list{batch_te_list}")
            adj_te_list = gen_trte_adj_mat1(batch_te_list, adj_parameter)
            batch_x1 = batch_x1.to(torch.float32)
            batch_x2 = batch_x2.to(torch.float32)
            batch_x3 = batch_x3.to(torch.float32)

            targets = targets.long()
            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)
            batch_x3 = batch_x3.to(device)
            targets = targets.to(device)

            exp_adj1 = adj_te_list[0]
            exp_adj2 = adj_te_list[1]
            exp_adj3 = adj_te_list[2]

            exp_adj1 = exp_adj1.to(device)
            exp_adj2 = exp_adj2.to(device)
            exp_adj3 = exp_adj3.to(device)
            evidences, evidence_a, loss_tmc, output1, output2, output3 = network(batch_x1, batch_x2, batch_x3,
                                                                                 exp_adj1, exp_adj2, exp_adj3, targets)
            te_pre_lab = torch.argmax(evidence_a, 1)

            all_targets = torch.cat((all_targets, targets), dim=0)
            all_predictions = torch.cat((all_predictions, te_pre_lab), dim=0)


    all_targets_cpu = all_targets.detach().cpu().numpy()
    all_predictions_cpu = all_predictions.detach().cpu().numpy()


    if num_class == 2:
        f1 = f1_score(all_targets_cpu, all_predictions_cpu, average='weighted')
        print(f1)
        # 如果是二分类问题，计算 F1 分数。
    else:
        f1 = f1_score(all_targets_cpu, all_predictions_cpu, average='macro')
        print(f1)
        # 对于多分类问题，计算宏平均 F1 分数。

    feat_imp_list = []

    featname_list = []

    for v in view_list:
        df = pd.read_csv(os.path.join(data_folder, str(v) + "_featname.csv"), header=None)
        featname_list.append(df.values.flatten())


    dim_list = [x.shape[1] for x in data_tr_list]

    print(len(featname_list))
    for i in range(len(featname_list)):
        feat_imp = {"feat_name": featname_list[i]}

        feat_imp['imp'] = np.zeros(dim_list[i])



        for j in range(dim_list[i]):
            feat_tr = data_tr_list[i][:, j].clone()
            feat_trte = data_te_list[i][:, j].clone()

            data_tr_list[i][:, j] = 0
            data_te_list[i][:, j] = 0


            te_data = torch.cat([data_te_list[0], data_te_list[1], data_te_list[2]], dim=1)
            te_dataset = torch.utils.data.TensorDataset(te_data, labels_te)
            te_data_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=64, shuffle=False)
            a_targets = torch.tensor([], dtype=torch.long, device=device)
            a_predictions = torch.tensor([], dtype=torch.long, device=device)
            with torch.no_grad():
                for k, data in enumerate(te_data_loader, 0):
                    batch_x, targets = data
                    batch_x1 = batch_x[:, 0:a]
                    batch_x2 = batch_x[:, a:a * 2]
                    batch_x3 = batch_x[:, a * 2:]
                    batch_te_list = [batch_x1, batch_x2, batch_x3]

                    adj_te_list = gen_trte_adj_mat1(batch_te_list, adj_parameter)
                    batch_x1 = batch_x1.to(torch.float32)
                    batch_x2 = batch_x2.to(torch.float32)
                    batch_x3 = batch_x3.to(torch.float32)

                    targets = targets.long()
                    batch_x1 = batch_x1.to(device)
                    batch_x2 = batch_x2.to(device)
                    batch_x3 = batch_x3.to(device)
                    targets = targets.to(device)

                    exp_adj1 = adj_te_list[0]
                    exp_adj2 = adj_te_list[1]
                    exp_adj3 = adj_te_list[2]

                    exp_adj1 = exp_adj1.to(device)
                    exp_adj2 = exp_adj2.to(device)
                    exp_adj3 = exp_adj3.to(device)
                    evidences, evidence_a, loss_tmc, output1, output2, output3 = network(batch_x1, batch_x2, batch_x3,
                                                                                         exp_adj1, exp_adj2, exp_adj3,
                                                                                         targets)
                    te_pre_lab = torch.argmax(evidence_a, 1)

                    a_targets = torch.cat((a_targets, targets), dim=0)
                    a_predictions = torch.cat((a_predictions, te_pre_lab), dim=0)


            a_targets_cpu = a_targets.detach().cpu().numpy()
            a_predictions_cpu = a_predictions.detach().cpu().numpy()


            if num_class == 2:
                f1_tmp = f1_score(a_targets_cpu, a_predictions_cpu, average='weighted')
            # 如果是二分类问题，计算 F1 分数。
            else:
                f1_tmp = f1_score(a_targets_cpu, a_predictions_cpu, average='macro')
            # 对于多分类问题，计算宏平均 F1 分数。

            feat_imp['imp'][j] = (f1 - f1_tmp) * dim_list[i]


            data_tr_list[i][:, j] = feat_tr.clone()
            data_te_list[i][:, j] = feat_trte.clone()


        feat_imp_list.append(pd.DataFrame(data=feat_imp))


    return feat_imp_list



def summarize_imp_feat(featimp_list_list, topn=30):
    num_rep = len(featimp_list_list)
    num_view = len(featimp_list_list[0])
    df_tmp_list = []
    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
        df_tmp_list.append(df_tmp.copy(deep=True))
    df_featimp = pd.concat(df_tmp_list).copy(deep=True)
    for r in range(1, num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
            df_featimp = df_featimp.append(df_tmp.copy(deep=True), ignore_index=True)
    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum()
    df_featimp_top = df_featimp_top.reset_index()
    df_featimp_top = df_featimp_top.sort_values(by='imp', ascending=False)
    df_featimp_top = df_featimp_top.iloc[:topn]
    print('{:}\t{:}'.format('Rank', 'Feature name'))
    for i in range(len(df_featimp_top)):
        print('{:}\t{:}'.format(i + 1, df_featimp_top.iloc[i]['feat_name']))


def summarize_imp_feat1(featimp_list_list, topn=10):
    num_rep = len(featimp_list_list)
    num_view = len(featimp_list_list[0])
    df_tmp_list = []
    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
        df_tmp_list.append(df_tmp.copy(deep=True))
    df_featimp = pd.concat(df_tmp_list).copy(deep=True)
    for r in range(1, num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
            df_featimp = df_featimp.append(df_tmp.copy(deep=True), ignore_index=True)
    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum()
    df_featimp_top = df_featimp_top.reset_index()


    df_featimp_top_list = []
    for v in range(num_view):
        df_view_top = df_featimp_top[df_featimp_top['omics'] == v].sort_values(by='imp', ascending=False).head(topn)
        df_featimp_top_list.append(df_view_top)


    for v in range(num_view):
        print(f"View {v}:")
        print('{:}\t{:}'.format('Rank', 'Feature name'))
        for i in range(len(df_featimp_top_list[v])):
            print('{:}\t{:}'.format(i + 1, df_featimp_top_list[v].iloc[i]['feat_name']))

    return df_featimp_top_list
