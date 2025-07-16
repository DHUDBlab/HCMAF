import logging
import time
import matplotlib.pyplot as plt

import pandas as pd
import random
import pickle
import copy
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from Lmodels import *
from utils import *

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



def prepare_trte_data1(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    label_tr=torch.tensor(labels_tr,dtype=torch.int)
    label_te=torch.tensor(labels_te,dtype=torch.int)

    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))


    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]

    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))

    num_mat=data_mat_list[0].shape[0]

    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].to(device)

    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_test_list=[]
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))


    labels = torch.cat([label_tr, label_te])
    return data_train_list, data_test_list,data_all_list, idx_dict, label_tr,label_te,labels


def gen_trte_adj_mat1(data_list, adj_parameter):
    adj_metric = "cosine"
    adj_list = []
    for i in range(len(data_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter1(adj_parameter, data_list[i], adj_metric)
        adj_list.append(gen_adj_mat_tensor(data_list[i], adj_parameter_adaptive, adj_metric))

    return adj_list


def train_test1(data_folder, view_list, num_class, num_epoch):


    num_view = len(view_list)

    dim_hvcdn = pow(num_class,num_view)
    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_he_list = [200,200,200,100]
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [1000,1000,503,200]

    if data_folder == 'KIPAN':
        adj_parameter = 10
        dim_he_list = [2000,2000,445,200]
    if data_folder == 'LGG':
        adj_parameter = 10
        dim_he_list = [2000,2000,548,200]
    data_tr_list,data_te_list ,data_trte_list, trte_idx, labels_tr,labels_te,labels = prepare_trte_data1(data_folder, view_list)
    tr_data=torch.cat([data_tr_list[0],data_tr_list[1],data_tr_list[2]],dim=1)
    te_data=torch.cat([data_te_list[0],data_te_list[1],data_te_list[2]],dim=1)
    print(tr_data.shape)
    print(f"data_tr_list0{data_tr_list[0].shape}data_tr_list1{data_tr_list[1].shape}data_tr_list2{data_tr_list[2].shape}")
    print(f"data_trte_list0{data_te_list[0].shape}data_trte_list1{data_te_list[1].shape}data_trte_list2{data_te_list[2].shape}")



    tr_dataset = torch.utils.data.TensorDataset(tr_data, labels_tr)
    tr_data_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=32, shuffle=True)
    te_dataset = torch.utils.data.TensorDataset(te_data, labels_te)
    te_data_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=32, shuffle=False)


    num_epochs = num_epoch
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    loss_function = nn.CrossEntropyLoss()
    network = Fusion(data_name=data_folder,classes=num_class, views=num_view, dim_list=dim_he_list,lambda_epochs=50)
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000], gamma=0.1)
    best_model_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0
    best_epoch = 0
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    best_f1 = 0.0
    best_auc = 0.0
    best_f1w=0.0
    best_f1m=0.0

    for epoch in range(0, num_epochs):

        print(' Epoch {}/{}'.format(epoch, num_epochs - 1))
        print("-" * 10)
        network.train()
        current_loss = 0.0
        train_loss = 0.0
        train_corrects = 0
        train_num = 0

        if data_folder == 'ROSMAP':
            a = 200
        if data_folder == 'BRCA':
            a=1000
        if data_folder == 'KIPAN':
            a=2000
        if data_folder == 'LGG':
            a=2000

        for i, data in enumerate(tr_data_loader, 0):
            batch_x, targets = data

            batch_x1=batch_x[:, 0:a]
            batch_x2=batch_x[:, a:a*2]
            batch_x3=batch_x[:, a*2:]
            batch_tr_list=[batch_x1,batch_x2,batch_x3]

            adj_tr_list= gen_trte_adj_mat1(batch_tr_list,  adj_parameter)

            batch_x1 = batch_x1.to(torch.float32)
            batch_x2 = batch_x2.to(torch.float32)
            batch_x3 = batch_x3.to(torch.float32)

            targets = targets.long()
            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)
            batch_x3 = batch_x3.to(device)
            targets = targets.to(device)
            exp_adj1 = adj_tr_list[0]
            exp_adj2 = adj_tr_list[1]
            exp_adj3 = adj_tr_list[2]
            exp_adj1 = exp_adj1.to(device)
            exp_adj2 = exp_adj2.to(device)
            exp_adj3 = exp_adj3.to(device)

            optimizer.zero_grad()
            evidences, evidence_a, loss_tmc, output1, output2, output3 = network(batch_x1, batch_x2, batch_x3, exp_adj1, exp_adj2, exp_adj3, targets, epoch)
            pre_lab = torch.argmax(evidence_a, 1)

            loss1 = loss_function(output1, targets)
            loss2 = loss_function(output2, targets)
            loss3 = loss_function(output3, targets)
            loss = loss_tmc+loss1+loss2+loss3

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x1.size(0)
            train_corrects += torch.sum(pre_lab == targets.data)
            train_num += batch_x1.size(0)

        network.eval()
        test_loss = 0.0
        test_corrects = 0
        test_num = 0

        all_targets = torch.tensor([], dtype=torch.long, device=device)
        all_predictions = torch.tensor([], dtype=torch.long, device=device)
        all_evidences = torch.tensor([], dtype=torch.float, device=device)
        with torch.no_grad():
            for i, data in enumerate(te_data_loader, 0):
                batch_x, targets = data
                batch_x1=batch_x[:, 0:a]
                batch_x2=batch_x[:, a:a*2]
                batch_x3=batch_x[:, a*2:]
                batch_te_list=[batch_x1,batch_x2,batch_x3]

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
                                                                                    exp_adj1, exp_adj2, exp_adj3, targets, epoch)
                te_pre_lab = torch.argmax(evidence_a, 1)

                loss1 = loss_function(output1, targets)
                loss2 = loss_function(output2, targets)
                loss3 = loss_function(output3, targets)
                loss = loss_tmc + loss1 + loss2 + loss3

                test_loss += loss.item() * batch_x1.size(0)
                test_corrects += torch.sum(te_pre_lab == targets.data)
                test_num += batch_x1.size(0)

                all_targets = torch.cat((all_targets, targets), dim=0)
                all_predictions = torch.cat((all_predictions, te_pre_lab), dim=0)
                all_evidences = torch.cat((all_evidences, evidence_a[:, 1]), dim=0)


        all_targets_cpu = all_targets.detach().cpu().numpy()
        all_predictions_cpu = all_predictions.detach().cpu().numpy()
        all_evidences_cpu = all_evidences.detach().cpu().numpy()
        if num_class==2:
            f1 = f1_score(all_targets_cpu, all_predictions_cpu, average='weighted')
            auc = roc_auc_score(all_targets_cpu, all_evidences_cpu)
            if f1>best_f1:
                best_f1 = f1
            if auc>best_auc:
                best_auc=auc
        else :
            f1_weighted = f1_score(all_targets_cpu, all_predictions_cpu, average='weighted')
            f1_macro = f1_score(all_targets_cpu, all_predictions_cpu, average='macro')
            if f1_weighted > best_f1w:
                best_f1w = f1_weighted
            if f1_macro > best_f1m:
                best_f1m = f1_macro


        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        test_loss_all.append(test_loss / test_num)
        test_acc_all.append(test_corrects.double().item() / test_num)
        print('{} Train ACC : {:.3f}'.format(epoch, train_acc_all[-1]))
        print('{} Test ACC : {:.3f}'.format(epoch,  test_acc_all[-1]))
        print('Best Test ACC : {:.3f}'.format( best_acc))
        if num_class==2:
            print(f'Best F1 Score: {best_f1:.3f}')
            print(f'Best AUC Score: {best_auc:.3f}')
        else:
            print(f'Best F1 W: {best_f1w:.3f}')
            print(f'Best F1 M: {best_f1m:.3f}')


        if test_acc_all[-1] > best_acc:
            best_acc = test_acc_all[-1]

            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(network.state_dict())

            save_path = f'./Bmodels.pth'
            state = {
                'net': best_model_wts,
                'epoch': best_epoch - 1,
                'loss': test_loss_all[best_epoch-1]
            }
            torch.save(state, save_path)

    print('num of epoch: {0}'.format(epoch))
    print('Best val Acc: {:.3f} Best epoch {:04d}'.format(best_acc, best_epoch-1))
    if num_class==2:
        print(f'Best F1 Score: {best_f1:.3f}')
        print(f'Best AUC Score: {best_auc:.3f}')
    else:
        print(f'Best F1 W: {best_f1w:.3f}')
        print(f'Best F1 M: {best_f1m:.3f}')
    print('end')



