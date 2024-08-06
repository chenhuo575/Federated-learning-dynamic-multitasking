#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from models_args import Global_models

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    G=Global_models()

    for i in range(args.num_models):
        G.add_model(args=args,train_dataset=train_dataset)
    # Set the model to train and send it to device.
    G.set_models(device=device)

    print_every = 2
    val_loss_pre, counter = 0, 0
    qk=0.5
    chosen_frequency=[0 for i in range(args.num_users)]
    for epoch in tqdm(range(args.epochs)):

        print(f"第{epoch+1}轮")
        print(f'\n | Global Training Round : {epoch+1} |\n')
        m = max(int(args.frac * args.num_users), 1)
        low_freq=[]
        high_freq=[]
        if epoch>=1:
            for j in range(len(chosen_frequency)):
                if chosen_frequency[j]/epoch<qk:
                    low_freq.append(j)
                else:
                    high_freq.append(j)
        else:
            high_freq=list(range(args.num_users))
        if len(low_freq)>=m*len(G.global_models):
            chosen_users=list(np.random.choice(low_freq, m*len(G.global_models), replace=False))
        else:
            chosen_users=[j for j in low_freq]
            chosen_users+=list(np.random.choice(high_freq, m*len(G.global_models)-len(low_freq), replace=False))
        for c in chosen_users:
            chosen_frequency[c]+=1
        idxs_users = []

        print(f"完成第{epoch+1}轮分配\n")
        for i in range(len(G.global_models)):

            idxs_users.append([chosen_users[j+i*m] for j in range(5)])
            print(f"第{epoch+1}轮第{i+1}个全局模型") 

            G.global_models[i].train()
            local_weights, local_losses = [], []
            flag=1
            for idx in idxs_users[i]:

                # if idxs_users[i].index(idx)!=0:                       #方便测试只执行第一个用户的本地模型
                #     continue

                print(f"第{flag}个本地模型")
                flag+=1
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx], logger=logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(G.global_models[i]), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            
            
            # update global weights
            G.global_weights[i] = average_weights(local_weights)

            # update global weights
            G.global_models[i].load_state_dict(G.global_weights[i])

            loss_avg = sum(local_losses) / len(local_losses)
            G.train_loss_s[i].append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            G.global_models[i].eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[c], logger=logger)
                acc, loss = local_model.inference(model=G.global_models[i])
                list_acc.append(acc)
                list_loss.append(loss)
            G.train_accuracys[i].append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(G.train_loss_s[i]))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*G.train_accuracys[i][-1]))
        
        print("--客户端的选择次数： ")
        print(chosen_frequency)

    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('every model train Accuracy')
    for i in range(len(G.global_models)):
    # Test inference after completion of training
        test_acc, test_loss = test_inference(args, G.global_models[i], test_dataset)

        print(f"model_{i}:")
        print(f' \n Results after {args.epochs} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*G.train_accuracys[i][-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        
        plt.plot(range(args.epochs),G.train_accuracys[i],label=f"{i+1}models")
 

        # Saving the objects train_loss and train_accuracy:
        file_name = '../save/objects/fed_avg_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            format(i,args.dataset, args.model, args.epochs, args.frac, args.iid,
                args.local_ep, args.local_bs)

        with open(file_name, 'wb') as f:
            pickle.dump([G.train_loss_s[i], G.train_accuracys[i],test_acc], f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    plt.legend()
    plt.savefig('../save/{}轮{}基线多任务[{}]独立同分布准确率多曲线图.png'.format(args.epochs,args.dataset,args.iid))
    
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('Average Accuracy')
    avg_acc=[0 for i in range(args.epochs)]
    for i in range(len(G.global_models)):
        for j in range(len(G.train_accuracys[i])):
            avg_acc[j]+=G.train_accuracys[i][j]
    avg_acc=[i/len(G.global_models) for i in avg_acc]
    print('-----------Final Average Train Accuracy: {:.2f}% \n'.format(100*avg_acc[-1]))
    plt.plot(range(args.epochs),avg_acc,label=f"{len(G.global_models)}models_avg_acc") 
    plt.legend()
    plt.savefig('../save/{}轮{}基线多任务[{}]独立同分布准确率.png'.format(args.epochs,args.dataset,args.iid))
    # plt.savefig('../save/fed_avg_{}_{}_{}_{}avgdata_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.format(args.dataset, args.model, args.epochs, len(G.global_models),args.frac,
    #             args.iid, args.local_ep, args.local_bs))
    

    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('Training loss')
    avg_loss=[0 for i in range(args.epochs)]
    for i in range(len(G.global_models)):
        for j in range(len(G.train_loss_s[i])):
            avg_loss[j]+=G.train_loss_s[i][j]
    avg_loss=[i/len(G.global_models) for i in avg_loss]
    plt.plot(range(args.epochs),avg_loss,label=f"{len(G.global_models)}models_avg_loss")  
    plt.legend()
    plt.savefig('../save/{}轮{}基线多任务[{}]独立同分布loss.png'.format(args.epochs,args.dataset,args.iid))
    # plt.savefig('../save/fed_avg_{}_{}_{}_{}avgdata_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.format(args.dataset, args.model, args.epochs, len(G.global_models),args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    
    file_name = '../save/objects/fed_avg_{}_{}_{}_{}avgdata_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            format(args.dataset, args.model, args.epochs,len(G.global_models), args.frac, args.iid,
                args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([avg_acc, avg_loss], f)