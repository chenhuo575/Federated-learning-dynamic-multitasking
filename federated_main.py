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
from allocation import allocate
from models_args import Global_models

def fore_update(E,F,en,weights,date,r,V=3,a=1,y=10):
    max_weight=weights[0][0]
    for j in range(len(weights)):
        for i in range(len(weights[0])):
            max_weight=max(max_weight,weights[j][i])
            if date[i]>0:
                weights[j][i]=round(y*F[j]-E[j]*en[j][i]-V*r[j]/date[i]-a*V,2)

    for j in range(len(weights)):
        for i in range(len(weights[0])):        
            if date[i]<=0:
                weights[j][i]=max_weight+500
    
    print("截止时间: ")
    print(date)
    print("声誉: ")
    print(r)
    print("能量： ")
    print(en)
    print("两个队列的值:")          
    print(E)
    print(F)
    for j in range(len(weights)):
        for i in range(len(weights[0])):
            print(f"{j}-{i} 第一部分：{round(y*F[j]-E[j]*en[j][i],2)}   第二部分: {round(V*r[j]/(date[i] if date[i] != 0 else 1) + a*V, 2)}")
    print("权重的值是：")
    print(weights)
    
    for i in range(len(date)):
        date[i]-=1
    return weights,date

def E_F_update(E,F,en,X,B=0.35,h=3):
    
    for j in range(len(X)):
        sum_t=0
        sum_x=0
        for i in range(len(X[0])):           
            sum_t+=X[j][i]*en[j][i]
            sum_x+=X[j][i]
        E[j]=max(0,E[j]+h-sum_t)
        F[j]=round(max(0,F[j]-B+sum_x),2)
        
    return E,F

def acc_compar(a):
    if a==1:
        a=0.99999999
    return (a/(1.00001-a))*100

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
    #创建各项参数，E，F是更新维护的权重参数，date是模型剩余轮次作为截止日期，ru为用户声誉值
    #X为用户j至模型i是否选择的表形式，eng是用户j执行模型i所消耗的能量，weights是权重
    E=[0 for i in range(args.num_users)]
    F=[0 for i in range(args.num_users)]
    date=[args.epochs for i in range(args.num_models)]                                             
    r=[30 for i in range(args.num_users)]
    X=[[0 for j in range(args.num_models)] for i in range(args.num_users)]
    eng=[[random.randint(2, 5) for j in range(args.num_models)] for i in range(args.num_users)]
    weights=[[0 for j in range(args.num_models)] for i in range(args.num_users)]
    #sum_epochs为总轮次，倘若中途增加模型，总轮次会增加,curmax_epoch为进度条当前进度
    sum_epochs=list(range(args.epochs))
    sum_progress=tqdm(total=args.epochs)
    curmax_epoch=0
    max_da=0
    chosen_frequency=[0 for i in range(args.num_users)]
    for epoch in (sum_epochs):
        #检查剩余轮次最多的模型，最大已进行轮次大于进度条再更新进度条
        for da in date:
            max_da=max(da,max_da)
        if curmax_epoch+max_da>=args.epochs:
            sum_progress.update(1)
            curmax_epoch+=1

        if len(G.global_models)<5:               #增加任务代码，暂设条件1000轮避免触发
            G.add_model(args=args,train_dataset=train_dataset)
            G.set_model(device=device,i=-1)
            date.append(args.epochs)
            leave_epoch=len(sum_epochs)-epoch
            for i in range(date[-1]-leave_epoch):
                sum_epochs.append(sum_epochs[-1]+1)
            for i in range(args.num_users):
                X[i].append(0)
                eng[i].append(random.randint(2, 5))
                weights[i].append(0)
            

        print(f"第{epoch+1}轮")
        print(f'\n | Global Training Round : {epoch+1} |\n')
        m = max(int(args.frac * args.num_users), 1)
        #weights_date_update更新权重，日期，权重是本轮次所需权重，日期则作为下轮权重计算所需提前更新
        weights,date=fore_update(E=E,F=F,en=eng,weights=weights,date=date,r=r)
        
        idxs_users,X=allocate(m=len(G.global_models),n=args.num_users,weights=weights,re_m=5).process()   
        #更新E，F参数
        E,F=E_F_update(E=E,F=F,en=eng,X=X)

        print(f"完成第{epoch+1}轮分配\n")
        for i in range(len(G.global_models)):
            
            print(f"第{epoch+1}轮第{i+1}个全局模型") 
            if date[i] < 0:                                           #检查该模型是否执行完成所有轮次，完成则跳过该模型
                print("该模型已完成所需轮次\n")
                continue
            G.global_models[i].train()
            local_weights, local_losses = [], []
            for idx in idxs_users[i]:

                # if idxs_users[i].index(idx)!=0:                       #方便测试只执行第一个用户的本地模型
                #     continue
                chosen_frequency[idx]+=1

                print(f"第{idxs_users[i].index(idx)+1}个本地模型")
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

            if args.fame_update :
                for idx in idxs_users[i]:
                    if len(G.train_accuracys[i])>=2:
                        r[idx]+=acc_compar(G.train_accuracys[i][-1])-acc_compar(G.train_accuracys[i][-2])
                    else:
                        r[idx]+=acc_compar(G.train_accuracys[i][-1])
                    r[idx]=round(r[idx],3)

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Training Loss : {np.mean(np.array(G.train_loss_s[i]))}')
                print('Train Accuracy: {:.2f}% \n'.format(100*G.train_accuracys[i][-1]))
    
        print("选择客户端的次数为： ")
        print(chosen_frequency)

    for i in range(len(G.global_models)):
    # Test inference after completion of training
        test_acc, test_loss = test_inference(args, G.global_models[i], test_dataset)

        print(f"model_{i}:")
        print(f' \n Results after {sum_epochs[-1]+1} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*G.train_accuracys[i][-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))


        # Saving the objects train_loss and train_accuracy:
        file_name = '../save/objects/fed_improves_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
            format(i,args.dataset, args.model, args.epochs, args.frac, args.iid,
                args.local_ep, args.local_bs)

        with open(file_name, 'wb') as f:
            pickle.dump([G.train_loss_s[i], G.train_accuracys[i]], f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    #分别保存训练精度和损失折线图
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
    plt.savefig('../save/{}轮{}动态多任务[{}]独立同分布准确率.png'.format(args.epochs,args.dataset,args.iid))
    # plt.savefig('../save/fed_improve_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.format(args.dataset, args.model, args.epochs, len(G.global_models),args.frac,
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
    plt.savefig('../save/{}轮{}动态多任务[{}]独立同分布loss.png'.format(args.epochs,args.dataset,args.iid))
    # plt.savefig('../save/fed_improve_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.format(args.dataset, args.model, args.epochs, len(G.global_models),args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    
    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))







