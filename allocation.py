import maxflow
import KM_min
import matplotlib.pyplot as plt
import numpy as np
import pickle


class allocate: 

    def __init__(self,n,m,weights,re_m=5,c=0,F=0,E=0,e=0,t=0,V=0,Q=0,allo_type="bipartite"):                           #包含各项输入的初始化
        self.V1,self.V2,self.E=[],[],[]                             #V1，V2，E分别为已完成的任务的集合，已完成的设备的集合，已完成的边的集合
        self.n,self.m=n,m                                           #n是用户数，m是任务数
        self.re_m=re_m                                              #任务的感知要求
        self.X=[[0 for i in range(m)] for i in range(n)]            #时间槽t中 X[i][j]为任务i到设备j是否可行，初始化为0不可行
        self.G=maxflow.Graph(n+m+2)                                 #初始化G图
        # connum=0.01                                               #各项输入格式
        # Fairness=[]
        # Energ=[]
        # e=[[0 for i in range(self.m)] for i in range(self.n)]
        # t=[]
        # V=
        # Q=[[0 for i in range(self.m)] for i in range(self.n)]
        self.connum=c                                               #常数γ
        self.Fairness=F                                             #时隙 t 的公平队列
        self.Energ=E                                                #时隙 t 的能量队列
        self.e=e                                                    #任务i到设备j的能量花费
        self.t=t                                                    #任务i的截止日期
        self.V=V                                                    #一个控制原始优化目标的重要性，可能是常数，我不确定
        self.Q=Q                                                    #时隙 t 任务i到设备j的群感质量
        self.allo_type=allo_type
        self.weights=weights
    
    def answer_user_groups(self,mid_answer):
        user_groups=[[] for i in range(self.m)]
        if self.allo_type == "maxflow":
            for x in self.X:
                user_groups.append([])
                for i in range(len(x)):
                    if x[i] == 1 :
                        user_groups[-1].append(i)
        elif self.allo_type=="bipartite":
            for mid in mid_answer:
                self.X[mid[0]][int(mid[1]/self.re_m)]=1
                user_groups[int(mid[1]/self.re_m)].append(mid[0])
        return user_groups
            
        

    def formula_wij(self,i,j):                                      #使用公式计算V上标1 i到V上标2 2的权重
        #answer_wieght=connum*Fairness[j]-Energ[j]*e[i][j]-V/t[i]-V*Q[i][j]   原公式暂放，方便后续改动公式使用
        return i*10+j                                               #公式未确定用简单式子替代结果
    
    def process(self):
        if self.allo_type=="maxflow":
            sr=self.srlist                                          
            data=[]                                                     #分别遍历V0到V上标1，Vd到V上标2，V上标1到V上标2的边，并分别将距离和权重作为单位费用和距离添加到data_f和data中
            data_f=[]                                                   #根据论文要求需要获取最小权重最大流，而导入Graph方法中计算的是最短距离路径，所以需要将设备和任务的权重作为距离导入Graph
            for i in range(self.n):                                     #遍历V0到所有V上标1(即所有任务)的边并添加至data和data_f中
                # self.V1.append(i)
                # self.E.append((0,i+1))                                
                # self.cap_0_i[i]=(0,sr[i])
                #
                data.append((0,i+1,0))
                data_f.append((0,i+1,sr[i]))
            for j in range(self.m):                                    #遍历Vd到所有V上标2(即所有设备)的边并添加至data和data_f中
                # self.V2.append(j)
                # self.E.append((self.n+1+j,self.n+self.m+1))
                # self.cap_j_d[j]=(0,'inf')
                #
                data.append((self.n+1+j,self.n+self.m+1,0))
                data_f.append((self.n+1+j,self.n+self.m+1,'inf'))
                
            for i in range(self.n):                                    #遍历所有V上标1(即所有任务)到所有V上标2(即所有设备)的边并添加至data和data_f中
                for j in range(self.m):
                    # self.E.append((i+1,self.n+1+j))
                    # self.cap_i_j[i][j]=(0,1)
                    # self.w_i_j[i][j]=self.formula_wij(i,j)
                    data.append((i+1,self.n+1+j,self.formula_wij(i,j)))
                    data_f.append((i+1,self.n+1+j,1))
            self.G.add_edge(data,data_f)                              #将所有边添加至G中
            F=self.G.get_sum_path(0,self.n+self.m+2)                  #计算所有最小权重最大流路径并保存在F中
            for i in range(self.n):
                for j in range(self.m): 
                    if [0,i+1,self.n+1+j,self.n+self.m+1] in F:       #如果路径在F中存在将对应的Xi,j设为1
                        self.X[i][j]=1
            return self.answer_user_groups()
        elif self.allo_type=="bipartite":
            values=[]
            for j in range(len(self.weights)):
                for i in range(len(self.weights[0])):
                    for m in range(self.re_m):
                        values.append((j,i*self.re_m+m,self.weights[j][i]))         
            
            
            mid_answer=KM_min.run_kuhn_munkres(values)
            print("选择的客户端结果是： ")
            print(self.answer_user_groups(mid_answer))

            return self.answer_user_groups(mid_answer),self.X
        
    