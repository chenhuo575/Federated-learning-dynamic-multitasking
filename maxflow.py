class Graph:
    def __init__(self, num):
        self.data_li = [['inf' for _ in range(num)] for _ in range(num)]          # 用于记录最短路径算法记录每条边的选择依据
        self.mark = []
        self.distance = ['inf' for _ in range(num)]
        self.path = ['inf' for _ in range(num)]                 # 记录前驱结点
        self.data_f_li = [[('inf', 0) for _ in range(num)] for _ in range(num)]   # 用于记录实际的流量变化情况
        self.expend_li = [['inf' for _ in range(num)] for _ in range(num)]        # 记录每条边的代价（费用）

    def add_edge(self, data, data_f):         # 这里data 中包含的是定点和当前流的信息。data_f 表示的是定点和允许的最大流量信息
        for i in data:
            self.data_li[i[0]][i[1]] = i[2]
            self.expend_li[i[0]][i[1]] = i[2]
        for i in data_f:
            self.data_f_li[i[0]][i[1]] = (i[2], 0)
        
    def dijkstra(self, start, num):                     # 计算最短距离路径
        self.mark = []                                  # 用于记录已经标记过的点
        self.distance = ['inf' for i in range(num)]     
        self.path = ['inf' for i in range(num)]
        self.mark.append(start)
        self.distance[start] = 0
        que = []
        que.append(start)
        while que:
            cur_node = que.pop(0)
            for i in range(len(self.data_li[cur_node])):
                if i not in self.mark:
                    if self.distance[i] == 'inf':
                        if self.data_li[cur_node][i] == 'inf':
                            continue
                        else:
                            self.distance[i] = self.distance[cur_node] + self.data_li[cur_node][i]
                            self.path[i] = cur_node
                            continue
                    if self.data_li[cur_node][i] == 'inf':
                        continue
                    else:
                        if self.distance[cur_node] + self.data_li[cur_node][i] < self.distance[i]:
                            self.distance[i] = self. distance[cur_node] + self.data_li[cur_node][i]
                            self.path[i] = cur_node
            cur_min_val = [self.distance[i] for i in range(len(self.distance)) if i not in self.mark and self.distance[i] != 'inf']
            if cur_min_val:
                cur_min_val = min(cur_min_val)
                for i in range(len(self.distance)):
                    if i not in self.mark:
                        if self.distance[i] == cur_min_val:
                            self.mark.append(i)
                            que.append(i)
        if self.path[-1] == 'inf':
            return
        shortest_path = []
        shortest_path.append(len(self.path)-1)
        a = len(self.path)-1                 # 利用a 递归寻找最短路径
        while a:
            if self.path[a] == 'inf':
                break
            else:
                shortest_path.insert(0,self.path[a])
                a = self.path[a]
        return shortest_path

    def get_sum_path(self, start, num):               # 统计所有最短路径
        s_path = self.dijkstra(start, num)
        sum_path=[]
        while s_path:
            sum_path.append(s_path)
            min_val = min([self.data_f_li[s_path[i]][s_path[i + 1]][0] -
                           self.data_f_li[s_path[i]][s_path[i + 1]][1] for i in range(len(s_path) - 1) if self.data_f_li[s_path[i]][s_path[i + 1]][0] != 'inf' ])
            for i in range(len(s_path) - 1):
                self.data_f_li[s_path[i]][s_path[i + 1]] = (
                self.data_f_li[s_path[i]][s_path[i + 1]][0],
                self.data_f_li[s_path[i]][s_path[i + 1]][1] + min_val)
            for i in range(len(s_path) - 1):
                if self.data_f_li[s_path[i]][s_path[i + 1]][0] == \
                        self.data_f_li[s_path[i]][s_path[i + 1]][1]:
                    self.data_li[s_path[i]][s_path[i + 1]] = 'inf'
            s_path = self.dijkstra(start, num)
        return sum_path