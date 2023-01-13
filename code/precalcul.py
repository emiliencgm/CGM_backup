"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import world
import torch
import numpy as np
import torch_scatter
from dataloader import dataset
import networkx as nx
from tqdm import tqdm
import torch.nn.functional as F
import os
import time
from scipy.sparse import csr_matrix

#=============================================================Overall Precalculate============================================================#
class precalculate():
    def __init__(self, config, dataset):
        
        self.P = Pop(dataset)
        self.C = None
        self.CN = None
        
        if config['adaptive_method'] in ['centroid']:
            self.C = Centroid(dataset, self.P)
        if config['adaptive_method'] in ['commonNeighbor']:
            self.CN = CommonNeighbor(dataset)
        
    @property
    def popularity(self):
        return self.P
    
    @property
    def centroid(self):
        return self.C

    @property
    def common_neighbor(self):
        return self.CN
        
#=============================================================Popularity============================================================#
class Pop():
    """
    precalculate popularity of users and items
    """
    def __init__(self, dataset:dataset):
        self.TrainPop_item = dataset.TrainPop_item #item's popularity (degree) in the training dataset
        self.TrainPop_user = dataset.TrainPop_user #user's popularity (degree) in the training dataset
        self.num_item = dataset.m_items
        self.num_user = dataset.n_users
        self.UInet = dataset.UserItemNet
        self.testDict = dataset.testDict

        #self.pop_statistic()
        self._ItemPopGroupDict, self._reverse_ItemPopGroupDict, self._testDict_PopGroup = self.build_pop_item()
        self._UserPopGroupDict, self._reverse_UserPopGroupDict = self.build_pop_user()
        #self.pop_label()
        self._pop_bias_Dict = self.pop_bias()

    @property
    def ItemPopGroupDict(self):
        '''
        {
            group0 : tensor([item0, ..., itemN])
            group9 : tensor([item0, ..., itemM])
        }
        '''
        return self._ItemPopGroupDict

    @property
    def reverse_ItemPopGroupDict(self):
        '''
        {
            item0 : groupN
            item9 : groupM
        }
        '''
        return self._reverse_ItemPopGroupDict

    @property
    def testDict_PopGroup(self):
        '''
        {
            group0 : {user0 : [item0, ..., itemN ],  user9 : [item0, ..., itemM ]}
            group9 : {user0 : [item0, ..., itemN'],  user9 : [item0, ..., itemM']}
        }
        '''
        return self._testDict_PopGroup

    @property
    def UserPopGroupDict(self):
        '''
        {
            group0 : tensor([user0, ..., userN])
            group9 : tensor([user0, ..., userM])
        }
        '''
        return self._UserPopGroupDict

    @property
    def reverse_UserPopGroupDict(self):
        '''
        {
            user0 : groupN
            user9 : groupM
        }
        '''
        return self._reverse_UserPopGroupDict
        
    @property
    def pop_bias_Dict(self):
        '''
        Not Implemented
        '''
        return self._pop_bias_Dict

    @property
    def item_pop_group_label(self):
        '''
        [pop_group_of_item_0, ..., pop_group_of_item_999]
        '''
        return self._item_pop_label
    
    @property
    def item_pop_degree_label(self):
        '''
        [pop_degree_of_item_0, ..., pop_degree_of_item_999]
        '''
        return self._item_pop
    
    @property
    def user_pop_group_label(self):
        '''
        [pop_group_of_user_0, ..., pop_group_of_user_999]
        '''
        return self._user_pop_label

    @property
    def user_pop_degree_label(self):
        '''
        [pop_degree_of_user_0, ..., pop_degree_of_user_999]
        '''
        return self._user_pop


    def build_pop_item(self):
        num_group = world.config['pop_group']
        item_per_group = int(self.num_item / num_group)
        TrainPopSorted = sorted(self.TrainPop_item.items(), key=lambda x: x[1])
        ItemPopGroupDict = {}#查询分组中有哪些item的字典
        testDict_PopGroup = {}#查询不同分组下用户在Test集中交互过的item的字典
        reverse_ItemPopGroupDict = {}#查询item属于哪个分组的字典
        self._item_pop_label = [0]*self.num_item
        self._item_pop = [0]*self.num_item
        #按照Pop分组，并存储至字典[0=Cold, 9=Hot]
        for group in range(num_group):
            ItemPopGroupDict[group] = []
            if group == num_group-1:
                for item, pop in TrainPopSorted[group * item_per_group:]:
                    ItemPopGroupDict[group].append(item)
                    reverse_ItemPopGroupDict[item] = group
                    self._item_pop_label[item] = group
                    self._item_pop[item] = pop
            else:
                for item, pop in TrainPopSorted[group * item_per_group: (group+1) * item_per_group]:
                    ItemPopGroupDict[group].append(item)
                    reverse_ItemPopGroupDict[item] = group
                    self._item_pop_label[item] = group
                    self._item_pop[item] = pop
        self._item_pop_label = np.array(self._item_pop_label)
        #转换为tensor格式
        for group, items in ItemPopGroupDict.items():
            ItemPopGroupDict[group] = torch.tensor(items)

        #初始化testDict_PopGroup的格式：testDict_PopGroup={0:{user:ColdItem}}
        for group in range(num_group):
            testDict_PopGroup[group] = {}
        #生成不同热度分组下的用户test交互item字典
        for user, items in self.testDict.items():                
            Hot = {}
            for group in range(num_group):
                Hot[group] = []
            for item in items:
                group = reverse_ItemPopGroupDict[item]
                Hot[group].append(item)
            for group in range(num_group):
                if Hot[group]:
                    testDict_PopGroup[group][user] = Hot[group]
                else:
                    testDict_PopGroup[group][user] = [999999999999999]#缺省值
        #print(testDict_PopGroup[0])
        return ItemPopGroupDict, reverse_ItemPopGroupDict, testDict_PopGroup

    def build_pop_user(self):
        num_group = world.config['pop_group']
        user_per_group = int(self.num_user / num_group)
        TrainPopSorted = sorted(self.TrainPop_user.items(), key=lambda x: x[1])
        UserPopGroupDict = {}#查询分组中有哪些item的字典
        reverse_UserPopGroupDict = {}#查询item属于哪个分组的字典
        self._user_pop_label = [0]*self.num_user
        self._user_pop = [0]*self.num_user
        #按照Pop分组，并存储至字典[0=Cold, 9=Hot]
        for group in range(num_group):
            UserPopGroupDict[group] = []
            if group == num_group-1:
                for user, pop in TrainPopSorted[group * user_per_group:]:
                    UserPopGroupDict[group].append(user)
                    reverse_UserPopGroupDict[user] = group
                    self._user_pop_label[user] = group
                    self._user_pop[user] = pop
            else:
                for user, pop in TrainPopSorted[group * user_per_group: (group+1) * user_per_group]:
                    UserPopGroupDict[group].append(user)
                    reverse_UserPopGroupDict[user] = group
                    self._user_pop_label[user] = group
                    self._user_pop[user] = pop
        self._user_pop_label = np.array(self._user_pop_label)
        #转换为tensor格式
        for group, users in UserPopGroupDict.items():
            UserPopGroupDict[group] = torch.tensor(users)
        
        return UserPopGroupDict, reverse_UserPopGroupDict

    def pop_bias(self):
        '''
        for M_xy in terms of popularity
        '''
        pop_bias_Dict = {}
        return pop_bias_Dict

#=============================================================Node & Edge Centroid============================================================#
class Centroid():
    def __init__(self, dataset:dataset, pop:Pop):
        self.dataset = dataset
        self.pop = pop
        self.mode = world.config['centroid_mode']

        #nodes_out == edge_index[0]
        #nodes_in == edge_index[1]
        self.nodes_out = torch.cat((torch.tensor(self.dataset._trainUser), torch.tensor(self.dataset._trainItem+self.dataset.n_users)))
        self.nodes_in =  torch.cat((torch.tensor(self.dataset._trainItem+self.dataset.n_users), torch.tensor(self.dataset._trainUser)))
        if world.config['centroid_mode'] in ['degree']:
            self._degree_item = torch.tensor(self.pop.item_pop_degree_label) #item's popularity (degree) in the training dataset
            self._degree_user = torch.tensor(self.pop.user_pop_degree_label) #user's popularity (degree) in the training dataset
            self._degree_item = self._degree_item.to(world.device)
            self._degree_user = self._degree_user.to(world.device)
        elif world.config['centroid_mode'] in ['pagerank']:
            self._pagerank_user, self._pagerank_item = self.compute_pr()
            self._pagerank_user, self._pagerank_item = self._pagerank_user.to(world.device), self._pagerank_item.to(world.device)
        elif world.config['centroid_mode'] in ['eigenvector']:
            self._eigenvector_user, self._eigenvector_item = self.eigenvector_centrality()
            self._eigenvector_user, self._eigenvector_item = self._eigenvector_user.to(world.device), self._eigenvector_item.to(world.device)
        else:
            raise TypeError('centroid mode not implemented')
        # look = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12])
        # print('degree_user', self._degree_user[look])
        # print('degree_item', self._degree_item[look])
        # print('PR_user',self._pagerank_user[look])
        # print('PR_item',self._pagerank_item[look])
        # print('eigenvector_user',self._eigenvector_user[look])
        # print('eigenvector_item',self._eigenvector_item[look])
        # print([self.get_edge_index(1,i) for i in range(600)])
        #print(self.get_edge_index_batch(torch.tensor([0,0,0,1,1]),torch.tensor([0,1,2,17,18])))


    def compute_pr(self, damp=0.85, k=15):
        '''
        For undirected graph, calculate twice pagerank in two direction\n
        U:|0   U-I|
        I:|I-U   0|
        '''
        start = time.time()
        num_nodes = self.dataset.n_users + self.dataset.m_items
        deg_out = torch.cat((torch.tensor(self.pop.user_pop_degree_label), torch.tensor(self.pop.item_pop_degree_label)))
        PR = torch.ones((num_nodes, )).to(torch.float32)
        nodes_out = self.nodes_out
        nodes_in =  self.nodes_in

        for i in range(k):
            edge_msg = PR[nodes_out] / deg_out[nodes_out]
            agg_msg = torch_scatter.scatter(edge_msg, nodes_in, reduce='sum')

            PR = (1 - damp) * PR + damp * agg_msg

        users_pagerank, items_pagerank = torch.split(PR, [self.dataset.n_users, self.dataset.m_items])
        end = time.time()
        print('pagerank_centrality cost: ',end-start)
        return users_pagerank, items_pagerank

    def eigenvector_centrality(self):
        start = time.time()
        precal_path = os.path.join(world.PRECALPATH,'EigenvectorCentroid')
        if os.path.exists(os.path.join(precal_path, 'Eigenvector.pt')):
            print(f'Loading Eigenvector.pt from {precal_path}')
            x = torch.load(os.path.join(precal_path, 'Eigenvector.pt'))
            eigenvector_centrality_user, eigenvector_centrality_item = torch.split(x, [self.dataset.n_users, self.dataset.m_items])
        else:
            nx_graph = self.dataset.nx_Graph #TODO 在dataset中用额外的nx_Graph存储networkx格式有点浪费内存！
            try:
                x = nx.eigenvector_centrality(nx_graph, max_iter=100, tol=1e-04)
            except:
                x = nx.eigenvector_centrality(nx_graph, max_iter=100, tol=5e-04)

            num_nodes = self.dataset.n_users + self.dataset.m_items
            x = torch.tensor([x[i] for i in range(num_nodes)])
            
            if not os.path.exists(precal_path):
                os.makedirs(precal_path, exist_ok=True)
            precal_path = os.path.join(precal_path, 'Eigenvector.pt')
            torch.save(x, precal_path)
            print(f'Save Eigenvector.pt to {precal_path}')

            eigenvector_centrality_user, eigenvector_centrality_item = torch.split(x, [self.dataset.n_users, self.dataset.m_items])
        end = time.time()
        print('eigenvector_centrality cost: ',end-start)
        return eigenvector_centrality_user, eigenvector_centrality_item
        

    
    # def cal_centroid_weights_all(self, centroid_user, centroid_item, aggr='mean', mode='GCA'):
    #     '''
    #     return weights: torch.tensor([w_edge_1, ..., w_edge_n])\n
    #     egde indiced by  self.nodes_out(row) --- self.nodes_in(col)\n
    #     outputs should selected by indices of edges !!!
    #     '''
    #     Centrality = torch.cat(centroid_user, centroid_item)
    #     Centrality_row = Centrality[self.nodes_out].to(torch.float32)
    #     Centrality_col = Centrality[self.nodes_in].to(torch.float32)
    #     s_row = torch.log(Centrality_row)
    #     s_col = torch.log(Centrality_col)
    #     if aggr == 'sink':
    #         s = s_col
    #     elif aggr == 'source':
    #         s = s_row
    #     elif aggr == 'mean':
    #         s = (s_col + s_row) * 0.5
    #     else:
    #         s = s_col
        
    #     if mode == 'GCA':
    #         weights = (s.max() - s) / (s.max() - s.mean())
    #     else:
    #         weights = s
        
    #     return weights

    def cal_centroid_weights_batch(self, batch_user:torch.Tensor, batch_item:torch.Tensor, centroid='degree', aggr='mean', mode='GCA'):
        '''
        input: batch_user and their pos items\n
        return weights: torch.tensor([w_edge_1, ..., w_edge_n])\n
        egde indiced by  self.nodes_out(row) --- self.nodes_in(col)\n
        edge between users and items are not guaranteed by this function.
        ''' 
        if centroid == 'degree':
            centroid_user = self._degree_user[batch_user]
            centroid_item = self._degree_item[batch_item]
        elif centroid == 'pagerank':
            centroid_user = self._pagerank_user[batch_user]
            centroid_item = self._pagerank_item[batch_item]
        elif centroid == 'eigenvector':
            centroid_user = self._eigenvector_user[batch_user]
            centroid_item = self._eigenvector_item[batch_item]
        else:
            raise TypeError('No demanded centroid')

        s_u = torch.log(centroid_user)
        s_i = torch.log(centroid_item)
        if aggr == 'item':
            s = s_i
        elif aggr == 'user':
            s = s_u
        elif aggr == 'mean':
            s = (s_u + s_i) * 0.5
        else:
            s = s_i
        
        if mode == 'GCA':
            weights = (s.max() - s) / (s.max() - s.mean())
            #TODO here max() is just in the batch. In the future version, should consider global information
        elif mode == 'mean':
            weights = F.softmax(s) #softmax( log(x) ) = mean(x)
        else:
            weights = s
        
        return weights




    def get_edge_index_batch(self, users:torch.Tensor, items:torch.Tensor):
        '''
        input: users index, items index, both start from 0.\n
        return: edges index in weights between user and item.\n
        -1 means no edge between them.
        '''
        return self.dataset.edge_indices.to_dense()[users, items] - 1

    def get_edge_index(self, user:int, item:int):
        '''
        input: user index, item index, both start from 0.\n
        return: edge index in weights between user and item.\n
        -1 means no edge between them.
        '''
        trainUser = self.dataset._trainUser.copy()
        trainItem = self.dataset._trainItem.copy()
        edge_index = (trainUser==user)*(trainItem==item)
        try:
            return np.nonzero(edge_index==True)[0][0]
        except:
            return -1
        return self.dataset.edge_indices[user, item].item() - 1
        # try:
        #     return self.dataset.edge_index.index((user,item))
        # except:
        #     return -1
        # edge_index = self.dataset.Graph.indices()
        # return edge_index[edge_index[0]==user,edge_index[1]==(item+self.dataset.n_users)]

#=============================================================Commmon Neighborhood============================================================#
class CommonNeighbor():
    def __init__(self, dataset:dataset):
        self.dataset = dataset
        self.mode = world.config['commonNeighbor_mode']
        mat_sp = self.CN_simi_unsymmetry_mat_sp(mode = self.mode)
        print('shape test :',mat_sp.nonzero()[0].shape[0],'==',2*self.dataset.trainDataSize)
        # self.CN_simi_mat_sp = mat_sp.to(world.device)
        self.CN_simi_mat_sp = mat_sp

    # def CN_simi_unsymmetry_mat(self, mode='SC'):
    #     """
    #     return SPARSE edge_weight:\n
    #     edge_weight[i,j] = importance of i to j\n
    #     |0        u_to_i|\n
    #     |i_to_u        0|\n
    #     Not symmetry !
    #     """
    #     start = time.time()
    #     precal_path = os.path.join(world.PRECALPATH,'CommonNeighbor')
    #     precal_path = os.path.join(precal_path, f'{mode}')
    #     #precal_path = os.path.join(precal_path, 'CommonNeighbor.pt')
    #     if os.path.exists(os.path.join(precal_path, 'CommonNeighbor.pt')):
    #         print(f'Loading CommonNeighbor.pt from {precal_path}')
    #         precal_path = os.path.join(precal_path, 'CommonNeighbor.pt')
    #         edge_weight = torch.load(precal_path)
    #     else:
    #         n_users = self.dataset.n_users
    #         n_items = self.dataset.m_items
    #         user_item_graph = self.dataset.Graph.cpu().to_dense()[:n_users, n_users:]
    #         user_item_graph[user_item_graph > 0] = 1

    #         edge_weight = torch.zeros((n_users + n_items, n_users + n_items))

    #         for i in tqdm(range(n_items), desc=f'Calculating importance of users to item in mode {mode}'):
    #             users = user_item_graph[:, i].nonzero().squeeze(-1)

    #             items = user_item_graph[users]
    #             user_user_cap = torch.matmul(items, items.t())
    #             user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

    #             if mode == 'JS':
    #                 simi = (user_user_cap / (user_user_cup - user_user_cap)).mean(dim=1)
    #             elif mode == 'CN':
    #                 simi =  user_user_cap.mean(dim=1)
    #             elif mode == 'SC':
    #                 simi = (user_user_cap / ((items.sum(dim=1) * items.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)
    #             elif mode == 'LHN':
    #                 simi = (user_user_cap / ((items.sum(dim=1) * items.sum(dim=1).unsqueeze(-1)))).mean(dim=1)
    #             else:
    #                 raise TypeError('No demanded Common Neighbor Method')

    #             edge_weight[users, i + n_users] = simi

    #         for i in tqdm(range(n_users), desc=f'Calculating importance of items to user in mode {mode}'):
    #             items = user_item_graph[i, :].nonzero().squeeze(-1)

    #             users = user_item_graph[:, items].t()
    #             item_item_cap = torch.matmul(users, users.t())
    #             item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

    #             if mode == 'JS':
    #                 simi = (item_item_cap / (item_item_cup - item_item_cap)).mean(dim=1)
    #             elif mode == 'CN':
    #                 simi =  item_item_cap.mean(dim=1)
    #             elif mode == 'SC':
    #                 simi = (item_item_cap / ((users.sum(dim=1) * users.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)
    #             elif mode == 'LHN':
    #                 simi = (item_item_cap / ((users.sum(dim=1) * users.sum(dim=1).unsqueeze(-1)))).mean(dim=1)
    #             else:
    #                 raise TypeError('No demanded Common Neighbor Method')

    #             edge_weight[items + n_users, i] = simi
    #         if not os.path.exists(precal_path):
    #             os.makedirs(precal_path, exist_ok=True)
    #         precal_path = os.path.join(precal_path, 'CommonNeighbor.pt')
    #         edge_weight = edge_weight.to_sparse()
    #         edge_weight = edge_weight.coalesce()
    #         torch.save(edge_weight, precal_path)
    #         print(f'Save CommonNeighbor.pt to {precal_path}')
        
    #     edge_weight = edge_weight.coalesce()
    #     end = time.time()
    #     print('CN_simi_unsymmetry_mat cost: ',end-start)
    #     return edge_weight


    def CN_simi_unsymmetry_mat_sp(self, mode='SC'):
        """
        return SPARSE edge_weight:\n
        edge_weight[i,j] = importance of i to j\n
        |0        u_to_i|\n
        |i_to_u        0|\n
        Not symmetry !
        """
        start = time.time()
        precal_path = os.path.join(world.PRECALPATH,'CommonNeighbor')
        precal_path = os.path.join(precal_path, f'{mode}')
        #precal_path = os.path.join(precal_path, 'CommonNeighbor.pt')
        if os.path.exists(os.path.join(precal_path, 'CommonNeighbor_sp.pt')):
            print(f'Loading CommonNeighbor_sp.pt from {precal_path}')
            precal_path = os.path.join(precal_path, 'CommonNeighbor_sp.pt')
            edge_weight = torch.load(precal_path)
        else:
            n_users = self.dataset.n_users
            n_items = self.dataset.m_items
            edge_index = self.dataset.Graph.cpu().indices()#item's index starts from n_users, not 0! 
            
            #edge_weight = torch.zeros((n_users + n_items, n_users + n_items))
            row = torch.tensor([])
            col = torch.tensor([])
            val = torch.tensor([])

            for i in tqdm(range(n_items), desc=f'Calculating importance of users to item in sparse mode {mode}'):
                users = edge_index[0, edge_index[1] == (i + n_users)]
                items = torch.zeros([users.shape[0], n_items])

                for j, user in enumerate(users):
                    items[j, torch.tensor(self.dataset.allPos[user.item()]).long()] = 1

                user_user_cap = torch.matmul(items, items.t())
                user_user_cup = items.sum(dim=1) + items.sum(dim=1).unsqueeze(-1)

                if mode == 'JS':
                    simi = (user_user_cap / (user_user_cup - user_user_cap)).mean(dim=1)
                elif mode == 'CN':
                    simi =  user_user_cap.mean(dim=1)
                elif mode == 'SC':
                    simi = (user_user_cap / ((items.sum(dim=1) * items.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)
                elif mode == 'LHN':
                    simi = (user_user_cap / ((items.sum(dim=1) * items.sum(dim=1).unsqueeze(-1)))).mean(dim=1)
                else:
                    raise TypeError('No demanded Common Neighbor Method')

                #edge_weight[users, i + n_users] = simi
                row = torch.cat((row, users))
                i_col = torch.tensor([i + n_users]*(users.shape[0]))
                col = torch.cat((col, i_col))
                val = torch.cat((val, simi))

            for i in tqdm(range(n_users), desc=f'Calculating importance of items to user in sparse mode {mode}'):
                items = edge_index[1, edge_index[0] == i]
                users = torch.zeros([items.shape[0], n_users])

                for j, item in enumerate(items):
                    users[j, torch.tensor(self.dataset.allPos_item[item.item()-n_users]).long()] = 1

                item_item_cap = torch.matmul(users, users.t())
                item_item_cup = users.sum(dim=1) + users.sum(dim=1).unsqueeze(-1)

                if mode == 'JS':
                    simi = (item_item_cap / (item_item_cup - item_item_cap)).mean(dim=1)
                elif mode == 'CN':
                    simi =  item_item_cap.mean(dim=1)
                elif mode == 'SC':
                    simi = (item_item_cap / ((users.sum(dim=1) * users.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)
                elif mode == 'LHN':
                    simi = (item_item_cap / ((users.sum(dim=1) * users.sum(dim=1).unsqueeze(-1)))).mean(dim=1)
                else:
                    raise TypeError('No demanded Common Neighbor Method')

                #edge_weight[items, i] = simi
                row = torch.cat((row, items))
                i_col = torch.tensor([i]*(items.shape[0]))
                col = torch.cat((col, i_col))
                val = torch.cat((val, simi))
            row = row.long()
            col = col.long()
            # index = torch.stack([row, col])
            # edge_weight = torch.sparse.FloatTensor(index, val, (n_users + n_items, n_users + n_items))
            # edge_weight = edge_weight.coalesce()
            edge_weight = csr_matrix((val, (row, col)), shape=(n_users + n_items, n_users + n_items))


            if not os.path.exists(precal_path):
                os.makedirs(precal_path, exist_ok=True)
            
            precal_path = os.path.join(precal_path, 'CommonNeighbor_sp.pt')
            torch.save(edge_weight, precal_path)
            print(f'Save CommonNeighbor_sp.pt to {precal_path}')
        # edge_weight = edge_weight.coalesce()
        end = time.time()
        print('CN_simi_unsymmetry_mat_sp cost: ',end-start)
        return edge_weight


    