"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import os
from os.path import join
import random
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
#自定义的数据集继承自Dataset，必须重写的方法:__init__()读取数据, __len__()自定义数据集的长度, __getitem__()对自定义数据集进行索引
#用Dataset制作好数据集后交给DataLoader可以自动输出每个batch的数据及标签
import world
from world import cprint
from time import time
import networkx as nx

class dataset(Dataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla\ yelp2018\ amazon-book\ MIND dataset
    """

    def __init__(self, config = world.config, path="../data/yelp2018"):
        # train or test
        print(f'loading [{path}]')
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        valid_file = path + '/valid.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validDataSize = 0

        self._TrainPop_item = {}#item's popularity (degree) in the training dataset
        self._TrainPop_user = {}#user's popularity (degree) in the training dataset
        
        self._allPos = {}
        self._allPos_item = {}

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
                    #================Pop=================#
                    for item in items:
                        if item in self._TrainPop_item.keys():
                            self._TrainPop_item[item] += 1
                        else:
                            self._TrainPop_item[item] = 1
                        
                        if item in self._allPos_item.keys():
                            self._allPos_item[item].append(uid)
                        else:
                            self._allPos_item[item] = [uid]

                    if uid in self._TrainPop_user.keys():
                        self._TrainPop_user[uid] += len(items)
                    else:
                        self._TrainPop_user[uid] = len(items)

                    self._allPos[uid] = items
                    #================Pop=================#
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1]:
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.n_user = max(self.n_user, uid)
                        self.testDataSize += len(items)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        
        if os.path.exists(valid_file):
            with open(valid_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        if l[1]:
                            items = [int(i) for i in l[1:]]
                            uid = int(l[0])
                            validUniqueUsers.append(uid)
                            validUser.extend([uid] * len(items))
                            validItem.extend(items)
                            self.m_item = max(self.m_item, max(items))
                            self.n_user = max(self.n_user, uid)
                            self.validDataSize += len(items)
            self.validUniqueUsers = np.array(validUniqueUsers)
            self.validUser = np.array(validUser)
            self.validItem = np.array(validItem)
        
        self.m_item += 1
        self.n_user += 1

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{config['dataset']} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")#针对无验证集时的稀疏度计算公式

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_user, self.m_item))
        # pre-calculate
        # self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # self._allPos_item = self.getItemPosUsers(list(range(self.m_item)))
        self.__testDict = self.__build_test()
        self._edge_indices = self.get_edge_indices()
        #将计算邻接矩阵的过程提前
        self.getSparseGraph()

        print(f"{config['dataset']} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def _trainUser(self):
        return self.trainUser
    
    @property
    def _trainItem(self):
        return self.trainItem
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos
    
    @property
    def allPos_item(self):
        return self._allPos_item

    @property
    def TrainPop_item(self):
        '''
        dict of items' popularity(degree) in training set
        '''
        return self._TrainPop_item

    @property
    def TrainPop_user(self):
        '''
        dict of users' popularity(degree) in training set
        '''
        return self._TrainPop_user
    
    @property
    def edge_indices(self):
        '''
        Edge's indice start from 1.\n
        Minus 1 while using, so that -1 means no edge.\n
        It's sparse, .to_dense() if many indices are needed.
        '''
        return self._edge_indices


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        """
        Graph = \n
        D^(-1/2) @ A @ D^(-1/2) \n
        A = \n
        |0,   R|\n
        |R.T, 0|\n
        度归一化相当于对user-item的交互R矩阵中u和i的位置乘上 1/sqrt(|u|) * 1/sqrt(|i|) |x|指节点x的邻居数
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix --- All train data")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                #此处会显存爆炸
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0]) TODO 无自连接
                
                rowsum = np.array(adj_mat.sum(axis=1))#度
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)#对角阵
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            if world.config['if_big_matrix']:
                print(f"split the adj matrix to {world.config['n_fold']} folds")
                self.Graph = self._split_matrix(norm_adj)
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
            self.nx_Graph = nx.from_scipy_sparse_matrix(norm_adj)#TODO 在dataset中用额外的nx_Graph存储networkx格式有点浪费内存！
        return self.Graph

    def _split_matrix(self, norm_adj):
        norm_adj_split = []
        fold_len = (self.n_user + self.m_items) // world.config['n_fold']
        for i_fold in range(world.config['n_fold']):
            start = i_fold * fold_len
            if i_fold == world.config['n_fold']-1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            norm_adj_split.append(self._convert_sp_mat_to_sp_tensor(norm_adj[start:end]).coalesce().to(world.device))
        return norm_adj_split

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems 
        
    def getItemPosUsers(self, items):
        posUsers = []
        for item in items:
            posUsers.append(self.UserItemNet[:,item].nonzero()[0])
        return posUsers 

    def get_edge_indices(self):
        '''
        edge's indices start from 1, so that 0 means no edge\n
        -1 when use this index
        '''
        index = torch.stack([torch.tensor(self.trainUser), torch.tensor(self.trainItem)])
        val =torch.arange(len(self.trainItem)) + 1
        edge_indice = torch.sparse.FloatTensor(index, val, (self.n_user, self.m_item))
        edge_indice = edge_indice.coalesce()
        return edge_indice

    '''
    Dataset的所有子类都应该重写方法__len__(), __getitem__()
    '''
    def __len__(self):
        return self.traindataSize

    def __getitem__(self, idx):
        '''
        input: user在trainUser列表中的idx
        output: 随机三元组(user, pos, neg)
        '''
        user = self.trainUser[idx]
        pos = random.choice(self._allPos[user])
        while True:
            neg = np.random.randint(0, self.m_item)
            if neg in self._allPos[user]:
                continue
            else:
                break
        return user, pos, neg
