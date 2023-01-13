"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
from model import LightGCN
import numpy as np
from utils import randint_choice
import scipy.sparse as sp
import world
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from precalcul import precalculate
import time
from k_means import kmeans


class Homophily:
    def __init__(self, model:LightGCN):
        self.model = model
        
    def get_homophily_batch(self, batch_user:torch.Tensor, batch_item:torch.Tensor, mode='not_in_batch'):
        '''
        return prob distribution of users and items in batch.
        '''
        with torch.no_grad():
            sigma = world.config['sigma_gausse']
            ncluster = world.config['n_cluster']
            #edge_index = self.model.dataset.Graph.cpu().indices()
            if mode == 'in_batch':
                embs_KMeans = torch.cat((self.model.embedding_user(batch_user), self.model.embedding_item(batch_item)), dim=0)
            else:
                embs_KMeans = torch.cat((self.model.embedding_user.weight, self.model.embedding_item.weight), dim=0)
            # embs_KMeans_numpy = embs_KMeans.cpu().numpy()#.detach()
            # kmeans_sk = KMeans(n_clusters=ncluster, random_state=0).fit(embs_KMeans_numpy)
            # #cluster_labels = kmeans.labels_
            # #homo = (cluster_labels[edge_index[0]] == cluster_labels[edge_index[1]])
            # centroids_sk = torch.FloatTensor(kmeans_sk.cluster_centers_).to(world.device)
            cluster_ids_x, cluster_centers = kmeans(X=embs_KMeans, num_clusters=ncluster, distance='euclidean', device=world.device, tqdm_flag=False)
            centroids = cluster_centers.to(world.device)
            logits = []
            embs_batch = torch.cat((self.model.embedding_user(batch_user), self.model.embedding_item(batch_item)), dim=0)
            for c in centroids:
                logits.append((-torch.square(embs_batch - c).sum(1)/sigma).view(-1, 1))
            logits = torch.cat(logits, axis=1)
            probs = F.softmax(logits, dim=1)
            #probs = F.normalize(logits, dim=1)# TODO
            #loss = F.l1_loss(probs[edge_index[0]], probs[edge_index[1]])
            batch_user_prob, batch_item_prob = torch.split(probs, [batch_user.shape[0], batch_item.shape[0]])
        return batch_user_prob, batch_item_prob



class ED_Uniform():
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precal = precal
        self.homophily = homophily
        self.augAdjMatrix1 = None
        self.augAdjMatrix2 = None

    def Edge_drop_random(self, p_drop):
        '''
        return: dropout后保留的交互构成的按度归一化的邻接矩阵(sparse)
        '''
        n_nodes = self.model.num_users + self.model.num_items
        #注意数组复制问题！
        trainUser = self.model.dataset.trainUser.copy()
        trainItem = self.model.dataset.trainItem.copy()
        keep_idx = randint_choice(len(self.model.dataset.trainUser), size=int(len(self.model.dataset.trainUser) * (1 - p_drop)), replace=False)
        user_np = trainUser[keep_idx]
        item_np = trainItem[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        if self.config['if_big_matrix']:
            g = self.model.dataset._split_matrix(adj_matrix)
            for fold in g:
                fold.requires_grad = False
        else:
            g = self.model.dataset._convert_sp_mat_to_sp_tensor(adj_matrix).coalesce().to(world.device)
            g.requires_grad = False
        return g

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Edge_drop_random(p_drop)
        self.augAdjMatrix2 =  self.Edge_drop_random(p_drop)



class RW_Uniform(ED_Uniform):
    def __init__(self, config, model, precal, homophily):
        super(RW_Uniform, self).__init__(config, model, precal, homophily)

    def Random_Walk(self, p_drop):
        aug_g = []
        for layer in self.config['num_layers']:
            aug_g.append(self.Edge_drop_random(p_drop))
        return aug_g

    # def computer(self, p_drop):
    #     aug_g = self.Random_Walk(p_drop)
    #     return self.model.view_computer(aug_g)

    def get_augAdjMatrix(self):
        p_drop = world.config['p_drop']
        self.augAdjMatrix1 =  self.Random_Walk(p_drop)
        self.augAdjMatrix2 =  self.Random_Walk(p_drop)




class Adaptive_Neighbor_Augment:
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precal = precal
        self.homophily = homophily
        self.L = self.config['num_layers']
        self.epsilon = self.config['epsilon_GCLRec']
        self.w = self.config['w_GCLRec']
    
    def get_adaptive_neighbor_augment(self, embs_per_layer, batch_users, batch_pos, batch_neg, k):
        '''
        return aug_all_users, aug_all_items of selected k-th layer\n
        u'(k) = (1-𝜀)*u(k) + (𝜀(L-k)/L)*u(L) + w Σ w_uv*v(L)
        '''
        aug_embs_k_layer = (1-self.epsilon) * embs_per_layer[k] + (self.epsilon*(self.L-k)/self.L) * embs_per_layer[self.L]
        Sigma = 0
        
        aug_embs_k_layer = aug_embs_k_layer + self.w * Sigma

        aug_user_embs_k_layer, aug_item_embs_k_layer = torch.split(aug_embs_k_layer, [self.model.num_users, self.model.num_items])
        return aug_user_embs_k_layer, aug_item_embs_k_layer
        
    def get_adaptive_neighbor_augment_batch(self, embs_per_layer, batch_users, batch_pos, batch_neg, k):
        '''
        return aug_all_users, aug_all_items of selected k-th layer\n
        u'(k) = (1-𝜀)*u(k) + (𝜀(L-k)/L)*u(L) + wΣ w_uv*v(L)
        '''

        return 
        
    def sample(self):
        '''
        sample several samples for each user or item
        '''
        return 

    def get_coef_adaptive(self, batch_user, batch_pos_item, method='centroid', mode='eigenvector'):
        '''
        input: index batch_user & batch_pos_item\n
        return tensor([adaptive coefficient of u_n-i_n])
        '''
        with torch.no_grad():
            if method == 'homophily':
                batch_user_prob, batch_item_prob = self.homophily.get_homophily_batch(batch_user, batch_pos_item)
                #inner_product
                batch_weight = torch.sum(torch.mul(batch_user_prob, batch_item_prob) ,dim=1)
                #cos_similarity
                #batch_weight = F.cosine_similarity(batch_user_prob, batch_item_prob)

            elif method == 'centroid':
                batch_weight = self.precal.centroid.cal_centroid_weights_batch(batch_user, batch_pos_item, centroid=mode, aggr='mean', mode='GCA')

            elif method == 'commonNeighbor':
                #commonNeighbor weights are not symmetry.
                #And index of item starts from n_users.
                n_users = self.model.num_users
                dense = self.precal.common_neighbor.CN_simi_mat_sp.to_dense()
                batch_weight1 = dense[batch_user, batch_pos_item+n_users]
                batch_weight2 = dense[batch_pos_item+n_users, batch_user]
                batch_weight = (batch_weight1 + batch_weight2) * 0.5
                # mat = self.precal.common_neighbor.CN_simi_mat_sp
                # batch_weight = []
                # for i in range(len(batch_user)):
                #     batch_weight1 = mat[batch_user[i], batch_pos_item[i]+n_users]
                #     batch_weight2 = mat[batch_pos_item[i]+n_users, batch_user[i]]
                #     batch_weight.append((batch_weight1+batch_weight2)*0.5)
                # batch_weight = torch.tensor(batch_weight)

            elif method == 'mlp':
                batch_weight = None
                raise TypeError('adaptive method not implemented')

            else:
                batch_weight = None
                raise TypeError('adaptive method not implemented')

            batch_weight = torch.sigmoid(batch_weight)

        return batch_weight.to(world.device)
