"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import os
import world
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from dataloader import dataset
from precalcul import precalculate
#=============================================================Basic LightGCN============================================================#
class LightGCN(nn.Module):
    def __init__(self, config, dataset:dataset, precal:precalculate):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.precal = precal
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("user:{}, item:{}".format(self.num_users, self.num_items))
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['num_layers']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        if self.config['loss'] in ['BC']:
            #For BC loss
            hotest_user = self.precal.popularity.UserPopGroupDict[self.config['pop_group']-1][-1]
            hotest_user = int(hotest_user)
            self.max_pop_user = self.precal.popularity.user_pop_degree_label[hotest_user]
            hotest_item = self.precal.popularity.ItemPopGroupDict[self.config['pop_group']-1][-1]
            hotest_item = int(hotest_item)
            self.max_pop_item = self.precal.popularity.item_pop_degree_label[hotest_item]
            #For BC loss
            self.embed_user_pop = nn.Embedding(self.max_pop_user+1, self.latent_dim)
            self.embed_item_pop = nn.Embedding(self.max_pop_item+1, self.latent_dim)
            nn.init.xavier_normal_(self.embed_user_pop.weight)
            nn.init.xavier_normal_(self.embed_item_pop.weight)
        
        if self.config['if_pretrain'] == 0:
            if world.config['init_method'] == 'Normal':
                world.cprint('use NORMAL distribution UI for Embedding')
                nn.init.normal_(self.embedding_user.weight, std=0.1)
                nn.init.normal_(self.embedding_item.weight, std=0.1)
            elif world.config['init_method'] == 'Xavier':
                world.cprint('use Xavier_uniform distribution UI for Embedding')
                nn.init.xavier_uniform_(self.embedding_user.weight, gain=1.0)
                nn.init.xavier_uniform_(self.embedding_item.weight, gain=1.0)
            else:
                raise TypeError('init method')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined Embedding')
        self.f = nn.Sigmoid()
        if self.dataset.Graph is None:
            self.Graph = self.dataset.getSparseGraph()
        else:
            self.Graph = self.dataset.Graph

        print(f"GCL Model is ready to go!")

    def computer(self):
        """
        vanilla LightGCN. No dropout used, return final embedding for rec. 
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        graph=self.Graph    
        for layer in range(self.n_layers):
            if world.config['if_big_matrix']:
                temp_emb = []
                for i_fold in range(len(graph)):
                    temp_emb.append(torch.sparse.mm(graph[i_fold], all_emb))
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    
    def getUsersRating(self, users):
        '''
        先执行一次model.computer().
        return rating=指定users对每个item做内积后过Sigmoid()
        '''
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))#TODO 为避免不同样本预测分值的数量级差异导致的梯度数值差异，BPRloss计算时通常用Sigmoid将正负样本预测分插值映射至01之间。
        return rating

    #================Pop=================#
    def getItemRating(self):
        '''
        获取输入items1, items2对全部user的平均得分
        return: rating1=Hot, rating2=Cold
        '''
        itemsPopDict = self.precal.popularity.ItemPopGroupDict
        all_users, all_items = self.computer()
        items_embDict = {}
        for group in range(world.config['pop_group']):
            items_embDict[group] = all_items[itemsPopDict[group].long()]
        users_emb = all_users
        #rating = self.f(torch.matmul(items_emb, users_emb.t()))#TODO 内积后过Sigmoid()作为输出Rating
        rating_Dict = {}
        for group in range(world.config['pop_group']):
            rating_Dict[group] = torch.matmul(items_embDict[group], users_emb.t())
            rating_Dict[group] = torch.mean(rating_Dict[group], dim=1)
            rating_Dict[group] = torch.mean(rating_Dict[group])
        return rating_Dict
    #================Pop=================#

    def getEmbedding(self, users, pos_items, neg_items):
        '''
        先执行一次model.computer().
        return: users, pos_items, neg_items各自的初始embedding(item在聚合KG信息前的embedding)和LightGCN更新后的embedding
        '''
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, None
    
    def bpr_loss(self, users, pos, neg):
        '''
        输入一个batch的users、pos_items、neg_items
        reG_loss = users、pos_items、neg_items初始embedding的L2正则化loss
        reC_loss = Σ{ softplus[ (ui,negi) - (ui,posi) ] }
        '''
        (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, _) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))#/float(len(users))
        #TODO 这里的reg数量级有问题？除以了batch？？
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        # mean or sum
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))#TODO SOFTPLUS()!!!
        '''
        self.tau = world.config['temp_tau']
        self.alpha = world.config['alpha']
        f = lambda x: torch.exp(x / self.tau)
        pos_scores = self.alpha * (self.sim(users_emb, pos_emb)) / self.tau
        neg_scores = (1-self.alpha) * (self.sim(users_emb, neg_emb)) / self.tau
        loss = -torch.sum(pos_scores - neg_scores)
        '''

        if(torch.isnan(loss).any().tolist()):
            print("user emb")
            print(userEmb0)
            print("pos_emb")
            print(posEmb0)
            print("neg_emb")
            print(negEmb0)
            print("neg_scores")
            print(neg_scores)
            print("pos_scores")
            print(pos_scores)
            return None, None
        return loss, reg_loss



#=============================================================SGL ED & RW============================================================#
class SGL(LightGCN):
    def __init__(self, config, dataset:dataset, precal):
        super(SGL, self).__init__(config, dataset, precal)

    def view_computer(self, augmentGraph):
        """
        在增强视图augmentGraph上用LightGCN传播user和item的信息。
        return: SGL用于推荐的user和item的嵌入。
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        if world.config['model'] == 'SGL':
            embs = [all_emb]
            if world.config['augment']=='RW':
                for layer in range(self.n_layers):
                    if world.config['if_big_matrix']:
                        temp_emb = []
                        for i_fold in range(len(augmentGraph[layer])):
                            temp_emb.append(torch.sparse.mm(augmentGraph[layer][i_fold], all_emb))
                        all_emb = torch.cat(temp_emb, dim=0)
                    else:
                        all_emb = torch.sparse.mm(augmentGraph[layer], all_emb)
                    embs.append(all_emb)
            elif world.config['augment']=='ED':
                for layer in range(self.n_layers):
                    if world.config['if_big_matrix']:
                        temp_emb = []
                        for i_fold in range(len(augmentGraph)):
                            temp_emb.append(torch.sparse.mm(augmentGraph[i_fold], all_emb))
                        all_emb = torch.cat(temp_emb, dim=0)
                    else:
                        all_emb = torch.sparse.mm(augmentGraph, all_emb)
                    embs.append(all_emb)
        else:
            raise TypeError('model-mode')
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def xxx(self):
        pass

#=============================================================SimGCL============================================================#
class SimGCL(LightGCN):
    def __init__(self, config, dataset:dataset, precal):
        super(SimGCL, self).__init__(config, dataset, precal)
        
    def view_computer(self):
        """
        return: SimGCL用于推荐的user和item的嵌入。
        """ 
        raw_graph = self.Graph
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        if world.config['model'] in ['SimGCL']:
            embs = []#SimGCL建议不使用初始嵌入
            if world.config['model']=='SimGCL':
                for layer in range(self.n_layers):
                    if world.config['if_big_matrix']:
                        temp_emb = []
                        for i_fold in range(len(raw_graph)):
                            temp_emb.append(torch.sparse.mm(raw_graph[i_fold], all_emb))
                        all_emb = torch.cat(temp_emb, dim=0)
                    else:
                        all_emb = torch.sparse.mm(raw_graph, all_emb)
                    low = torch.zeros_like(all_emb).float()
                    high = torch.ones_like(all_emb).float()
                    random_noise = torch.distributions.uniform.Uniform(low, high).sample()
                    noise = torch.mul(torch.sign(all_emb),torch.nn.functional.normalize(random_noise, dim=1)) * world.config['eps_SimGCL']
                    all_emb += noise
                    embs.append(all_emb)
            else:
                raise TypeError('model-mode')
        else:
            raise TypeError('model-mode')
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def xxx(self):
        pass


#=============================================================Myself GCLRec============================================================#
class GCLRec(LightGCN):
    def __init__(self, config, dataset:dataset, precal):
        super(GCLRec, self).__init__(config, dataset, precal)

    def computer(self):
        """
        return: users(mean(0~3)), items(mean(0~3)), embs_per_layer(0~3)
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        graph=self.Graph    
        for layer in range(self.n_layers):
            if world.config['if_big_matrix']:
                temp_emb = []
                for i_fold in range(len(graph)):
                    temp_emb.append(torch.sparse.mm(graph[i_fold], all_emb))
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        embs_per_layer = embs
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items, embs_per_layer
    
    
    def getUsersRating(self, users):
        '''
        '''
        all_users, all_items, _ = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))#TODO 为避免不同样本预测分值的数量级差异导致的梯度数值差异，BPRloss计算时通常用Sigmoid将正负样本预测分插值映射至01之间。
        return rating


    def getItemRating(self):
        '''
        '''
        itemsPopDict = self.precal.popularity.ItemPopGroupDict
        all_users, all_items, _ = self.computer()
        items_embDict = {}
        for group in range(world.config['pop_group']):
            items_embDict[group] = all_items[itemsPopDict[group].long()]
        users_emb = all_users
        #rating = self.f(torch.matmul(items_emb, users_emb.t()))#TODO 内积后过Sigmoid()作为输出Rating
        rating_Dict = {}
        for group in range(world.config['pop_group']):
            rating_Dict[group] = torch.matmul(items_embDict[group], users_emb.t())
            rating_Dict[group] = torch.mean(rating_Dict[group], dim=1)
            rating_Dict[group] = torch.mean(rating_Dict[group])
        return rating_Dict


    def getEmbedding(self, users, pos_items, neg_items):
        '''
        return 
        users_emb: [ego, layer1, layer2, layer3, mean_for_rec]\n
        embs_per_layer: [ego, layer1, layer2, layer3]
        '''
        all_users_rec, all_items_rec, embs_per_layer = self.computer()
        users_emb, pos_emb, neg_emb = [], [], []
        for i in range(len(embs_per_layer)):
            all_embs_current_layer = embs_per_layer[i]
            user_embs_current_layer, item_embs_current_layer = torch.split(all_embs_current_layer, [self.num_users, self.num_items])
            users_emb.append(user_embs_current_layer[users])
            pos_emb.append(item_embs_current_layer[pos_items])
            neg_emb.append(item_embs_current_layer[neg_items])
        
        users_emb.append(all_users_rec[users])
        pos_emb.append(all_items_rec[pos_items])
        neg_emb.append(all_items_rec[neg_items])

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, embs_per_layer
