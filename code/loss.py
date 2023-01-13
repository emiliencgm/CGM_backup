"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
from model import LightGCN
from precalcul import precalculate
import torch
import torch.nn.functional as F
import world
from augment import Homophily
import numpy as np
#=============================================================BPR loss============================================================#
class BPR_loss():
    def __init__(self, config, model:LightGCN, precalculate:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precalculate = precalculate

    def bpr_loss(self, batch_user, batch_pos, batch_neg):
        loss, reg_loss = self.model.bpr_loss(batch_user, batch_pos, batch_neg)
        reg_loss = reg_loss * self.config['weight_decay']
        loss = loss + reg_loss
        loss = loss/self.config['batch_size']
        return loss

#=============================================================BPR + CL loss============================================================#
class BPR_Contrast_loss(BPR_loss):
    def __init__(self, config, model:LightGCN, precalculate:precalculate, homophily:Homophily):
        super(BPR_Contrast_loss, self).__init__(config, model, precalculate, homophily)
        self.tau = config['temp_tau']

    def bpr_contrast_loss(self, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, batch_user, batch_pos, batch_neg, aug_users1, aug_items1, aug_users2, aug_items2):

        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        bprloss = (loss + reg_loss * self.config['weight_decay'])/self.config['batch_size']
        
        contrastloss = self.info_nce_loss_overall(aug_users1[batch_user], aug_users2[batch_user], aug_users2) \
                        + self.info_nce_loss_overall(aug_items1[batch_pos], aug_items2[batch_pos], aug_items2)
        return bprloss + self.config['lambda1']*contrastloss

    def info_nce_loss_overall(self, z1, z2, z_all):
        '''
        z1--z2: pos,  z_all: neg\n
        return: InfoNCEloss
        '''
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))
        all_sim = f(self.sim(z1, z_all))
        positive_pairs = between_sim
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / (negative_pairs)))
        #print('positive_pairs / negative_pairs',max(positive_pairs / negative_pairs))
        loss = loss/self.config['batch_size']
        return loss


    def sim(self, z1: torch.Tensor, z2: torch.Tensor, mode='cos'):
        '''
        计算一个z1和一个z2两个向量的相似度/或者一个z1和多个z2的各自相似度。
        即两个输入的向量数（行数）可能不同。
        '''
        if mode == 'inner_product':
            if z1.size()[0] == z2.size()[0]:
                #return F.cosine_similarity(z1,z2)
                x = F.normalize(z1)
                y = F.normalize(z2)
                return torch.sum(torch.mul(z1,z2) ,dim=1)
            else:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                #return ( torch.mm(z1, z2.t()) + 1 ) / 2
                return torch.mm(z1, z2.t())
        elif mode == 'cos':
            if z1.size()[0] == z2.size()[0]:
                return F.cosine_similarity(z1,z2)
            else:
                z1 = F.normalize(z1)
                z2 = F.normalize(z2)
                #return ( torch.mm(z1, z2.t()) + 1 ) / 2
                return torch.mm(z1, z2.t())

#=============================================================BC loss============================================================#       
class BC_loss():
    def __init__(self, config, model:LightGCN, precalculate:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precalculate = precalculate
        self.tau = config['temp_tau']
        self.f = lambda x: torch.exp(x / self.tau)
        self.decay = self.config['weight_decay']
        self.batch_size = self.config['batch_size']
        self.alpha = self.config['alpha']#BC loss下的alpha与Adaptive下的alpha不同，是用于对pop_loss和主bc_loss加权用的。

    def bc_loss(self, batch_target, batch_pos, batch_negs, mode):
        #只使用正样本（一个user的正样本往往是其他user的负样本），无需负采样
        users_pop = torch.tensor(self.precalculate.popularity.user_pop_degree_label)[batch_target].to(world.device)
        pos_items_pop = torch.tensor(self.precalculate.popularity.item_pop_degree_label)[batch_pos].to(world.device)
        bc_loss, pop_loss, reg_pop_emb_loss, reg_pop_loss, reg_emb_loss = self.calculate_loss(batch_target, batch_pos, users_pop, pos_items_pop)
        if mode == 'only_bc':
            loss = bc_loss + reg_emb_loss
        elif mode == 'pop_bc':
            loss = bc_loss + pop_loss + reg_pop_emb_loss
        elif mode =='only_pop':
            loss = pop_loss + reg_pop_loss
        return loss
    
    #From BC loss
    def calculate_loss(self, users, pos_items, users_pop, pos_items_pop):

        # popularity branch
        users_pop_emb = self.model.embed_user_pop(users_pop)
        pos_pop_emb = self.model.embed_item_pop(pos_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim = -1)

        users_pop_emb = F.normalize(users_pop_emb, dim = -1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim = -1)

        ratings = torch.matmul(users_pop_emb, torch.transpose(pos_pop_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        loss2 = self.alpha * torch.mean(torch.negative(torch.log(numerator/denominator)))
        #loss2 = self.alpha * torch.sum(torch.negative(torch.log(numerator/denominator)))

        # main bc branch
        all_users, all_items = self.model.computer()

        userEmb0 = self.model.embedding_user(users)
        posEmb0 = self.model.embedding_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))+\
                                (1-torch.sigmoid(pos_ratings_margin)))
        #ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))+\
        #                        (torch.zeros_like(pos_ratings_margin)))
        
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        loss1 = (1-self.alpha) * torch.mean(torch.negative(torch.log(numerator/denominator)))
        #loss1 = (1-self.alpha) * torch.sum(torch.negative(torch.log(numerator/denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2
        regularizer1 = regularizer1/self.batch_size
        #regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2
        #regularizer1 = regularizer1

        regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + self.batch_size * 0.5 * torch.norm(pos_pop_emb) ** 2
        regularizer2  = regularizer2/self.batch_size
        #regularizer2= 0.5 * torch.norm(users_pop_emb) ** 2 + self.batch_size * 0.5 * torch.norm(pos_pop_emb) ** 2
        #regularizer2  = regularizer2
        reg_loss = self.decay * (regularizer1+regularizer2)

        reg_loss_freeze=self.decay * (regularizer2)
        reg_loss_norm=self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm


#=============================================================Myself Adaptive loss============================================================#
class Adaptive_softmax_loss():
    def __init__(self, config, model:LightGCN, precal:precalculate, homophily:Homophily):
        self.config = config
        self.model = model
        self.precal = precal
        self.homophily = homophily
        self.tau = config['temp_tau']
        self.alpha = config['alpha']
        self.f = lambda x: torch.exp(x / self.tau)

    def adaptive_softmax_loss(self, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0, batch_user, batch_pos, batch_neg, aug_users1, aug_items1, aug_users2, aug_items2):
        # (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0) = self.model.getEmbedding(batch_user.long(), batch_pos.long(), batch_neg.long())
        # reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2))#/float(len(batch_user))
        
        # #loss1 = self.calculate_loss(users_emb, pos_emb, neg_emb)
        # #loss1 = self.calculate_loss(users_emb, pos_emb, pos_emb)#改为全正样本，分母自动多了自己一项，见BC_loss中loss1的实现
        # loss1, _ = self.model.bpr_loss(batch_user, batch_pos, batch_neg)
        # #loss1 = self.info_nce_loss_overall(users_emb, pos_emb, pos_emb)
        # loss2 = self.calculate_loss(aug_users1[batch_user], aug_users2[batch_user], aug_users2)
        # loss3 = self.calculate_loss(aug_items1[batch_pos], aug_items2[batch_pos], aug_items2)
        # #print(batch_user, batch_pos, batch_neg)
        # #print('reg',reg_loss,'bpr',loss1,'user',loss2,'item',loss3)
        # return self.config['weight_decay']*reg_loss + 1.0*loss1 + self.config['lambda1']*loss2 + self.config['lambda1']*loss3 
        
        reg = (0.5 * torch.norm(userEmb0) ** 2 + len(batch_pos) * 0.5 * torch.norm(posEmb0) ** 2)/len(batch_pos)
        loss1 = self.calculate_loss(users_emb, pos_emb, neg_emb, batch_user, batch_pos, self.config['adaptive_method'], self.config['centroid_mode'])
        #loss1 = self.calculate_loss(users_emb, pos_emb, neg_emb, None, None, None, None)
        if not (aug_users1 is None):
            loss2 = self.calculate_loss(aug_users1[batch_user], aug_users2[batch_user], aug_users2, None, None, None, None)
            loss3 = self.calculate_loss(aug_items1[batch_pos], aug_items2[batch_pos], aug_items2, None, None, None, None)
            loss = self.config['weight_decay']*reg + loss1 + self.config['lambda1']*(loss2 + loss3)
        else:
            loss = self.config['weight_decay']*reg + loss1
        
        return loss


    def calculate_loss(self, batch_target_emb, batch_pos_emb, batch_negs_emb, batch_target, batch_pos, method, mode):
        '''
        input : embeddings, not index.
        '''
        # '''
        # pos_term = self.alpha * self.sim_adaptive(batch_target, batch_pos) / self.tau
        # neg_term = self.f(self.sim(batch_target, batch_negs, mode='1:all'))
        # neg_term = (1-self.alpha) * torch.log(torch.sum(neg_term, 1))
        # return -torch.sum(pos_term - neg_term)
        # '''
        users_emb = batch_target_emb
        pos_emb = batch_pos_emb

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        if not (method is None):
            #Adaptive coef between User and Item
            pos_ratings_margin = self.get_coef_adaptive(batch_target, batch_pos, method=method, mode=mode)
            ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7))+(pos_ratings_margin))
            #reliable / important ==> big margin ==> small theta ==> big simi between u,i 
        else:
            ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag,-1+1e-7,1-1e-7)))
        
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        #loss = torch.mean(torch.negative(torch.log(numerator/denominator)))
        loss = torch.mean(torch.negative(2*self.alpha * torch.log(numerator) -  2*(1-self.alpha) * torch.log(denominator)))

        # '''
        # positive_pairs = self.f(self.sim_adaptive(batch_target, batch_pos))
        # all_sim = self.f(self.sim(batch_target, batch_negs, mode='1:all'))
        # negative_pairs = torch.sum(all_sim, 1)
        # loss = -torch.sum(2*self.alpha * torch.log(positive_pairs) - 2*(1-self.alpha) * torch.log(negative_pairs))
        # #loss = 2.0 * torch.sum(-(self.alpha * torch.log(positive_pairs) - (1-self.alpha) * torch.log(negative_pairs)))
        # return loss
        # '''

        return loss

    def get_coef_adaptive(self, batch_user, batch_pos_item, method='centroid', mode='eigenvector'):
        '''
        input: index batch_user & batch_pos_item\n
        return tensor([adaptive coefficient of u_n-i_n])\n
        the bigger, the more reliable, the more important
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
                csr_matrix_CN_simi = self.precal.common_neighbor.CN_simi_mat_sp
                batch_user, batch_pos_item = np.array(batch_user.cpu()), np.array(batch_pos_item.cpu())
                batch_weight1 = csr_matrix_CN_simi[batch_user, batch_pos_item+n_users]
                batch_weight2 = csr_matrix_CN_simi[batch_pos_item+n_users, batch_user]
                batch_weight1 = torch.tensor(np.array(batch_weight1).reshape((-1,)))
                batch_weight2 = torch.tensor(np.array(batch_weight2).reshape((-1,)))
                batch_weight = (batch_weight1 + batch_weight2) * 0.5
                batch_weight = batch_weight.to(world.device)
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

        return batch_weight

