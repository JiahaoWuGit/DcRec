import torch
from dataloader import BasicDataset
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import Dropout, Linear, LayerNorm
import datetime
import random
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class Model(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(Model, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        if self.config['ssl_enable']:
            if self.config['social_enable']:
                self.model = 'CycleContra'
            else:
                self.model = 'ssl_LightGCN'
        else:
            self.model = 'LightGCN'

        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['model_n_layers']
        self.social_n_layers = self.config['social_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.contrastMLP = self.config['contrastMLP']
        self.social_contrastMLP = self.config['social_contrastMLP']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_user_social = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print('======================use NORMAL distribution initializer')
        self.f = nn.Sigmoid()
        # initialization for preference part
        self.Graph = self.dataset.getSparseGraph(self.config['lOrls'])
        self.Graph_View1 = None
        self.Graph_View2 = None
        if self.config['ssl_enable']:
            self.Graph_View1 = self.dataset.getSparseGraph_aug(self.config['aug_type1'], self.config['edge_drop_rate1'],
                                                               self.config['node_drop_rate1'], self.config['sglOrsgls'])
            self.Graph_View2 = self.dataset.getSparseGraph_aug(self.config['aug_type2'], self.config['edge_drop_rate2'],
                                                               self.config['node_drop_rate2'], self.config['sglOrsgls'])

            self.input_dim = self.config['bpr_batch_size'] * self.latent_dim
            self.output_dim = self.config['bpr_batch_size'] * self.latent_dim
            self.hidden_dim = self.config['cycle_hidden_dim']
            self.mlp_dropout_value = self.config['dropout_mlp']


            self.layerNorm = LayerNorm(self.hidden_dim, eps=1e-6)
            self.mlp_dropout = Dropout(self.mlp_dropout_value)

            self.mlp_prefer_user_layer1 = Linear(self.input_dim, self.hidden_dim)
            self.mlp_prefer_user_layer2 = Linear(self.hidden_dim, self.output_dim)
            self.mlp_prefer_user_act = torch.nn.functional.elu


            self.mlp_prefer_item_layer1 = Linear(self.input_dim, self.hidden_dim)
            self.mlp_prefer_item_layer2 = Linear(self.hidden_dim, self.output_dim)
            self.mlp_prefer_item_act = torch.nn.functional.elu


        print(f"{self.model} is already to go(dropout:{self.config['dropout']})")

        # Initialization for social part
        self.Graph_socialView1 = None
        self.Graph_socialView2 = None
        if self.config['social_enable']:
            self.Graph_socialView1 = self.dataset.getSparseGraph_aug_social(self.config['aug_social1'],
                                                                            self.config['ed_social1'],
                                                                            self.config['nd_social1'])
            self.Graph_socialView2 = self.dataset.getSparseGraph_aug_social(self.config['aug_social2'],
                                                                            self.config['ed_social2'],
                                                                            self.config['nd_social2'])
            self.mlp_social_layer1 = Linear(self.input_dim, self.hidden_dim)
            self.mlp_social_layer2 = Linear(self.hidden_dim, self.output_dim)
            self.mlp_social_act = torch.nn.functional.elu

            self.propagation_weight = []
            for i in range(self.social_n_layers):
                self.propagation_weight.append(Parameter(torch.FloatTensor(self.latent_dim, self.latent_dim).to(self.config['DEVICE'])))
            for i in range(self.social_n_layers):
                nn.init.normal_(self.propagation_weight[i],std=0.1)

        # Initialization for cycle part
        if self.config['social_enable']:
            self.mlp_social2prefer_layer1 = Linear(self.input_dim, self.hidden_dim)
            self.mlp_social2prefer_layer2 = Linear(self.hidden_dim, self.output_dim)
            self.mlp_social2prefer_act = nn.functional.elu

            self.mlp_prefer2social_layer1 = Linear(self.input_dim, self.hidden_dim)
            self.mlp_prefer2social_layer2 = Linear(self.hidden_dim, self.output_dim)
            self.mlp_prefer2social_act = nn.functional.elu

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def mlp_social(self, input_emb, mlpOrNot):
        if mlpOrNot:
            x = self.mlp_social_layer1(input_emb)
            x = self.mlp_social_act(x)
            y = self.mlp_social_layer2(x)
        else:
            y = input_emb
        return y
    def mlp_prefer_user(self, input_emb, mlpOrNot):
        if mlpOrNot:
            x = self.mlp_prefer_user_layer1(input_emb)
            x = self.mlp_prefer_user_act(x)
            y = self.mlp_prefer_user_layer2(x)
        else:
            y = input_emb
        return y
    def mlp_prefer_item(self, input_emb, mlpOrNot):
        if mlpOrNot:
            x = self.mlp_prefer_item_layer1(input_emb)
            x = self.mlp_prefer_item_act(x)
            y = self.mlp_prefer_item_layer2(x)
        else:
            y = input_emb
        return y
    def mlp_social2prefer(self, input_emb, mlpOrNot):
        if mlpOrNot:
            x = self.mlp_social2prefer_layer1(input_emb)
            x = self.mlp_social2prefer_act(x)
            y = self.mlp_social2prefer_layer2(x)
        else:
            y = input_emb
        return y
    def mlp_prefer2social(self, input_emb, mlpOrNot):
        if mlpOrNot:
            x = self.mlp_prefer2social_layer1(input_emb)
            x = self.mlp_prefer2social_act(x)
            y = self.mlp_prefer2social_layer2(x)
        else:
            y = input_emb
        return y


    def propagation(self, users_id, items_id, g_dropped, social):
        all_emb = torch.cat([users_id, items_id])

        embs = [all_emb]
        if not social:
            n_layers = self.n_layers
        else:
            n_layers = self.social_n_layers
        for i in range(n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    temp_emb.append(torch.sparse.mm(g_dropped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_dropped, all_emb)
                if social:
                    all_emb = torch.mm(all_emb, self.propagation_weight[i])
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs,
                               dim=1)  
        if not social:
            users, items = torch.split(light_out, [self.num_users, self.num_items])
            return users, items
        else:
            users1, users2 = torch.split(light_out, [self.num_users, self.num_users])
            return users1

    def computer(self):
        """
        get the embeddings after propagation in the rules of LightGCN.
        """
        if self.config['dropout']:
            if self.training:
                g_dropped = self.__dropout(self.keep_prob)
            else:
                g_dropped = self.Graph
        else:
            g_dropped = self.Graph
        users, items = self.propagation(self.embedding_user.weight, self.embedding_item.weight, g_dropped, False)
        if not self.config['ssl_enable']:
            users1, items1, users2, items2 = None, None, None, None
        else:
            users1, items1 = self.propagation(self.embedding_user.weight, self.embedding_item.weight, self.Graph_View1,
                                              False)
            users2, items2 = self.propagation(self.embedding_user.weight, self.embedding_item.weight, self.Graph_View2,
                                              False)
        if not self.config['social_enable']:
            user_social1, user_social2 = None, None
        else:
            user_social1 = self.propagation(self.embedding_user_social.weight, self.embedding_user_social.weight,
                                            self.Graph_socialView1, True)
            user_social2 = self.propagation(self.embedding_user_social.weight, self.embedding_user_social.weight,
                                            self.Graph_socialView2, True)
        return users, items, users1, items1, users2, items2, user_social1, user_social2

    def getUsersRating(self, users):
        all_users, all_items, users1, items1, users2, items2, user_social1, user_social2 = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        '''
        :return: users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
        the embeddings returned are used to calculate the bpr loss
        '''
        all_users, all_items, all_users1, all_items1, all_users2, all_items2, user_social1, user_social2 = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getEmbeddingSSL(self, users, items, preference):
        '''
        :return: the embeddings for view1, view2 to calculate the ssl loss for the preference view
        '''

        all_users, all_items, all_users1, all_items1, all_users2, all_items2, user_social1, user_social2 = self.computer()
        users_emb1 = all_users1[users]
        users_emb2 = all_users2[users]
        if preference:  # If it's used to calculate the preference ssl loss
            items_emb1 = all_items1[items]
            items_emb2 = all_items2[items]
            return users_emb1, users_emb2, items_emb1, items_emb2
        else:
            return users_emb1, users_emb2

    def getEmbeddingSSL_social(self, users):
        all_users, all_items, all_users1, all_items1, all_users2, all_items2, user_social1, user_social2 = self.computer()
        user_social_emb1 = user_social1[users]
        user_social_emb2 = user_social2[users]
        return user_social_emb1, user_social_emb2

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        userEmbSocial = self.embedding_user_social(users.long())
        reg_social_loss = (1 / 2) * ((userEmbSocial.norm(2).pow(2)) / float(len(users)))
        # (len(users_emb), len(pos_emb), len(neg_emb))
        # x = input()
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss, reg_social_loss  # loss, reg_loss: bpr_loss, regularization loss


    def calc_ssl_sim(self, emb1, emb2, tau, normalization = False):
        # (emb1, emb2) = (F.normalize(emb1, p=2, dim=0), F.normalize(emb2, p=2, dim=0))\
        if normalization:
            emb1 = nn.functional.normalize(emb1, p=2, dim=1, eps=1e-12)
            emb2 = nn.functional.normalize(emb2, p=2, dim=1, eps=1e-12)

        (emb1_t, emb2_t) = (emb1.t(), emb2.t())

        pos_scores_users = torch.exp(torch.div(F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8), tau))  # Sum by row
        # denominator cosine_similarity: following codes
        if self.config['interOrIntra'] == 'inter':

            denominator_scores = torch.mm(emb1, emb2_t)
            norm_emb1 = torch.norm(emb1, dim=-1)
            norm_emb2 = torch.norm(emb2, dim=-1)
            norm_emb = torch.mm(norm_emb1.unsqueeze(1), norm_emb2.unsqueeze(1).t())

            denominator_scores1 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(1)  # Sum by row
            denominator_scores2 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(0)  # Sum by column
            # denominator cosine_similarity: above codes

            ssl_loss1 = -torch.mean(torch.log(pos_scores_users / denominator_scores1))
            ssl_loss2 = -torch.mean(torch.log(pos_scores_users / denominator_scores2))
        else:  # interAintra
            denominator_scores = torch.mm(emb1, emb2_t)
            norm_emb1 = torch.norm(emb1, dim=-1)
            norm_emb2 = torch.norm(emb2, dim=-1)
            norm_emb = torch.mm(norm_emb1.unsqueeze(1), norm_emb2.unsqueeze(1).t())
            denominator_scores1 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(1)  # Sum by row
            denominator_scores2 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(0)  # Sum by column

            denominator_scores_intraview1 = torch.mm(emb1, emb1_t)
            norm_intra1 = torch.mm(norm_emb1.unsqueeze(1), norm_emb1.unsqueeze(1).t())
            denominator_intra_scores1 = torch.exp(torch.div(denominator_scores_intraview1 / norm_intra1, tau))
            diag1 = torch.diag(denominator_intra_scores1)
            d_diag1 = torch.diag_embed(diag1)
            denominator_intra_scores1 = denominator_intra_scores1 - d_diag1  # here we set the elements on diagonal to be 0.
            intra_denominator_scores1 = denominator_intra_scores1.sum(1)  # Sum by row#
            # .sum(1)

            denominator_scores_intraview2 = torch.mm(emb2, emb2_t)
            norm_intra2 = torch.mm(norm_emb2.unsqueeze(1), norm_emb2.unsqueeze(1).t())
            denominator_intra_scores2 = torch.exp(torch.div(denominator_scores_intraview2 / norm_intra2, tau))
            diag2 = torch.diag(denominator_intra_scores2)
            d_diag2 = torch.diag_embed(diag2)
            denominator_intra_scores2 = denominator_intra_scores2 - d_diag2
            intra_denominator_scores2 = denominator_intra_scores2.sum(1)

            # denominator cosine_similarity: above codes
            ssl_loss1 = -torch.mean(torch.log(pos_scores_users / (denominator_scores1 + intra_denominator_scores1)))
            ssl_loss2 = -torch.mean(torch.log(pos_scores_users / (denominator_scores2 + intra_denominator_scores2)))


        return ssl_loss1 + ssl_loss2

    def preference_ssl_loss(self, batch_users, batch_items, tau):
        (users_emb1, users_emb2, items_emb1, items_emb2) = self.getEmbeddingSSL(batch_users.long(), batch_items.long(),
                                                                                True)

        input_len_user = len(users_emb1) * len(users_emb1[0])
        origin_shape_user = (len(users_emb1), len(users_emb1[0]))

        input_len_item = len(items_emb1) * len(items_emb1[0])
        origin_shape_item = (len(items_emb1), len(items_emb1[0]))
        users_emb1 = self.mlp_prefer_user(users_emb1.reshape(1, input_len_user),
                                                           self.contrastMLP).reshape(origin_shape_user)
        users_emb2 = self.mlp_prefer_user(users_emb2.reshape(1, input_len_user),
                                                           self.contrastMLP).reshape(origin_shape_user)
        items_emb1 = self.mlp_prefer_item(items_emb1.reshape(1, input_len_item),
                                                           self.contrastMLP).reshape(origin_shape_item)
        items_emb2 = self.mlp_prefer_item(items_emb2.reshape(1, input_len_item),
                                                           self.contrastMLP).reshape(origin_shape_item)
        normalized_users_emb1 = nn.functional.normalize(users_emb1, p=2, dim=1,eps=1e-12)
        normalized_users_emb2 = nn.functional.normalize(users_emb2, p=2, dim=1, eps=1e-12)
        normalized_items_emb1 = nn.functional.normalize(items_emb1, p=2, dim=1, eps=1e-12)
        normalized_items_emb2 = nn.functional.normalize(items_emb2, p=2, dim=1, eps=1e-12)
        ssl_loss_user = self.calc_ssl_sim(normalized_users_emb1, normalized_users_emb2, tau)
        ssl_loss_item = self.calc_ssl_sim(normalized_items_emb1, normalized_items_emb2, tau)

        return ssl_loss_user, ssl_loss_item

    def social_ssl_loss(self, batch_users, tau):

        user_social1, user_social2 = self.getEmbeddingSSL_social(batch_users.long())
        input_len = len(user_social1) * len(user_social1[0])
        origin_shape = (len(user_social1), len(user_social1[0]))
        user_social1 = self.mlp_social(user_social1.reshape(1, input_len),
                                                        self.social_contrastMLP).reshape(origin_shape)
        user_social2 = self.mlp_social(user_social2.reshape(1, input_len),
                                                        self.social_contrastMLP).reshape(origin_shape)
        normalized_user_social1 = nn.functional.normalize(user_social1, p=2, dim=1, eps=1e-12)
        normalized_user_social2 = nn.functional.normalize(user_social2, p=2, dim=1, eps=1e-12)
        ssl_loss_social = self.calc_ssl_sim(normalized_user_social1, normalized_user_social2, tau)
        return ssl_loss_social

    def cycle_ssl_loss(self, batch_users, tau):
        '''
        :return:
        '''
        if len(batch_users) != self.config['bpr_batch_size']:
            return 0, 0
        users_preference1, users_preference2 = self.getEmbeddingSSL(batch_users.long(), None, False)
        users_social1, users_social2 = self.getEmbeddingSSL_social(batch_users.long())

        input_len_social1 = len(users_preference1) * len(users_preference1[0])
        input_len_preference2 = len(users_social2) * len(users_social2[0])

        origin_shape_social1 = (len(users_preference1), len(users_preference1[0]))
        origin_shape_preference2 = (len(users_social2), len(users_social2[0]))


        converted_prefer1 = self.mlp_prefer2social(users_preference1.reshape(1, input_len_social1),
                                                                   self.config['cycleMLP']).reshape(
            origin_shape_social1)
        converted_prefer2 = self.mlp_prefer2social(users_preference2.reshape(1, input_len_social1),
                                                   self.config['cycleMLP']).reshape(
            origin_shape_social1)
        converted_social1 = self.mlp_social2prefer(users_social1.reshape(1, input_len_preference2),
                                                   self.config['cycleMLP']).reshape(
            origin_shape_preference2)
        converted_social2 = self.mlp_social2prefer(users_social2.reshape(1, input_len_preference2),
                                                                       self.config['cycleMLP']).reshape(
            origin_shape_preference2)

        cycle_loss_social = self.calc_ssl_sim(converted_prefer1, converted_social1, tau) + self.calc_ssl_sim(converted_prefer1, converted_social2, tau)
        cycle_loss_preference = self.calc_ssl_sim(converted_prefer2, converted_social1, tau) + self.calc_ssl_sim(converted_prefer2, converted_social2, tau)
        # ===================
        return cycle_loss_social, cycle_loss_preference

    def forward(self, users, items):  # I was wondering if here we should change accordingly
        # compute embedding
        all_users, all_items, users1, items1, users2, items2, user_social1, user_social2 = self.computer()

        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

