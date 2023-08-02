import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
import multiprocessing



class BigLoader(BasicDataset):
    def __init__(self, select="yelp", device = 0):
        self.select = select
        self.path = "../data/" + str(self.select)
        self.device = device
        cprint(f"loading [{self.select}]")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        trainData = pd.read_table(join(self.path, 'ratings_train.txt'), sep='\t', header=None)
        testData = pd.read_table(join(self.path, 'ratings_test.txt'), sep='\t', header=None)
        trustNet = pd.read_table(join(self.path, 'trusts.txt'), sep='\t', header=None)
        self.trustNet = trustNet
        self.trustor = np.array(self.trustNet[:][0])
        self.trustee = np.array(self.trustNet[:][1])
        self.trainData = trainData
        self.testData = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        self.testUser = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(testData[:][1])

        self.statU = np.unique(np.append(self.trainUser, self.testUser))
        self.statU1 = np.append(self.trustee, self.trustor)
        self.statU = np.unique(np.append(self.statU, self.statU1))
        self.statI = np.unique(np.append(self.trainItem, self.testItem))

        self.Graph = None
        self.GraphPreferenceAug1 = None  # Used to indicate whether we have calculated the laplacian matrix for the propagation
        self.GraphPreferenceAug2 = None  # Used to indicate whether we have calculated the laplacian matrix for the propagation
        self.GraphSocialAug1 = None  # Used to indicate whether we have calculated the laplacian matrix for the propagation
        self.GraphSocialAug2 = None  # Used to indicate whether we have calculated the laplacian matrix for the propagation
        '''
        {'augmentation type': 'laplacian of the augmentation graph'}
        augmentation type: edge drop, node drop, random walk with restart.
        '''
        print(f"{self.select} Sparsity : {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items}")
        # (users,users)
        try:
            self.socialNet = csr_matrix((np.ones(len(trustNet)), (self.trustor, self.trustee)),
                                        shape=(self.n_users, self.n_users))
            # (users,items), bipartite graph
            # print('trainUser:',len(self.trainUser), 'trainItem', len(self.trainItem), self.n_users, self.m_items)
            self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                          shape=(self.n_users, self.m_items))
        except ValueError as e:
            print(e)
            for i in range(0, len(self.trustor)):
                if self.trustor[i] < 0:
                    print(str(i) + ':' + str(self.trustor[i]) + ',' + str(self.trustee[i]))
            for i in range(0, len(self.trustee)):
                if self.trustee[i] < 0:
                    print(str(i) + ':' + str(self.trustor[i]) + ',' + str(self.trustee[i]))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()
    @property
    def n_users(self):
        return self.statU.max() + 1
    @property
    def m_items(self):
        return self.statI.max() + 1
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    @property
    def testDict(self):
        return self.__testDict
    @property
    def allPos(self):
        return self._allPos

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                preAdjMat = sp.load_npz(self.path+'/pref_s_pre_adj_mat.npz')
                print("successfully loaded...")
                normAdj = preAdjMat
            except:
                print("generating adjacency matrix")
                s = time()
                adjMat = sp.dok_matrix((self.n_users+self.m_items, self.n_users+self.m_items), dtype=np.float32)
                adjMat = adjMat.tolil()
                R = self.UserItemNet.tolil()
                adjMat[:self.n_users, self.n_users:] = R
                adjMat[self.n_users:, :self.n_users] = R.T
                adjMat = adjMat.todok()

                rowsum = np.array(adjMat.sum(axis=1))
                dInv = np.power(rowsum, -0.5).flatten()
                dInv[np.isinf(dInv)] = 0.
                dMat = sp.diags(dInv)

                normAdj = dMat.dot(adjMat)
                normAdj = normAdj.dot(dMat)
                normAdj = normAdj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/pref_s_pre_adj_mat.npz', normAdj)
        self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
        self.Graph = self.Graph.coalesce().to(world.device)
        return  self.Graph

    def getSparseGraph_aug(self, aug_type=None, edge_drop_rate=0, node_drop_rate=0):
        print('loading augmentation adjacency matrix')
        try:
            preAdjMat = sp.load_npz(self.path + '/pref_s_pre_adj_mat_' + str(aug_type) + '_edRate_' + str(
                edge_drop_rate) + '_ndRate_' + str(edge_drop_rate) + '.npz')
            print("successfully loaded...")
            normAdj = preAdjMat
        except:
            print("generating augmentation adjacency matrix")
            s = time()
            adjMat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adjMat = adjMat.tolil()

            user_dim = self.trainUser
            item_dim = self.trainItem
            # ================Augmentation=================
            if aug_type == 'edge_drop':
                user_dim, item_dim = edgeDrop(user_dim, item_dim, edge_drop_rate)
            elif aug_type == 'node_drop':
                user_dim, item_dim = nodeDrop(self.n_users, self.m_items, user_dim, item_dim, node_drop_rate)
            elif aug_type == 'random_walk':
                user_dim, item_dim = nodeDrop(self.n_users, self.m_items, user_dim, item_dim, node_drop_rate)
                user_dim, item_dim = edgeDrop(user_dim, item_dim, edge_drop_rate)
            # ================Augmentation=================
            Aug_UserItemNet = csr_matrix((np.ones(len(user_dim)), (user_dim, item_dim)),
                                         shape=(self.n_users, self.m_items))

            R = Aug_UserItemNet.tolil()
            adjMat[:self.n_users, self.n_users:] = R
            adjMat[self.n_users:, :self.n_users] = R.T
            adjMat = adjMat.todok()

            rowsum = np.array(adjMat.sum(axis=1))
            dInv = np.power(rowsum, -0.5).flatten()
            dInv[np.isinf(dInv)] = 0.
            dMat = sp.diags(dInv)

            normAdj = dMat.dot(adjMat)
            normAdj = normAdj.dot(dMat)
            normAdj = normAdj.tocsr()
            end = time()
            print(f"costing {end - s}s, saved norm_mat...")
            sp.save_npz(self.path + '/pref_s_pre_adj_mat_' + str(aug_type) + '_edRate_' + str(
                edge_drop_rate) + '_ndRate_' + str(edge_drop_rate) + '.npz', normAdj)
        self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
        self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def getSparseGraph_aug_social(self, aug_type=None, edge_drop_rate=0, node_drop_rate=0):
        print('loading augmentation social adjacency matrix')
        try:
            preAdjMat = sp.load_npz(
                self.path + '/social_s_pre_adj_mat_' + str(aug_type) + '_edRate_' + str(
                    edge_drop_rate) + '_ndRate_' + str(
                    edge_drop_rate) + '.npz')
            print("successfully loaded...")
            normAdj = preAdjMat
        except:
            print("generating augmentation social adjacency matrix")
            s = time()
            adjMat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adjMat = adjMat.tolil()

            trustor = self.trustor
            trustee = self.trustee
            # ================Augmentation=================
            if aug_type == 'edge_drop':
                trustor, trustee = edgeDrop(trustor, trustee, edge_drop_rate)
            elif aug_type == 'node_drop':
                trustor, trustee = nodeDrop_social(self.n_users, trustor, trustee, node_drop_rate)
            elif aug_type == 'random_walk':
                trustor, trustee = nodeDrop_social(self.n_users, trustor, trustee, node_drop_rate)
                trustor, trustee = edgeDrop(trustor, trustee, edge_drop_rate)
            # ================Augmentation=================

            Aug_SocialNet = csr_matrix((np.ones(len(self.trustNet)), (trustor, trustee)),
                                       shape=(self.n_users, self.n_users))

            R = Aug_SocialNet.tolil()
            adjMat[:self.n_users, self.n_users:] = R
            adjMat[self.n_users:, :self.n_users] = R.T
            adjMat = adjMat.todok()

            rowsum = np.array(adjMat.sum(axis=1))
            dInv = np.power(rowsum, -0.5).flatten()
            dInv[np.isinf(dInv)] = 0.
            dMat = sp.diags(dInv)

            normAdj = dMat.dot(adjMat)
            normAdj = normAdj.dot(dMat)
            normAdj = normAdj.tocsr()
            end = time()
            print(f"costing {end - s}s, saved norm_mat...")
            sp.save_npz(
                self.path + '/social_s_pre_adj_mat_' + str(aug_type) + '_edRate_' + str(
                    edge_drop_rate) + '_ndRate_' + str(
                    edge_drop_rate) + '.npz', normAdj)
        self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
        self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

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

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

