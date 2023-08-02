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


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset to return all the negative items
        """
        raise NotImplementedError
    
    def getSparseGraph(self, lOrls):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

    def getSparseGraph_aug(self, aug_type, edge_drop_rate, node_drop_rate, sglOrsgls='sgl'):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
    def getSparseGraph_aug_social(self, aug_type=None, edge_drop_rate = 0, node_drop_rate = 0):#使用random walk的时候，需要同时specify edge_drop_rate和node_drop_rate.
        '''

        :param aug_type:
        :param edge_drop_rate:
        :param node_drop_rate:
        :return:
        '''
        raise NotImplementedError

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

class Loader(BasicDataset):
    def __init__(self, select="lastfm", device = 0, ratioLowPreserve = 0.3, prejudice = False):
        # train or test
        self.ratioLowPreserve = ratioLowPreserve
        self.prejudice = prejudice  # Used to decide use prejudice or not, which decides whether out data augmentation will be conducted on low degree nodes.
        self.select =select
        self.path = "../data/"+str(self.select)
        self.device = device
        cprint(f"loading [{self.select}]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(self.path, 'ratings_train.txt'), sep='\t', header=None)
        testData  = pd.read_table(join(self.path, 'ratings_test.txt'), sep='\t', header=None)
        trustNet  = pd.read_table(join(self.path, 'trusts.txt'), sep='\t', header=None)
        self.trustNet  = trustNet
        self.trustor = np.array(self.trustNet[:][0])
        self.trustee = np.array(self.trustNet[:][1])
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])

        self.statU = np.unique(np.append(self.trainUser, self.testUser))
        self.statU1 = np.append(self.trustee, self.trustor)
        self.statU = np.unique(np.append(self.statU, self.statU1))
        self.statI = np.unique(np.append(self.trainItem, self.testItem))

        self.Graph = None
        self.GraphPreferenceAug1 = None#
        self.GraphPreferenceAug2 = None#
        self.GraphSocialAug1 = None#
        self.GraphSocialAug2 = None#

        '''
        {'augmentation type': 'laplacian of the augmentation graph'}
        augmentation type: edge drop, node drop, random walk with restart.
        '''

        print(f"{self.select} Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        # (users,users)
        try:
            self.socialNet    = csr_matrix((np.ones(len(trustNet)), (self.trustor, self.trustee)), shape=(self.n_users,self.n_users))
            self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_users,self.m_items))
        except ValueError as e:
            print(e)
            for i in range(0,len(self.trustor)):
                if self.trustor[i] < 0:
                    print(str(i)+':'+str(self.trustor[i])+','+str(self.trustee[i]))
            for i in range(0,len(self.trustee)):
                if self.trustee[i] < 0:
                    print(str(i) + ':' + str(self.trustor[i])+','+str(self.trustee[i]))


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
        return self.statU.max()+1

    @property
    def m_items(self):
        return self.statI.max()+1
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self, lOrls='lightGCN'):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            if lOrls == 'lightGCN':
                first_sub = torch.stack([user_dim, item_dim + self.n_users])
                second_sub = torch.stack([item_dim+self.n_users, user_dim])
                index = torch.cat([first_sub, second_sub], dim=1)
            else:#'lightGCN_social'
                trustor_dim = torch.LongTensor(self.trustor)
                trustee_dim = torch.LongTensor(self.trustee)
                first_sub = torch.stack([user_dim, item_dim + self.n_users])
                second_sub = torch.stack([item_dim + self.n_users, user_dim])
                first_sub_social = torch.stack([trustor_dim, trustee_dim])
                index = torch.cat([first_sub, second_sub, first_sub_social], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items])).to(self.device)
            #print(self.Graph)
            dense = self.Graph.to_dense()
            #print(dense.shape)
            D = torch.sum(dense, dim=1).float().to(self.device)
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0).to(self.device)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9].to(self.device)
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(self.device)
        return self.Graph

    def getSparseGraph_aug(self, aug_type=None, edge_drop_rate = 0, node_drop_rate = 0, sglOrsgls='sgl'):#使用random walk的时候，需要同时specify edge_drop_rate和node_drop_rate.
        assert len(self.trainUser) == len(self.trainItem)
        user_dim = torch.LongTensor(self.trainUser)
        item_dim = torch.LongTensor(self.trainItem)
        #================Augmentation=================
        if aug_type == 'edge_drop':
            low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
            # print('Preference edge drop:')
            # print('len of low_degree_users and low_degree_items:', len(low_degree_users), '&', len(low_degree_items))
            user_dim, item_dim = edgeDrop(user_dim, item_dim, edge_drop_rate, low_degree_users, self.prejudice)
        elif aug_type == 'node_drop':
            user_dim, item_dim = nodeDrop(self.n_users, self.m_items, user_dim, item_dim, node_drop_rate)
        elif aug_type == 'random_walk':
            user_dim, item_dim = nodeDrop(self.n_users, self.m_items, user_dim, item_dim, node_drop_rate)
            low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
            user_dim, item_dim = edgeDrop(user_dim, item_dim, edge_drop_rate, low_degree_users, self.prejudice)
        elif aug_type == 'edge_add':
            low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
            user_dim, item_dim = edgeAdd(user_dim, item_dim, edge_drop_rate, low_degree_users, self.prejudice, self.path)
            user_dim = torch.LongTensor(user_dim)
            item_dim = torch.LongTensor(item_dim)

        # ================Augmentation=================
        if sglOrsgls == 'sgl':
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])  # 因为要把items和users合并到一起，所以要给items的编号都往后推n_users个
            index = torch.cat([first_sub, second_sub], dim=1)
        else:
            trustor_dim = torch.LongTensor(self.trustor)
            trustee_dim = torch.LongTensor(self.trustee)
            if aug_type == 'edge_drop':
                low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
                trustor_dim, trustee_dim = edgeDrop(trustor_dim, trustee_dim, edge_drop_rate, low_degree_users, self.prejudice)
            elif aug_type == 'node_drop':
                trustor_dim, trustee_dim = nodeDrop_social(self.n_users, trustor_dim, trustee_dim, node_drop_rate)
            elif aug_type == 'random_walk':
                trustor_dim, trustee_dim = nodeDrop_social(self.n_users, trustor_dim, trustee_dim, node_drop_rate)
                low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
                trustor_dim, trustee_dim = edgeDrop(trustor_dim, trustee_dim, edge_drop_rate, low_degree_users, self.prejudice)
            elif aug_type == 'edge_add':
                low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
                user_dim, item_dim = edgeAdd(user_dim, item_dim, edge_drop_rate, low_degree_users, self.prejudice, self.path)
                user_dim = torch.LongTensor(user_dim)
                item_dim = torch.LongTensor(item_dim)
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])  
            first_sub_social = torch.stack([trustor_dim, trustee_dim])
            index = torch.cat([first_sub, second_sub, first_sub_social], dim=1)
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data,
                                            torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()#Degree matrix.
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero()
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
            [self.n_users + self.m_items, self.n_users + self.m_items]))
        Graph = Graph.coalesce().to(self.device)
        '''
        Returns a coalesced copy of self if self is an uncoalesced tensor.
        Returns self if self is a coalesced tensor.
        '''
        return Graph

    def getSparseGraph_aug_social(self, aug_type=None, edge_drop_rate = 0, node_drop_rate = 0):
        user_dim = torch.LongTensor(self.trainUser)
        item_dim = torch.LongTensor(self.trainItem)
        trustor = torch.LongTensor(self.trustor)
        trustee = torch.LongTensor(self.trustee)
        #================Augmentation=================
        low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
        if aug_type == 'edge_drop':
            trustor, trustee = edgeDrop(trustor, trustee, edge_drop_rate, low_degree_users, self.prejudice)
        elif aug_type == 'node_drop':
            trustor, trustee = nodeDrop_social(self.n_users, trustor, trustee, node_drop_rate, low_degree_users, self.prejudice)
        elif aug_type == 'random_walk':
            trustor, trustee = nodeDrop_social(self.n_users, trustor, trustee, node_drop_rate, low_degree_users, self.prejudice)
            low_degree_users= []
            trustor, trustee = edgeDrop(trustor, trustee, edge_drop_rate, low_degree_users, False)
        elif aug_type == 'edge_add':
            trustor, trustee = edgeAdd(trustor, trustee, edge_drop_rate, low_degree_users, self.prejudice, self.path, True)
            trustor = torch.LongTensor(trustor)
            trustee = torch.LongTensor(trustee)
        # ================Augmentation=================
        first_sub = torch.stack([trustor, trustee])
        second_sub = torch.stack([trustee, trustor])
        index = torch.cat([first_sub,second_sub], dim=1)
        data = torch.ones(index.size(-1))
        Graph = torch.sparse.IntTensor(index, data, torch.Size([2*self.n_users, 2*self.n_users]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D==0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        dense = dense/D_sqrt.t()
        index = dense.nonzero()
        data = dense[dense>=1e-9]
        assert  len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([2*self.n_users, 2*self.n_users]))
        Graph = Graph.coalesce().to(self.device)
        return Graph

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
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class BigLoader(BasicDataset):
    def __init__(self, select="yelp", device = 0, ratioLowPreserve = 0.3, prejudice = False):
        self.ratioLowPreserve = ratioLowPreserve
        self.prejudice = prejudice#Used to decide use prejudice or not, which decides whether out data augmentation will be conducted on low degree nodes.
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
        if self.select!='epinions':
            self.trustor = np.array(self.trustNet[:][0])
            self.trustee = np.array(self.trustNet[:][1])
        else:
            self.trustor = np.array(self.trustNet[:][1])
            self.trustee = np.array(self.trustNet[:][2])
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
        self.GraphPreferenceAug1 = None  # 
        self.GraphPreferenceAug2 = None  # 
        self.GraphSocialAug1 = None  # 
        self.GraphSocialAug2 = None  # 

        print(f"{self.select} Sparsity : {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items}")
        # (users,users)
        try:
            self.socialNet = csr_matrix((np.ones(len(trustNet)), (self.trustor, self.trustee)),
                                        shape=(self.n_users, self.n_users))
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

    def getSparseGraph(self, lOrls='lightGCN'):
        print("loading adjacency matrix")
        if self.Graph is None:

            try:
                if lOrls == 'lightGCN':
                    preAdjMat = sp.load_npz(self.path+'/pref_pre_adj_mat.npz')
                else:
                    preAdjMat = sp.load_npz(self.path+'/pref_lightSocial_pre_adj_mat.npz')
                print("successfully loaded...")
                normAdj = preAdjMat
            except:

                print("generating adjacency matrix")
                s = time()
                adjMat = sp.dok_matrix((self.n_users+self.m_items, self.n_users+self.m_items), dtype=np.float32)
                adjMat = adjMat.tolil()
                if lOrls == 'lightGCN':
                    print('lightGCN_encoder')
                    R = self.UserItemNet.tolil()
                    adjMat[:self.n_users, self.n_users:] = R
                    adjMat[self.n_users:, :self.n_users] = R.T
                else:
                    print('lightGCN_social_encoder')
                    R = self.UserItemNet.tolil()
                    adjMat[:self.n_users, self.n_users:] = R
                    adjMat[self.n_users:, :self.n_users] = R.T
                    adjMat[:self.n_users,:self.n_users] = self.socialNet.tolil()
                adjMat = adjMat.todok()

                rowsum = np.array(adjMat.sum(axis=1))
                #print('Getting sparse graph')
                #print(rowsum)
                dInv = np.power(rowsum, -0.5).flatten()
                dInv[np.isinf(dInv)] = 0.
                dMat = sp.diags(dInv)

                normAdj = dMat.dot(adjMat)
                normAdj = normAdj.dot(dMat)
                normAdj = normAdj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")

                if lOrls == 'lightGCN':
                    sp.save_npz(self.path + '/pref_s_pre_adj_mat.npz', normAdj)
                else:
                    sp.save_npz(self.path + '/pref_lightSocial_pre_adj_mat.npz', normAdj)

        self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
        self.Graph = self.Graph.coalesce().to(self.device)
        return  self.Graph

    def getSparseGraph_aug(self, aug_type=None, edge_drop_rate=0, node_drop_rate=0, sglOrsgls='sgl'):
        print('loading augmentation adjacency matrix')

        print("generating augmentation adjacency matrix")
        s = time()
        adjMat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adjMat = adjMat.tolil()

        user_dim = self.trainUser
        item_dim = self.trainItem
        # ================Augmentation=================
        if aug_type == 'edge_drop':
            low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
            user_dim, item_dim = edgeDrop(user_dim, item_dim, edge_drop_rate, low_degree_users, self.prejudice)
        elif aug_type == 'node_drop':
            user_dim, item_dim = nodeDrop(self.n_users, self.m_items, user_dim, item_dim, node_drop_rate)
        elif aug_type == 'random_walk':
            user_dim, item_dim = nodeDrop(self.n_users, self.m_items, user_dim, item_dim, node_drop_rate)
            low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
            user_dim, item_dim = edgeDrop(user_dim, item_dim, edge_drop_rate, low_degree_users, self.prejudice)
        elif aug_type == 'edge_add':
            low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
            user_dim, item_dim = edgeAdd(user_dim, item_dim, edge_drop_rate, low_degree_users, self.prejudice, self.path)
            user_dim = torch.LongTensor(user_dim)
            item_dim = torch.LongTensor(item_dim)

        # ================Augmentation=================
        Aug_UserItemNet = csr_matrix((np.ones(len(user_dim)), (user_dim, item_dim)),
                                     shape=(self.n_users, self.m_items))

        R = Aug_UserItemNet.tolil()
        if sglOrsgls == 'sgl':
            adjMat[:self.n_users, self.n_users:] = R
            adjMat[self.n_users:, :self.n_users] = R.T
        else:
            adjMat[:self.n_users, self.n_users:] = R
            adjMat[self.n_users:, :self.n_users] = R.T
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
            elif aug_type == 'edge_add':
                low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
                trustor, trustee = edgeAdd(trustor, trustee, edge_drop_rate, low_degree_users, self.prejudice, self.path)
                trustor = torch.LongTensor(trustor)
                trustee = torch.LongTensor(trustee)
            # ================Augmentation=================
            Aug_SocialNet = csr_matrix((np.ones(len(trustor)), (trustor, trustee)),
                                       shape=(self.n_users, self.n_users))
            adjMat[:self.n_users, :self.n_users] = Aug_SocialNet.tolil()

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
        if sglOrsgls == 'sgl':
            sp.save_npz(self.path + '/pref_aug_pre_adj_mat_' + str(aug_type) + '_edRate_' + str(
                edge_drop_rate) + '_ndRate_' + str(edge_drop_rate) + '.npz', normAdj)
        else:
            sp.save_npz(self.path + '/sgls_pref_aug_pre_adj_mat_' + str(aug_type) + '_edRate_' + str(
                edge_drop_rate) + '_ndRate_' + str(edge_drop_rate) + '.npz', normAdj)

        self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
        self.Graph = self.Graph.coalesce().to(self.device)
        return self.Graph

    def getSparseGraph_aug_social(self, aug_type=None, edge_drop_rate=0, node_drop_rate=0):
        print('loading augmentation social adjacency matrix')
        user_dim = torch.LongTensor(self.trainUser)
        item_dim = torch.LongTensor(self.trainItem)
        try:
            preAdjMat = sp.load_npz(
                self.path + '/social_aug_pre_adj_mat_' + str(aug_type) + '_edRate_' + str(
                    edge_drop_rate) + '_ndRate_' + str(
                    edge_drop_rate) + '.npz')
            print("successfully loaded...")
            normAdj = preAdjMat
        except:

            print("generating augmentation social adjacency matrix")
            s = time()
            adjMat = sp.dok_matrix((self.n_users + self.n_users, self.n_users + self.n_users), dtype=np.float32)
            adjMat = adjMat.tolil()

            trustor = self.trustor
            trustee = self.trustee
            # ================Augmentation=================
            low_degree_users, low_degree_items = degreeCount(user_dim, item_dim, self.ratioLowPreserve)
            if aug_type == 'edge_drop':
                trustor, trustee = edgeDrop(trustor, trustee, edge_drop_rate, low_degree_users, self.prejudice)
            elif aug_type == 'node_drop':
                trustor, trustee = nodeDrop_social(self.n_users, trustor, trustee, node_drop_rate, low_degree_users,
                                                   self.prejudice)
            elif aug_type == 'random_walk':
                trustor, trustee = nodeDrop_social(self.n_users, trustor, trustee, node_drop_rate, low_degree_users,
                                                   self.prejudice)
                low_degree_users = []
                trustor, trustee = edgeDrop(trustor, trustee, edge_drop_rate, low_degree_users, False)
            elif aug_type == 'edge_add':
                trustor, trustee = edgeAdd(trustor, trustee, edge_drop_rate, low_degree_users, self.prejudice, self.path, True)
                trustor = torch.LongTensor(trustor)
                trustee = torch.LongTensor(trustee)
            # ================Augmentation=================
            Aug_SocialNet = csr_matrix((np.ones(len(trustor)), (trustor, trustee)),
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
                self.path + '/social_aug_pre_adj_mat_' + str(aug_type) + '_edRate_' + str(
                    edge_drop_rate) + '_ndRate_' + str(
                    edge_drop_rate) + '.npz', normAdj)

        self.Graph = self._convert_sp_mat_to_sp_tensor(normAdj)
        self.Graph = self.Graph.coalesce().to(self.device)
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

def degreeCount(heads, tails, ratioLowPreserve):
    head_degrees = {}
    tail_degrees = {}
    for head in heads:
        if head not in head_degrees.keys():
            head_degrees[head] = 1
        else:
            head_degrees[head] +=1
    for tail in tails:
        if tail not in tail_degrees.keys():
            tail_degrees[tail] = 1
        else:
            tail_degrees[tail] += 1

    sorted_heads = sorted(head_degrees.items(), key = lambda kv:(kv[1], kv[0]))
    sorted_tails = sorted(tail_degrees.items(), key= lambda kv:(kv[1], kv[0]))

    num_lowest = int(len(sorted_heads)*ratioLowPreserve)
    lowest_heads = sorted_heads[0:num_lowest]
    lowest_tails = sorted_tails[0:num_lowest]
    for i in range(0,len(lowest_heads)):
        lowest_heads[i] = lowest_heads[i][0]
    for i in range(0, len(lowest_tails)):
        lowest_tails[i] = lowest_tails[i][0]

    return lowest_heads, lowest_tails
# ====================Augmentation=========================
# =========================================================


def edgeAdd(user_dim, item_dim, edge_add_rate, lowest_degree_users=[], prejudice=False, selfPath='', socialOrNot = False):#可以initiate一个所有节点互相连接的边的数量，然后随机从里面抽取对应数量的边加到现有的edge里面去
    num_edge = len(user_dim)
    num_sample = int(num_edge * edge_add_rate)
    if socialOrNot:
        negSamples = pd.read_table(join(selfPath, 'negSamples_social.txt'), sep='\t', header=None)
    else:
        negSamples = pd.read_table(join(selfPath, 'negSamples.txt'), sep='\t', header=None)
    neg_users = np.array(negSamples[:][0])
    neg_items = np.array(negSamples[:][1])
    num_neg = len(neg_users)
    add_index = np.random.choice(num_neg, size=num_sample, replace=False)  # replace = False means that we sample without returning the sampled numbers

    add_users = neg_users[add_index]
    add_items = neg_items[add_index]

    user_dim = np.append(user_dim, add_users)
    item_dim = np.append(item_dim, add_items)

    return user_dim, item_dim

def edgeDrop(user_dim, item_dim, edge_drop_rate, lowest_degree_users=[], prejudice=False):
    num_edge = len(user_dim)
    #print(user_dim[0].item())
    drop_index = np.random.choice(num_edge, size=int(num_edge * edge_drop_rate), replace=False)  # replace = False means that we sample without returning the sampled numbers
    #the type of drop_index is <class 'numpy.ndarray'>
    if prejudice:
        drop_index = drop_index[~np.in1d(drop_index, np.array(lowest_degree_users))]
        rest_num = drop_index.size
        times_num = int(num_edge*edge_drop_rate/rest_num)
        for j in range(0, times_num):
            sub_drop_index = np.random.choice(num_edge, size=int(num_edge * edge_drop_rate), replace=False)
            sub_drop_index = sub_drop_index[~np.in1d(sub_drop_index, np.array(lowest_degree_users))]
            drop_index = np.concatenate((drop_index, sub_drop_index),axis=0)
            drop_index = np.unique(drop_index)
    user_dim = np.delete(user_dim, drop_index)
    item_dim = np.delete(item_dim, drop_index)
    return user_dim, item_dim

def nodeDrop(n_users, m_items, user_dim, item_dim, node_drop_rate):
    num_node = n_users + m_items
    drop_index = np.random.choice(num_node, size=int(num_node * node_drop_rate), replace=False)
    drop_user_id = drop_index[np.where(drop_index < n_users)[0]]
    drop_item = drop_index[np.where(drop_index >= n_users)[0]]
    drop_item_id = [x - n_users for x in drop_item]

    drop_interaction_index = np.array([], dtype=int)
    if len(drop_user_id) != 0:
        drop_user_index = np.array([], dtype=int)
        for user_index in drop_user_id:  
            tmp_user_index = np.where(user_dim == user_index)[0]
            drop_user_index = np.concatenate((drop_user_index, tmp_user_index), axis=0)
        drop_interaction_index = np.concatenate((drop_interaction_index, drop_user_index), axis=0)
    if len(drop_item_id) != 0:
        drop_item_index = np.array([], dtype=int)
        for item_index in drop_item_id:
            tmp_item_index = np.where(item_dim == item_index)[0]
            drop_item_index = np.concatenate((drop_item_index, tmp_item_index), axis=0)
        drop_interaction_index = np.concatenate((drop_interaction_index, drop_item_index), axis=0)
    drop_interaction_index = np.unique(drop_interaction_index)
    user_dim = np.delete(user_dim, drop_interaction_index)
    item_dim = np.delete(item_dim, drop_interaction_index)
    return user_dim, item_dim

def nodeDrop_social(n_users, trustor_dim, trustee_dim, node_drop_rate, lowest_degree_users=[], prejudice=False):
    num_node = n_users
    drop_index = np.random.choice(num_node, size = int(num_node*node_drop_rate), replace=False)
    if prejudice:
        drop_index = drop_index[~np.in1d(drop_index, np.array(lowest_degree_users))]

        rest_num = drop_index.size
        times_num = int(num_node*node_drop_rate/rest_num)
        for j in range(0, times_num):
            sub_drop_index = np.random.choice(num_node, size = int(num_node*node_drop_rate), replace=False)
            sub_drop_index = sub_drop_index[~np.in1d(sub_drop_index, np.array(lowest_degree_users))]
            
            drop_index = np.concatenate((drop_index, sub_drop_index), axis=0)
            drop_index = np.unique(drop_index)
    drop_trust_index = np.array([], dtype=int)
    for index in drop_index:
        tmp_user_index1 = np.where(trustor_dim == index)[0]
        tmp_user_index2 = np.where(trustee_dim == index)[0]
        drop_trust_index = np.concatenate((drop_trust_index,tmp_user_index1), axis=0)
        drop_trust_index = np.concatenate((drop_trust_index, tmp_user_index2), axis=0)
    drop_trust_index = np.unique(drop_trust_index)
    trustor_dim = np.delete(trustor_dim, drop_trust_index)
    trustee_dim = np.delete(trustee_dim, drop_trust_index)
    return trustor_dim, trustee_dim
# =========================================================
# ====================Augmentation=========================

