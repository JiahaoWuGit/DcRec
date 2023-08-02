import sys

import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import Model
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
import datetime


sample_ext = False


class DCCLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        
        self.pretrain = config['pretrain']
        self.pretrain_epochs = config['PRETRAIN_epochs']
        self.model = recmodel
        self.preference_ssl_enable = config['ssl_enable']
        self.social_ssl_enable = config['social_enable']
        self.normalization_decay = config['normalization_decay']
        self.cycle_decay = config['cycle_decay']
        self.social_ssl_decay = config['social_ssl_decay']
        self.preference_ssl_decay = config['preference_ssl_decay']
        self.tau = config['tau']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)#

    def stageOne(self, users, pos, neg, epoch):
        starttime = datetime.datetime.now()
        loss, reg_loss, reg_social_loss = self.model.bpr_loss(users, pos, neg)
        bpr_loss_forRecord = loss
        reg_loss = reg_loss*self.normalization_decay
        reg_social_loss = reg_social_loss*self.normalization_decay
        loss = loss + reg_loss + reg_social_loss
        endtime = datetime.datetime.now()
        bpr_cal_time = (endtime-starttime)
        bpr_cal_time = bpr_cal_time.seconds

        ssl_loss_rate = 0
        cycle_loss = 0
        #Preference ssl
        if self.preference_ssl_enable:
            starttime = datetime.datetime.now()
            ssl_loss_user, ssl_loss_item = self.model.preference_ssl_loss(users, pos, self.tau)
            ssl_loss_rate = (0.5*ssl_loss_user+0.5*ssl_loss_item)*self.preference_ssl_decay
            loss += ssl_loss_rate
            endtime = datetime.datetime.now()
            preference_ssl_cal_time = (endtime-starttime).seconds
        if self.social_ssl_enable:
            starttime = datetime.datetime.now()
            ##Social ssl
            ssl_loss_social = self.model.social_ssl_loss(users, self.tau)
            ssl_loss_social = ssl_loss_social*self.social_ssl_decay
            loss += ssl_loss_social
            #cycle loss
            cycle_loss_social, cycle_loss_preference = self.model.cycle_ssl_loss(users, self.tau)
            ssl_loss_cycle_print = cycle_loss_social + cycle_loss_preference
            cycle_loss = (0.5*cycle_loss_social+0.5*cycle_loss_preference)*self.cycle_decay
            loss += cycle_loss
            endtime = datetime.datetime.now()
            social_ssl_cal_time = (endtime-starttime).seconds

        starttime = datetime.datetime.now()


        if self.pretrain:
            if not self.preference_ssl_enable:
                words = 'Hyper-parameter ERROR: you cannot pretrain when ssl is disabled!!!!! '
                print(f"\033[0;30;43m{words}\033[0m")
                sys.exit(0)
            elif epoch < self.pretrain_epochs:
                #print('Pretraining!')
                loss = ssl_loss_rate + cycle_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        endtime = datetime.datetime.now()
        back_propagation_time = (endtime-starttime).seconds
        if not self.preference_ssl_enable:
            preference_ssl_cal_time = 0
        if not self.social_ssl_enable:
            social_ssl_cal_time = 0
        return loss.cpu().item(), bpr_loss_forRecord, bpr_cal_time, preference_ssl_cal_time, social_ssl_cal_time, back_propagation_time



def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    sample_ext = False
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)

    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):

        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size')
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    #print(shuffle_indices)
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        '''
        print('============True')
        x = input()
        '''
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# ===========================================================
