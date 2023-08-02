import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
from functools import partial

def DCC_train_original(dataset, recommend_model, loss_class, epoch, tf_writer=None):
    Recmodel = recommend_model
    Recmodel.train()
    dccloss: utils.DCCLoss = loss_class
    with timer(name="Sample"):  # Use Timer every time you call the function
        S = utils.UniformSample_original(dataset)  
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    device = Recmodel.config['DEVICE']
    users = users.to(device)
    posItems = posItems.to(device)
    negItems = negItems.to(device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // recommend_model.config['bpr_batch_size'] + 1
    aver_loss = 0.
    bpr_loss_record = 0.
    bpr_cal_time =0.
    preference_ssl_cal_time=0.
    social_ssl_cal_time = 0.
    back_propagation_time =0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=recommend_model.config['bpr_batch_size'])):
        if len(batch_users) < Recmodel.config['bpr_batch_size']:
            break
        cri,bpr_loss_forRecord, tmp_bpr_cal_time, tmp_preference_ssl_cal_time, tmp_social_ssl_cal_time, tmp_back_propagation_time = dccloss.stageOne(batch_users, batch_pos, batch_neg, epoch)
        aver_loss += cri
        bpr_loss_record += bpr_loss_forRecord
        bpr_cal_time += tmp_bpr_cal_time
        preference_ssl_cal_time += tmp_preference_ssl_cal_time
        social_ssl_cal_time += tmp_social_ssl_cal_time
        back_propagation_time += tmp_back_propagation_time

        if tf_writer:
                tf_writer.add_scalar(f'DCCLoss/DCC', cri, epoch * int(len(users) / recommend_model.config['bpr_batch_size']) + batch_i)

    aver_loss = aver_loss / total_batch
    bpr_loss_record = bpr_loss_record /total_batch
    time_info = timer.dict()
    timer.zero()

    returnMessage = f"loss{aver_loss:.3f}-bpr_loss{bpr_loss_record:.3f}-{time_info}"
    if Recmodel.config['pretrain'] and epoch<Recmodel.config['PRETRAIN_epochs']:
        returnMessage = 'Pretraining_' + returnMessage
    return returnMessage


def test_one_batch(topks, X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}

def Test(dataset, Recmodel, epoch, tf_writer=None,  multicore=0):#, topks=[5]
    u_batch_size = Recmodel.config['test_u_batch_size']
    config = Recmodel.config
    topks = config['TOPKS']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.DCCLoss
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            u_batch_size = u_batch_size/2
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config['DEVICE'])#world

            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            func = partial(test_one_batch, topks)
            pre_results = pool.map(func, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(topks, x))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        if tf_writer:
            tf_writer.add_scalars(f'Test/Recall@{topks}',
                          {str(topks[i]): results['recall'][i] for i in range(len(topks))}, epoch)
            tf_writer.add_scalars(f'Test/Precision@{topks}',
                          {str(topks[i]): results['precision'][i] for i in range(len(topks))}, epoch)
            tf_writer.add_scalars(f'Test/NDCG@{topks}',
                          {str(topks[i]): results['ndcg'][i] for i in range(len(topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results

def Multi_TOPKS_Test(dataset, Recmodel, epoch, tf_writer=None,  multicore=0):
    u_batch_size = Recmodel.config['test_u_batch_size']
    config = Recmodel.config
    topks = config['TOPKS']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.DCCLoss
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            u_batch_size = u_batch_size / 2
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config['DEVICE'])  # world

            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            func = partial(test_one_batch, topks)
            pre_results = pool.map(func, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(topks, x))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        if tf_writer:
            tf_writer.add_scalars(f'Test/Recall@{topks}',
                                  {str(topks[i]): results['recall'][i] for i in range(len(topks))}, epoch)
            tf_writer.add_scalars(f'Test/Precision@{topks}',
                                  {str(topks[i]): results['precision'][i] for i in range(len(topks))}, epoch)
            tf_writer.add_scalars(f'Test/NDCG@{topks}',
                                  {str(topks[i]): results['ndcg'][i] for i in range(len(topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
