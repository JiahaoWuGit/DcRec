dataName = ['dianping']#['ciao', 'douban',  'yelp']#['ciao',  'epinions', 'filmtrust']#
import random
import pandas as pd
import numpy as np
#train ratio: 0.8, test ratio: 0.2

def negSample(Us, Is, negStorePath, socialOrNot, dataList1):
    users_shuffle = Us
    items_shuffle = Is

    np.random.shuffle(items_shuffle)

    delete_exist = []
    for i in range(0, len(users_shuffle)):
        if items_shuffle[i] in dataList1[data][users_shuffle[i]]:
            delete_exist.append(i)
    users_neg_sampled = np.delete(users_shuffle, delete_exist)
    items_neg_sampled = np.delete(items_shuffle, delete_exist)
    if socialOrNot:
        print('ratio of re-sampled negative samples in trustnet:', float(len(users_neg_sampled))/float(len(users_shuffle)))
    else:
        print('ratio of re-sampled negative samples:', float(len(users_neg_sampled))/float(len(users_shuffle)))

    storeNeg = []
    for i in range(0, len(users_neg_sampled)):
        storeNeg.append(str(users_neg_sampled[i])+'\t'+str(items_neg_sampled[i])+'\n')
    with open(negStorePath, 'w') as f:
        f.writelines(storeNeg)

dataList = {}
for data in dataName:
    dataList[data] = {}
print('dataList:', dataList)
dataList_social = {}
for data in dataName:
    dataList_social[data] = {}
print('dataList_social:', dataList_social)
for data in dataName:
    negPath = str(data)+'/negSamples.txt'
    negPath_social = str(data)+'/negSamples_social.txt'
    if data !='epinions':
        train_filePath = str(data) + '/ratings_train.txt'
        test_filePath = str(data) + '/ratings_test.txt'
        trust_filePath = str(data) + '/trusts.txt'
        train_source = pd.read_table(train_filePath, sep='\t', header=None)
        test_source = pd.read_table(test_filePath, sep='\t', header=None)
        trust_source = pd.read_table(trust_filePath, sep='\t', header=None)
    else:
        train_filePath = str(data) + '/ratings_train_origin.txt'
        test_filePath = str(data) + '/ratings_test_origin.txt'
        trust_filePath = str(data) + '/trusts_origin.txt'
        train_source = pd.read_table(train_filePath, sep=' ', header=None)
        test_source = pd.read_table(test_filePath, sep=' ', header=None)
        trust_source = pd.read_table(trust_filePath, sep='\t', header=None)

    users_train = np.array(train_source[:][0])
    items_train = np.array(train_source[:][1])
    users_test = np.array(test_source[:][0])
    items_test = np.array(test_source[:][1])

    trustors = np.array(trust_source[:][0])
    trustees = np.array(trust_source[:][1])

    print(len(users_train))
    print(len(users_test))
    users = np.append(users_train, users_test)
    items = np.append(items_train, items_test)

    length = len(users)

    print('===========', data, '===========')
    for i in range(0, length):
        if users[i] not in dataList[data].keys():
            dataList[data][users[i]] = []
        dataList[data][users[i]].append(items[i])
    print('num of new users:', len(users))
    for i in range(0, len(trustors)):
        if trustors[i] not in dataList_social[data].keys():
            dataList_social[data][trustors[i]] = []
        dataList_social[data][trustors[i]].append(trustees[i])
    print('num of trust relationships:', len(trustors))

    negSample(users, items, negPath, False, dataList)
    negSample(trustors, trustees, negPath_social, True, dataList_social)


print('finished!')