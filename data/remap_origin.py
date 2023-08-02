import random
import pandas as pd
import numpy as np
# dataNames = ['ciao', 'douban', 'yelp', 'epinions']
def remap(dataName):
    #ratingPath = str(dataName)+'/ratings.txt'
    trainOriginPath = str(dataName) + '/ratings_train_origin.txt'#divide by '\t'
    testOriginPath = str(dataName) + '/ratings_test_origin.txt'#divide by '\t'
    trustOriginPath = str(dataName) + '/trusts_origin.txt'#divide by '\t'
    valOriginPath = str(dataName) + '/ratings_valid_origin.txt'

    trainOrigin = pd.read_table(trainOriginPath, sep='\t', header=None)
    usersTrain = np.array(trainOrigin[:][0])
    itemsTrain = np.array(trainOrigin[:][1])
    testOrigin = pd.read_table(testOriginPath, sep ='\t', header=None)
    usersTest = np.array(testOrigin[:][0])
    itemsTest = np.array(testOrigin[:][1])
    trustOrigin = pd.read_table(trustOriginPath, sep='\t', header=None)
    users1Trust = np.array(trustOrigin[:][0])
    users2Trust = np.array(trustOrigin[:][1])
    valOrigin = pd.read_table(valOriginPath, sep='\t', header=None)
    usersVal = np.array(valOrigin[:][0])
    itemsVal = np.array(valOrigin[:][0])


    users = np.unique(np.append(usersTrain, usersTest))
    users = np.unique(np.append(users, usersVal))
    trusts = np.unique(np.append(users1Trust, users2Trust))


    users = np.unique(np.append(users, trusts))
    items = np.unique(np.append(itemsTrain, itemsTest))
    items = np.unique(np.append(items, itemsVal))
    userDic = {}
    itemDic = {}
    lu = len(users)

    for i in range(lu):
        #print('users: ',i, '/',l)
        userDic[users[i]] = i
    li = len(items)
    for i in range(li):
        #print('items: ', l, '/', l)
        itemDic[items[i]] = i

    print('Number of users:', lu)
    print('Number of items:', li)

    train = []
    test = []
    val = []
    trust = []
    for i in range(0,len(usersTrain)):
        train.append(str(userDic[usersTrain[i]])+'\t'+str(itemDic[itemsTrain[i]])+'\n')
    for i in range(0, len(usersTest)):
        test.append(str(userDic[usersTest[i]])+'\t'+str(itemDic[itemsTest[i]])+'\n')
    for i in range(0, len(usersVal)):
        val.append(str(userDic[usersVal[i]])+'\t'+str(itemDic[itemsVal[i]])+'\n')
    print('Number of train:', len(usersTrain))
    print('Number of test:', len(usersTest))
    print('Number of val:', len(usersVal))
    print('Number of ratings:',len(usersTrain)+len(usersTest)+len(usersVal))
    print('Number of trust relations',len(users1Trust))
    for i in range(0,len(users1Trust)):
        trust.append(str(userDic[users1Trust[i]])+'\t'+str(userDic[users2Trust[i]])+'\n')



    trainPath = str(dataName) + '/ratings_train.txt'
    testPath = str(dataName) + '/ratings_test.txt'
    trustPath = str(dataName) + '/trusts.txt'
    valPath = str(dataName) + '/ratings_valid.txt'

    with open(trainPath, 'w') as f:
        f.writelines(train)
    with open(testPath, 'w') as f:
        f.writelines(test)
    with open(trustPath, 'w') as f:
        f.writelines(trust)
    with open(valPath, 'w') as f:
        f.writelines(val)
    print('Finished ',dataName, '!')

# for dataName in dataNames:
#     remap(dataName)
remap('dianping')
#remap('epinions')
print('finished!')