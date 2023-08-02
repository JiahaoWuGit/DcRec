dataName = ['yelp']#['ciao', 'douban', 'epinions', 'filmtrust', 'yelp']
import random
import pandas as pd
import numpy as np
#train ratio: 0.8, test ratio: 0.2
test_ratio = 0.2
for data in dataName:
    filePath = str(data)+'/ratings.txt'
    trainPath = str(data)+'/ratings_train.txt'
    testPath = str(data)+'/ratings_test.txt'
    source = pd.read_table(filePath, sep=' ', header=None)
    users = np.array(source[:][0])
    items = np.array(source[:][1])
    ratings = np.array(source[:][2])
    length = len(users)
    delete_index = []
    for i in range(0,length):
        if data == 'yelp':
            break
        if ratings[i] < 4:
            delete_index.append(i)
    users_new = np.delete(users, delete_index)
    items_new = np.delete(items, delete_index)
    ratings_new = np.delete(ratings, delete_index)
    length = len(users_new)
    test = []
    train = []
    for i in range(0,length):
        if random.random() > test_ratio:
            train.append(str(users_new[i])+'\t'+str(items_new[i])+'\n')
        else:
            test.append(str(users_new[i])+'\t'+str(items_new[i])+'\n')
    with open(trainPath, 'w') as f:
        f.writelines(train)
    with open(testPath, 'w') as f:
        f.writelines(test)
print('finished!')
#axiba