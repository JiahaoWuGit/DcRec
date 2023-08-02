dataName = ['yelp']#['ciao', 'douban', 'epinions', 'filmtrust', 'yelp']
import random
import pandas as pd
import numpy as np
import re
for data in dataName:
    new = []
    filePath = str(data)+'/trusts.txt'
    with open(filePath) as f:
        i = 0
        fList = list(f)
        l = len(fList)
        for line in fList:
            if data == 'yelp':
                newline = re.findall(r"\d+",line)
                x1 = newline[0]+'\t'+newline[1]+'\n'
                x2 = newline[1]+'\t'+newline[0]+'\n'
                print(i,'/',l)
                i = i+1
                if x1 not in new:
                    new.append(x1)
                if x2 not in new:
                    new.append(x2)
            else:
                newline = line.replace(' ', '\t')
                new.append(newline)
    #break
    with open(filePath, 'w') as f:
        f.writelines(new)

