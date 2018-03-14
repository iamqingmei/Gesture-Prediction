
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from collections import Counter 
import math
import operator


# In[46]:

def user_independent_split(data_df, tester_info_df, ratio):
    data_groups = data.groupby(['TagName','tester_id'])
    all_keys = list(data_groups.groups.keys())


    len(tester_info[tester_info.Tester_Name == 'qm'])
    c = Counter(tester_info.Tester_Name)


    sorted_c = sorted(c.items(), key=operator.itemgetter(1))
    name_count_pair = sorted_c.copy()
    np.random.shuffle(name_count_pair)



    N = math.ceil(len(all_keys)*ratio/10)

    test_name = []
    test_idx = []
    for i in name_count_pair:
        if i[1]<= (N - len(test_idx)):
            test_name.append(i[0])
            test_idx += tester_info[tester_info.Tester_Name == i[0]].index.values.tolist()
        if len(test_idx) >= N:
            print("finished")
            break
    train_idx = [i for i in tester_info.index.values if i not in test_idx]

    test_keys = [k for k in all_keys if k[1] in test_idx]
    train_keys = [k for k in all_keys if k[1] in train_idx]
    
    train_data = pd.DataFrame()
    for k in train_keys:
        train_data.append(data_groups.get_group(k), ignore_index = True)

    test_data = pd.DataFrame()
    for k in test_keys:
        test_data.append(data_groups.get_group(k), ignore_index = True)
        
    train_data.to_csv("../../data/database/train_data.csv")
    test_data.to_csv("../../data/database/test_data.csv")

