import pandas as pd
import numpy as np
from collections import Counter
import math
import operator


def user_independent_split(data_df, tester_info_df, ratio):
    data_groups = data_df.groupby(['TagName', 'tester_id'])
    all_keys = list(data_groups.groups.keys())

    c = Counter(tester_info_df.Tester_Name)

    sorted_c = sorted(c.items(), key=operator.itemgetter(1))
    name_count_pair = sorted_c.copy()
    np.random.shuffle(name_count_pair)

    n = math.ceil(len(all_keys) * ratio / 10)

    test_name = []
    test_idx = []
    for i in name_count_pair:
        if i[1] <= (n - len(test_idx)):
            test_name.append(i[0])
            test_idx += tester_info_df[tester_info_df.Tester_Name == i[0]].index.values.tolist()
        if len(test_idx) >= n:
            print("seperating idx finished")
            break
    train_idx = [i for i in tester_info_df.index.values if i not in test_idx]

    test_keys = [k for k in all_keys if k[1] in test_idx]
    train_keys = [k for k in all_keys if k[1] in train_idx]

    train_data = pd.DataFrame(columns=data_df.columns)
    for k in train_keys:
        train_data = train_data.append(data_groups.get_group(k), ignore_index=True)

    test_data = pd.DataFrame(columns=data_df.columns)
    for k in test_keys:
        test_data = test_data.append(data_groups.get_group(k), ignore_index=True)

    train_data.to_csv("../../data/database/train_data.csv")
    test_data.to_csv("../../data/database/test_data.csv")
    print("saved")


data = pd.DataFrame.from_csv("../../data/global_acc_features_df.csv")
tester_info = pd.DataFrame.from_csv("../../data/database/tester_info.csv")
user_independent_split(data, tester_info, 0.2)

quit()
