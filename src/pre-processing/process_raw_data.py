
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import datetime
import sys
import logging


# In[ ]:

import argparse


# In[2]:

useful_sensor = [1, 2, 3, 4, 11, 26, 17, 9, 10]


# In[3]:

def count_fequency(df):
    # only select the useful information
    df = df[df.SENSORTYPE.isin(useful_sensor)]
    min_time = df.TIMESTAMP.min()
    max_time = df.TIMESTAMP.max()
    c = Counter(df.SENSORTYPE)
    for i in c.keys():
        c[i] = c[i] / (float(max_time - min_time) /1000.0)

    print(c)


# In[48]:

def save_user_info_into_database(tag_df, database_dir):
    if (os.path.exists(database_dir) is not True):
        os.mkdir(database_dir)
    files = os.listdir(database_dir)

    cur_user_info = pd.DataFrame([tag_df.iloc[0].tolist()[2:]], columns=tag_df.columns[2:].values.tolist(), index=[0])
    cur_user_info['start_time'] = pd.Series([tag_df.TimeStamp.min()])
    cur_user_info['end_time'] = pd.Series([tag_df.TimeStamp.max() + pd.Timedelta('9 seconds')])
    if 'tester_info.csv' not in files:
        cur_user_info.to_csv(os.path.join(database_dir, "tester_info.csv"))
        return int(0)
    else:
        all_tester_info = pd.DataFrame.from_csv(os.path.join(database_dir, "tester_info.csv"))
        all_tester_info = all_tester_info.append(cur_user_info, ignore_index = True)
        all_tester_info.to_csv(os.path.join(database_dir, "tester_info.csv"))
        return int(all_tester_info.index.max())


# In[63]:

def save_sensor_data_into_database(sensor_df, database_dir):
    cur_sensor_df = sensor_df[~sensor_df.TagName.isnull()]
    if (os.path.exists(database_dir) is not True):
        os.mkdir(database_dir)
    files = os.listdir(database_dir)
    if 'sensor_data.csv' not in files:
        cur_sensor_df.to_csv(os.path.join(database_dir, 'sensor_data.csv'))
    else:
        all_sensor_df = pd.DataFrame.from_csv(os.path.join(database_dir, 'sensor_data.csv'))
        all_sensor_df = all_sensor_df.append(cur_sensor_df, ignore_index = True)
        all_sensor_df.to_csv(os.path.join(database_dir, 'sensor_data.csv'))


# In[49]:

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_data_dir', type=str, help='sensor data directory')
    parser.add_argument('--tag_data_dir', type=str, help='tag data directory')
    parser.add_argument('--database_dir', type=str, default='../../data/database/', help='database directory')

    args = parser.parse_args()
    process(args)


def process(args):
    sensor_data = pd.read_csv(args.sensor_data_dir, skiprows=13, skipinitialspace= True)
    count_fequency(sensor_data)
    sensor_data.TIMESTAMP = pd.DataFrame(index = pd.to_datetime(sensor_data.TIMESTAMP, unit='ms', utc = 'True')).tz_localize('utc').tz_convert('Asia/Singapore').index
    tag_data = pd.read_csv(args.tag_data_dir, skipinitialspace= True)
    tag_data.TimeStamp = pd.DataFrame(index = pd.to_datetime(tag_data.TimeStamp, utc = 'True')).tz_localize('Asia/Singapore').index
    
    user_groups = list(set(tag_data['Tester_Name'].values.tolist()))

    for user in user_groups:
        cur_user_tag_df = tag_data[tag_data['Tester_Name'] == user]
        if (cur_user_tag_df.iloc[0]['TagName'] != 'wear_start'):
            logging.error("Cannot find wear_start tag! User: " + str(user))
            sys.exit()
        if (len(cur_user_tag_df[cur_user_tag_df['TagName'] == 'wear_start']) > 1):
            logging.error("There are more than 1 wear_start tags! User: " + str(user))
            sys.exit()

        cur_user_id = save_user_info_into_database(cur_user_tag_df, args.database_dir)

        time_different_between_wear_phone =             cur_user_tag_df.iloc[0].TimeStamp - sensor_data.TIMESTAMP.min()
        sensor_data.TIMESTAMP = sensor_data.TIMESTAMP + time_different_between_wear_phone

        tags = cur_user_tag_df.TagName.tolist()
        for i in range(len(tags)): 
            if i == 0:
                continue  # skip the first tag as the first tag is always 'wear_start'
            cur_tag = tags[i]
            if cur_tag == 'ACTION_FINISH':
                continue 
            cur_tag_start_time = cur_user_tag_df.iloc[i].TimeStamp + pd.Timedelta('4 seconds')
            if (i + 1 < len(tags)):
                if (tags[i + 1] == 'ACTION_FINISH'):
                    cur_tag_end_time = cur_user_tag_df.iloc[i+1].TimeStamp
            else:
                cur_tag_end_time = cur_tag_start_time + pd.Timedelta('5 seconds')
            sensor_data.loc[(sensor_data.TIMESTAMP < cur_tag_end_time) & (sensor_data.TIMESTAMP > cur_tag_start_time), 'TagName'] = cur_tag
            sensor_data.loc[(sensor_data.TIMESTAMP < cur_tag_end_time) & (sensor_data.TIMESTAMP > cur_tag_start_time), 'tester_id'] = cur_user_id

    save_sensor_data_into_database(sensor_data, args.database_dir)


# In[50]:




# In[ ]:

if __name__ == '__main__':
    main()

