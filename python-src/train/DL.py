
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import math
from collections import Counter
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import linregress
import datetime
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, concatenate, Conv2D
from keras.models import Model
from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler


# In[4]:

sensor_data = pd.DataFrame.from_csv("../../data/database/sensor_data.csv")
sensor_data = sensor_data[~((sensor_data.TagName == 'Start1') | (sensor_data.TagName == 'Start2'))]
Counter(sensor_data.TagName)


# In[3]:

def min_max_normalization(feature, mini = None, maxi = None):
    if ((maxi is None) or (mini is None)):
        maxi = np.max(feature)
        mini = np.min(feature)
#         print(maxi)
#         print(mini)
    else:
        if (maxi == mini):
            # all the values is same:
            return np.array([1] * feature.size).reshape(feature.shape)
        if type(feature) == list:
            feature = np.array(feature)
        feature[feature < mini] = mini
        feature[feature > maxi] = maxi

    feature = ((feature - mini) / (maxi - mini))
        
    
    return feature

if os.path.exists("../../data/database/normalized_sensor_data.csv") is False:
    
    percentile_df = pd.DataFrame.from_csv("../../Results/percentiles_sensortype.txt")

    normalized_sensor_data = pd.DataFrame(columns=sensor_data.columns,index=sensor_data.index)
    normalized_sensor_data.loc[:,'SENSORTYPE'] = sensor_data['SENSORTYPE'].values
    normalized_sensor_data.loc[:,'TagName'] = sensor_data['TagName'].values
    normalized_sensor_data.loc[:,'tester_id'] = sensor_data['tester_id'].values
    normalized_sensor_data.loc[:,'TIMESTAMP'] = sensor_data['TIMESTAMP'].values
#     for i in range(len(percentile_df)):
#         cur = percentile_df.iloc[i].values
#         sensor = cur[0]
#         val = ['VALUES1','VALUES2', 'VALUES3']
#         p97 = cur[1]
#         p03 = cur[2]
#         normalized_sensor_data.loc[(sensor_data.SENSORTYPE == sensor), val] = min_max_normalization(sensor_data[(sensor_data.SENSORTYPE == sensor)][val].values, p03, p97)
#         print("processing: " + str(cur))
    for sensor in (percentile_df[' SENSORTYPE'].values):
        if sensor == 26:
            continue
        val = ['VALUES1','VALUES2', 'VALUES3']
        print("processing sensortype: " + str(sensor))

        data = sensor_data[sensor_data.SENSORTYPE == sensor][val].values.reshape(-1,1)
        scaler = StandardScaler()
        scaler.fit(data)
        normalized_sensor_data.loc[(sensor_data.SENSORTYPE == sensor), val] = scaler.transform(data).reshape(int(data.size / 3),3)
     
#     normalized_sensor_data.to_csv("../../data/database/normalized_sensor_data.csv")
#     del sensor_data
else:
    normalized_sensor_data = pd.DataFrame.from_csv("../../data/database/normalized_sensor_data.csv")


# In[9]:

normalized_sensor_data


# In[4]:

sensor_data_option = "Normalize"
if sensor_data_option == "Normalize":
    # tag_id_groupby = sensor_data_acc_tag12.groupby(['TagName', 'tester_id'])
    tag_id_groupby_acc = normalized_sensor_data[(normalized_sensor_data.SENSORTYPE == 1)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_magnetic = normalized_sensor_data[(normalized_sensor_data.SENSORTYPE == 2)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_orientation = normalized_sensor_data[(normalized_sensor_data.SENSORTYPE == 3)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_gyro = normalized_sensor_data[(normalized_sensor_data.SENSORTYPE == 4)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_gravity = normalized_sensor_data[(normalized_sensor_data.SENSORTYPE == 9)].groupby(['TagName', 'tester_id'])
    
    tag_id_linear_acc = normalized_sensor_data[(normalized_sensor_data.SENSORTYPE == 10)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_quaternion = normalized_sensor_data[(normalized_sensor_data.SENSORTYPE == 11)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_tilt = normalized_sensor_data[(normalized_sensor_data.SENSORTYPE == 26)].groupby(['TagName', 'tester_id'])
else:

    # tag_id_groupby = sensor_data_acc_tag12.groupby(['TagName', 'tester_id'])
    tag_id_groupby_acc = sensor_data[(sensor_data.SENSORTYPE == 1)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_magnetic = sensor_data[(sensor_data.SENSORTYPE == 2)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_orientation = sensor_data[(sensor_data.SENSORTYPE == 3)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_gyro = sensor_data[(sensor_data.SENSORTYPE == 4)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_gravity = sensor_data[(sensor_data.SENSORTYPE == 9)].groupby(['TagName', 'tester_id'])

    tag_id_linear_acc = sensor_data[(sensor_data.SENSORTYPE == 10)].groupby(['TagName', 'tester_id'])
    
    tag_id_groupby_quaternion = sensor_data[(sensor_data.SENSORTYPE == 11)].groupby(['TagName', 'tester_id'])

    tag_id_groupby_tilt = sensor_data[(sensor_data.SENSORTYPE == 26)].groupby(['TagName', 'tester_id'])


# In[5]:

def gesture_features(accs):
#     for i in range(accs.shape[0]): # x, y, z
#         accs[i] = min_max_normalization(accs[i], np.min(accs[i]), np.max(accs[i]))
        
    if N_frame_no > 1:
        Ls = math.floor(len(accs)/ (N_frame_no + 1))
        segments = None
        for i in range(N_frame_no + 1):
            if segments is None:
                segments = np.array([accs[i*Ls:(i+1)*Ls]])
            else:
                segments = np.append(segments, np.array([accs[i*Ls:(i+1)*Ls]]), axis=0)

        frames = None
        for i in range(N_frame_no):
            cur_frame = segments[i:i+2]
            cur_frame = cur_frame.reshape((cur_frame.shape[0]*cur_frame.shape[1],cur_frame.shape[2]))
            if frames is None:
                frames = np.array([cur_frame])
            else:
                frames = np.append(frames, np.array([cur_frame]), axis = 0)
        return np.array([frame_features(f) for f in frames]).reshape(-1)
    else:
        return frame_features(accs).reshape(-1)


# In[6]:

def frame_features(cur_frame):
    dft_cur_frame = np.fft.fftn(cur_frame)
    
    mean_cur_frame = dft_cur_frame[0]

    energy_cur_frame=[]
    for T in range(cur_frame.shape[1]): #x,y,z
        T_sum = 0
        for i in range(1,len(cur_frame)):
            T_sum += math.pow(abs(dft_cur_frame[i,T]),2)

        energy_cur_frame.append(T_sum / (len(cur_frame)-1))
    energy_cur_frame = np.array(energy_cur_frame)
    

    std_cur_frame = []
    for T in range(cur_frame.shape[1]): #x,y,z
        std_cur_frame.append(np.std(cur_frame))
    std_cur_frame = np.array(std_cur_frame)
    
    coorelation_cur_frame = []
    for T1,T2 in [(0,1),(1,2),(0,2)]:
        coorelation_cur_frame.append(np.correlate(cur_frame[:,T1], cur_frame[:,T2])[0])
    coorelation_cur_frame = np.array(coorelation_cur_frame)
    
    return np.array([mean_cur_frame])


# In[7]:

def shrink_array(array,size):
    
    ratio = float(len(array)) / float(size+1)
    res = []
    for i in range(size):
        res.append(np.mean(array[math.floor(i*ratio):math.ceil((i+1.0)*ratio)], axis = 0))
    return np.array(res)


# In[35]:

# X = []
# for key in list(tag_id_dict.keys()):
# #     frame_feature = gesture_features(tag_id_groupby_acc.get_group(key)[['VALUES1', 'VALUES2', 'VALUES3']].values).reshape(-1)
#     acc_feature = shrink_array(tag_id_groupby_acc.get_group(key)[['VALUES1', 'VALUES2', 'VALUES3']].values, 30)
# #     acc_feature = min_max_normalization(acc_feature)
# #     gyro_feature = shrink_array(tag_id_linear_acc.get_group(key)[['VALUES1', 'VALUES2', 'VALUES3']].values, 30)
# #     gyro_feature = min_max_normalization(gyro_feature)
# #     t = pd.to_datetime(tag_id_groupby_acc.get_group(key)['TIMESTAMP']).values
# #     time_dif = (np.max(t) - np.min(t)).item()/1000000000
#     X.append(acc_feature)
# #     X.append(np.concatenate((acc_feature, gyro_feature), axis = 1))
# X = np.array(X)

X = []
y=[]
for key in list(tag_id_linear_acc.groups.keys()):
    linear_acc_feature = shrink_array(tag_id_linear_acc.get_group(key)[['VALUES1', 'VALUES2', 'VALUES3']].values, 30)
#     acc_feature = shrink_array(tag_id_groupby_acc.get_group(key)[['VALUES1', 'VALUES2', 'VALUES3']].values, 30)
    X.append(linear_acc_feature)
    y.append(key[0])
#     X.append(np.concatenate((acc_feature, linear_acc_feature), axis = 1))
y = np.array(y)
X = np.array(X)


# In[36]:

tag_list = []
for i in range(10):
    tag_list.append(['Tag'+str(i),i])
for i in tag_list:
    tag_str = i[0]
    tag_int = i[1]
    y[y==tag_str] = tag_int
y_categorical = to_categorical(y)
idx = list(range(len(X)))
np.random.shuffle(idx)
X = X[idx]
y_categorical = y_categorical[idx]


# In[38]:

y.shape


# In[39]:

cv = 5
if cv > 1:
    scores = []
    tests = []
    predicts = []
    chunk = math.floor(len(X)/cv)
    for i in range(1,1+cv):
        test_idx = list(range((i-1)*chunk,i*chunk))
        train_idx = [i for i in range(len(X)) if i not in test_idx]
        train_x = X[train_idx].reshape(len(train_idx),-1)
        train_y = y_categorical[train_idx]
        test_x = X[test_idx].reshape(len(test_idx),-1)
        test_y = y_categorical[test_idx]

        # This returns a tensor
        inputs = Input(shape=(train_x.shape[1:]))

        # a layer instance is callable on a tensor, and returns a tensor
        # con1 = Conv1D(filters=30,kernel_size=10)(inputs)
        layer1 = Dense(64, activation='relu')(inputs)
        layer2 = Dense(128, activation='relu')(layer1)
        layer3 = Dense(64, activation='relu')(layer2)
        layer4 = Dense(32, activation='relu')(layer3)
        predictions = Dense(len(set(y)), activation='softmax')(layer4)


        model = Model(inputs=inputs, outputs=predictions)
    #     print(model.summary())
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model_his = model.fit(train_x, train_y, batch_size=32, epochs=40, verbose = 0)  # starts training
        pred_y = model.predict(test_x)

        # print(classification_report(np.argmax(test_y,1), np.argmax(pred_y, 1)))
        # print(confusion_matrix(np.argmax(test_y,1), np.argmax(pred_y, 1)))
        scores.append(accuracy_score(np.argmax(test_y,1), np.argmax(pred_y, 1)))
        tests += np.argmax(test_y,1).tolist()
        predicts += np.argmax(pred_y, 1).tolist()
print(classification_report(tests, predicts))
print(confusion_matrix(tests, predicts))
print(scores)
print(np.mean(np.array(scores)))


# In[40]:

cv = 5
if cv > 1:
    scores = []
    tests = []
    predicts = []
    chunk = math.floor(len(X)/cv)
    for i in range(1,1+cv):
        test_idx = list(range((i-1)*chunk,i*chunk))
        train_idx = [i for i in range(len(X)) if i not in test_idx]
        train_x = X[train_idx]
        train_y = y_categorical[train_idx]
        test_x = X[test_idx]
        test_y = y_categorical[test_idx]# This returns a tensor
        input_val1 = Input(shape=train_x.shape[1:])

        con1 = Conv1D(filters=30,kernel_size=10)(input_val1)
        max_pooling_1d_1 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(con1)
        # con2 = Conv1D(filters=30,kernel_size=10)(max_pooling_1d_1)
        # max_pooling_1d_2 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(con2)
        flat_1 = Flatten()(max_pooling_1d_1)

        # input_val2 = Input(shape=(200,1))

        # con3 = Conv1D(filters=30,kernel_size=10)(input_val2)
        # max_pooling_1d_3 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(con3)
        # con4 = Conv1D(filters=30,kernel_size=10)(max_pooling_1d_3)
        # max_pooling_1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(con4)
        # flat_2 = Flatten()(max_pooling_1d_4)

        # input_val3 = Input(shape=(200,1))
        # con6 = Conv1D(filters=30,kernel_size=10)(input_val3)
        # max_pooling_1d_5 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(con5)
        # con5 = Conv1D(filters=30,kernel_size=10)(max_pooling_1d_5)
        # max_pooling_1d_6 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(con6)
        # flat_3 = Flatten()(max_pooling_1d_6)

        # concat = concatenate([flat_1,flat_2, flat_3])
        # layer1 = Dense(64, activation='relu')(inputs)
        layer2 = Dense(128, activation='relu')(flat_1)
        # layer3 = Dense(64, activation='relu')(layer2)
        layer4 = Dense(32, activation='relu')(layer2)
        predictions = Dense(y_categorical.shape[-1], activation='softmax')(layer4)

        # model = Model(inputs=[input_val1,input_val2, input_val3], outputs=predictions)
        model = Model(inputs = input_val1, outputs=predictions)
#         model.summary()

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # model_his = model.fit([train_x[:,:,[0]],train_x[:,:,[1]],train_x[:,:,[2]]], train_y, batch_size=32, epochs=40, verbose = 0)  # starts training
        model_his = model.fit(train_x, train_y, batch_size=32, epochs=40, verbose = 0)  # starts training
        # pred_y = model.predict([test_x[:,:,[0]],test_x[:,:,[1]],test_x[:,:,[2]]])
        pred_y = model.predict(test_x)

    #     print(classification_report(np.argmax(test_y,1), np.argmax(pred_y, 1)))
    #     print(confusion_matrix(np.argmax(test_y,1), np.argmax(pred_y, 1)))
        scores.append(accuracy_score(np.argmax(test_y,1), np.argmax(pred_y, 1)))
        tests += np.argmax(test_y,1).tolist()
        predicts += np.argmax(pred_y, 1).tolist()
print(classification_report(tests, predicts))
print(confusion_matrix(tests, predicts))
print(scores)
print(np.mean(np.array(scores)))


# In[ ]:



