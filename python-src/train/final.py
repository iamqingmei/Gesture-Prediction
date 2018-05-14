# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import matplotlib.pyplot as plt


# In[85]:

def load_5models_from_disk():
    models = []
    for idx in range(5):
        json_file = open("./model" + str(idx) + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("temp" + str(idx) + ".hdf5")
        #         print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
        models.append(loaded_model)
    return models


# In[3]:

def shrink_array(array, size):
    ratio = float(len(array)) / float(size + 1)
    res = []
    for idx in range(size):
        res.append(np.mean(array[math.floor(idx * ratio):math.ceil((idx + 2.0) * ratio)], axis=0))
    return np.array(res)


# In[4]:

train_data = pd.DataFrame.from_csv("../../data/database/train_data.csv")
test_data = pd.DataFrame.from_csv("../../data/database/test_data.csv")
f_df = pd.DataFrame.from_csv("../../data/gesture_feature_df.csv")

# In[6]:

feature_cols = ['global_acc3', 'acc_12_square']


def get_feature_label(data_groups, keys, frame_number):
    y = []
    x = []
    f = []
    for k in keys:
        frame_feature = shrink_array(data_groups.get_group(k)[feature_cols].values, frame_number)
        x.append(frame_feature)
        y.append(k[0])
        f.append(f_df[(f_df.TagName == k[0]) & (f_df.tester_id == k[1])].d_change.values[0])
    return np.array(x), np.array(y), np.array(f).reshape(-1, 1)


def SVC_training(feature_train, y_train, x_test, y_test, f_test):
    if len(feature_train.shape) > 2:
        feature_train = feature_train.reshape(list(feature_train.shape)[0], -1)
        x_test = x_test.reshape(list(x_test.shape)[0], -1)
    main_score = []
    overall_score = []
    for k in [['poly',0.7], ['rbf',2.0], ['linear',0.2]]:
        clf4 = SVC(kernel=k[0], C=k[1], degree=3, verbose=False, probability=True)
        clf4.fit(feature_train, y_train)
        res = clf4.predict(x_test)
        score = accuracy_score(y_test, res)
        print(k[0] + " score: " + str(score))
        main_score.append(score)

        if (print_conf):
            print(classification_report(y_test, res))
            print(confusion_matrix(y_test, res))

        if ensemble_06 is True:
            for idx in range(len(res)):
                if (res[idx] == 'Tag0') or (res[idx] == "Tag6"):
                    res[idx] = rf_clf.predict([f_test[idx]])[0]
            print("-----ensembled---------")
            score = accuracy_score(y_test, res)
            print(score)
            if (print_conf):
                print(classification_report(y_test, res))
                print(confusion_matrix(y_test, res))
        overall_score.append(score)
    return main_score, overall_score


def RF_training(X_train, y_train, x_test, y_test, f_test):
    if len(X_train.shape) > 2:
        X_train = X_train.reshape(list(X_train.shape)[0], -1)
        x_test = x_test.reshape(list(x_test.shape)[0], -1)

        clf4 = RandomForestClassifier(n_estimators=30)

        clf4.fit(X_train, y_train)
        # joblib.dump(clf4, '../../Results/baseline SVC 0.80 raw data acc with gyro 200 chunk.pkl') 
        res = clf4.predict(x_test)
        score = accuracy_score(y_test, res)
        if print_conf:
            print(classification_report(y_test, res))
            print(confusion_matrix(y_test, res))
        print("RF score: " + str(score))

        if ensemble_06 is True:
            for i in range(len(res)):
                if (res[i] == 'Tag0') or (res[i] == "Tag6"):
                    res[i] = rf_clf.predict([f_test[i]])[0]
            print("-----ensembled---------")
            print((accuracy_score(y_test, res)))
            if (print_conf):
                print(classification_report(y_test, res))
                print(confusion_matrix(y_test, res))
    return score, accuracy_score(y_test, res)


def DL_training(X_train, y_train, x_test, y_test, f_test):
    y = np.concatenate([y_train, y_test])
    tag_list = []
    for i in range(10):
        tag_list.append(['Tag' + str(i), i])
    for i in tag_list:
        tag_str = i[0]
        tag_int = i[1]
        y[y == tag_str] = tag_int
    y_categorical = to_categorical(y)

    y_train_cate = y_categorical[:len(y_train)]
    y_test_cate = y_categorical[len(y_train):]

    X_train = X_train.reshape(list(X_train.shape)[0], -1)
    x_test = x_test.reshape(list(x_test.shape)[0], -1)

    main_scores = []
    overall_scores = []
    for i in range(5):
        # This returns a tensor
        inputs = Input(shape=(X_train.shape[1:]))

        # a layer instance is callable on a tensor, and returns a tensor
        layer1 = Dense(128, activation='relu')(inputs)
        layer2 = Dense(96, activation='relu')(layer1)
        layer3 = Dense(64, activation='relu')(layer2)
        layer4 = Dense(32, activation='relu')(layer3)
        predictions = Dense(10, activation='softmax')(layer4)

        mcp = ModelCheckpoint("./temp" + str(i) + ".hdf5", monitor='val_acc', verbose=0, save_best_only=True,
                              save_weights_only=False, mode='auto', period=1)
        model = Model(inputs=inputs, outputs=predictions)
        #         print(model.summary())
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train_cate, batch_size=32, epochs=40, verbose=0,
                  validation_data=(x_test, y_test_cate), callbacks=[mcp]
                  )  # starts training

        model.load_weights("./temp" + str(i) + ".hdf5")

        res = model.predict(x_test)
        predict = np.argmax(res, 1).tolist()
        score = accuracy_score(np.argmax(y_test_cate, 1), predict)
        print("DL score:" + str(score))
        main_scores.append(score)

        if (print_conf):
            print(classification_report(np.argmax(y_test_cate, 1), np.argmax(res, 1)))
            print(confusion_matrix(np.argmax(y_test_cate, 1), np.argmax(res, 1)))

        if ensemble_06 is True:
            for i in range(len(res)):
                if (predict[i] == 0) or (predict[i] == 6):
                    if rf_clf.predict([f_test[i]])[0] == 'Tag0':
                        predict[i] = 0
                    else:
                        predict[i] = 6

            print("-----ensembled---------")
            score = accuracy_score(np.argmax(y_test_cate, 1), predict)
            print(score)
            overall_scores.append(score)
            if (print_conf):
                print(classification_report(np.argmax(y_test_cate, 1), predict))
                print(confusion_matrix(np.argmax(y_test_cate, 1), predict))

    return main_scores, overall_scores


def CONV1d_training(X_train, y_train, x_test, y_test, f_test):
    y = np.concatenate([y_train, y_test])
    tag_list = []
    for i in range(10):
        tag_list.append(['Tag' + str(i), i])
    for i in tag_list:
        tag_str = i[0]
        tag_int = i[1]
        y[y == tag_str] = tag_int
    y_categorical = to_categorical(y)

    y_train_cate = y_categorical[:len(y_train)]
    y_test_cate = y_categorical[len(y_train):]

    main_scores = []
    overall_scores = []

    for i in range(5):
        input_val1 = Input(shape=X_train.shape[1:])

        con1 = Conv1D(filters=30, kernel_size=3)(input_val1)
        max_pooling_1d_1 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(con1)
        con2 = Conv1D(filters=30, kernel_size=3)(max_pooling_1d_1)
        max_pooling_1d_2 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(con2)
        # con3 = Conv1D(filters=30, kernel_size=3)(max_pooling_1d_2)
        # max_pooling_1d_3 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(con3)
        flat_1 = Flatten()(max_pooling_1d_2)
        layer2 = Dense(128, activation='relu')(flat_1)
        layer3 = Dense(64, activation='relu')(layer2)
        layer4 = Dense(32, activation='relu')(layer3)
        predictions = Dense(y_categorical.shape[-1], activation='softmax')(layer4)

        model = Model(inputs=input_val1, outputs=predictions)
        #         print(model.summary())
        mcp = ModelCheckpoint("./temp" + str(i) + ".hdf5", monitor='val_acc', verbose=0, save_best_only=True,
                              save_weights_only=False, mode='auto', period=1)

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train_cate, batch_size=32, epochs=40, verbose=0,
                              validation_data=(x_test, y_test_cate), callbacks=[mcp]
                              )  # starts training

        model.load_weights("./temp" + str(i) + ".hdf5")

        res = model.predict(x_test)
        predict = np.argmax(res, 1).tolist()
        score = accuracy_score(np.argmax(y_test_cate, 1), predict)
        print("CONV score:" + str(score))
        main_scores.append(score)

        if (print_conf):
            print(classification_report(np.argmax(y_test_cate, 1), np.argmax(res, 1)))
            print(confusion_matrix(np.argmax(y_test_cate, 1), np.argmax(res, 1)))

        if ensemble_06 is True:
            for i in range(len(res)):
                if (predict[i] == 0) or (predict[i] == 6):
                    if rf_clf.predict([f_test[i]])[0] == 'Tag0':
                        predict[i] = 0
                    else:
                        predict[i] = 6

            print("-----ensembled---------")
            score = accuracy_score(np.argmax(y_test_cate, 1), predict)
            print(score)
            overall_scores.append(score)
            if (print_conf):
                print(classification_report(np.argmax(y_test_cate, 1), predict))
                print(confusion_matrix(np.argmax(y_test_cate, 1), predict))

    return main_scores, overall_scores


# In[35]:

test_groups = test_data.groupby(['TagName', 'tester_id'])
test_keys = list(test_groups.groups.keys())
train_groups = train_data.groupby(['TagName', 'tester_id'])
train_keys = list(train_groups.groups.keys())
# In[ ]:


# In[122]:

ensemble_06 = True
print_conf = True
if ensemble_06 is True:
    rf_clf = joblib.load('./binary_model.pkl')
# In[76]:

svc_kernals = ['SVC-poly', 'SVC-rbf', 'SVC-linear']

log = "N,MODEL,MAIN_RES,OVERALL_RES\n"
for N in [26]:
    np.random.shuffle(train_keys)
    vali_keys = train_keys[:int(len(train_keys)/10)]
    X_train, y_train, f_train = get_feature_label(train_groups, train_keys[int(len(train_keys)/10):], N)
    X_test, y_test, f_test = get_feature_label(train_groups, vali_keys, N)
    # X_train, y_train, f_train = get_feature_label(train_groups, train_keys, N)
    # X_test, y_test, f_test = get_feature_label(test_groups, test_keys, N)
    # main, overall = RF_training(X_train, y_train, X_test, y_test, f_test)
    # log += str(N) + "," + "RF," + str(main) + "," + str(overall) + "\n"
    # svc_main_res, svc_overall_res = SVC_training(X_train, y_train, X_test, y_test, f_test)
    # for i in range(3):
    #     log += str(N) + "," + svc_kernals[i] + "," + str(svc_main_res[i]) + "," + str(svc_overall_res[i]) + "\n"
    # dl_main, dl_overall = DL_training(X_train, y_train, X_test, y_test, f_test)
    # log += str(N) + "," + "DNN," + str(np.max(dl_main)) + "," + str(np.max(dl_overall)) +  "\n"
    dl_main, dl_overall = CONV1d_training(X_train, y_train, X_test, y_test, f_test)
    log +=str(N) + "," +  "CNN," + str(np.max(dl_main)) + "," + str(np.max(dl_overall)) +  "\n"
#
#
print(log)
# with open("./log_N_small3.csv",'w') as file:
#     file.write(log)

# N=26
# np.random.shuffle(train_keys)
# vali_keys = train_keys[:int(len(train_keys) / 10)]
# X_train, y_train, f_train = get_feature_label(train_groups, train_keys[int(len(train_keys) / 10):], N)
# X_test, y_test, f_test = get_feature_label(train_groups, vali_keys, N)
#
# dl_main, dl_overall = DL_training(X_train, y_train, X_test, y_test, f_test)
# print("DNN," + str(np.max(dl_main)) + "," + str(np.max(dl_overall)))

# In[124]:


# ## Determine C for SVM
#
# # In[49]:


# def SVC_training_c(X_train, y_train,x_test, y_test, f_test,c):
#     if len(X_train.shape) > 2:
#         X_train = X_train.reshape(list(X_train.shape)[0],-1)
#         x_test = x_test.reshape(list(x_test.shape)[0],-1)
#     main_score = []
#     overal_score = []
#     for k in ['poly','rbf','linear']:
#         clf4 = SVC(kernel=k, C=c, degree=3, verbose = False)
#         clf4.fit(X_train, y_train)
#         # joblib.dump(clf4, '../../Results/baseline SVC 0.80 raw data acc with gyro 200 chunk.pkl')
#         res = clf4.predict(x_test)
#         score = accuracy_score(y_test, res)
#         print(k)
#         print("score: " + str(score) + " C = " + str(c))
#         main_score.append(score)
# #         print(classification_report(y_test, max_res))
# #         print(confusion_matrix(y_test, max_res))
#         if ensemble_06 is True:
#             for i in range(len(res)):
#                 if (res[i] =='Tag0') or (res[i] == "Tag6"):
#                     res[i] = rf_clf.predict([f_test[i]])[0]
#                 score = accuracy_score(y_test, res)
# #             print("-----ensembled---------")
# #             print(classification_report(y_test, max_res))
# #             print(confusion_matrix(y_test, max_res))
#             overal_score.append(score)
#     return main_score,overal_score
# #
# #
# #
# # # # In[64]:
# # #
# C = [2.0+i*0.1 for i in range(20)]
# C = [math.pow(2,i) for i in range(-3,6)]
# res = []
# for i in C:
#     main, overall = SVC_training_c(X_train, y_train, X_test, y_test, f_test,i)
#     res.append([main,overall])


# plt.plot(C,[i[0][0] for i in res],'x',color = 'b')
# plt.plot(C,[i[1][0] for i in res],'o',color = 'b')
# plt.legend(['Main Model Accuracy','Final Result'])
# plt.xlabel("C")
# plt.ylabel("Accuracy")
# plt.savefig("C_poly_small.png")
# plt.show()

# plt.plot(C,[i[0][1] for i in res],'x',color = 'r')
# plt.plot(C,[i[1][1] for i in res],'o',color = 'r')
# plt.legend(['Main Model Accuracy','Final Result'])
# plt.xlabel("C")
# plt.ylabel("Accuracy")
# plt.savefig("C_rbf_small.png")
# plt.show()

# plt.plot(C,[i[0][2] for i in res],'x',color = 'g')
# plt.plot(C,[i[1][2] for i in res],'o',color = 'g')
# plt.legend(['Main Model Accuracy','Final Result'])
# plt.xlabel("C")
# plt.ylabel("Accuracy")
# plt.savefig("C_linear_small.png")
# plt.show()


# In[65]:

#
#
# # In[ ]:
#
# from scipy.ndimage.filters import gaussian_filter
# blurred = gaussian_filter(acc3, sigma=np.std(acc3))
#
#
# # In[ ]:
#
# train_data['global_acc3_filterred'] = blurred
#
#
# # In[ ]:
#
# u = train_data[(train_data.TagName == 'Tag0') & (train_data.tester_id == 0.0)]
#
#
# # In[ ]:
#
# plt.plot(u.global_acc3)
# plt.plot(u.global_acc3_filterred)
# plt.show()
#
#
# # In[ ]:
#
#
#
