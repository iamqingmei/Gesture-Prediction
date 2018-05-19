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


def load_5models_from_disk():
    """
    Load the 5 keras models saved in disk
    :return: the model loaded
    """
    models = []
    for idx in range(5):
        json_file = open("./model" + str(idx) + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("temp" + str(idx) + ".hdf5")

        # evaluate loaded model on test data
        loaded_model.compile(optimizer='rmsprop',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
        models.append(loaded_model)
    return models


def shrink_array(array, size):
    """
    Convert the array to the given size (frame feature)
    :param array: input feature array
    :param size: int, size of the output array
    :return: the converted array
    """
    ratio = float(len(array)) / float(size + 1)
    res = []
    for idx in range(size):
        res.append(np.mean(array[math.floor(idx * ratio):math.ceil((idx + 2.0) * ratio)], axis=0))
    return np.array(res)


# In[4]:

train_data = pd.DataFrame.from_csv("../../data/database/train_data.csv")
test_data = pd.DataFrame.from_csv("../../data/database/test_data.csv")
displacement_df = pd.DataFrame.from_csv("../../data/gesture_feature_df.csv")  # z-axis displacement

# In[6]:

feature_cols = ['global_acc3', 'acc_12_square']


def get_feature_label(data_groups, keys, frame_number):
    """
    From pandas groups, generate frame features, labels, and the corresponding z-axis displacement
    :param data_groups: pandas groups
    :param keys: the selected key of groups
    :param frame_number: number of frames to describe a activity
    :return: frame features, labels, z-axis displacement
    """
    y = []
    x = []
    f = []
    for k in keys:
        frame_feature = shrink_array(data_groups.get_group(k)[feature_cols].values, frame_number)
        x.append(frame_feature)
        y.append(k[0])
        f.append(displacement_df[(displacement_df.TagName == k[0])
                                 & (displacement_df.tester_id == k[1])].d_change.values[0])
    return np.array(x), np.array(y), np.array(f).reshape(-1, 1)


def SVC_training(feature_train, label_train, feature_test, labels_test, displacement_test):
    """
    the training process of SVC, including 3 SVC kernels: ['poly', 'rbf', linear']

    :param feature_train: the features of training data
    :param label_train: the labels of training data
    :param feature_test: the features of testing data
    :param labels_test: the labels of testing data
    :param displacement_test: the z-axis displacement of testing data
    :return: main_score: the result of main model
             overall_score: the result of the whole system
    """
    if len(feature_train.shape) > 2:
        feature_train = feature_train.reshape(list(feature_train.shape)[0], -1)
        feature_test = feature_test.reshape(list(feature_test.shape)[0], -1)
    main_score = []
    overall_score = []
    for k in [['poly', 0.7], ['rbf', 2.0], ['linear', 0.2]]:
        clf4 = SVC(kernel=k[0], C=k[1], degree=3, verbose=False, probability=True)
        clf4.fit(feature_train, label_train)
        res = clf4.predict(feature_test)
        score = accuracy_score(labels_test, res)
        print(k[0] + " score: " + str(score))
        main_score.append(score)

        if print_conf:
            print(classification_report(labels_test, res))
            print(confusion_matrix(labels_test, res))

        if ensemble_06 is True:
            for idx in range(len(res)):
                if (res[idx] == 'Tag0') or (res[idx] == "Tag6"):
                    res[idx] = rf_clf.predict([displacement_test[idx]])[0]
            print("-----ensembled---------")
            score = accuracy_score(labels_test, res)
            print(score)
            if print_conf:
                print(classification_report(labels_test, res))
                print(confusion_matrix(labels_test, res))
        overall_score.append(score)
    return main_score, overall_score


def RF_training(feature_train, label_train, feature_test, label_test, displacement_test):
    """
    the training process of Random Forest
    :param feature_train: the features of training data
    :param label_train: the labels of training data
    :param feature_test: the features of testing data
    :param label_test: the labels of testing data
    :param displacement_test: the z-axis displacement of testing data
    :return: main_score: the result of main model
             overall_score: the result of the whole system
    """
    if len(feature_train.shape) > 2:
        feature_train = feature_train.reshape(list(feature_train.shape)[0], -1)
        feature_test = feature_test.reshape(list(feature_test.shape)[0], -1)

    clf4 = RandomForestClassifier(n_estimators=30)

    clf4.fit(feature_train, label_train)
    res = clf4.predict(feature_test)
    score = accuracy_score(label_test, res)
    if print_conf:
        print(classification_report(label_test, res))
        print(confusion_matrix(label_test, res))
    print("RF score: " + str(score))

    if ensemble_06 is True:
        for idx in range(len(res)):
            if (res[idx] == 'Tag0') or (res[idx] == "Tag6"):
                res[idx] = rf_clf.predict([displacement_test[idx]])[0]
        print("-----ensembled---------")
        print((accuracy_score(label_test, res)))
        if print_conf:
            print(classification_report(label_test, res))
            print(confusion_matrix(label_test, res))
    return score, accuracy_score(label_test, res)


def DL_training(feature_train, label_train, feature_test, label_test, displacement_test):
    """
    the training process of Fully Connected Neural Network
    :param feature_train: the features of training data
    :param label_train: the labels of training data
    :param feature_test: the features of testing data
    :param label_test: the labels of testing data
    :param displacement_test: the z-axis displacement of testing data
    :return: main_score: the result of main model
             overall_score: the result of the whole system
    """
    y = np.concatenate([label_train, label_test])
    tag_list = []
    for idx in range(10):
        tag_list.append(['Tag' + str(idx), idx])
    for idx in tag_list:
        tag_str = idx[0]
        tag_int = idx[1]
        y[y == tag_str] = tag_int
    y_categorical = to_categorical(y)

    y_train_cate = y_categorical[:len(label_train)]
    y_test_cate = y_categorical[len(label_train):]

    feature_train = feature_train.reshape(list(feature_train.shape)[0], -1)
    feature_test = feature_test.reshape(list(feature_test.shape)[0], -1)

    main_scores = []
    overall_scores = []
    for run in range(5):
        # This returns a tensor
        inputs = Input(shape=(feature_train.shape[1:]))

        # a layer instance is callable on a tensor, and returns a tensor
        layer1 = Dense(128, activation='relu')(inputs)
        layer2 = Dense(96, activation='relu')(layer1)
        layer3 = Dense(64, activation='relu')(layer2)
        layer4 = Dense(32, activation='relu')(layer3)
        predictions = Dense(10, activation='softmax')(layer4)

        mcp = ModelCheckpoint("./temp" + str(idx) + ".hdf5", monitor='val_acc', verbose=0, save_best_only=True,
                              save_weights_only=False, mode='auto', period=1)
        model = Model(inputs=inputs, outputs=predictions)
        #         print(model.summary())
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(feature_train, y_train_cate, batch_size=32, epochs=40, verbose=0,
                  validation_data=(feature_test, y_test_cate), callbacks=[mcp]
                  )  # starts training

        model.load_weights("./temp" + str(idx) + ".hdf5")

        res = model.predict(feature_test)
        predict = np.argmax(res, 1).tolist()
        score = accuracy_score(np.argmax(y_test_cate, 1), predict)
        print("DL score:" + str(score))
        main_scores.append(score)

        if print_conf:
            print(classification_report(np.argmax(y_test_cate, 1), np.argmax(res, 1)))
            print(confusion_matrix(np.argmax(y_test_cate, 1), np.argmax(res, 1)))

        if ensemble_06 is True:
            for idx in range(len(res)):
                if (predict[idx] == 0) or (predict[idx] == 6):
                    if rf_clf.predict([displacement_test[idx]])[0] == 'Tag0':
                        predict[idx] = 0
                    else:
                        predict[idx] = 6

            print("-----ensembled---------")
            score = accuracy_score(np.argmax(y_test_cate, 1), predict)
            print(score)
            overall_scores.append(score)
            if print_conf:
                print(classification_report(np.argmax(y_test_cate, 1), predict))
                print(confusion_matrix(np.argmax(y_test_cate, 1), predict))

    return main_scores, overall_scores


def CONV1d_training(feature_train, label_train, feature_test, label_test, displacement_test):
    """
    the training process of 1D CNN
    :param feature_train: the features of training data
    :param label_train: the labels of training data
    :param feature_test: the features of testing data
    :param label_test: the labels of testing data
    :param displacement_test: the z-axis displacement of testing data
    :return: main_score: the result of main model
             overall_score: the result of the whole system
    """
    y = np.concatenate([label_train, label_test])
    tag_list = []
    for idx in range(10):
        tag_list.append(['Tag' + str(idx), idx])
    for idx in tag_list:
        tag_str = idx[0]
        tag_int = idx[1]
        y[y == tag_str] = tag_int
    y_categorical = to_categorical(y)

    y_train_cate = y_categorical[:len(label_train)]
    y_test_cate = y_categorical[len(label_train):]

    main_scores = []
    overall_scores = []

    for run in range(5):
        input_val1 = Input(shape=feature_train.shape[1:])

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
        mcp = ModelCheckpoint("./temp" + str(idx) + ".hdf5", monitor='val_acc', verbose=0, save_best_only=True,
                              save_weights_only=False, mode='auto', period=1)

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(feature_train, y_train_cate, batch_size=32, epochs=40, verbose=0,
                  validation_data=(feature_test, y_test_cate), callbacks=[mcp]
                  )  # starts training

        model.load_weights("./temp" + str(idx) + ".hdf5")

        res = model.predict(feature_test)
        predict = np.argmax(res, 1).tolist()
        score = accuracy_score(np.argmax(y_test_cate, 1), predict)
        print("CONV score:" + str(score))
        main_scores.append(score)

        if print_conf:
            print(classification_report(np.argmax(y_test_cate, 1), np.argmax(res, 1)))
            print(confusion_matrix(np.argmax(y_test_cate, 1), np.argmax(res, 1)))

        if ensemble_06 is True:
            for idx in range(len(res)):
                if (predict[idx] == 0) or (predict[idx] == 6):
                    if rf_clf.predict([displacement_test[idx]])[0] == 'Tag0':
                        predict[idx] = 0
                    else:
                        predict[idx] = 6

            print("-----ensembled---------")
            score = accuracy_score(np.argmax(y_test_cate, 1), predict)
            print(score)
            overall_scores.append(score)
            if print_conf:
                print(classification_report(np.argmax(y_test_cate, 1), predict))
                print(confusion_matrix(np.argmax(y_test_cate, 1), predict))

    return main_scores, overall_scores


test_groups = test_data.groupby(['TagName', 'tester_id'])
test_keys = list(test_groups.groups.keys())
train_groups = train_data.groupby(['TagName', 'tester_id'])
train_keys = list(train_groups.groups.keys())


ensemble_06 = True # option to add binary model
print_conf = True # option to print confusion matrix
if ensemble_06 is True:
    rf_clf = joblib.load('./binary_model.pkl')

svc_kernals = ['SVC-poly', 'SVC-rbf', 'SVC-linear']

log = "N,MODEL,MAIN_RES,OVERALL_RES\n"

N = 26
X_train, y_train, f_train = get_feature_label(train_groups, train_keys, N)
X_test, y_test, f_test = get_feature_label(test_groups, test_keys, N)
main, overall = RF_training(X_train, y_train, X_test, y_test, f_test)
log += str(N) + "," + "RF," + str(main) + "," + str(overall) + "\n"
svc_main_res, svc_overall_res = SVC_training(X_train, y_train, X_test, y_test, f_test)
for i in range(3):
    log += str(N) + "," + svc_kernals[i] + "," + str(svc_main_res[i]) + "," + str(svc_overall_res[i]) + "\n"
dl_main, dl_overall = DL_training(X_train, y_train, X_test, y_test, f_test)
log += str(N) + "," + "DNN," + str(np.max(dl_main)) + "," + str(np.max(dl_overall)) + "\n"
cnn_main, cnn_overall = CONV1d_training(X_train, y_train, X_test, y_test, f_test)
log += str(N) + "," + "CNN," + str(np.max(cnn_main)) + "," + str(np.max(cnn_overall)) + "\n"

#########
# validation
# for N in [26]:
#     np.random.shuffle(train_keys)
#     vali_keys = train_keys[:int(len(train_keys)/10)]
#     X_train, y_train, f_train = get_feature_label(train_groups, train_keys[int(len(train_keys)/10):], N)
#     X_test, y_test, f_test = get_feature_label(train_groups, vali_keys, N)
# X_train, y_train, f_train = get_feature_label(train_groups, train_keys, N)
# X_test, y_test, f_test = get_feature_label(test_groups, test_keys, N)
# main, overall = RF_training(X_train, y_train, X_test, y_test, f_test)
# log += str(N) + "," + "RF," + str(main) + "," + str(overall) + "\n"
# svc_main_res, svc_overall_res = SVC_training(X_train, y_train, X_test, y_test, f_test)
# for i in range(3):
#     log += str(N) + "," + svc_kernals[i] + "," + str(svc_main_res[i]) + "," + str(svc_overall_res[i]) + "\n"
# dl_main, dl_overall = DL_training(X_train, y_train, X_test, y_test, f_test)
# log += str(N) + "," + "DNN," + str(np.max(dl_main)) + "," + str(np.max(dl_overall)) +  "\n"
# dl_main, dl_overall = CONV1d_training(X_train, y_train, X_test, y_test, f_test)
# log +=str(N) + "," +  "CNN," + str(np.max(dl_main)) + "," + str(np.max(dl_overall)) +  "\n"

print(log)
