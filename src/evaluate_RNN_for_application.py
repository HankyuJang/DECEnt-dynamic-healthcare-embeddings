"""
Author: -
Email: -
Last Modified: Oct, 2021

"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, LSTM, SimpleRNN
# from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
# import keras_metrics
import tensorflow as tf
from keras.utils import to_categorical
import os
import pickle

from imblearn.under_sampling import RandomUnderSampler

def evaluate(y_true, y_pred):
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    # print(metrics.classification_report(y_true, y_pred, digits=4))
    return f1_micro, f1_macro

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-label', '--label', default='CDI', type=str, help='Filename of the label dataframe. CDI | mortality | severity | MICU_transfer')
    parser.add_argument('-network', '--network', default='patient_DECEnt_PF_2010-01-01', type=str, help='name of the network (interaction data)')
    parser.add_argument('-gpu', '--gpu', default=3, type=int, help='ID of the gpu. Default is 0')
    parser.add_argument('-epochs', '--epochs', default=30, type=int, help='epochs')
    parser.add_argument('-method', '--method', default="LSTM", type=str, help='RNN or LSTM')
    args = parser.parse_args()

    label = args.label
    network = args.network
    method = args.method
    epochs = args.epochs
    n_repetition = 30
    # n_repetition = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    folder = args.method

    # columns of df_label is ['user_id', 'timestamp', 'label']
    df_label = pd.read_csv("../data/patient_label/df_{}.csv".format(label))

    # read interaction dataset
    df_interaction_data = pd.read_csv("../DECEnt/data/{}.csv".format(network))
    # drop unused columns
    df_interaction_data.drop(["item_id", "state_label", "label"], axis=1, inplace=True)

    if label in ["mortality", "severity"]:
        # Drop duplicates in ['user_id']
        df_interaction_data.drop_duplicates(subset=["user_id"], inplace=True)
        df_dataset = pd.merge(left=df_label, right=df_interaction_data, how="inner", on=["user_id"])
        scoring_metric = 'f1_micro'
    else:
        # Drop duplicates in ['user_id', 'timestamp']
        df_interaction_data.drop_duplicates(subset=["user_id", "timestamp"], inplace=True)
        df_dataset = pd.merge(left=df_label, right=df_interaction_data, how="inner", on=["user_id", "timestamp"])
        scoring_metric = 'roc_auc'


    # Do the sorting in this way to keep the sequence of instances to be exactly the same over different models
    df_dataset.sort_values(by="timestamp", inplace=True)
    df_dataset.sort_values(by="user_id", inplace=True)
    df_dataset.reset_index(drop=True, inplace=True)
    print("df_dataset.shape: {}".format(df_dataset.shape))

    ####################
    # Prep datasets as input for RNN, LSTM
    n_instances = df_dataset.shape[0]
    n_timesteps=5
    n_features = df_dataset.shape[1] - 3 # user id and timestamp
    X = np.zeros((n_instances, n_timesteps, n_features))
    y = df_dataset["label"].values

    n_classes = np.unique(y).shape[0]

    print("Preparing input for {}...".format(method))
    for idx, row in df_dataset.iterrows(): # integers change to float in iterrows command.
        timestep_idx = n_timesteps-1
        X[idx, timestep_idx, :] = row.values[3:]

        user_id = int(row["user_id"])
        timestamp = int(row["timestamp"])

        df_temp = df_interaction_data[df_interaction_data["user_id"]==user_id]
        if df_temp.shape[0]==1: # only one record
            continue

        for t in range(timestamp-1, timestamp-n_timesteps, -1):
            timestep_idx -= 1
            # Find the record
            prev_timestamp_record = df_temp[df_temp["timestamp"]==t]
            if prev_timestamp_record.empty: # no record
                break
            X[idx, timestep_idx, :] = prev_timestamp_record.values[0, 2:]

    ##############################
    # TRAINING
    if label in ["CDI", "MICU_transfer"]:
        network_eval_results = {"train_auc":[], "test_auc":[], "test_fpr":[], "test_tpr":[]}
    else:
        network_eval_results = {"train_f1_micro":[], "train_f1_macro":[], "test_f1_micro":[], "test_f1_macro":[]}

    # standardize
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # undersample = RandomUnderSampler(sampling_strategy='majority')

    y_onehot = to_categorical(y, n_classes)
    # 5-fold cv
    skf = StratifiedKFold(n_splits=5)
    for rep in range(n_repetition):
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # if label in ["CDI", "MICU_transfer"]:
                # X_train, y_train = undersample.fit_resample(X_train, y_train)

            y_onehot_train = to_categorical(y_train, n_classes)
            y_onehot_test = to_categorical(y_test, n_classes)

            model=Sequential()
            if method == "LSTM":
                model.add(LSTM(n_features, activation='relu', input_shape=(n_timesteps, n_features)))
            elif method == "RNN":
                model.add(SimpleRNN(n_features, activation='relu', input_shape=(n_timesteps, n_features)))
            model.add(Dense(n_classes, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy')

            # model.fit(X, y_onehot, epochs=epochs, batch_size=20, verbose=1, validation_split=0.2, callbacks=[es])
            model.fit(X_train, y_onehot_train, epochs=epochs, batch_size=20, verbose=0)

            # PREDICT
            pred = model.predict_classes(X_test)
            pred_prob = model.predict(X_test)
            probs = pred_prob[:,1]

            if label in ["CDI", "MICU_transfer"]:
                fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
                auc = metrics.auc(fpr, tpr)
                network_eval_results["test_auc"].append(auc)
                network_eval_results["test_fpr"].append(fpr)
                network_eval_results["test_tpr"].append(tpr)
            else:
                # evaluate prediction
                f1_micro, f1_macro = evaluate(y_test, pred)
                network_eval_results["test_f1_micro"].append(f1_micro)
                network_eval_results["test_f1_macro"].append(f1_macro)

            train_pred = model.predict_classes(X_train)
            train_pred_prob = model.predict(X_train)
            train_probs = train_pred_prob[:,1]

            if label in ["CDI", "MICU_transfer"]:
                fpr, tpr, thresholds = metrics.roc_curve(y_train, train_probs)
                auc = metrics.auc(fpr, tpr)
                network_eval_results["train_auc"].append(auc)
            else:
                train_f1_micro, train_f1_macro = evaluate(y_train, train_pred)
                network_eval_results["train_f1_micro"].append(train_f1_micro)
                network_eval_results["train_f1_macro"].append(train_f1_macro)

    ###################
    # Training complete

    filename = "../result/{}/{}/{}_tpr_fpr_{}.pickle".format(folder, network, label, method)
    outfile = open(filename, "wb")
    pickle.dump(network_eval_results, outfile)
    outfile.close()
    # 

    # network_eval_results contains tpr and fpr. remove these before creating the result dataframe
    network_eval_results.pop("test_fpr", None)
    network_eval_results.pop("test_tpr", None)
    network_eval_df = pd.DataFrame(network_eval_results)
    # print(network_eval_df.mean())

    network_eval_mean_series = np.mean(network_eval_df)
    network_eval_std_series = np.std(network_eval_df)
    result_series = network_eval_mean_series.round(3).astype(str) + " (" + network_eval_std_series.round(3).astype(str) + ")"
    result_df = pd.DataFrame(data=result_series, columns=["{}".format(network)]).T
    print(result_df)

    result_df.to_csv("../result/{}/{}/{}_{}.csv".format(folder, network, label, method))

