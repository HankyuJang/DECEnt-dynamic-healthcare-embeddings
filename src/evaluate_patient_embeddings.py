'''
Author: -
Email: -
Last Modified: Sep, 2021

Required two inputs: label dataframe and dynamic embedding

This script trains classification algorithms based on the learned embedding 
and predicts outcomes.
5-fold CV

Script usage:
    # patient_hidep

    $ python -i evaluate_patient_embeddings.py -network patient_hidep_PF_2010-01-01 -label CDI -clf_name logit
    $ python -i evaluate_patient_embeddings.py -network patient_hidep_PF_2010-01-01 -label severity -clf_name logit
    $ python -i evaluate_patient_embeddings.py -network patient_hidep_PF_2010-01-01 -label mortality -clf_name logit
    $ python -i evaluate_patient_embeddings.py -network patient_hidep_PF_2010-01-01 -label MICU_transfer -clf_name logit
'''

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import statistics

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint  
from keras import backend as K
# import keras_metrics
import tensorflow as tf
import pickle

from imblearn.under_sampling import RandomUnderSampler

#####################################################################
# MLP
def define_model(X_train, n_classes):
    K.clear_session()
    # Neural Network model
    model = Sequential()
    # model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
    # model.add(Dropout(.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(n_classes, activation='softmax'))
    # model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=[keras_metrics.precision()])
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    # model.summary()
    return model

#####################################################################

def prepare_feature_label(input_df, embedding_colnames, label_colname):
    X = np.array(input_df[embedding_colnames])
    y = np.array(input_df[label_colname])

    return X, y

def classification(clf, clf_name, X, y, label, n_repetition):

    if label in ["CDI", "MICU_transfer"]:
        eval_results = {"train_auc":[], "test_auc":[], "test_fpr":[], "test_tpr":[]}
    else:
        eval_results = {"train_f1_micro":[], "train_f1_macro":[], "test_f1_micro":[], "test_f1_macro":[]}

    # standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Undersample the training set on majority class for highly imbalanced binary classification
    # undersample = RandomUnderSampler(sampling_strategy='majority', random_state=123)
    undersample = RandomUnderSampler(sampling_strategy='majority')

    if clf_name=="MLP":
        n_classes = np.unique(y).shape[0]
        # 5-fold cross validation
        # skf = StratifiedKFold(n_splits=5, random_state=123)
        skf = StratifiedKFold(n_splits=5)
        for rep in range(n_repetition):
            for train_idx, test_idx in skf.split(X,y):
                X_train_val, X_test = X[train_idx], X[test_idx]
                y_train_val, y_test = y[train_idx], y[test_idx]
                X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=123, stratify=y_train_val)

                if label in ["CDI", "MICU_transfer"]:
                    X_train, y_train = undersample.fit_resample(X_train, y_train)

                y_onehot_train = to_categorical(y_train, n_classes)
                y_onehot_val = to_categorical(y_val, n_classes)
                y_onehot_test = to_categorical(y_test, n_classes)


                model = define_model(X_train, n_classes)
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)
                # model.fit(X_train, y_onehot_train, epochs=200, batch_size=20, verbose=0, validation_split=0.5, callbacks=[es])
                model.fit(X_train, y_onehot_train, epochs=200, batch_size=20, verbose=0, validation_data=(X_val, y_onehot_val), callbacks=[es])
                pred = model.predict_classes(X_test)
                pred_prob = model.predict(X_test)
                probs = pred_prob[:,1]

                if label in ["CDI", "MICU_transfer"]:
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
                    auc = metrics.auc(fpr, tpr)
                    eval_results["test_auc"].append(auc)
                    eval_results["test_fpr"].append(fpr)
                    eval_results["test_tpr"].append(tpr)
                else:
                    # evaluate prediction
                    f1_micro, f1_macro = evaluate(y_test, pred)
                    eval_results["test_f1_micro"].append(f1_micro)
                    eval_results["test_f1_macro"].append(f1_macro)

                train_pred = model.predict_classes(X_train)
                train_pred_prob = model.predict(X_train)
                train_probs = train_pred_prob[:,1]

                if label in ["CDI", "MICU_transfer"]:
                    fpr, tpr, thresholds = metrics.roc_curve(y_train, train_probs)
                    auc = metrics.auc(fpr, tpr)
                    eval_results["train_auc"].append(auc)
                else:
                    train_f1_micro, train_f1_macro = evaluate(y_train, train_pred)
                    eval_results["train_f1_micro"].append(train_f1_micro)
                    eval_results["train_f1_macro"].append(train_f1_macro)

    else:
        # 5-fold cross validation
        # skf = StratifiedKFold(n_splits=5, random_state=123)
        skf = StratifiedKFold(n_splits=5)
        for rep in range(n_repetition):
            for train_idx, test_idx in skf.split(X,y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                # unique,counts = np.unique(y_train, return_counts=True)
                # print(dict(zip(unique,counts)))
                # unique,counts = np.unique(y_test, return_counts=True)
                # print(dict(zip(unique,counts)))

                if label in ["CDI", "MICU_transfer"]:
                    X_train, y_train = undersample.fit_resample(X_train, y_train)

                # clf = OneVsRestClassifier(LogisticRegression())
                # clf = OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                pred_prob = clf.predict_proba(X_test)
                probs = pred_prob[:,clf.classes_ == True].flatten()

                # unique,counts = np.unique(pred, return_counts=True)
                # print(dict(zip(unique,counts)))

                if label in ["CDI", "MICU_transfer"]:
                    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
                    auc = metrics.auc(fpr, tpr)
                    eval_results["test_auc"].append(auc)
                    eval_results["test_fpr"].append(fpr)
                    eval_results["test_tpr"].append(tpr)
                else:
                    # evaluate prediction
                    f1_micro, f1_macro = evaluate(y_test, pred)
                    eval_results["test_f1_micro"].append(f1_micro)
                    eval_results["test_f1_macro"].append(f1_macro)

                # training evaluation
                train_pred = clf.predict(X_train)
                train_pred_prob = clf.predict_proba(X_train)
                train_probs = train_pred_prob[:,clf.classes_ == True].flatten()
                # unique,counts = np.unique(train_pred, return_counts=True)
                # print(dict(zip(unique,counts)))

                if label in ["CDI", "MICU_transfer"]:
                    fpr, tpr, thresholds = metrics.roc_curve(y_train, train_probs)
                    auc = metrics.auc(fpr, tpr)
                    eval_results["train_auc"].append(auc)
                else:
                    train_f1_micro, train_f1_macro = evaluate(y_train, train_pred)
                    eval_results["train_f1_micro"].append(train_f1_micro)
                    eval_results["train_f1_macro"].append(train_f1_macro)

    return eval_results

def evaluate(y_true, y_pred):
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    # print(metrics.classification_report(y_true, y_pred, digits=4))
    return f1_micro, f1_macro

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', '--folder', required=True, help='name of the folder')
    parser.add_argument('-network', '--network', required=True, help='name of the network')
    parser.add_argument('-label', '--label', type=str, help='Filename of the label dataframe. CDI | mortality | severity | MICU_transfer')
    parser.add_argument('-clf_name', '--clf_name', type=str, help='Name of the clf_name: logit, rf, MLP')

    args = parser.parse_args()

    folder = args.folder
    network = args.network
    label = args.label
    clf_name = args.clf_name
    n_repetition = 30

    # Classifier to use
    if clf_name=="logit":
        clf = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500)
    elif clf_name=="rf":
        # clf = RandomForestClassifier(n_estimators=1000, max_depth=2, random_state=123)
        clf = RandomForestClassifier(n_estimators=1000, max_depth=2)
    elif clf_name=="MLP":
        clf = "MLP"

    # columns of df_label is ['user_id', 'timestamp', 'label']
    df_label = pd.read_csv("../data/patient_label/df_{}.csv".format(label))

    # columns of df_patient_embedding and df_patient_embedding_per_day is ['user_id', 'timestamp', 'e0', 'e1', ..., 'e127']
    if label in ["mortality", "severity"]:
        df_patient_embedding = pd.read_csv("../data/{}/{}/df_patient_embedding.csv".format(folder, network))

        df_dataset = pd.merge(left=df_label, right=df_patient_embedding, how="inner", on=["user_id"])

    else:
        df_patient_embedding_per_day = pd.read_csv("../data/{}/{}/df_patient_embedding_per_day.csv".format(folder, network))
        # columns of df_dataset is ['user_id', 'timestamp', 'label', 'e0', 'e1', ..., 'e127']
        df_dataset = pd.merge(left=df_label, right=df_patient_embedding_per_day, how="inner", on=["user_id", "timestamp"])

    # Do the sorting in this way to keep the sequence of instances to be exactly the same over different models
    df_dataset.sort_values(by="user_id", inplace=True)
    df_dataset.sort_values(by="timestamp", inplace=True)
    print("df_dataset.shape: {}".format(df_dataset.shape))

    # # specify feature and label column names
    emb_dim = df_dataset.shape[1] - 3
    embedding_colnames = ['e{}'.format(i) for i in range(0, emb_dim)]
    label_colname = "label"

    ###################################
    print("\nlabel: {}, clf: {}".format(label, clf))
    print("### Proposed method {}###".format(clf_name))
    network_X, network_y = prepare_feature_label(df_dataset, embedding_colnames, label_colname)
    print("Shape of feature matrix: {}".format(network_X.shape))
    network_eval_results = classification(clf, clf_name, network_X, network_y, label, n_repetition)

    # Save the results for later use
    filename = "../result/{}/{}/{}_tpr_fpr_{}.pickle".format(folder, network, label, clf_name)
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

    result_df.to_csv("../result/{}/{}/{}_{}.csv".format(folder, network, label, clf_name))


