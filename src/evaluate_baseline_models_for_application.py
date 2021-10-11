"""
Author: -
Email: -
Last Modified: Sep, 2021

Required two inputs: label dataframe and intereaction dataset

This script trains classification algorithms using the engineered features.
Also, does a abalation study to find the set of features that gives the best performance.
Serves as a baseline method (for domain specific methods)
and predicts outcomes.

NOTE: data instances from this script should match the data instances from `evaluate_patient_embeddings.py` script
NOTE: Instead of 2-fold, do 5-fold - because it's somewhat standard that others do
"""

import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import statistics

from sklearn.model_selection import cross_validate

# These are added for MLP
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint  
from keras import backend as K
# import keras_metrics
import tensorflow as tf

from sklearn.neural_network import MLPClassifier
from evaluate_patient_embeddings import define_model
from evaluate_patient_embeddings import prepare_feature_label
from evaluate_patient_embeddings import classification
from evaluate_patient_embeddings import evaluate

import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-label', '--label', default='CDI', type=str, help='Filename of the label dataframe. CDI | mortality | severity | MICU_transfer')
    parser.add_argument('-clf_name', '--clf_name', default='rf', type=str, help='Name of the clf_name: logit, rf, MLP')
    parser.add_argument('-network', '--network', default='patient_DECEnt_PF_2010-01-01', type=str, help='name of the network (interaction data)')

    args = parser.parse_args()

    label = args.label
    clf_name = args.clf_name
    network = args.network
    n_repetition = 30
    folder = "domain_specific"

    # Classifier to use
    clf_for_feature_importance = RandomForestClassifier(n_estimators=1000, max_depth=2)
    if clf_name=="logit":
        clf = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500)
    elif clf_name=="rf":
        clf = RandomForestClassifier(n_estimators=1000, max_depth=2)
    elif clf_name=="MLP":
        clf = "MLP"

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
    df_dataset.sort_values(by="user_id", inplace=True)
    df_dataset.sort_values(by="timestamp", inplace=True)
    print("df_dataset.shape: {}".format(df_dataset.shape))

    # Prepare X and y
    feature_names = list(df_dataset.iloc[:, 3:].columns)
    X = df_dataset.iloc[:, 3:].values
    y = df_dataset["label"].values

    ################################################################################################################
    # Step1: Compute feature importance. Prepare a sorted version of features in decreasing order of importance.
    print("\n Step1: compute feature importance")
    n_cv = 5 # 5-fold CV
    feature_importances_array = np.zeros((n_cv, len(feature_names)))
    # output = cross_validate(clf_for_feature_importance, X, y, cv=n_cv, scoring = 'roc_auc', return_estimator=True) # this uses stratified K-fold
    output = cross_validate(clf_for_feature_importance, X, y, cv=n_cv, scoring = scoring_metric, return_estimator=True) # this uses stratified K-fold

    for idx, estimator in enumerate(output['estimator']):
        print("Features sorted by their score for estimator {}:".format(idx))
        feature_importances_array[idx, :] = estimator.feature_importances_

    df_mean_feature_importance_kfold = pd.DataFrame(
            data=np.mean(feature_importances_array, axis=0),
            index=feature_names,
            columns=["importance"]).sort_values('importance', ascending=False)
    print(df_mean_feature_importance_kfold)

    feature_names_sorted_by_importance_desc = list(df_mean_feature_importance_kfold.index)
    # End of Step1
    ################################################################################################################

    ################################################################################################################
    # Step2: Do the linear search over the features.
    # Start w/ all features, then remove features w/ most least importance. Stop if model performance degrades
    print("\n Step2: linear search over the features.")
    n_features = len(feature_names)
    best_score = 0.0
    best_std = 0.0
    best_feature_names = feature_names_sorted_by_importance_desc[:n_features]
    while True:
        print("# features: {}".format(n_features))
        feature_names = feature_names_sorted_by_importance_desc[:n_features] # selecting top n_feature important features in modeling
        X = df_dataset[feature_names].values

        if clf_name=="MLP":
            model = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', solver='adam')
            # output = cross_validate(model, X, y, cv=n_cv, scoring = 'roc_auc', return_estimator=True) # this uses stratified K-fold
            output = cross_validate(model, X, y, cv=n_cv, scoring = scoring_metric, return_estimator=True) # this uses stratified K-fold
        else:
            # output = cross_validate(clf, X, y, cv=n_cv, scoring = 'roc_auc', return_estimator=True) # this uses stratified K-fold
            output = cross_validate(clf, X, y, cv=n_cv, scoring = scoring_metric, return_estimator=True) # this uses stratified K-fold
        mean_score = np.mean(output["test_score"])
        std_score = np.std(output["test_score"])
        print("mean_score: {:.3f}".format(mean_score))
        print("std_score: {:.3f}".format(std_score))

        if mean_score > best_score:
            best_score = mean_score
            best_std = std_score
            best_feature_names = feature_names
        else:
            break
        n_features -= 1

    print("Best score of the Baseline model: {:.3f}".format(best_score))
    print("std: {:.3f}".format(best_std))
    # End of Step2
    ################################################################################################################

    ################################################################################################################
    # Step3
    # At this point, we have the dataset X, with features optimized to give the best performance.
    # Now, run the classification models using the `classification` function
    # Use the best set of features in the experiment
    X = df_dataset[best_feature_names].values
    print("\n Step3, train the model again, with the feature set w/ repetition.")
    print("\nlabel: {}, clf: {}".format(label, clf))
    print("### Domain specific method {}###".format(clf_name))
    print("Shape of feature matrix: {}".format(X.shape))
    network_eval_results = classification(clf, clf_name, X, y, label, n_repetition)

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

    network_eval_mean_series = np.mean(network_eval_df)
    network_eval_std_series = np.std(network_eval_df)
    result_series = network_eval_mean_series.round(3).astype(str) + " (" + network_eval_std_series.round(3).astype(str) + ")"
    result_df = pd.DataFrame(data=result_series, columns=["{}".format(network)]).T
    print(result_df)

    result_df.to_csv("../result/{}/{}/{}_{}.csv".format(folder, network, label, clf_name))
    # End of Step3
    ################################################################################################################

