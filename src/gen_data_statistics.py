'''
Author: -
Email: -
Last Modified: May, 2021

This script generates statistics of data
'''

import pandas as pd
import numpy as np

if __name__ == "__main__":

    df_interactions = pd.read_csv("../DECEnt_3M_auto_v1/data/patient_DECEnt_PF_2010-01-01.csv")

    df_D_interactions = df_interactions[df_interactions["label"]=='D']
    df_M_interactions = df_interactions[df_interactions["label"]=='M']
    df_R_interactions = df_interactions[df_interactions["label"]=='R']
    
    n_interactions = df_interactions.shape[0]
    n_D_interactions = df_D_interactions.shape[0]
    n_M_interactions = df_M_interactions.shape[0]
    n_R_interactions = df_R_interactions.shape[0]

    n_patients = df_interactions.user_id.unique().shape[0]
    n_D = df_D_interactions.item_id.unique().shape[0]
    n_M = df_M_interactions.item_id.unique().shape[0]
    n_R = df_R_interactions.item_id.unique().shape[0]

    features = list(df_interactions.columns)[5:]
    n_static_features = 2
    n_dynamic_features = len(features) - n_static_features
    
    min_timestamp = df_interactions.timestamp.min()
    max_timestamp = df_interactions.timestamp.min()
    n_distinct_timestamps = max_timestamp - min_timestamp + 1

    index_list = [
            "Number of patients", "Number of doctors", "Number of medications", "Number of rooms",
            "Total interactions", "patient - doctors interactions", "patient - medication interactions", "patient - room interactions",
            "Time resolution", "# of distinct time-stamps"]
    values = [
            n_patients, n_D, n_M, n_R,
            n_interactions, n_D_interactions, n_M_interactions, n_R_interactions,
            "day", n_distinct_timestamps]
    df_data_summary = pd.DataFrame(data=values, index=index_list)
    df_data_summary.to_csv("../table/data_summary.csv")
