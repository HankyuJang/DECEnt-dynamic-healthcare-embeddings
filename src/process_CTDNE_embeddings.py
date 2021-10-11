"""
Author: -
Email: -
Last Modified: Oct, 2021

Process CTDNE
"""

import argparse
import numpy as np
import pandas as pd
import os

import numpy as np
import operator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-network', '--network', default='patient_DECEnt_PF_2010-01-01', type=str, help='name of the network (interaction data)')
    args = parser.parse_args()

    network = args.network

    #-----------------------------------------------------------------
    # STEP1
    # copy - paster code block from prepare_interactions_for_net_emb.py
    # to generate mapping and unique user_id_array
    # read interaction dataset
    df_interaction_data = pd.read_csv("../DECEnt/data/{}.csv".format(network))
    
    # df_interactions = df_interaction_data[["user_id", "item_id", "timestamp", "label"]]
    df_interactions = df_interaction_data[["user_id", "item_id", "timestamp"]].copy(deep=True)
    df_interactions = df_interactions.drop_duplicates()

    # Mapping
    unique_user_id_array = df_interactions.user_id.unique()
    unique_item_id_array = df_interactions.item_id.unique()
    unique_useritem_id_array = np.concatenate((unique_user_id_array, unique_item_id_array))

    # note: idx starts from 1
    id_idx_mapping = dict()
    idx_id_mapping = dict()
    for i, useritem_id in enumerate(unique_useritem_id_array):
        idx_id_mapping[i+1] = useritem_id
        id_idx_mapping[useritem_id] = i+1

    min_timestamp = df_interactions.timestamp.min()
    max_timestamp = df_interactions.timestamp.max()

    #-----------------------------------------------------------------
    # STEP2
    # Load learned embeddings
    method = "CTDNE"

    print("Procedding {} embeddings...".format(method))
    learned_emb_list = []

    df_interactions["user_idx"] = df_interactions["user_id"].map(id_idx_mapping)
    for t in range(min_timestamp, max_timestamp+1):
        # Get the users in each day. 
        df_temp = df_interactions[df_interactions.timestamp==t]
        users_in_the_day = df_temp.user_idx.unique()

        filename = "~/CTDNE/emb/day{}.emb".format(t)
        learned_emb = pd.read_csv(filename, sep=" ", header=None, skiprows=1)
        # filter in only those that are actually in the day
        learned_emb = learned_emb[learned_emb[0].isin(users_in_the_day)]
        
        learned_emb.insert(value=t, loc=1, column="timestamp")
        learned_emb_list.append(learned_emb)

    df_embedding = pd.concat([df for df in learned_emb_list], ignore_index=True)

    n_emb_dim = df_embedding.shape[1] - 2
    column_list = ["node_id", "timestamp"] + ["e{}".format(i) for i in range(n_emb_dim)]
    df_embedding.columns = column_list

    #-----------------------------------------------------------------
    # STEP3
    # filter in only users
    df_embedding["node_id"] = df_embedding["node_id"].map(idx_id_mapping)
    df_patient_embedding_per_day = df_embedding[df_embedding["node_id"].isin(unique_user_id_array)]
    # rename 'node_id' to 'user_id'
    df_patient_embedding_per_day = df_patient_embedding_per_day.rename(columns={"node_id": "user_id"})
    # change dtype of user_id to int
    df_patient_embedding_per_day = df_patient_embedding_per_day.astype({"user_id":int})
    df_patient_embedding = df_patient_embedding_per_day.drop_duplicates(subset=["user_id"], keep="last")

    # Save
    folder=method
    network="patient_DECEnt_PF_2010-01-01"

    # df_patient_embedding_per_day.to_csv("../data/{}/{}/df_patient_embedding_per_day.csv".format(folder, network), index=False)
    df_patient_embedding.to_csv("../data/{}/{}/df_patient_embedding.csv".format(folder, network), index=False)

