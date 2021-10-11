"""
Author: -
Email: -
Last Modified: Oct, 2021

TODO: 
Map user_id and item_id to integers 1, 2, 3, ... Save the mapping dictionary (both ways).
Save each edgelist.
Write .sh to get embedding matrices of 90 timesteps
Load the 90 matrices. Make the two datasets in the following format:

user_id | timestamp | e0 | e1 | ...
"""

import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-network', '--network', default='patient_DECEnt_PF_2010-01-01', type=str, help='name of the network (interaction data)')
    args = parser.parse_args()

    network = args.network

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

    # Convert user_id and item_id to integers 1, 2, 3, ...
    df_interactions["user_idx"] = df_interactions["user_id"].map(id_idx_mapping)
    df_interactions["item_idx"] = df_interactions["item_id"].map(id_idx_mapping)

    min_timestamp = df_interactions.timestamp.min()
    max_timestamp = df_interactions.timestamp.max()

    for t in range(min_timestamp, max_timestamp+1):
        print(t)
        df_temp = df_interactions[df_interactions.timestamp==t]
        df_temp[["user_idx", "item_idx"]].to_csv("../data/node2vec_G/day{}.edgelist".format(t), sep=' ', header=False, index=False)

        # Save the edgelist
        df_temp = df_interactions[df_interactions.timestamp<=t]
        df_temp[["user_idx", "item_idx", "timestamp"]].to_csv("../data/CTDNE_G/day{}.edgelist".format(t), sep=' ', header=False, index=False)
        # Save the edgelist w/ timesteps

