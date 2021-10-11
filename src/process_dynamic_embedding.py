"""
Author: -
Email: -
Last Modified: Oct, 2021

Process dynamic embedding
"""
import torch
import pickle
import pandas as pd
import numpy as np
import operator

def process_dynamic_embeddings_DECEnt(folder, network, ep):
    # Load user_embedding_time_series
    if folder in ["DECEnt_3M_auto_v1", "DECEnt_3M_auto_v3"]:
        filename = "../{}/saved_models/{}/checkpoint.ep{}.pth.tar".format(folder, network, ep)
    elif folder=="jodie":
        filename = "../{}/saved_models/{}/checkpoint.{}.ep{}.pth.tar".format(folder, network, folder, ep)
    checkpoint_saved_model = torch.load(filename, map_location=torch.device('cpu'))
    user_embedding_time_series = checkpoint_saved_model["user_embeddings_time_series"] 
    user_embeddings = checkpoint_saved_model["user_embeddings"] 
    emb_dim = user_embedding_time_series.shape[1]

    # Load interaction data
    df_interaction = pd.read_csv("../{}/data/{}.csv".format(folder, network))

    dynamic_embedding_array = np.hstack([df_interaction[["user_id", "timestamp"]].values, user_embedding_time_series])
    column_list = ["user_id", "timestamp"] + ["e{}".format(i) for i in range(emb_dim)]
    df_dynamic_embedding = pd.DataFrame(data=dynamic_embedding_array, columns=column_list)
    df_dynamic_embedding = df_dynamic_embedding.astype({"user_id":int, "timestamp":int})

    df_patient_embedding_per_day = df_dynamic_embedding.drop_duplicates(subset=["user_id", "timestamp"], keep="last")
    df_patient_embedding = df_patient_embedding_per_day.drop_duplicates(subset=["user_id"], keep="last")

    df_patient_embedding_per_day.to_csv("../data/{}/{}/df_patient_embedding_per_day.csv".format(folder, network), index=False)
    df_patient_embedding.to_csv("../data/{}/{}/df_patient_embedding.csv".format(folder, network), index=False)
    return df_patient_embedding_per_day, df_patient_embedding

def get_epoch_w_min_loss(folder, network, patience):
    filename = "../{}/loss/loss_{}.npz".format(folder, network)
    npzfile = np.load(filename)
    loss_per_timestep = npzfile["loss_per_timestep"]
    npzfile.close()

    # Find idx with smallest loss, but greater than 0.00001 (because initialization of the loss array is zero, and if training terminated before the end of epoch, thoes would have loss=0
    valid_idx = np.where(loss_per_timestep > 0.00001)[0]
    ep = valid_idx[np.argmin(loss_per_timestep[valid_idx])]

    max_ep = loss_per_timestep.shape[0]-1 # epoch starts from 0. so max ep is 1 minus the total epochs.
    if ep + patience > max_ep:
        ep = max_ep
    else: # If there was an early stopping,
        if valid_idx[-1] == ep:
            ep = ep
        else:
            ep = ep + patience

    return ep

if __name__ == "__main__":
    start_date = pd.Timestamp(2010,1,1)
    start_date_str = start_date.strftime('%Y-%m-%d')

    patience = 5

    print("\nJODIE")
    ep = get_epoch_w_min_loss("jodie", "patient_jodie_2010-01-01", patience)
    df_patient_embedding_per_day, df_patient_embedding = process_dynamic_embeddings_DECEnt("jodie", "patient_jodie_2010-01-01", ep)
    print("Dataset sizes: df_patient_embedding_per_day: {}, df_patient_embedding: {}".format(df_patient_embedding_per_day.shape, df_patient_embedding.shape))

    #####################################################
    # our method from here

    patience = 10

    print("DECEnt_3M_auto_v1")
    ep = get_epoch_w_min_loss("DECEnt_3M_auto_v1", "patient_DECEnt_PF_2010-01-01", patience)
    df_patient_embedding_per_day, df_patient_embedding = process_dynamic_embeddings_DECEnt("DECEnt_3M_auto_v1", "patient_DECEnt_PF_2010-01-01", ep)
    print("Dataset sizes: df_patient_embedding_per_day: {}, df_patient_embedding: {}".format(df_patient_embedding_per_day.shape, df_patient_embedding.shape))

    print("DECEnt_3M_auto_v3")
    ep = get_epoch_w_min_loss("DECEnt_3M_auto_v3", "patient_DECEnt_PF_2010-01-01", patience)
    df_patient_embedding_per_day, df_patient_embedding = process_dynamic_embeddings_DECEnt("DECEnt_3M_auto_v3", "patient_DECEnt_PF_2010-01-01", ep)
    print("Dataset sizes: df_patient_embedding_per_day: {}, df_patient_embedding: {}".format(df_patient_embedding_per_day.shape, df_patient_embedding.shape))

