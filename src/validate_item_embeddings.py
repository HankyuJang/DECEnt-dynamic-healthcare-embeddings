
"""
Author: -
Email: -
Last Modified: Sep 2021

Validate item embeddings by projecting the embedding into 2 dimensions using t-SNE, plot a scatter plot and color the nodes based on their category.
For the entities that have too many categories, we select top k categories.
This script also computes dispersion matrices per entity
"""

import argparse
import torch
import pickle
import operator
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.query_from_db import *
import mysql.connector

def get_n_combination(x):
    return int(x * (x-1) / 2)

def get_embedding(folder, network, ep):

    filename = "../{}/saved_models/{}/checkpoint.ep{}.pth.tar".format(folder, network, ep)

    with open("../{}/pickle/{}_item2id.pickle".format(folder, network), "rb") as handle:
        item2id = pickle.load(handle)
    with open("../{}/pickle/{}_item2itemtype.pickle".format(folder, network), "rb") as handle:
        item2itemtype = pickle.load(handle)

    item2id_itemtype = {}
    for item in item2id:
        item2id_itemtype[item] = (item2id[item], item2itemtype[item])
    sorted_item2id_itemtype = sorted(item2id_itemtype.items(), key=operator.itemgetter(1))
    item_array = np.array([item[0] for item in sorted_item2id_itemtype])
    itemtype_array = np.array([item[1][1] for item in sorted_item2id_itemtype])
    # sorted_item2id = sorted(item2id.items(), key=operator.itemgetter(1))
    # item_array = np.array([item[0] for item in sorted_item2id])
    n_items = item_array.shape[0]

    checkpoint_medication = torch.load(filename, map_location=torch.device('cpu'))
    item_embedding_medication = checkpoint_medication["item_embeddings"] 
    embedding_dim = checkpoint_medication["item_embeddings_time_series"].shape[1] 
    # column_embedding = ["d{}".format(x) for x in range(1, embedding_dim+1)]

    # remove the last row (it's not actual item embedding)
    embedding = item_embedding_medication[:n_items,:embedding_dim]
    return item_array, itemtype_array, embedding

def plot_embedding(embedding_2d, groups, label_colname, outfile):

    fig, ax=plt.subplots()
    # color_list = ["r", "g", "b"]
    # color_idx = 0
    idx = 0
    marker_list=['o', '^', '1', '2', '3', '4', '8']
    for name, group in groups:
        ax.scatter(
                x=embedding_2d[group.index,0], 
                y=embedding_2d[group.index,1], 
                s=50, cmap="tab20", 
                label=group[label_colname].iloc[0],
                marker=marker_list[idx])
        idx += 1
        print(idx)
        # color_idx += 1
    box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
    ax.set_position([box.x0, box.y0, box.width, box.height])
    # ax.legend(loc="center left", bbox_to_anchor=(1,0.5), prop={'size':7})
    ax.legend(loc="best")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.savefig(outfile, dpi=300)
    plt.close()

def plot_embedding_v2(embedding_2d, embedding_2d_others, groups, label_colname, outfile):

    fig, ax=plt.subplots()
    # color_list = ["r", "g", "b"]
    # color_idx = 0
    # plot all the nodes as small dots
    ax.scatter(
            x=embedding_2d_others[:,0],
            y=embedding_2d_others[:,1],
            s = 10,
            label="Others",
            marker=".",
            color="black",
            zorder=100
            )
    idx = 0
    color_list = ["tab:orange", "tab:green", "tab:red", "tab_blue"]
    marker_list=['o', '^', '1', '2', '3', '4', '8']
    for name, group in groups:
        ax.scatter(
                x=embedding_2d[group.index,0], 
                y=embedding_2d[group.index,1], 
                # s=50, cmap="tab20", 
                s=50, color=color_list[idx],
                label=group[label_colname].iloc[0],
                marker=marker_list[idx])
        idx += 1
        print(idx)
        # color_idx += 1
    box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
    ax.set_position([box.x0, box.y0, box.width, box.height])
    # ax.legend(loc="center left", bbox_to_anchor=(1,0.5), prop={'size':7})
    ax.legend(loc="best", prop={'size':15})
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.savefig(outfile, dpi=300)
    plt.close()

def prepare_df_medication_mapping(item_array):
    df_medication_mapping = pd.read_csv("../data/mapping_medication.csv")
    df_item = pd.DataFrame({"mid":item_array})
    df_medication_mapping = pd.merge(left=df_item, right=df_medication_mapping, on="mid", how="left")
    # for items that do not have mid-majid mapping, set value = -1
    df_medication_mapping = df_medication_mapping.fillna(-1)
    df_medication_mapping = df_medication_mapping.astype({"majid":int})
    df_medication_mapping = df_medication_mapping.astype({"minid":int})
    df_medication_mapping = df_medication_mapping.astype({"subminid":int})

    # unique_label = np.unique(df_medication_mapping.majid.values)
    # majid_label_mapping = dict(zip(unique_label, np.arange(unique_label.shape[0])))
    # add a column 'label' that has unique mapping of majid to integer starting from 0
    # df_medication_mapping = df_medication_mapping.assign(label=df_medication_mapping.majid.map(majid_label_mapping))
    return df_medication_mapping

# 'clip': physician id
# 'sid': specialty id
def prepare_df_physician_mapping(item_array):
    df_physician_mapping = pd.read_csv("../data/mapping_physician_specialty.csv")
    df_physician_mapping = df_physician_mapping.drop_duplicates(subset="clip")

    df_item = pd.DataFrame({"clip":item_array})
    df_physician_mapping = pd.merge(left=df_item, right=df_physician_mapping, on="clip", how="left")
    df_physician_mapping = df_physician_mapping.fillna(-1)
    df_physician_mapping = df_physician_mapping.astype({"sid":int})

    # unique_label = np.unique(df_physician_mapping.sid.values)
    # sid_label_mapping = dict(zip(unique_label, np.arange(unique_label.shape[0])))
    # df_physician_mapping = df_physician_mapping.assign(label=df_physician_mapping.sid.map(sid_label_mapping))
    return df_physician_mapping

def prepare_df_room_mapping(item_array):
    df_room_mapping = pd.read_csv("../data/mapping_room_unit.csv")
    df_room_mapping = df_room_mapping.drop_duplicates(subset="rid")

    df_item = pd.DataFrame({"rid":item_array.astype(int)})
    df_room_mapping = pd.merge(left=df_item, right=df_room_mapping, on="rid", how="left")
    df_room_mapping = df_room_mapping.fillna(-1)
    df_room_mapping = df_room_mapping.astype({"uid":int})

    # unique_label = np.unique(df_room_mapping.uid.values)
    # uid_label_mapping = dict(zip(unique_label, np.arange(unique_label.shape[0])))
    # df_room_mapping = df_room_mapping.assign(label=df_room_mapping.uid.map(uid_label_mapping))
    return df_room_mapping

def get_item_arrays_and_emb(item_array, itemtype_array, DECEnt_embedding):
    medication_index = [itemtype=='M' for itemtype in itemtype_array]
    physician_index = [itemtype=='D' for itemtype in itemtype_array]
    room_index = [itemtype=='R' for itemtype in itemtype_array]

    medication_array = item_array[medication_index]
    physician_array = item_array[physician_index]
    room_array = item_array[room_index]

    medication_emb = DECEnt_embedding[medication_index]
    physician_emb = DECEnt_embedding[physician_index]
    room_emb = DECEnt_embedding[room_index]
    return medication_array, physician_array, room_array, medication_emb, physician_emb, room_emb

def plot_2d_medication_manual(df_medication_mapping):
    # extract medication mapping and embedding with specific labels
    df_medication_mapping_manual = df_medication_mapping[df_medication_mapping.majclass.isin(selected_medication)]
    medication_manual_label_index = df_medication_mapping_manual.index
    medication_emb_manual = medication_emb[medication_manual_label_index]

    df_medication_mapping_manual.reset_index(drop=True, inplace=True)
    medication_groups = df_medication_mapping_manual.groupby('majid')

    # t-SNE
    tsne = TSNE(n_components=2)
    medication_emb_2d = tsne.fit_transform(medication_emb_manual)
    outfile = "../plots/{}/medication_embedding_manual.png".format(folder)
    plot_embedding(medication_emb_2d, medication_groups, "majclass", outfile)

def plot_2d_physician_manual(df_physician_mapping):
    # extract physician mapping and embedding with specific labels
    df_physician_mapping_manual = df_physician_mapping[df_physician_mapping.name.isin(selected_physician)]
    physician_manual_label_index = df_physician_mapping_manual.index
    physician_emb_manual = physician_emb[physician_manual_label_index]

    df_physician_mapping_manual.reset_index(drop=True, inplace=True)
    physician_groups = df_physician_mapping_manual.groupby('sid')

    # t-SNE
    tsne = TSNE(n_components=2)
    physician_emb_2d = tsne.fit_transform(physician_emb_manual)
    outfile = "../plots/{}/physician_embedding_manual.png".format(folder)
    plot_embedding(physician_emb_2d, physician_groups, "name", outfile)

def plot_2d_physician_manual_v2(df_physician_mapping):
    # t-SNE
    for rand_state in range(30):
        tsne = TSNE(n_components=2, random_state=rand_state)
        physician_emb_2d = tsne.fit_transform(physician_emb)

        # extract physician mapping and embedding with specific labels
        df_physician_mapping_manual = df_physician_mapping[df_physician_mapping.name.isin(selected_physician)]
        physician_manual_label_index = df_physician_mapping_manual.index
        physician_emb_2d_manual = physician_emb_2d[physician_manual_label_index]
        # print(physician_emb_manual.shape)
        
        df_physician_mapping_others = df_physician_mapping[~df_physician_mapping.name.isin(selected_physician)]
        other_index = df_physician_mapping_others.index
        physician_emb_2d_others = physician_emb_2d[other_index]
        # print(physician_emb_others.shape)

        df_physician_mapping_manual.reset_index(drop=True, inplace=True)
        physician_groups = df_physician_mapping_manual.groupby('sid')

        # t-SNE
        # tsne = TSNE(n_components=2)
        # physician_emb_2d = tsne.fit_transform(physician_emb_manual)
        outfile = "../plots/{}/physician_embedding_manual_rs{}.png".format(folder, rand_state)
        plot_embedding_v2(physician_emb_2d_manual, physician_emb_2d_others, physician_groups, "name", outfile)

def plot_2d_room_manual(df_room_mapping):
    # extract room mapping and embedding with specific labels
    df_room_mapping_manual = df_room_mapping[df_room_mapping.name.isin(selected_room)]
    room_manual_label_index = df_room_mapping_manual.index
    room_emb_manual = room_emb[room_manual_label_index]

    df_room_mapping_manual.reset_index(drop=True, inplace=True)
    room_groups = df_room_mapping_manual.groupby('uid')

    # t-SNE
    tsne = TSNE(n_components=2)
    room_emb_2d = tsne.fit_transform(room_emb_manual)
    outfile = "../plots/{}/room_embedding_manual.png".format(folder)
    plot_embedding(room_emb_2d, room_groups, "name", outfile)

# This version, plot the rest of the rooms as in the small dots
def plot_2d_room_manual_v2(df_room_mapping):
    # t-SNE
    tsne = TSNE(n_components=2, random_state=100)
    room_emb_2d = tsne.fit_transform(room_emb)

    # extract room mapping and embedding with specific labels
    df_room_mapping_manual = df_room_mapping[df_room_mapping.name.isin(selected_room)]
    room_manual_label_index = df_room_mapping_manual.index
    room_emb_2d_manual = room_emb_2d[room_manual_label_index]
    # print(room_emb_manual.shape)
    
    df_room_mapping_others = df_room_mapping[~df_room_mapping.name.isin(selected_room)]
    other_index = df_room_mapping_others.index
    room_emb_2d_others = room_emb_2d[other_index]
    # print(room_emb_others.shape)

    df_room_mapping_manual.reset_index(drop=True, inplace=True)
    room_groups = df_room_mapping_manual.groupby('uid')

    # t-SNE
    # tsne = TSNE(n_components=2)
    # room_emb_2d = tsne.fit_transform(room_emb_manual)
    outfile = "../plots/{}/room_embedding_manual_v2.png".format(folder)
    plot_embedding_v2(room_emb_2d_manual, room_emb_2d_others, room_groups, "name", outfile)

def plot_2d_medication_top(df_medication_mapping, k):
    medication_value_counts = df_medication_mapping.majid.value_counts()
    medication_top_labels = medication_value_counts.head(k).index # top 3 frequent labels of medications
    # medication_top_labels = list(medication_top_labels)

    # extract medication mapping and embedding with specific labels
    df_medication_mapping_top = df_medication_mapping[df_medication_mapping.majid.isin(medication_top_labels)]
    medication_top_label_index = df_medication_mapping_top.index
    medication_emb_top = medication_emb[medication_top_label_index]

    df_medication_mapping_top.reset_index(drop=True, inplace=True)
    medication_groups = df_medication_mapping_top.groupby('majid')
    # t-SNE
    tsne = TSNE(n_components=2)
    medication_emb_2d = tsne.fit_transform(medication_emb_top)
    outfile = "../plots/{}/medication_embedding_top.png".format(folder)
    plot_embedding(medication_emb_2d, medication_groups, "majclass", outfile)

def plot_2d_physician_top(df_physician_mapping, k):
    physician_value_counts = df_physician_mapping.sid.value_counts()
    physician_top_labels = physician_value_counts.head(k).index # top k frequent labels of medications
    physician_top_labels = list(physician_top_labels)

    # extract physician mapping and embedding with specific labels
    df_physician_mapping_top = df_physician_mapping[df_physician_mapping.sid.isin(physician_top_labels)]
    physician_top_label_index = df_physician_mapping_top.index
    physician_emb_top = physician_emb[physician_top_label_index]

    df_physician_mapping_top.reset_index(drop=True, inplace=True)
    physician_groups = df_physician_mapping_top.groupby('sid')

    # # physician
    tsne = TSNE(n_components=2, random_state=123)
    physician_emb_2d = tsne.fit_transform(physician_emb_top)
    outfile = "../plots/{}/physician_embedding_top.png".format(folder)
    plot_embedding(physician_emb_2d, physician_groups, "name", outfile)
    return df_physician_mapping_top, physician_emb_2d

def plot_2d_room_top(df_room_mapping, k):
    room_value_counts = df_room_mapping.uid.value_counts()
    room_top_uids = room_value_counts.head(5).index # top 3 frequent uids of medications
    room_top_uids = list(room_top_uids)

    # extract room mapping and embedding with specific labels
    df_room_mapping_top = df_room_mapping[df_room_mapping.uid.isin(room_top_uids)]
    room_top_label_index = df_room_mapping_top.index
    room_emb_top = room_emb[room_top_label_index]

    df_room_mapping_top.reset_index(drop=True, inplace=True)
    room_groups = df_room_mapping_top.groupby('uid')

    # # room
    tsne = TSNE(n_components=2)
    room_emb_2d = tsne.fit_transform(room_emb_top)
    outfile = "../plots/{}/room_embedding_top.png".format(folder)
    plot_embedding(room_emb_2d, room_groups, "uid", outfile)

# aside from dataframes, return a matrix of MSW
def compute_SSW_SSB(df_mapping, label_col, dict_label_desc, item_emb):

    print("Computing SSW...")
    label_list = df_mapping[label_col].unique()
    label_name_list = [dict_label_desc[label] for label in label_list]
    n_labels = label_list.shape[0]
    SSW_array = np.zeros((n_labels))
    SSW_len = np.zeros((n_labels)).astype(int)
    SSW_n_items = np.zeros((n_labels)).astype(int)

    SS_2d_array = np.zeros((n_labels, n_labels))
    SS_len_2d_array = np.zeros((n_labels, n_labels))

    for idx_label, label in enumerate(label_list):
        SSW_temp = 0
        index_list = (df_mapping[df_mapping[label_col] == label]).index
        item_emb_one_label = item_emb[index_list]
        n_items = item_emb_one_label.shape[0]
        SSW_len[idx_label] = get_n_combination(n_items)
        SSW_n_items[idx_label] = n_items
        # Compute pairwise SSW 
        for i in range(n_items):
            for j in range(i+1, n_items):
                SSW_pair = np.sum(np.power(item_emb_one_label[i,:] - item_emb_one_label[j,:], 2))
                SSW_temp += SSW_pair
        SSW_array[idx_label] = SSW_temp

        SS_2d_array[idx_label, idx_label] = SSW_array[idx_label]
        SS_len_2d_array[idx_label, idx_label] = SSW_len[idx_label]

    print("Computing SSB...")
    SSB_list = []
    SSB_len = []
    SSB_label_pairs = []
    SSB_label_name_pairs = []
    for idx_label1, (label1, label1_name) in enumerate(zip(label_list, label_name_list)):
        index1_list = (df_mapping[df_mapping[label_col] == label1]).index
        item_emb_label1 = item_emb[index1_list]
        n_item1 = item_emb_label1.shape[0]

        for idx_label2, (label2, label2_name) in enumerate(zip(label_list[idx_label1+1 : ], label_name_list[idx_label1+1 : ])):
            index2_list = (df_mapping[df_mapping[label_col] == label2]).index
            item_emb_label2 = item_emb[index2_list]
            n_item2 = item_emb_label2.shape[0]

            SSB_label_pairs.append(tuple([label1, label2]))
            SSB_label_name_pairs.append(tuple([label1_name, label2_name]))
            SSB_len.append(n_item1 * n_item2)

            SSB_temp = 0
            for i in range(n_item1):
                for j in range(n_item2):
                    SSB_pair = np.sum(np.power(item_emb_label1[i,:] - item_emb_label2[j,:], 2))
                    SSB_temp += SSB_pair
            SSB_list.append(SSB_temp)

            SS_2d_array[idx_label1, idx_label1+1+idx_label2] = SSB_temp
            SS_len_2d_array[idx_label1, idx_label1+1+idx_label2] = n_item1 * n_item2
            SS_2d_array[idx_label1+1+idx_label2, idx_label1] = SSB_temp
            SS_len_2d_array[idx_label1+1+idx_label2, idx_label1] = n_item1 * n_item2

    SSB_array = np.array(SSB_list)
    df_SSW = pd.DataFrame({"label": label_list, "SSW": SSW_array, "n_items": SSW_n_items, "n_pairs": SSW_len})
    df_SSW.insert(loc=df_SSW.shape[1], column="MSW", value=df_SSW.SSW/df_SSW.n_pairs)
    df_SSW.insert(loc=1, column="desc", value=df_SSW["label"].map(dict_label_desc))

    df_SSB = pd.DataFrame({"label": SSB_label_pairs, "label_name": SSB_label_name_pairs, "SSB": SSB_array, "n_pairs": SSB_len})
    df_SSB.insert(loc=df_SSB.shape[1], column="MSB", value=df_SSB.SSB/df_SSB.n_pairs)

    MS_2d_array = SS_2d_array / SS_len_2d_array
    df_MS = pd.DataFrame(data=MS_2d_array, columns=label_list, index=label_list)
    index_desc_list = [dict_label_desc[l] for l in label_list]
    df_MS_desc = pd.DataFrame(data=MS_2d_array, columns=index_desc_list, index=index_desc_list)

    # return df_SSW.sort_values(by="MSW"), df_SSB.sort_values(by="MSB"), df_MS, df_MS_desc
    return df_SSW, df_SSB, df_MS, df_MS_desc

def save_dataframes():
    df_physician_SSW.to_csv("../dispersion/{}/physician_SSW.csv".format(folder), index=False)
    df_physician_SSB.to_csv("../dispersion/{}/physician_SSB.csv".format(folder), index=False)
    df_physician_MS.to_csv("../dispersion/{}/physician_MS_matrix.csv".format(folder), index=True)
    df_physician_MS_desc.to_csv("../dispersion/{}/physician_MS_desc_matrix.csv".format(folder), index=True)

    df_medication_SSW.to_csv("../dispersion/{}/medication_SSW.csv".format(folder), index=False)
    df_medication_SSB.to_csv("../dispersion/{}/medication_SSB.csv".format(folder), index=False)
    df_medication_MS.to_csv("../dispersion/{}/medication_MS_matrix.csv".format(folder), index=True)
    df_medication_MS_desc.to_csv("../dispersion/{}/medication_MS_desc_matrix.csv".format(folder), index=True)

    df_room_SSW.to_csv("../dispersion/{}/room_SSW.csv".format(folder), index=False)
    df_room_SSB.to_csv("../dispersion/{}/room_SSB.csv".format(folder), index=False)
    df_room_MS.to_csv("../dispersion/{}/room_MS_matrix.csv".format(folder), index=True)
    df_room_MS_desc.to_csv("../dispersion/{}/room_MS_desc_matrix.csv".format(folder), index=True)

    df_medication_mid_SSB.to_csv("../dispersion/{}/medication_mid_SSB.csv".format(folder), index=False)
    df_medication_mid_SSB[["label_name", "SSB"]].head(100).to_csv("../dispersion/{}/medication_mid_SSB_top100.csv".format(folder), index=False)
    df_medication_mid_SSB[["label_name", "SSB"]].tail(100).to_csv("../dispersion/{}/medication_mid_SSB_bottom100.csv".format(folder), index=False)

def get_epoch_w_min_loss(folder, network):
    filename = "../{}/loss/loss_{}.npz".format(folder, network)
    npzfile = np.load(filename)
    loss_per_timestep = npzfile["loss_per_timestep"]
    npzfile.close()

    # Find idx with smallest loss, but greater than 0.00001 (because initialization of the loss array is zero, and if training terminated before the end of epoch, thoes would have loss=0
    valid_idx = np.where(loss_per_timestep > 0.00001)[0]
    ep = valid_idx[np.argmin(loss_per_timestep[valid_idx])]

    max_ep = loss_per_timestep.shape[0]-1 # epoch starts from 0. so max ep is 1 minus the total epochs.
    patience = 10 # Used 5 epoch as the patience in training
    if ep + patience > max_ep:
        ep = max_ep
    else: # If there was an early stopping,
        ep = ep + patience
    return ep

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', '--folder', required=True, help='name of the folder')

    args = parser.parse_args()

    folder = args.folder

    top_10percent_mid = np.load("../data/top_mid.npy", allow_pickle=True)
    
    start_date = pd.Timestamp(2010,1,1)
    start_date_str = start_date.strftime('%Y-%m-%d')

    # Item embedding from DECEnt
    network= "patient_DECEnt_PF_{}".format(start_date_str)
    ep_with_min_loss = get_epoch_w_min_loss(folder, network)

    item_array, itemtype_array, DECEnt_embedding = get_embedding(folder, network, ep_with_min_loss)

    # Divide the item_array and itemtype_array into three
    medication_array, physician_array, room_array, medication_emb, physician_emb, room_emb = get_item_arrays_and_emb(item_array, itemtype_array, DECEnt_embedding)
    n_emb = medication_emb.shape[1]

    ##################
    # Modify medication_array and medication_emb with top 10%
    df_med_emb = pd.DataFrame(data=medication_emb, columns=["e{}".format(i) for i in range(n_emb)])
    df_med_emb.insert(loc=0, column="mid", value=medication_array)
    df_med_emb["mid"].isin(top_10percent_mid)

    df_top_10percent_mid_emb = df_med_emb[df_med_emb["mid"].isin(top_10percent_mid)]
    # NOTE: OVERWRITE medication array and medication emb!!!!!
    medication_array = df_top_10percent_mid_emb.mid.values
    medication_emb = df_top_10percent_mid_emb.iloc[:, 1:].values
    ##################

    # Medication mapping 
    df_medication_mapping = prepare_df_medication_mapping(medication_array)
    df_medication_mapping = df_medication_mapping[df_medication_mapping.majclass != -1]
    # 1. manual selection (majid or majclass)
    selected_medication = ["CARDIOVASCULAR AGENTS", "RENAL AND GENITOURINARY AGENTS", "EYE EAR NOSEAND THROAT PREP"]
    plot_2d_medication_manual(df_medication_mapping)
    # 2. top 3 medications
    # plot_2d_medication_top(df_medication_mapping, 3)

    # Physician mapping 
    df_physician_mapping = prepare_df_physician_mapping(physician_array)
    df_physician_mapping = df_physician_mapping[df_physician_mapping.name != -1]
    df_physician_mapping_top, physician_emb_2d = plot_2d_physician_top(df_physician_mapping, 3)

    selected_physician = ["General Internal Med", "General Pediatrics"]
    plot_2d_physician_manual(df_physician_mapping)
    plot_2d_physician_manual_v2(df_physician_mapping)

    # Room mapping
    df_room_mapping = prepare_df_room_mapping(room_array)
    selected_room = ["7JPP PEDS ICU", "2RCARVER WEST 2"]
    plot_2d_room_manual(df_room_mapping)
    plot_2d_room_manual_v2(df_room_mapping)
    # plot_2d_room_top(df_room_mapping, 5)

    ##################################################################################
    # Compute measure of dispersion 
    # 1. Mean pairwise distance of doctors wihtin group. Sum up the pairwise distances. divide it to number of pairs. (normalization)
    # 2. Compute the distance of the group from the rest of the points. distance between the two means?

    # NOTE: Uncomment the following for dispersion matrices
    dict_physician_label_desc = dict(df_physician_mapping[["sid", "name"]].values)
    # dict_medication_label_desc = dict(df_medication_mapping[["majid", "majclass"]].values)
    dict_medication_label_desc = dict(df_medication_mapping[["minid", "minclass"]].values)
    dict_room_label_desc = dict(df_room_mapping[["uid", "name"]].values)

    df_physician_SSW, df_physician_SSB, df_physician_MS, df_physician_MS_desc = compute_SSW_SSB(df_physician_mapping, "sid", dict_physician_label_desc, physician_emb)
    # df_medication_SSW, df_medication_SSB, df_medication_MS, df_medication_MS_desc = compute_SSW_SSB(df_medication_mapping, "majid", dict_medication_label_desc, medication_emb)
    df_medication_SSW, df_medication_SSB, df_medication_MS, df_medication_MS_desc = compute_SSW_SSB(df_medication_mapping, "minid", dict_medication_label_desc, medication_emb)
    df_room_SSW, df_room_SSB, df_room_MS, df_room_MS_desc = compute_SSW_SSB(df_room_mapping, "uid", dict_room_label_desc, room_emb)

    # Compute medication dispersion matrix on the lowest level
    print("Computing medication dispersion matrix on the lowest level...")
    query1 = 'SELECT * from medications'
    column_names, result = query_from_db(query1)
    df_medications = pd.DataFrame(result, columns=column_names)
    medication_mid_name_mapping = dict(zip(df_medications.mid.values, df_medications.name.values))

    df_medication_mid_SSW, df_medication_mid_SSB, df_medication_mid_MS, df_medication_mid_MS_desc = compute_SSW_SSB(df_medication_mapping, "mid", medication_mid_name_mapping, medication_emb)
    df_medication_mid_SSB.sort_values(by="SSB", inplace=True)

    save_dataframes()
