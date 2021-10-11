'''
Author: -
Email: -
Last Modified: Oct, 2021

This script generates result tables 

'''

import numpy as np
import pandas as pd

if __name__ == "__main__":

    # network_list = ["patient_hidep_PF_2010-01-01", "patient_jodie_PF_2010-01-01", "patient_jodie_2010-01-01"]
    label_list = ["CDI", "mortality", "severity", "MICU_transfer"]
    clf_name_list = ["logit", "rf", "MLP"]

    # index_list = ["Domain", "JODIE", "JODIE_pf", "DECEnt_3M_auto_v1", "DECEnt_3M_auto_v1_LAPx", "DECEnt_3M_auto_v2", "DECEnt_3M_auto_v2_LAPx", "DECEnt_3M_auto_v3", "DECEnt_3M_auto_v3_LAPx", "DECEnt_3M_auto_v4", "DECEnt_3M_auto_v4_LAPx"]
    index_list = ["Domain", "deepwalk", "node2vec_BFS", "node2vec_DFS", \
                    "CTDNE", "JODIE", "DECEnt_3M_auto_v1", "DECEnt_3M_auto_v3"]

    for label in label_list:
        for clf_name in clf_name_list:
            df_domain = pd.read_csv("../result/{}/{}/{}_{}.csv".format("domain_specific", "patient_DECEnt_PF_2010-01-01", label, clf_name), index_col=0)
            df_deepwalk = pd.read_csv("../result/{}/{}/{}_{}.csv".format("deepwalk", "patient_DECEnt_PF_2010-01-01", label, clf_name), index_col=0)
            df_node2vec_BFS = pd.read_csv("../result/{}/{}/{}_{}.csv".format("node2vec_BFS", "patient_DECEnt_PF_2010-01-01", label, clf_name), index_col=0)
            df_node2vec_DFS = pd.read_csv("../result/{}/{}/{}_{}.csv".format("node2vec_DFS", "patient_DECEnt_PF_2010-01-01", label, clf_name), index_col=0)

            df_CTDNE = pd.read_csv("../result/{}/{}/{}_{}.csv".format("CTDNE", "patient_DECEnt_PF_2010-01-01", label, clf_name), index_col=0)
            df_jodie = pd.read_csv("../result/{}/{}/{}_{}.csv".format("jodie", "patient_jodie_2010-01-01", label, clf_name), index_col=0)
            df_DECEnt_3M_auto_v1 = pd.read_csv("../result/{}/{}/{}_{}.csv".format("DECEnt_3M_auto_v1", "patient_DECEnt_PF_2010-01-01", label, clf_name), index_col=0)
            df_DECEnt_3M_auto_v3 = pd.read_csv("../result/{}/{}/{}_{}.csv".format("DECEnt_3M_auto_v3", "patient_DECEnt_PF_2010-01-01", label, clf_name), index_col=0)

            df_results_over_methods = pd.concat([df_domain, df_deepwalk, df_node2vec_BFS, df_node2vec_DFS, \
                                                df_CTDNE, df_jodie, df_DECEnt_3M_auto_v1, df_DECEnt_3M_auto_v3], ignore_index=True)
            df_results_over_methods.insert(loc=0, column="index", value=index_list)
            df_results_over_methods.set_index(keys="index", inplace=True)
            df_results_over_methods.to_csv("../result/summary/{}_{}.csv".format(label, clf_name))

    index_list_others = ["RNN", "LSTM"]
    for label in label_list:
        df_RNN = pd.read_csv("../result/{}/{}/{}_{}.csv".format("RNN", "patient_DECEnt_PF_2010-01-01", label, "RNN"), index_col=0)
        df_LSTM = pd.read_csv("../result/{}/{}/{}_{}.csv".format("LSTM", "patient_DECEnt_PF_2010-01-01", label, "LSTM"), index_col=0)

        df_results_over_methods = pd.concat([df_RNN, df_LSTM], ignore_index=True)
        df_results_over_methods.insert(loc=0, column="index", value=index_list_others)
        df_results_over_methods.set_index(keys="index", inplace=True)
        df_results_over_methods.to_csv("../result/summary/{}_RNN_LSTM.csv".format(label))
