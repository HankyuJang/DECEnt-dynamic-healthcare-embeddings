
"""
Author: -
Email: -
Last Modified: Oct 2021

Plot ROC curves

the length of tpr and fpr differ per experiment. 

Fix the fpr array with size 100, and interp the tpr with the same shape.
Reference:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

TODO: plot MLP once baseline training is complete.
"""

import numpy as np
import pandas as pd
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import auc

if __name__ == "__main__":

    clf_name_list = ["MLP", "rf", "logit"]
    clf_name_to_show_list = ["MLP", "RF", "LR"]
    # clf_name="MLP"
    label_list = ["MICU_transfer", "CDI"]

    for label in label_list:
        for clf_name, clf_name_to_show in zip(clf_name_list, clf_name_to_show_list):
            mean_fpr = np.linspace(0, 1, 100)
            # method_idx = 0
            label_list = ["DECEnt+", "DECEnt", "JODIE", "DeepWalk", "node2vec", "Domain", "RNN", "LSTM"]
            color_list = ["tab:red", "tab:orange", "tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:gray", "black"]
            marker_list = ["o", "^", "1", "+", "x", "2", "3", "4"]

            folder_list = ["DECEnt_3M_auto_v3", "DECEnt_3M_auto_v1", "jodie", \
                            "deepwalk", "node2vec_BFS", "domain_specific", "RNN", "LSTM"]
            network_list = ["patient_DECEnt_PF_2010-01-01", "patient_DECEnt_PF_2010-01-01", "patient_jodie_2010-01-01"] + \
                            ["patient_DECEnt_PF_2010-01-01"]*5
            # n_methods = len(network_list)

            plt.figure(figsize=(8,8))
            for method_idx, (folder, network) in enumerate(zip(folder_list, network_list)):
                tprs = []
                aucs = []
                if folder in ["RNN", "LSTM"]:
                    filename = "../result/{}/{}/{}_tpr_fpr_{}.pickle".format(folder, network, label, folder)
                else:
                    filename = "../result/{}/{}/{}_tpr_fpr_{}.pickle".format(folder, network, label, clf_name)
                inputfile = open(filename, "rb")
                network_eval_results = pickle.load(inputfile)
                inputfile.close()
                
                test_fpr = network_eval_results["test_fpr"]
                test_tpr = network_eval_results["test_tpr"]
                n_test_results = len(test_fpr)

                for i in range(n_test_results):
                    interp_tpr = np.interp(mean_fpr, network_eval_results["test_fpr"][2], network_eval_results["test_tpr"][2])
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                
                plt.plot(mean_fpr, mean_tpr,
                        label="{} (AUC: {:.3f})".format(label_list[method_idx], mean_auc),
                        color=color_list[method_idx],
                        marker=marker_list[method_idx])
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('FPR', fontsize=20)
            plt.ylabel('TPR', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.title('{} ROC ({})'.format(label, clf_name_to_show), fontsize=25)
            plt.legend(loc="lower right", prop={'size':18})
            plt.savefig("../plots/roc_curve/{}_{}.png".format(label, clf_name_to_show), dpi = 300)
            plt.close()
                
