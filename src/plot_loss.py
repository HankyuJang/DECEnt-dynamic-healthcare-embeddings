
"""
Author: -
Email: -
Last Modified: Oct, 2021

Plot loss function
"""
import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', '--folder', required=True, help='name of the folder')
    parser.add_argument('-network', '--network', required=True, help='name of the network')
    parser.add_argument('-ep', '--ep', type=int, default=99, help='max epoch')

    args = parser.parse_args()

    folder = args.folder
    network = args.network
    ep = args.ep

    # network = "patient_DECEnt_no_features_lap_2010-01-01"
    # ep = 380
    # network = "patient_DECEnt_PF_2010-01-01"
    npzfile = np.load("../{}/loss/loss_{}.npz".format(folder, network))
    loss_per_timestep = npzfile["loss_per_timestep"]
    prediction_loss_per_timestep = npzfile["prediction_loss_per_timestep"]
    user_update_loss_per_timestep = npzfile["user_update_loss_per_timestep"]
    item_update_loss_per_timestep = npzfile["item_update_loss_per_timestep"]
    if folder in ["DECEnt_3M_auto_v1", "DECEnt_3M_auto_v3"]:
        D_loss_per_timestep = npzfile["D_loss_per_timestep"]
        M_loss_per_timestep = npzfile["M_loss_per_timestep"]
        R_loss_per_timestep = npzfile["R_loss_per_timestep"]
    npzfile.close()

    # -----------------------------------------
    if folder in ["DECEnt_3M_auto_v1", "DECEnt_3M_auto_v3"]:
        plt.title("Loss per epoch (Total, D, M, R)")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        p0, = plt.plot(loss_per_timestep[:ep], label="Total loss", color="C0")
        p1, = plt.plot(D_loss_per_timestep[:ep], label="D loss", color="C1")
        p2, = plt.plot(M_loss_per_timestep[:ep], label="M loss", color="C2")
        p3, = plt.plot(R_loss_per_timestep[:ep], label="R loss", color="C3")
        plt.legend(handles=[p0, p2, p1, p3], loc='best')
        plt.savefig("../plots/loss/{}/loss_{}_TDMR.png".format(folder, network))
        plt.close()

        plt.title("log(Loss) per epoch (Total, D, M, R)")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        p0, = plt.plot(np.log(loss_per_timestep[:ep]+1), label="Total loss", color="C0")
        p1, = plt.plot(np.log(D_loss_per_timestep[:ep]+1), label="D loss", color="C1")
        p2, = plt.plot(np.log(M_loss_per_timestep[:ep]+1), label="M loss", color="C2")
        p3, = plt.plot(np.log(R_loss_per_timestep[:ep]+1), label="R loss", color="C3")
        plt.legend(handles=[p0, p2, p1, p3], loc='best')
        plt.savefig("../plots/loss/{}/logloss_{}_TDMR.png".format(folder, network))
        plt.close()
        # -----------------------------------------
        # -----------------------------------------
        # All together
        plt.title("Loss per epoch")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        p0, = plt.plot(loss_per_timestep[:ep], label="Total loss", color="C0")
        p1, = plt.plot(D_loss_per_timestep[:ep], label="D loss", color="C1")
        p2, = plt.plot(M_loss_per_timestep[:ep], label="M loss", color="C2")
        p3, = plt.plot(R_loss_per_timestep[:ep], label="R loss", color="C3")
        p4, = plt.plot(prediction_loss_per_timestep[:ep], label="Prediction loss", color="C4")
        p5, = plt.plot(user_update_loss_per_timestep[:ep], label="User update loss", color="C5")
        p6, = plt.plot(item_update_loss_per_timestep[:ep], label="Item update loss", color="C6")
        plt.legend(handles=[p0, p2, p1, p3, p5, p6, p4], loc='best')
        plt.savefig("../plots/loss/{}/loss_{}.png".format(folder, network))
        plt.close()

        plt.title("log(Loss) per epoch")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        p0, = plt.plot(np.log(loss_per_timestep[:ep]+1), label="Total loss", color="C0")
        p1, = plt.plot(np.log(D_loss_per_timestep[:ep]+1), label="D loss", color="C1")
        p2, = plt.plot(np.log(M_loss_per_timestep[:ep]+1), label="M loss", color="C2")
        p3, = plt.plot(np.log(R_loss_per_timestep[:ep]+1), label="R loss", color="C3")
        p4, = plt.plot(np.log(prediction_loss_per_timestep[:ep]+1), label="Prediction loss", color="C4")
        p5, = plt.plot(np.log(user_update_loss_per_timestep[:ep]+1), label="User update loss", color="C5")
        p6, = plt.plot(np.log(item_update_loss_per_timestep[:ep]+1), label="Item update loss", color="C6")
        plt.legend(handles=[p0, p2, p1, p3, p5, p6, p4], loc='best')
        plt.savefig("../plots/loss/{}/logloss_{}.png".format(folder, network))
        plt.close()

    # -----------------------------------------
    plt.title("Loss per epoch (Prediction, Uupdate, Iupdate)")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    p4, = plt.plot(prediction_loss_per_timestep[:ep], label="Prediction loss", color="C4")
    p5, = plt.plot(user_update_loss_per_timestep[:ep], label="User update loss", color="C5")
    p6, = plt.plot(item_update_loss_per_timestep[:ep], label="Item update loss", color="C6")
    plt.legend(handles=[p5, p6, p4], loc='best')
    plt.savefig("../plots/loss/{}/loss_{}_PUI.png".format(folder, network))
    plt.close()

    # Here, add 1 in care there is early stopping
    plt.title("log(Loss) per epoch (Prediction, Uupdate, Iupdate)")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    p4, = plt.plot(np.log(prediction_loss_per_timestep[:ep]+1), label="Prediction loss", color="C4")
    p5, = plt.plot(np.log(user_update_loss_per_timestep[:ep]+1), label="User update loss", color="C5")
    p6, = plt.plot(np.log(item_update_loss_per_timestep[:ep]+1), label="Item update loss", color="C6")
    plt.legend(handles=[p5, p6, p4], loc='best')
    plt.savefig("../plots/loss/{}/logloss_{}_PUI.png".format(folder, network))
    plt.close()
    # -----------------------------------------

