"""
Author: 
Email: 
Last Modified: 

Plot dispersion.
NOTE: the csv files are generated in the excel file.
"""

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_dispersion(title, x, y, outfile):

    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Distance")
    plt.bar(x, y)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

if __name__ == "__main__":
    #----------------------------------------
    df_physician_dispersion = pd.read_csv("../table/physician_dispersion_matrix_distance_specialty.csv")
    distance_array = df_physician_dispersion["Distance"]
    specialty_array = df_physician_dispersion["Specialty"]
    title="Physician dispersion (partial)"
    outfile="../plots/dispersion/physician_dispersion_bar_plot.png"

    plot_dispersion(title, specialty_array, distance_array, outfile)

    #----------------------------------------
    df_room_dispersion = pd.read_csv("../table/room_dispersion_matrix_distance_unit.csv")
    distance_array = df_room_dispersion["Distance"]
    specialty_array = df_room_dispersion["Unit"]
    title="Room dispersion (partial)"
    outfile="../plots/dispersion/room_dispersion_bar_plot.png"

    plot_dispersion(title, specialty_array, distance_array, outfile)


