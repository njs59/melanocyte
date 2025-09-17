import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import re
from skimage import filters

import os


# Use existing imports from user's code
import read_tif_file_operator as tif  # Custom module
import pre_pro_operators as pre_oper  # Custom module

# Parameters
basedir = '/Users/Nathan/Documents/Oxford/DPhil/melanocyte/'
saving_loc = 'specialised_pipeline/pre_processing_output/VID289/'
data_folder = 'data/Still_Images_with_BF_for_Nathan/'
fileID = '.tif'
time = 0
min_clus_size = 150


base_str = "VID289"

# experiment_id = "B4"  # Change this as needed
# experiment_ids = ["A5","B4","D5","E2"]  # Change this as needed
experiment_ids = ["A1","A3","A5","B2","B4","B6","D1","D3","D5","E2","E4","E6"]  # Change this as needed


for experiment_id in experiment_ids:
    print('Running on file: ', experiment_id)
    
    filenames = pre_oper.generate_filenames(experiment_id=experiment_id, base="VID289",
                       start_day = 0, start_hour = 6, start_minute = 0, 
                       end_day = 5, end_hour = 0, end_minute = 0,
                       lowest_day = 0, lowest_hour = 0, lowest_minute = 0,
                       highest_day = 5, highest_hour = 21, highest_minute = 0, 
                       gap_days = 1, gap_hours = 3, gap_minutes = 15)
    for name in filenames:
        print(name)

                # Save area array to csv file

        # Construct full file paths
        area_csv_path = os.path.join(basedir, saving_loc, name + '_area.csv')
        index_csv_path = os.path.join(basedir, saving_loc, name + '_indexed.csv')

        # Read the CSV files into DataFrames
        df_area = pd.read_csv(area_csv_path, header=None)
        df_index = pd.read_csv(index_csv_path, header=None)

        array_area = df_area.to_numpy()
        array_index = df_index.to_numpy()

        plt.imshow(array_area)
        plt.show()

        plt.imshow(array_index)
        plt.show()