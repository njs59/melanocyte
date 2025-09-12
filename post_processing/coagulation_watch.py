# Write code to watch each coagulation events
# Go a few frames before and after each coagulation event and watch
# just the clusters involved in the event

import numpy as np
import pandas as pd
from ast import literal_eval

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

from pylab import *
from scipy.ndimage import *


basedir = '/Users/Nathan/Documents/Oxford/DPhil/'
exp_type = 'In_vitro_homogeneous_data/'
exp_date = '2017-02-03'
time_array = range(1,98)
num_times = len(time_array)
# Rename single digit values with 0 eg 1 to 01 for consistency
time_list = [str(x).zfill(2) for x in time_array]
well_loc = 's10'

cols = ["Tag number", "Cluster size", "Cluster Centre x", "Cluster Centre y", 
           "Event", "Clusters in event", "Timestep", "Date", "Well ID"]


coagulation_df = pd.DataFrame(columns = cols)

for i in range(97, 50, -1) :
    print('i is', i)
    time_i = str(i).zfill(2)
    df_step_csv_name_list = basedir, exp_type, 'post_processing_output/', exp_date, '/', well_loc, 't', time_i, 'c2_post_processing', '.csv'
    df_step_csv_name_list_2  =''.join(df_step_csv_name_list)
    df_step = pd.read_csv(df_step_csv_name_list_2)

    if (df_step['Event'] == 'Coagulation').any():

        rows_to_add = df_step.loc[df_step['Event'].isin(['Coagulation'])]

        coagulation_df = pd.concat([coagulation_df, rows_to_add])

        
print('Yay')

for j in range(coagulation_df.shape[0]):

    coag_event = coagulation_df.iloc[j]

    time_of_event = coag_event['Timestep']
    clusters_in_event = literal_eval(coag_event['Clusters in event'])

    frames_before = time_of_event - 10
    frames_after = time_of_event + 5

    frames_before = max(1,frames_before)
    frames_after = min(97, frames_after)

    for k in range(frames_before, frames_after+1):

        time_k = str(k).zfill(2)

        # Find index for tag
        df_step_csv_name_list = basedir, exp_type, 'post_processing_output/', exp_date, '/', well_loc, 't', time_k, 'c2_post_processing', '.csv'
        df_step_csv_name_list_2  =''.join(df_step_csv_name_list)
        df_step = pd.read_csv(df_step_csv_name_list_2)

        df_clus_in_event = df_step.loc[df_step['Tag number'].isin(clusters_in_event)]
        index_to_keep = df_clus_in_event.index
        index_to_keep += 1


        # Read in indexed array at current time
        area_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, 't', time_k, 'c2', '_indexed', '.csv'
        area_csv_name_list_2  =''.join(area_csv_name_list)
        df_slice = pd.read_csv(area_csv_name_list_2, header=None)
        current_array = df_slice.to_numpy()

        clusters_in_event_arr = np.zeros([current_array.shape[0], current_array.shape[1]])
        for k in range(len(index_to_keep)):
            loc_keep = np.where(current_array == index_to_keep[k])
            for l in range(len(loc_keep[0])):
                clusters_in_event_arr[loc_keep[0][l], loc_keep[1][l]] = 1


        # Plot a heatmap of the array on a log scale and savre the image for use in a gif
        my_cmap = mpl.colormaps['spring']
        my_cmap.set_under('w')
        # plt.imshow(clusters_in_event_arr, cmap=my_cmap, norm = LogNorm(vmin=150, vmax=25000))
        plt.imshow(clusters_in_event_arr)
        # plt.imshow(area_slice, cmap=my_cmap, norm=matplotlib.colors.LogNorm(vmin=100,vmax=25000))
        plt.axis([0, current_array.shape[1], 0, current_array.shape[0]])
        plt.colorbar()
        plt.show()
        # plt.savefig(f'{basedir}images/cluster_sizes_log/frame-{i:03d}.png', bbox_inches='tight', dpi=300)
        plt.clf()