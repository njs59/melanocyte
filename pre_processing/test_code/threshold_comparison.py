import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


import matplotlib.animation as animation
from IPython import display

import glob
import contextlib
import os
from PIL import Image


from pylab import *
from scipy.ndimage import *

import read_tif_file_operator as tif
import pre_pro_operators as pre_oper

t_before = time.time()


########### Input parameters ##################

basedir = '/Users/Nathan/Documents/Oxford/DPhil/In_vitro_homogeneous_data/RAW_data/'
basedir_2 = '/Users/Nathan/Documents/Oxford/DPhil/'
experiment = '2017-02-03_sphere_timelapse/'
exp_date = '2017-02-03_'
folder = 'RAW/Timelapse/sphere_timelapse_useful_wells/'
folder_3 = 'sphere_timelapse/'
fileID = '.tif'

time_array = range(67,68)

# Rename single digit values with 0 eg 1 to 01 for consistency
time_list = [str(x).zfill(2) for x in time_array]
# time_list= ['21','22','23','24','25','26','27','28','29','30']

well_loc = 's09'
threshold_pixel = [250, 297, 300, 310, 320, 350]
thresholds = 1
min_clus_size = 150
use_existing_file = False

#################################################


# Function to update indexed array to array displaying areas
def update_arr(arr):
    global area_new, index_keep
    index = np.where(index_keep == arr)
    if len(index[0]) != 0:        
        i = area_new[index[0]][0]
    else:
        i = 0
    return i


################### Code from tif file to txt file ###########################
    # Convert tif file to 3D array with values between 0 and 1 (1 is maximum intensity point)
    # ? Problem with thresholding, need constant value not constant proportion
    # Might be ok as we're doing it over whole 3D array so probably ok


for i in range(len(threshold_pixel)):
    # time_step = 67

    raw_arr_3D = tif.tif_to_arr(basedir, experiment, folder, well_loc, str(time_array[0]), fileID)
    print('Shape:', raw_arr_3D.shape)
    if i == 0:
        plt.imshow(raw_arr_3D, cmap='gray')
        plt.axis('off')
        plt.show()
    # Threshold 3D array to boolean array
    tf_bool_3D = pre_oper.threshold_arr_supervised(raw_arr_3D, threshold_pixel[i], 2)

        # print(tf_bool_3D)
    print(tf_bool_3D.shape)




    ######### Adapt array ################
    t_mid = time.time()

    cluster_areas = np.array([])
    fig_1 = plt.figure()
    num_clusters = []


    t_step_before = time.time()



    current_array_holes = tf_bool_3D

    current_array = binary_fill_holes(current_array_holes).astype(int)

    label_arr, num_clus = label(current_array)

    # plt.imshow(label_arr, interpolation=None)
    # plt.show()

    area_list = sum(current_array, label_arr, index=arange(label_arr.max() + 1))

    area_arr = label_arr

    global area_new, index_keep
    area_new, index_keep = pre_oper.remove_fragments(area_list, num_clus, min_clus_size)

    num_clusters = np.append(num_clusters,len(area_new))

    applyall = np.vectorize(update_arr)
    area_slice = applyall(area_arr)
    my_cmap = mpl.colormaps['spring']
    my_cmap.set_under('k')
    plt.imshow(area_slice, cmap=my_cmap, vmin=1)
    # plt.imshow(area_slice, cmap=my_cmap, norm=matplotlib.colors.LogNorm(vmin=100,vmax=25000))
    plt.axis([0, area_slice.shape[1], 0, area_slice.shape[0]])
    plt.colorbar()
    plt.savefig(f'{basedir_2}clusters/data_analysis/pre-processing/test_code/images/threshold_PRO_t_67_pixel-' + str(threshold_pixel[i]) + '.png', bbox_inches='tight', dpi=300)
    plt.clf()

    
    # plt.show()
    # animation_1 = animation.FuncAnimation(fig_1, pre_oper.update_heat_map, len(time_array), interval=500, fargs=(area_slice) )



    ##### Re-binarize array
    slice_binary = np.where(area_slice>0)

    output_label_arr, nc = label(slice_binary)


 
    # df_area = pd.DataFrame(area_slice)
    # area_csv_name_list = basedir, 'csv_folder/', exp_date, 'sphere_timelapse_', well_loc, 't', time_list[i], 'c2', '_area', '.csv'
    # area_csv_name_list_2  =''.join(area_csv_name_list)
    # df_area.to_csv(area_csv_name_list_2, index=False, header=False)

    # df_index = pd.DataFrame(area_slice)
    # area_csv_name_list = basedir, 'csv_folder/', exp_date, 'sphere_timelapse_', well_loc, 't', time_list[i], 'c2', '_indexed', '.csv'
    # area_csv_name_list_2  =''.join(area_csv_name_list)
    # df_index.to_csv(area_csv_name_list_2, index=False, header=False)

    # cluster_areas = pre_oper.save_clus_areas(i, area_new, cluster_areas)


    # t_step_after = time.time()

    # t_step = t_step_after - t_step_before

    # print('Time for step', i, t_step)


print('Number of clusters:', num_clusters)



    