import math
import numpy as np
import pandas as pd
from ast import literal_eval

import matplotlib.pyplot as plt

import glob
from PIL import Image

from matplotlib.colors import ListedColormap
import matplotlib

from cycler import cycler
from pylab import *
from scipy.ndimage import *

import post_pro_operators as post_oper


def generate_colormap(N):
    arr = np.arange(N)/N
    N_up = int(math.ceil(N/7)*7)
    arr.resize(N_up)
    arr = arr.reshape(7,N_up//7).T.reshape(-1)
    ret = matplotlib.cm.hsv(arr)
    n = ret[:,3].size
    a = n//2
    b = n-a
    for i in range(3):
        ret[0:n//2,i] *= np.arange(0.2,1,0.8/a)
    ret[n//2:,3] *= np.arange(1,0.1,-0.9/b)
#     print(ret)
    return ret



basedir = '/Users/Nathan/Documents/Oxford/DPhil/'
exp_type = 'In_vitro_homogeneous_data/'
# experiment = 'RAW_data/2017-02-03_sphere_timelapse/'
experiment = 'RAW_data/2017-02-13_sphere_timelapse_2/'
exp_date = '2017-02-03'
# exp_date = '2017-02-13'
# exp_date = '2017-03-10'
# exp_date = '2017-03-16'
folder = 'RAW/Timelapse/sphere_timelapse_useful_wells/'
folder_3 = 'sphere_timelapse/'
fileID = '.tif'

time_array = range(1,98)
# time_array = range(1,95)
# time_array = range(1,143)

last_time = 97
# last_time = 94
# last_time = 142
# Rename single digit values with 0 eg 1 to 01 for consistency
time_list = [str(x).zfill(2) for x in time_array]
# time_list = [str(x).zfill(3) for x in time_array]
# time_list= ['21','22','23','24','25','26','27','28','29','30']

# 2017-02-13 sphere timelapse 2_s13t01c2_ORG

# well_loc = 's10'
well_loc = 's11'
# well_loc = 's13'
# well_loc = 's27'

# well_loc = 's073'

# well_loc = 's04'

cols = ["Tag number", "Cluster size", "Cluster Centre x", "Cluster Centre y", 
           "Event", "Clusters in event", "Timestep", "Date", "Well ID"]

df_end_now_csv_name_list = basedir, exp_type, 'post_processing_output/' , exp_date, '/', well_loc, 't', str(last_time), 'c2_post_processing', '.csv'
df_end_now_csv_name_list_2  =''.join(df_end_now_csv_name_list)
df_end_now = pd.read_csv(df_end_now_csv_name_list_2)
centres_end_2D = df_end_now[['Cluster Centre x', 'Cluster Centre y']]
cluster_tags = df_end_now["Tag number"].to_numpy().astype(int)

index_shape_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, 't', str(last_time), 'c2', '_indexed', '.csv'
index_shape_csv_name_list_2  =''.join(index_shape_csv_name_list)
df_shape_slice = pd.read_csv(index_shape_csv_name_list_2, header=None)
shape_array = df_shape_slice.to_numpy()

lineage_old_arr = np.zeros(shape_array.shape)

# centres_end_store = []
centres_start_store = []

# cluster_lineage = [56]
# cluster_lineage = [125]
# cluster_lineage = [103]

cluster_lineage_store = []
all_cluster_lineage = []

for h in range(len(cluster_tags)):
    if h == 192:
        print('Pause')
    cluster_lineage = [cluster_tags[h]]
    for i in range(last_time, 50, -1) :
        if i == 61:
            print('Pause')
        time_i = str(i).zfill(2)
        # time_i = str(i).zfill(3)
        df_step_csv_name_list = basedir, exp_type, 'post_processing_output/' , exp_date, '/', well_loc, 't', time_i, 'c2_post_processing', '.csv'
        df_step_csv_name_list_2  =''.join(df_step_csv_name_list)
        df_step = pd.read_csv(df_step_csv_name_list_2)
        # cluster_2D_areas = df_clus_areas.to_numpy()
        for j in range(len(cluster_lineage)):
            row_interest = df_step.loc[df_step['Tag number'] == cluster_lineage[j]]
            if (row_interest['Event'] == 'Coagulation').any():
            # if (row_interest.iloc[:, [4]] == 'Coagulation').any() == True:
                locs = row_interest.iloc[:, [5]]
                str_locs = locs.iloc[0]['Clusters in event']
                list_locs = literal_eval(str_locs)
                arr_locs = np.array(list_locs)
                res = [item for item in arr_locs if item not in cluster_lineage]
                cluster_lineage = np.append(cluster_lineage, res)

    if h == 0:
        all_cluster_lineage = cluster_lineage
    else:
        all_cluster_lineage = np.append(all_cluster_lineage,cluster_lineage)

    print(cluster_lineage)
    if h == 0:
        cluster_lineage_store = cluster_lineage
    elif h == 1:
        if len(cluster_lineage) <= shape(cluster_lineage_store)[0]:
            difference = shape(cluster_lineage_store)[0] - len(cluster_lineage)
            cluster_lineage = np.pad(cluster_lineage, (0, difference), 'constant')
        else:
            difference = len(cluster_lineage) - shape(cluster_lineage_store)[0]
            zeros_to_add = np.zeros((shape(cluster_lineage_store)[0],difference))
            cluster_lineage_store = np.hstack((cluster_lineage_store,zeros_to_add))

        cluster_lineage_store = np.vstack((cluster_lineage_store, cluster_lineage))        
    else:
        if len(cluster_lineage) <= shape(cluster_lineage_store)[1]:
            difference = shape(cluster_lineage_store)[1] - len(cluster_lineage)
            cluster_lineage = np.pad(cluster_lineage, (0, difference), 'constant')
        else:
            difference = len(cluster_lineage) - shape(cluster_lineage_store)[1]
            zeros_to_add = np.zeros((shape(cluster_lineage_store)[0],difference))
            cluster_lineage_store = np.hstack((cluster_lineage_store,zeros_to_add))

        cluster_lineage_store = np.vstack((cluster_lineage_store, cluster_lineage))

for k in range(51, last_time+1, 1) :
    time_k = str(k).zfill(2)

    centres_current_store = []

    # Find locations of clusters at current time
    df_step_csv_name_list = basedir, exp_type, 'post_processing_output/', exp_date, '/', well_loc, 't', time_k, 'c2_post_processing', '.csv'
    df_step_csv_name_list_2  =''.join(df_step_csv_name_list)
    df_step_interest = pd.read_csv(df_step_csv_name_list_2)

    rows_of_interest = pd.DataFrame(columns=cols)
    for l in range(len(all_cluster_lineage)):
        new_row_of_interest = df_step_interest.loc[df_step['Tag number'] == all_cluster_lineage[l]]

        # to append df2 at the end of df1 dataframe
        rows_of_interest = pd.concat([rows_of_interest, new_row_of_interest])

    


    # Store centres in array
    df1 = rows_of_interest[['Cluster Centre x', 'Cluster Centre y']]
    centres_2D_lineage = df1.to_numpy()
    centres_current_store = np.append(centres_current_store,centres_2D_lineage)
    centres_current_2D = np.reshape(centres_current_store, (-1, 2))

    print('INT1')

    # Read in array for given timestep

    # Locate indexes of clusters, print
    index_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, 't', time_k, 'c2', '_indexed', '.csv'
    index_csv_name_list_2  =''.join(index_csv_name_list)
    df_slice = pd.read_csv(index_csv_name_list_2, header=None)
    current_array = df_slice.to_numpy()

    lineage_tags = []
    lineage_tag_arr = np.zeros(current_array.shape)
    # Loop over all Clusters at timepoint
    for c in range(df_step_interest.shape[0]):
        if np.any(rows_of_interest.index == c) == True:
            lineage_tag = np.where(cluster_lineage_store == df_step_interest["Tag number"][c])[0] + 1
        else:
            lineage_tag = 0

        lineage_tags = np.append(lineage_tags, lineage_tag)
        cluster_locs_this_tag = np.where(current_array==c+1)
        if lineage_tag > 0:
            for d in range(np.shape(cluster_locs_this_tag)[1]):
                x_coord = cluster_locs_this_tag[0][d]
                y_coord = cluster_locs_this_tag[1][d]
                lineage_tag_arr[x_coord,y_coord] = lineage_tag


    print('INT3')
    final_index_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, 't', str(last_time), 'c2', '_indexed', '.csv'
    final_index_csv_name_list_2  =''.join(final_index_csv_name_list)
    final_df_slice = pd.read_csv(final_index_csv_name_list_2, header=None)
    final_current_array = final_df_slice.to_numpy()

    max_value = final_current_array.max()
    plt.figure()

    # For colour maps to line-up maximum values of the arrays must be the same 
    # So label a single pixel with max value
    if lineage_tag_arr.max() != max_value:
        lineage_tag_arr[-1][-1] = max_value
    my_cmap = mpl.colormaps['tab20']
    my_cmap.set_under('w')
    plt.imshow(lineage_tag_arr, vmin=1, cmap=my_cmap)
        # Loop over data dimensions and create text annotations.
    for j in range(centres_current_2D.shape[0]):
            centre_coord = centres_current_2D[j]
            x = int(centre_coord[0])
            y = int(centre_coord[1])
            index_print = int(lineage_tag_arr[x-1, y-1])
            if index_print == 0:
                near_clus, clus_distances = post_oper.nearby_clusters(x-1, y-1, 50, lineage_tag_arr)
                if len(clus_distances) == 0:
                    continue
                else:
                    index_dist_min = np.argmin(clus_distances)  
                    index_print = int(near_clus[index_dist_min])
                    plt.text(y-1, x-1,
                                    index_print,
                            ha="center", va="center", color="k")
            else:
                plt.text(y-1, x-1,
                                    index_print,
                            ha="center", va="center", color="k")

    e = k - 50
    plt.axis([0, current_array.shape[1], 0, current_array.shape[0]])
    plt.savefig(f'{basedir}clusters/data_analysis_3D/post-processing/plot_to_gif/frame-{e:03d}.png', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.clf()


###   -----------------  Gif code  ----------------- ###

# create an empty list called images
images = []

# get the current time to use in the filename
timestr = time.strftime("%Y%m%d-%H%M%S")

# get all the images in the 'images for gif' folder
for filename in sorted(glob.glob(basedir + 'clusters/data_analysis_3D/post-processing/plot_to_gif/frame-*.png')): # loop through all png files in the folder
    im = Image.open(filename) # open the image
    images.append(im) # add the image to the list

# calculate the frame number of the last frame (ie the number of images)
last_frame = (len(images)) 

# create 10 extra copies of the last frame (to make the gif spend longer on the most recent data)
for x in range(0, 9):
    im = images[last_frame-1]
    images.append(im)

# save as a gif   
images[0].save(basedir + 'clusters/data_analysis_3D/post-processing/gif_plots/' + timestr + '.gif',
            save_all=True, append_images=images[1:], optimize=False, duration=300, loop=0)



