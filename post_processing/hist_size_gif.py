import math
import numpy as np
import pandas as pd

import glob
from PIL import Image
import time

import matplotlib.pyplot as plt

def hist_size_plotter(basedir, exp_type, exp_date, multi_loc, start_time, end_time, timestep):
    for i in range(start_time, end_time + 1, timestep):
        cluster_areas = []
        cluster_number = []
        for j in range(len(multi_loc)):
            well_loc = multi_loc[j]
            df_step_csv_name_list = basedir, exp_type, 'post_processing_output/' , exp_date, '/', well_loc, 't', str(i).zfill(3) , 'c2_post_processing', '.csv'
            df_step_csv_name_list_2  =''.join(df_step_csv_name_list)
            df_step = pd.read_csv(df_step_csv_name_list_2)


            cluster_areas_well_ID = df_step["Cluster size"]

            cluster_areas = np.append(cluster_areas, cluster_areas_well_ID)

        cluster_number = cluster_areas/189

        plt.hist(cluster_number, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120])
        # plt.xlim(0,120)
        plt.ylim(0,600)
        plt.xlabel('Cluster size')
        plt.ylabel('Number of clusters')        
        # plt.show()
        plt.savefig(f'/Users/Nathan/Documents/Oxford/DPhil/clusters/data_analysis/post-processing/histogram/frame-{i:03d}.png', bbox_inches='tight', dpi=300)
        plt.clf()

    images_hist = []
    # get all the images in the 'images for gif' folder
    for filename_hist in sorted(glob.glob('/Users/Nathan/Documents/Oxford/DPhil/clusters/data_analysis/post-processing/histogram/frame-*.png')): # loop through all png files in the folder
        im_hist = Image.open(filename_hist) # open the image
        images_hist.append(im_hist) # add the image to the list

    # calculate the frame number of the last frame (ie the number of images)
    last_frame = (len(images_hist)) 

    # create 10 extra copies of the last frame (to make the gif spend longer on the most recent data)
    for x in range(0, 9):
        im_hist = images_hist[last_frame-1]
        images_hist.append(im_hist)
    
    # get the current time to use in the filename
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # save as a gif   
    images_hist[0].save('/Users/Nathan/Documents/Oxford/DPhil/clusters/data_analysis/post-processing/histogram/' + timestr + '.gif',
                save_all=True, append_images=images_hist[1:], optimize=False, duration=500, loop=0)


        # cluster_number_2D = np.round(cluster_number)
        # plt.hist(cluster_number_2D)
        # plt.show()

        # cluster_radius = np.sqrt(cluster_areas/math.pi)
        # cluster_volume_3D = (4/3)*math.pi*(cluster_radius**3)

        # single_cell_volume = (4/3)*math.pi*((189/math.pi)**(3/2))
        # cluster_number_3D = np.round(cluster_volume_3D/single_cell_volume)

        # chance_of_cell_in_clus = 

        # plt.hist(cluster_number_2D)
        # plt.show()

        # plt.hist(cluster_number_3D)
        # plt.show()