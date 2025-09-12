import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glob
from PIL import Image
import time
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from IPython import display

from pylab import *
from scipy.ndimage import *


import pre_pro_operators as pre_oper


### ------------   Input parameters    -----------------  ###
basedir = '/Users/Nathan/Documents/Oxford/DPhil/'
exp_type = 'In_vitro_homogeneous_data/'
experiment = 'RAW_data/2017-02-03_sphere_timelapse/'
experiment = 'RAW_data/2017-02-13_sphere_timelapse_2/'
multi_dates = ['2017-03-16', '2017-03-10']
# multi_dates = ['2017-02-03', '2017-02-13', '2017-03-16', '2017-03-10']
folder = 'RAW/Timelapse/sphere_timelapse_useful_wells/'
folder_3 = 'sphere_timelapse/'
fileID = '.tif'

# time_array = range(1,98)
time_array = range(1,143)
# time_array = range(1,146)
# time_array = range(1,143)

# Rename single digit values with 0 eg 1 to 01 for consistency
# time_list = [str(x).zfill(2) for x in time_array]
time_list = [str(x).zfill(3) for x in time_array]
# time_list= ['21','22','23','24','25','26','27','28','29','30']



multi_well_multi_dates = [['s073', 's074'], ['s04', 's05']]
# multi_well_multi_dates = [['s11', 's12'], ['s27', 's28'], ['s073', 's074'], ['s04', 's05']]

plt_num = 0
for j in range(len(multi_dates)):
    exp_date = multi_dates[j]
    multi_well = multi_well_multi_dates[j]
    for i in range(len(multi_well)):
        well_loc = multi_well[i]
        num_clusters = []
        cluster_areas = np.array([])


        # Read in 2D array of cluster areas over time
        cluster_2D_areas_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, '_cluster_areas', '.csv'
        cluster_2D_areas_csv_name_list_2  =''.join(cluster_2D_areas_csv_name_list)
        df_clus_areas = pd.read_csv(cluster_2D_areas_csv_name_list_2, header=None)
        cluster_2D_areas = df_clus_areas.to_numpy()
        print('Shape', cluster_2D_areas.shape)

        # Read in mean area of cluster
        mean_areas_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, '_mean_areas', '.csv'
        mean_areas_csv_name_list_2  =''.join(mean_areas_csv_name_list)
        df_mean_areas = pd.read_csv(mean_areas_csv_name_list_2, header=None)
        mean_areas = df_mean_areas.to_numpy()

        # Read in total cluster coverage area
        total_areas_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, '_total_areas', '.csv'
        total_areas_csv_name_list_2  =''.join(total_areas_csv_name_list)
        df_total_areas = pd.read_csv(total_areas_csv_name_list_2, header=None)
        total_areas = df_total_areas.to_numpy()

        # Read in number of clusters
        number_clusters_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, '_number_clusters', '.csv'
        number_clusters_csv_name_list_2  =''.join(number_clusters_csv_name_list)
        df_number_clusters = pd.read_csv(number_clusters_csv_name_list_2, header=None)
        number_clusters = df_number_clusters.to_numpy()

        # colors = ['tab20']
        cm = plt.cm.get_cmap('tab20')
        # Plot mean size of cluster
        plt.figure(1)
        # plt.colormaps('tab20')
        plt.xlim(20,142)
        plt.plot(mean_areas/189, color = cm.colors[plt_num])
        # plt.legend(('2017-02-03', '2017-02-03', '2017-02-13', '2017-02-13',
        #            '2017-03-16', '2017-03-16', '2017-03-10', '2017-03-10'))
        plt.title("Mean number of cells in a cluster")
        plt.savefig(basedir + 'clusters/data_analysis/pre-processing/Mean_areas_' + 'multi' + '.png', dpi=300)
        # plt.show()
        # plt.clf()

        # Plot total coverage of clusters over array
        plt.figure(2)
        plt.xlim(20,142)
        plt.plot(total_areas/(1025*1344), color = cm.colors[plt_num])
        # plt.legend(('2017-02-03', '2017-02-03', '2017-02-13', '2017-02-13',
        #            '2017-03-16', '2017-03-16', '2017-03-10', '2017-03-10',))
        plt.title("Confluence of well")
        plt.savefig(basedir + 'clusters/data_analysis/pre-processing/Total_2D_cell_area_' + 'multi' + '.png', dpi=300)
        # plt.show()
        # plt.clf()

        # Plot number of clusters
        plt.figure(3)
        plt.xlim(20,142)
        plt.plot(number_clusters, color = cm.colors[plt_num])
        # plt.legend(('2017-02-03', '2017-02-03', '2017-02-13', '2017-02-13',
        #            '2017-03-16', '2017-03-16', '2017-03-10', '2017-03-10'))
        plt.title("Number of clusters over time")
        plt.savefig(basedir + 'clusters/data_analysis/pre-processing/Number_clusters_' + 'multi' + '.png', dpi=300)
        # plt.show()
        # plt.clf()

        # Convert 2D circular areas to 3D spherical volumes
        cluster_3D_area = (4/3)*pi*((sqrt(cluster_2D_areas/(189*pi)))**3)

        tot_3D_volume = []
        mean_3D_volume = []
        # Calculate mean and total 3D volumes
        for p in range(len(time_array)):
            time_3D = cluster_3D_area[p,:]
            tot_3D_volume = np.append(tot_3D_volume,sum(time_3D))
            time_3D[time_3D == 0] = np.nan
            mean_curr = np.nanmean(time_3D)
            mean_3D_volume = np.append(mean_3D_volume,mean_curr)

        # Plot total 3D
        plt.figure(4)
        plt.xlim(20,142)
        plt.plot(tot_3D_volume, color = cm.colors[plt_num])
        # plt.legend(('2017-02-03', '2017-02-03', '2017-02-13', '2017-02-13',
        #            '2017-03-16', '2017-03-16', '2017-03-10', '2017-03-10'))
        plt.ylim(0,10000)
        plt.title("Total 3D Volume")
        plt.savefig(basedir + 'clusters/data_analysis/pre-processing/Total_3D_Number_cells_' + 'multi' + '.png', dpi=300)
        # plt.show()
        # plt.clf()

        # Plot mean 3D volume of cluster
        plt.figure(5)
        plt.xlim(20,142)
        plt.plot(mean_3D_volume, color = cm.colors[plt_num])
        # plt.legend(('2017-02-03', '2017-02-03', '2017-02-13', '2017-02-13',
        #            '2017-03-16', '2017-03-16', '2017-03-10', '2017-03-10'))
        plt.savefig(basedir + 'clusters/data_analysis/pre-processing/Mean_volume_cluster_' + 'multi' + '.png', dpi=300)
        # plt.show()
        # plt.clf()
        plt_num += 1
