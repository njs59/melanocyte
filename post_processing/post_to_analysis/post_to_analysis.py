import numpy as np
import pandas as pd
import math

# These three parameters are needed for accessing data and saving to files
basedir = '/Users/Nathan/Documents/Oxford/DPhil/'
exp_type = 'In_vitro_homogeneous_data/'
# exp_date = '2017-02-03'
# exp_date = '2017-03-16'
exp_date = '2018-06-21'

# well_loc = 's11'
# multi_loc = ['s11', 's12']
# multi_loc = ['s09', 's10']
# multi_loc = ['s073','s074']
# multi_loc = ['s037', 's038']
multi_loc = ['s24']
start_time = 35
end_time = 145
timestep = 1

ODE_out = np.zeros((100,end_time+1-start_time))
ODE_out_multi = np.zeros((100,end_time+1-start_time))
for k in range(len(multi_loc)):
    well_loc = multi_loc[k]
    for i in range(start_time, end_time + 1, timestep):

        cluster_areas = []
        cluster_volumes = []
        cluster_number = []
        df_step_csv_name_list = basedir, exp_type, 'post_processing_output_3D/' , exp_date, '/', well_loc, 't', str(i).zfill(3) , 'c2_post_processing', '.csv'
        df_step_csv_name_list_2  =''.join(df_step_csv_name_list)
        df_step = pd.read_csv(df_step_csv_name_list_2)


        cluster_areas_well_ID = df_step["Cluster size"]
        cluster_volumes_well_ID = df_step["Cluster volume"]

        cluster_areas = np.append(cluster_areas, cluster_areas_well_ID)
        cluster_volumes = np.append(cluster_volumes, cluster_volumes_well_ID)

        # cluster_3D_area = (4/3)*math.pi*((np.sqrt(cluster_areas/(189*math.pi)))**3)

        # cluster_number = np.round(cluster_volumes/1955)
        cluster_number = np.round(cluster_volumes)
        bin = list(range(1,101))
        bin.append(100000)
        hist_arr = np.histogram(cluster_number,bin)

        time_ODE_output = hist_arr[0]
        ODE_out[:,i-start_time] = time_ODE_output
        print('Total cell count', sum(cluster_number))
        print('Cluster number', cluster_number)
        # print('Output for ODE', hist_arr)

    ODE_out_multi += ODE_out


ODE_out_multi = ODE_out_multi/len(multi_loc)

# save array into csv file 
np.savetxt("homogeneous_3D/2018-06-21_co_culture_t_35.csv", ODE_out_multi,  
            delimiter = ",")
# ODE_out.tofile('s11_inference_input.csv', sep = ',')