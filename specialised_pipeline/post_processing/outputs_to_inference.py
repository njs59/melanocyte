import numpy as np
import pandas as pd

# These three parameters are needed for accessing data and saving to files
basedir = '/Users/Nathan/Documents/Oxford/DPhil/clusters/worked_example/'
csv_loc = 'data_analysis_outputs/csv_files/'
# exp_date = '2017-03-16'
exp_date = '2018-06-21'

# thresh_method = 'li_method'
thresh_method = 'otsu_method'

# multi_loc = ['s037', 's038']
# multi_loc = ['s073','s074']
multi_loc = ['s24']
start_time = 1
end_time = 145
timestep = 1

ODE_out = np.zeros((100,end_time+1-start_time))
ODE_out_multi = np.zeros((100,end_time+1-start_time))
for k in range(len(multi_loc)):
    well_loc = multi_loc[k]
    for i in range(start_time, end_time + 1, timestep):

        cluster_volumes = []
        cluster_volume_out = []
        df_step_csv_name_list = basedir, csv_loc, exp_date, '/', thresh_method, '/', well_loc, 't', str(i).zfill(3), 'c2', '_tracking', '.csv'
        df_step_csv_name_list_2  =''.join(df_step_csv_name_list)
        df_step = pd.read_csv(df_step_csv_name_list_2)


        cluster_volumes_well_ID = df_step["Cluster volume"]

        cluster_volumes = np.append(cluster_volumes, cluster_volumes_well_ID)

        cluster_volume_out = np.round(cluster_volumes)
        bin = list(range(1,101))
        bin.append(200)
        hist_arr = np.histogram(cluster_volume_out,bin)

        time_ODE_output = hist_arr[0]
        ODE_out[:,i-start_time] = time_ODE_output
        print('Total cell count', sum(cluster_volume_out))
        print('Cluster number', cluster_volume_out)
        # print('Output for ODE', hist_arr)

    ODE_out_multi += ODE_out


ODE_out_multi = ODE_out_multi/len(multi_loc)

save_name_list = basedir, csv_loc, exp_date, '/', thresh_method, '/', '3D_COMBO_multi_t_1', '.csv'
save_name_list_2  =''.join(save_name_list)

# save array into csv file 
np.savetxt(save_name_list_2, ODE_out_multi,  
            delimiter = ",")
# ODE_out.tofile('s11_inference_input.csv', sep = ',')