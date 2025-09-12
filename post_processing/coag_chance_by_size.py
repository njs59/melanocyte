import math
import numpy as np
import pandas as pd
from ast import literal_eval

import matplotlib.pyplot as plt

# These three parameters are needed for accessing data and saving to files
basedir = '/Users/Nathan/Documents/Oxford/DPhil/'
exp_type = 'In_vitro_homogeneous_data/'

# exp_date = '2017-02-03'
# start_time = 37
# end_time = 97
# multi_well = ['s11', 's12']

exp_date = '2017-03-16'
start_time = 20
end_time = 145
multi_well = ['s073', 's074']


cols = ["Tag number", "Cluster size", "Cluster Centre x", "Cluster Centre y", 
        "Event", "Clusters in event", "Timestep", "Date", "Well ID"]

event_cols = ["Move", "Coagulation", "Move large", "Splitting", 
              "Move large and grow", "Possible Coagulation",
              "Edge Appearance type 1", "Appearance type 1",
              "Edge Appearance type 2", "Appearance type 2", 
              "Edge Appearance type 3", "Appearance type 3", "Appearance Error"]

event_cols_plot = ["Move", "Coag", "Move l", "Splitting", 
              "Move l & g", "Poss Coag",
              "Edge App 1", "App 1",
              "Edge App 2", "App 2", 
              "Edge App 3", "App 3", "App Error"]


# coagulation_df = pd.DataFrame(columns = cols)

cluster_coag_sizes = []
cluster_not_coag_sizes = []

cluster_appear_sizes = []
number_event = np.zeros(len(event_cols))
for h in range(len(multi_well)):
    well_loc = multi_well[h]
    for i in range(end_time, start_time - 1, -1) :
        coag_IDs = []
        # print('i is', i)
        time_i = str(i).zfill(3)
        time_old = str(i-1).zfill(3)
        df_step_csv_name_list = basedir, exp_type, 'post_processing_output_3D/', exp_date, '/', well_loc, 't', time_i, 'c2_post_processing', '.csv'
        df_step_csv_name_list_2  =''.join(df_step_csv_name_list)
        df_step = pd.read_csv(df_step_csv_name_list_2)
        # cluster_2D_areas = df_clus_areas.to_numpy()
        df_step_old_csv_name_list = basedir, exp_type, 'post_processing_output_3D/', exp_date, '/', well_loc, 't', time_old, 'c2_post_processing', '.csv'
        df_step_old_csv_name_list_2  =''.join(df_step_old_csv_name_list)
        df_step_old = pd.read_csv(df_step_old_csv_name_list_2)
        
        if (df_step['Event'] == 'Coagulation').any():

            rows_to_add = df_step.loc[df_step['Event'].isin(['Coagulation'])]
            rows_to_add = rows_to_add.reset_index()

            # coagulation_df = pd.concat([coagulation_df, rows_to_add])

            for j in range(len(rows_to_add["Clusters in event"])):
                coag_IDs_add = literal_eval(rows_to_add["Clusters in event"][j])
                coag_IDs = np.append(coag_IDs, coag_IDs_add)


        if (df_step['Event'] == 'Possible Coagulation').any():
            print('Possible at:', i)

        print('Coag IDs are', coag_IDs)

        cluster_step_sizes = df_step_old['Cluster volume']

        coag_future_indexes = df_step_old.index[np.in1d(df_step_old['Tag number'],coag_IDs)]

        cluster_step_sizes_coag = cluster_step_sizes[coag_future_indexes].to_numpy()
        cluster_step_sizes_not_coag = cluster_step_sizes.drop(coag_future_indexes, axis=0).to_numpy()


        cluster_coag_sizes = np.append(cluster_coag_sizes, cluster_step_sizes_coag)
        cluster_not_coag_sizes = np.append(cluster_not_coag_sizes, cluster_step_sizes_not_coag)



# cluster_coag_cell_num = np.round(cluster_coag_sizes/189)
cluster_coag_cell_num = np.round(cluster_coag_sizes)

# cluster_coag_cell_num = np.round(((4/3)*np.pi*((np.sqrt(cluster_coag_sizes/(np.pi)))**3))/1955)
# cluster_not_coag_cell_num = np.round(cluster_not_coag_sizes/189)
cluster_not_coag_cell_num = np.round(cluster_not_coag_sizes)
# cluster_not_coag_cell_num = np.round(((4/3)*np.pi*((np.sqrt(cluster_not_coag_sizes/(np.pi)))**3))/1955)
max_size = max(max(cluster_coag_cell_num),max(cluster_not_coag_cell_num))

chance_array = np.zeros((int(max_size),1))
for k in range(1, int(max_size)):
    coag_count = 0
    non_coag_count = 0
    if k < 21:
        coag_count = np.count_nonzero(cluster_coag_cell_num == k)
        non_coag_count = np.count_nonzero(cluster_not_coag_cell_num == k)

    elif 20 < k < 30:
        # search_arr = [k-2,k-1,k,k+1,k+2]
        for l in range(k-2, k+3):
            coag_count += np.count_nonzero(cluster_coag_cell_num == l)
            non_coag_count += np.count_nonzero(cluster_not_coag_cell_num == l)
    elif 30 <= k < 50:
        # search_arr = [k-2,k-1,k,k+1,k+2]
        for l in range(k-5, k+6):
            coag_count += np.count_nonzero(cluster_coag_cell_num == l)
            non_coag_count += np.count_nonzero(cluster_not_coag_cell_num == l)

    elif 50 <= k:
        # search_arr = [k-2,k-1,k,k+1,k+2]
        for l in range(k-20, k+21):
            coag_count += np.count_nonzero(cluster_coag_cell_num == l)
            non_coag_count += np.count_nonzero(cluster_not_coag_cell_num == l)

    if coag_count > 0 and non_coag_count > 0:
        chance_array[k] = coag_count/(coag_count + non_coag_count)

    else:
        print('Inconclusive size:', k)


print('Yes')

plt.plot(chance_array)

# plt.xlim(0,20)
plt.savefig(basedir + 'clusters/data_analysis_3D/post-processing/Coag_chance_by_size.png', dpi=300)

plt.show()


chance_array = np.zeros((300,1))
for k in range(1, 301):
    coag_count = 0
    non_coag_count = 0
    # if k < 51:
    #     coag_count = np.count_nonzero(cluster_coag_cell_num == k)
    #     non_coag_count = np.count_nonzero(cluster_not_coag_cell_num == k)

    if 0 < k < 51:
        # search_arr = [k-2,k-1,k,k+1,k+2]
        for l in range(k-2, k+3):
            coag_count += np.count_nonzero(cluster_coag_cell_num == l)
            non_coag_count += np.count_nonzero(cluster_not_coag_cell_num == l)
    elif 51 <= k < 90:
        # search_arr = [k-2,k-1,k,k+1,k+2]
        for l in range(k-5, k+6):
            coag_count += np.count_nonzero(cluster_coag_cell_num == l)
            non_coag_count += np.count_nonzero(cluster_not_coag_cell_num == l)

    elif 90 <= k:
        # search_arr = [k-2,k-1,k,k+1,k+2]
        for l in range(k-20, k+21):
            coag_count += np.count_nonzero(cluster_coag_cell_num == l)
            non_coag_count += np.count_nonzero(cluster_not_coag_cell_num == l)

    if coag_count > 0 and non_coag_count > 0:
        chance_array[k] = coag_count/(coag_count + non_coag_count)

    else:
        print('Inconclusive size:', k)


print('Yes')

x = np.linspace(0,299,300).astype('int')

plt.plot(x[1:],chance_array[1:,:])

# plt.xlim(0,20)
plt.savefig(basedir + 'clusters/data_analysis_3D/post-processing/Coag_chance_by_size_2.png', dpi=300)

plt.show()

plt.plot(x[1:151],chance_array[1:151,:])

# plt.xlim(0,20)
plt.savefig(basedir + 'clusters/data_analysis_3D/post-processing/Coag_chance_by_size_3.png', dpi=300)

plt.show()






chance_array = np.zeros((int(max_size),1))
for k in range(1, int(max_size)):
    coag_count = 0
    non_coag_count = 0
    coag_count = np.count_nonzero(cluster_coag_cell_num == k)
    non_coag_count = np.count_nonzero(cluster_not_coag_cell_num == k)

    if coag_count > 0 and non_coag_count > 0:
        chance_array[k] = coag_count/(coag_count + non_coag_count)

    else:
        print('Inconclusive single size:', k)


plt.plot(chance_array)

# plt.xlim(0,20)
plt.savefig(basedir + 'clusters/data_analysis_3D/post-processing/Single_Coag_chance_by_size.png', dpi=300)

plt.show()

plt.plot(chance_array[0:280])

# plt.xlim(0,20)
plt.savefig(basedir + 'clusters/data_analysis_3D/post-processing/Single_Coag_chance_by_size_zoomed.png', dpi=300)

plt.show()



# plt.hist(cluster_appear_sizes)
# plt.show()

# plt.bar(event_cols_plot, number_event)
# plt.show()


