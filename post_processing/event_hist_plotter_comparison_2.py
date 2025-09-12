import numpy as np
import pandas as pd
from ast import literal_eval

import matplotlib.pyplot as plt

# These three parameters are needed for accessing data and saving to files
basedir = '/Users/Nathan/Documents/Oxford/DPhil/'
exp_type = 'In_vitro_homogeneous_data/'

exp_dates = ['2017-03-16','2017-03-16','2018-06-21']
well_locs = ['s037','s073','s24']
start_times = [104,51,21]
end_times = [142,142,142]


'''
Lineage tracer traces each cluster in turn back in time using event to find contributing clusters

Input arguments:
start_time: Timepoint to be traced back to and plotted
end_time: Timepoint to trace from and plot cluster 
basedir,
exp_date,
well_loc,

Output:
Series of subplots of start_time and end_time 
for each cluster's lineage next to each other

'''

# df_end_now_csv_name_list = basedir, exp_type, 'post_processing_output_3D/', exp_date, '/', well_loc, 't', str(end_time).zfill(3), 'c2_post_processing', '.csv'
# df_end_now_csv_name_list_2  =''.join(df_end_now_csv_name_list)
# df_end_now = pd.read_csv(df_end_now_csv_name_list_2)
# cluster_tags = df_end_now["Tag number"].to_numpy().astype(int)




cols = ["Tag number", "Cluster size", "Cluster Centre x", "Cluster Centre y", 
        "Event", "Clusters in event", "Timestep", "Date", "Well ID"]

# event_cols = ["Move", "Coagulation", "Move large", "Splitting", 
#               "Move large and grow", "Possible Coagulation",
#               "Edge Appearance type 1", "Appearance type 1",
#               "Edge Appearance type 2", "Appearance type 2", 
#               "Edge Appearance type 3", "Appearance type 3", "Appearance Error"]

# event_cols_plot = ["Move", "Coag", "Move l", "Splitting", 
#               "Move l & g", "Poss Coag",
#               "Edge App 1", "App 1",
#               "Edge App 2", "App 2", 
#               "Edge App 3", "App 3", "App Error"]

event_cols = ["Coagulation", "Move large", "Splitting", 
              "Move large and grow", "Possible Coagulation",
              "Edge Appearance type 1", "Appearance type 1",
              "Edge Appearance type 2", "Appearance type 2", 
              "Edge Appearance type 3", "Appearance type 3", "Appearance Error"]

event_cols_plot = ["Coag", "Move l", "Splitting", 
              "Move l & g", "Poss Coag",
              "Edge App 1", "App 1",
              "Edge App 2", "App 2", 
              "Edge App 3", "App 3", "App Error"]

for g in range(len(well_locs)):
        exp_date = exp_dates[g]
        well_loc = well_locs[g]
        start_time = start_times[g]
        end_time = end_times[g]

        cluster_appear_sizes = []
        number_event = np.zeros(len(event_cols))
        for i in range(end_time, start_time - 1, -1) :
                # print('i is', i)
                time_i = str(i).zfill(3)
                df_step_csv_name_list = basedir, exp_type, 'post_processing_output_3D/', exp_date, '/', well_loc, 't', time_i, 'c2_post_processing', '.csv'
                df_step_csv_name_list_2  =''.join(df_step_csv_name_list)
                df_step = pd.read_csv(df_step_csv_name_list_2)
                # cluster_2D_areas = df_clus_areas.to_numpy()

                
                df_appear = df_step.loc[df_step['Event'] == 'Appearance type 1']
                # df_appear = df_step.loc[df_step['Event'] == 'Splitting']
                sizes_to_add = df_appear['Cluster size']
                cluster_appear_sizes = np.append(cluster_appear_sizes, sizes_to_add)

                for j in range(len(event_cols)):
                        df_add = df_step.loc[df_step['Event'] == event_cols[j]]
                        # df_appear = df_step.loc[df_step['Event'] == 'Splitting']
                        number_events_adding = df_add.shape[0]
                        number_event[j] += number_events_adding

        ratio_1 = number_event[0]/number_event[2]

        ratio_2 = (number_event[0]+ number_event[4])/number_event[2]

        print('Ratios are:', ratio_1, ratio_2)
        plt.figure(1)
        plt.hist(cluster_appear_sizes/189, alpha=0.7)

        plt.figure(2)
        plt.bar(event_cols_plot, number_event, alpha=0.7)

plt.show()
plt.show()



