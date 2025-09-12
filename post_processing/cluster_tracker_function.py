import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def cluster_tracker(start_time, end_time, timejump, cluster_index_final_time, basedir, exp_type, exp_date, well_loc):
    '''
Cluster tracker tracks an individually taggged cluster over time
Input arguments: 
    start_time: first timepoint to plot
    end_time: final timepoint to plot
    timejump: number of timesteps between each plot
    cluster_index_final_time: row in final time to select cluster ID tag from
    basedir,
    exp_type,
    exp_date,
    well_loc

Output:
    Series of plots 
'''
    df_end_csv_name_list = basedir, exp_type, 'post_processing_output/', exp_date, '/', well_loc, 't', str(end_time).zfill(2) , 'c2_post_processing', '.csv'
    df_end_csv_name_list_2  =''.join(df_end_csv_name_list)
    df_end = pd.read_csv(df_end_csv_name_list_2)

    cluster_tags = df_end["Tag number"].to_numpy().astype(int)

    # for h in range(len(cluster_tags)):
    #     cluster_lineage = [cluster_tags[h]]

    cluster_tag_to_track = cluster_tags[cluster_index_final_time]

    cluster_size = []
    cluster_location_x = []
    cluster_location_y = []

    for i in range(start_time - 1, end_time, timejump):
        # Read in csv
        time_i = str(i).zfill(3)
        index_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, 't', time_i, 'c2', '_indexed', '.csv'
        index_csv_name_list_2  =''.join(index_csv_name_list)
        df_slice = pd.read_csv(index_csv_name_list_2, header=None)
        current_array = df_slice.to_numpy()

        df_storage_csv_name_list = basedir, exp_type, 'post_processing_output/', exp_date, '/', well_loc, 't', time_i , 'c2_post_processing', '.csv'
        df_storage_csv_name_list_2  =''.join(df_storage_csv_name_list)
        df_storage = pd.read_csv(df_storage_csv_name_list_2)

        cluster_current_row_of_interest = df_storage.loc[df_storage['Tag number'] == cluster_tag_to_track]
        # Store centres in array
        df1_slice = cluster_current_row_of_interest[['Cluster Centre x', 'Cluster Centre y']]
        centres_end_2D_lineage = df1_slice.to_numpy()
        if cluster_current_row_of_interest.shape[0] == 0:
            cluster_size = np.append(cluster_size, 0)
            cluster_location_x = np.append(cluster_location_x, None)
            cluster_location_y = np.append(cluster_location_y, None)
            end_index = 0
        else:
            cluster_size = np.append(cluster_size, cluster_current_row_of_interest['Cluster size'])
            cluster_location_x = np.append(cluster_location_x, int(centres_end_2D_lineage[0][0]))
            cluster_location_y = np.append(cluster_location_y, int(centres_end_2D_lineage[0][1]))
            end_index = current_array[int(centres_end_2D_lineage[0][0]),int(centres_end_2D_lineage[0][1])]
        if end_index > 0:
            index_locs = np.where(current_array == int(end_index))
            single_index_arr = np.asarray(index_locs)

            bool_index = np.zeros(current_array.shape)

            for r in range(single_index_arr.shape[1]):
                bool_index[single_index_arr[0,r], single_index_arr[1,r]] = 1


            plt.imshow(bool_index)
            plt.axis([0, current_array.shape[1], 0, current_array.shape[0]])
            plt.show()


    plt.plot(cluster_size)
    plt.show()

    plt.plot(cluster_location_x, cluster_location_y)
    plt.show()
