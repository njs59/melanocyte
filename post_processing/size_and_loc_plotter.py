import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import lineage_tracer_function as ltf

def size_and_loc_tracker(start_time, end_time, timejump,  basedir, exp_type, exp_date, well_loc, cluster_lineage):
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
    cluster_lineage = ltf.lineage_tracer(start_time, end_time, basedir, exp_type, exp_date, well_loc, plots = False)

    if len(cluster_lineage) > 0:
        cluster_tags = cluster_lineage
    else:
        df_end_csv_name_list = basedir, exp_type, 'post_processing_output/' , exp_date, '/', well_loc, 't', str(end_time).zfill(3) , 'c2_post_processing', '.csv'
        df_end_csv_name_list_2  =''.join(df_end_csv_name_list)
        df_end = pd.read_csv(df_end_csv_name_list_2)

        cluster_tags = df_end["Tag number"].to_numpy().astype(int)


    colour_ticker = -1
    for h in range(len(cluster_tags)):

        cluster_tag_to_track = cluster_tags[h]

        x = []
        cluster_size = []
        cluster_location_x = []
        cluster_location_y = []
        cluster_change_from_init_x = []
        cluster_change_from_init_y = []

        for i in range(start_time, end_time + 1, timejump):
            x = np.append(x, i)
            # Read in csv
            time_i = str(i).zfill(2)
            index_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, 't', str(time_i).zfill(2), 'c2', '_indexed', '.csv'
            index_csv_name_list_2  =''.join(index_csv_name_list)
            df_slice = pd.read_csv(index_csv_name_list_2, header=None)
            current_array = df_slice.to_numpy()

            df_storage_csv_name_list = basedir, exp_type, 'post_processing_output/', exp_date, '/', well_loc, 't', str(time_i).zfill(2) , 'c2_post_processing', '.csv'
            df_storage_csv_name_list_2  =''.join(df_storage_csv_name_list)
            df_storage = pd.read_csv(df_storage_csv_name_list_2)

            
            cluster_current_row_of_interest = df_storage.loc[df_storage['Tag number'] == cluster_tag_to_track]
            # Store centres in array
            if cluster_current_row_of_interest.shape[0] == 0:
                cluster_size = np.append(cluster_size, 0)
                cluster_location_x = np.append(cluster_location_x, None)
                cluster_location_y = np.append(cluster_location_y, None)
                # cluster_change_from_init_x = np.append(cluster_change_from_init_x,0)
                # init_x = cluster_location_x
                # cluster_change_from_init_y = np.append(cluster_change_from_init_y, 0)
                # init_y = cluster_location_y
                continue
            df1_slice = cluster_current_row_of_interest[['Cluster Centre x', 'Cluster Centre y']]
            centres_end_2D_lineage = df1_slice.to_numpy()

            

            cluster_size = np.append(cluster_size, cluster_current_row_of_interest['Cluster size'])
            cluster_location_x = np.append(cluster_location_x, int(centres_end_2D_lineage[0][0]))
            cluster_location_y = np.append(cluster_location_y, int(centres_end_2D_lineage[0][1]))

            init_x = next(item for item in cluster_location_x if item is not None)
            init_y = next(item for item in cluster_location_y if item is not None)

            cluster_change_from_init_x = np.append(cluster_change_from_init_x, int(centres_end_2D_lineage[0][0]) - init_x)
            cluster_change_from_init_y = np.append(cluster_change_from_init_y, int(centres_end_2D_lineage[0][1]) - init_y)



        plt.figure(1)
        # Location plot
        # plt.plot(cluster_location_x, cluster_location_y)
        # Spider plot
        plt.plot(cluster_change_from_init_x, cluster_change_from_init_y, lw=2)
        # plt.show()
        
        plt.figure(2)
        # plt.plot(x, cluster_size)
        plt.plot(cluster_size)

        
        plt.figure(3)
        # fig, ax = plt.subplots()
        for i in range(len(cluster_location_x)-1):
            g = i/len(cluster_location_x)
            if i < len(cluster_location_x)/2:
                b = i/ (len(cluster_location_x)/2)
            else:
                b = 2 - (i/ (len(cluster_location_x)/2))
            # print('B is', b)
            r = 1-g
            color = (r, g, b)
            # Location plot
            plt.plot(cluster_location_x[i:i+2], cluster_location_y[i:i+2], c=color, lw=2)
            # Spider plot
            # plt.plot(cluster_change_from_init_x[i:i+2], cluster_change_from_init_y[i:i+2], c=color, lw=2)
            # plt.plot(cluster_location_x, cluster_location_y)

        plt.figure(4)
        # if colour_ticker == 11:
        #     colour_ticker = 0
        # else:
        colour_ticker+=1        
        cmap = plt.colormaps['hsv']
        # Take colors at regular intervals spanning the colormap.
        colors = cmap(np.linspace(0, 1, len(cluster_tags)))
        # colors = plt.colormaps['Set3'].colors
        for i in range(len(cluster_change_from_init_x)-1):
            plt.plot(cluster_change_from_init_x[i:i+2], cluster_change_from_init_y[i:i+2],
                     color=colors[colour_ticker],linewidth=((cluster_size[i+1]/189)/20))
    
        # plt.figure(5)
        # points = np.array([cluster_change_from_init_x, cluster_change_from_init_y]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # lc = LineCollection(segments, linewidths=cluster_size,color='blue')
        # # fig,a = plt.subplots()
        # plt.add_collection(lc)

    plt.show()
    plt.show()
    plt.show()
    plt.show()
    plt.show()
    

        
    
