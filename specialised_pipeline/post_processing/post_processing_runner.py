import csv
import math
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt

from scipy.ndimage import *

import post_pro_operators as post_oper

t_before = time.time()

###    -----------   Input parameters   --------------     ###
# Parameters
basedir = '/Users/Nathan/Documents/Oxford/DPhil/melanocyte/'
data_folder = 'data/Still_Images_with_BF_for_Nathan/'
fileID = '.tif'
# time = 0
min_clus_size = 150


base_str = "VID289"

# experiment_id = "B4"  # Change this as needed
# experiment_ids = ["A5","B4","D5","E2"]  # Change this as needed
experiment_ids = ["A1","A3","A5","B2","B4","B6","D1","D3","D5","E2","E4","E6"]

# Initialize storage for all experiments
summary_data = {eid: {"days": [], "total": [], "mean": [], "count": [], "vol_total": [], "vol_mean": []} for eid in experiment_ids}


# Column titles to be used in dataframes
cols = ["Tag number","Cluster size", "Cluster volume", "Cluster Centre x", "Cluster Centre y", 
           "Event", "Clusters in event", "Timestep", "Exp ID", "Well ID"]


# Variable to store next tag ID number
next_available_tag = 1

# Constant for area to search around a cluster for nearby clusters
search_radius = 50

# Constant for maximum percentage change in size to be 'Move large' event
percent_diff_max = 20

###   --------------------   Main Code section   -------------------   ###

for experiment_id in experiment_ids:
    
    filenames = post_oper.generate_filenames(experiment_id=experiment_id, base="VID289",
                       start_day = 0, start_hour = 6, start_minute = 0, 
                       end_day = 5, end_hour = 0, end_minute = 0,
                       lowest_day = 0, lowest_hour = 0, lowest_minute = 0,
                       highest_day = 5, highest_hour = 21, highest_minute = 0, 
                       gap_days = 1, gap_hours = 3, gap_minutes = 15)
    for name in filenames:
        print(name)

    # days = list(range(0, len(filenames)))
    full_times_info, full_times_days, full_times_hours, full_times_minutes = post_oper.extract_time_components(filenames)
    times = full_times_info[0]
    print("Times in days are", times)

    # Process each timestep
    ticker = 0
    for filename in filenames:

        current_day = full_times_days[ticker]
        current_hour = full_times_hours[ticker]
        current_minute = full_times_minutes[ticker]
        current_timing = f"{current_day:02d}d{current_hour:02d}h{current_minute:02d}m"

        current_read_in_dir_list = basedir, 'specialised_pipeline/', 'pre_processing_output/', base_str, '/'
        current_read_in_dir  =''.join(current_read_in_dir_list)

        t_before_step = time.time()

        ###   ---------------   Step 0: Initialisation   --------------   ###
        # Read in area array for given time
        csv_name_list = current_read_in_dir, filename, '_area', '.csv'
        csv_name_list_2  =''.join(csv_name_list)
        df_area = pd.read_csv(csv_name_list_2, header=None)
        array_area_current_time = df_area.to_numpy()

        # Read in index array for given time
        csv_name_list_index = current_read_in_dir, filename, '_indexed', '.csv'
        csv_name_list_2_index  =''.join(csv_name_list_index)
        df_index = pd.read_csv(csv_name_list_2_index, header=None)
        array_index_current_time = df_index.to_numpy()

        # Initialise dataframe for given time
        df_step = pd.DataFrame(np.nan, index=range(array_index_current_time.max()), columns=cols)
        df_step["Event"] = ""
        df_step["Clusters in event"] = ""

        # Calculate the centres of the clusters (to nearest integer value in x,y)
        centres_2D_current = post_oper.calc_clus_centre(array_index_current_time)

        # Generate 1D array of cluster areas
        area_2D_current = []
        for j in range(1,array_index_current_time.max()+1):
            # Find location for each cluster at current time
            loc_x = np.where(array_index_current_time==j)[0][0]
            loc_y = np.where(array_index_current_time==j)[1][0]
            # Append the list of areas using the area array
            area_2D_current.append(array_area_current_time[loc_x,loc_y])



        # Initial timestep treated differently
        if ticker == 0:
            # Create list of tagged indices
            tag_number_current = range(1,array_index_current_time.max()+1)

            # Store variables for use in next step
            centres_2D_old = centres_2D_current
            next_available_tag = array_index_current_time.max()+1

        else:
            ###   --------------   Step 1: Find simple and clear coagulations and movements   ----------   ###
            tag_number_current = []
            no_same_locs_index = []
            # Loop over each cluster
            for k in range(1,array_index_current_time.max()+1):
                # Find how many cluster centres from previous timestep overlap with this cluster
                same_locs, same_locs_store = post_oper.previous_clusters_at_loc(array_index_current_time, centres_2D_old, k)
                if same_locs == 1:
                    # Simple, it's just movement case

                    # Get row from previous timestep
                    mask = (df_old['Cluster Centre x'] == same_locs_store[0]) & (df_old['Cluster Centre y'] == same_locs_store[1])
                    matching_row = df_old[mask]

                    # Keep same Tag number as cluster from previous timestep
                    cluster_tag_number = matching_row['Tag number'].tolist()[0]
                    tag_number_current.append(cluster_tag_number)

                    # Remove cluster ID from list of non-assigned clusters
                    non_assigned_old_tags_list = list(non_assigned_old_tags_list)
                    non_assigned_old_tags_list.remove(cluster_tag_number)

                    # Event is move, no clusters involved
                    df_step.iloc[k-1,5] = 'Move'

                elif same_locs == 0:
                    # Splitting and other cases, need to be considered in a separate loop
                    # Save indexes with no cluster centres from the previous timestep matching and
                    # deal with them after the main loop
                    no_same_locs_index.append(k)
                    # Assign next available tag number (which might later be overwritten)
                    tag_number_current.append(next_available_tag)
                    next_available_tag += 1
                    


                else:
                    # Coagulation case
                    # same_locs will be greater than 1 so we have coagulation
                    rows_to_save = []

                    # Generate df of coagulating clusters from previous timestep
                    for s in range(same_locs):
                        mask = (df_old['Cluster Centre x'] == same_locs_store[s,0]) &\
                            (df_old['Cluster Centre y'] == same_locs_store[s,1])
                        rows_to_save.append(np.where(mask == True)[0][0])
                    clusters_coagulating = df_old.iloc[rows_to_save]

                    # Remove cluster IDs from list of non-assigned clusters
                    clusters_in_event = clusters_coagulating['Tag number'].tolist()
                    non_assigned_old_tags_list = list(non_assigned_old_tags_list)
                    for t in range(len(clusters_in_event)):
                        non_assigned_old_tags_list.remove(clusters_in_event[t])
                    # Keep lowest tag number of the coagulating clusters
                    cluster_tag_number = min(clusters_coagulating['Tag number'])
                    

                    # Event is coagulation, store clusters involved and tag number
                    df_step.iloc[k-1,5] = 'Coagulation'
                    df_step.iloc[k-1,6] = str(clusters_in_event)
                    tag_number_current.append(cluster_tag_number)
            
            ###   ------------------      Step 2: No cluster centres case (2nd loop)     ---------------  ###
            
            # Now deal separately with same_locs = 0 case after simple move and coagulations have been sorted
            
            # Generate array of just non-assigned clusters
            non_assigned_cluster_array = np.zeros([array_index_old.shape[0], array_index_old.shape[1]])
            locs_not_considered = 0
            move_large_indexes = []
            for m in range(len(non_assigned_old_tags_list)):
                row_corresponding_to_tag = df_old.loc[df_old['Tag number'] == non_assigned_old_tags_list[m]].index[0]
                index_corresponding = row_corresponding_to_tag + 1
                old_locs_of_arrs = np.where(array_index_old == index_corresponding) 
                if old_locs_of_arrs[0].shape[0] != 0:
                    for n in range(len(old_locs_of_arrs[0])):
                        non_assigned_cluster_array[old_locs_of_arrs[0][n], old_locs_of_arrs[1][n]] = array_index_old[old_locs_of_arrs[0][n], old_locs_of_arrs[1][n]]
                
            # Main 2nd loop over all clusters that don't match with a previous cluster centre
            for l in range(len(no_same_locs_index)): 
                
                # Find index and centre of the cluster
                index_of_interest = no_same_locs_index[l]           
                x_cen = int(centres_2D_current[index_of_interest-1][0])
                y_cen = int(centres_2D_current[index_of_interest-1][1])

                # Compares with previous timestep array for all clusters and for non-assigned clusters respectively 
                near_clus, clus_distances = post_oper.nearby_clusters(x_cen, y_cen, search_radius, array_index_old)
                near_non_assigned_clus, clus_distances_non_assigned = post_oper.nearby_clusters(x_cen, y_cen, search_radius, non_assigned_cluster_array)


                if len(near_clus) == 0:
                    # Absolutely no nearby clusters
                    # Appearance code
                    print('No nearby clusters')
                    # Check if cluster at edge of field of view
                    if x_cen < 10 or x_cen > 1015 or y_cen < 10 or y_cen > 1334:
                        # Cluster appeared at edge (type 1 is no neaarby clusters at all)
                        df_step.iloc[index_of_interest-1,5] = 'Edge Appearance type 1'
                        
                    else:
                        # Cluster has just appeared (type 1 is no neaarby clusters at all)
                        df_step.iloc[index_of_interest-1,5] = 'Appearance type 1'
                        

                else:
                    # Search for non-assigned clusters
                    if len(near_non_assigned_clus) == 1:
                        # Solo non-assigned cluster
                        # Large move event if close in size else split or large move and grow
                        
                        # Find ID of nearby non-assigned cluster
                        cluster_index_split_from = post_oper.pick_cluster_inverse_dist(near_non_assigned_clus, clus_distances_non_assigned)
                        # Keeps ID of old cluster
                        cluster_ID = float(df_old.iloc[cluster_index_split_from - 1 , 0].item())

                        # Find and compare sizes of old and new cluster
                        old_cluster_size = float(df_old.iloc[cluster_index_split_from - 1 , 1].item())
                        new_cluster_size = area_2D_current[index_of_interest-1]
                        percent_diff = 100*(abs(new_cluster_size - old_cluster_size))/((old_cluster_size+new_cluster_size)/2)
                        
                        if percent_diff < percent_diff_max:
                            # If close in size then it's the same cluster but it has just
                            # Moved a large distance so we have 'Move large' event and the same cluster ID
                            if cluster_ID in move_large_indexes:
                                df_step.iloc[index_of_interest-1,5] = 'Move large'
                                df_step.iloc[index_of_interest-1,6] = str([cluster_ID])
                            else:
                                tag_number_current[no_same_locs_index[l]-1] = cluster_ID
                                df_step.iloc[index_of_interest-1,5] = 'Move large'
                                df_step.iloc[index_of_interest-1,6] = str([cluster_ID])
                                move_large_indexes = np.append(move_large_indexes, cluster_ID)

                        elif old_cluster_size > new_cluster_size and percent_diff >= 20:
                            # The new cluster is significantly smaller than the old nearby cluster so has split
                            df_step.iloc[index_of_interest-1,5] = 'Splitting'
                            df_step.iloc[index_of_interest-1,6] = str([cluster_ID])
                        else:
                            # Cluster is significantly larger than nearby cluster so has both moved and 
                            # grown without coagulation
                            if cluster_ID in move_large_indexes:
                                df_step.iloc[index_of_interest-1,5] = 'Move large and grow'
                                df_step.iloc[index_of_interest-1,6] = str([cluster_ID])
                            else:
                                tag_number_current[no_same_locs_index[l]-1] = cluster_ID
                                df_step.iloc[index_of_interest-1,5] = 'Move large and grow'
                                df_step.iloc[index_of_interest-1,6] = str([cluster_ID])
                                move_large_indexes = np.append(move_large_indexes, cluster_ID)
                    
                    elif len(near_non_assigned_clus) > 1:
                        # Multiple nearby non-assigned clusters

                        # Generate df of these clusters
                        for s in range(len(near_non_assigned_clus)):                                        
                            clusters_coagulating_df = df_old.iloc[near_non_assigned_clus - 1]
                        
                        # Find cluster sizes
                        max_near_non_assigned_size = max(clusters_coagulating_df['Cluster size'])
                        new_cluster_size = area_2D_current[index_of_interest-1]

                        if max_near_non_assigned_size < new_cluster_size:  
                            # If new cluster larger than all non-assigned nearby clusters
                            # Possible Coagulation Event 
                            # Keeps ID of lowest ID of the coagulating clusters                        
                            cluster_tag_number = min(clusters_coagulating_df['Tag number'])
                            if cluster_tag_number in move_large_indexes:
                                clusters_in_event = clusters_coagulating_df['Tag number'].tolist()
                                df_step.iloc[index_of_interest-1,5] = 'Possible Coagulation'
                                df_step.iloc[index_of_interest-1,6] = str(clusters_in_event)
                            else:
                                clusters_in_event = clusters_coagulating_df['Tag number'].tolist()
                                df_step.iloc[index_of_interest-1,5] = 'Possible Coagulation'
                                df_step.iloc[index_of_interest-1,6] = str(clusters_in_event)
                                # Overwrite cluster ID
                                tag_number_current[no_same_locs_index[l]-1] = cluster_tag_number
                                move_large_indexes = np.append(move_large_indexes, cluster_tag_number)

                        else:
                            # Pick random event with single cluster inversely proportional to distance
                            cluster_index_split_from = post_oper.pick_cluster_inverse_dist(near_non_assigned_clus, clus_distances_non_assigned)
                            # Get ID of selected cluster
                            cluster_ID = float(df_old.iloc[cluster_index_split_from - 1 , 0].item())

                            # Find and compare sizes of old and new cluster
                            old_cluster_size = float(df_old.iloc[cluster_index_split_from - 1 , 1].item())
                            new_cluster_size = area_2D_current[index_of_interest-1]
                            percent_diff = 100*(abs(new_cluster_size - old_cluster_size))/((old_cluster_size+new_cluster_size)/2)
                            
                            if percent_diff < 20:
                                # If close in size then it's the same cluster but it has just
                                # Moved a large distance so we have 'Move large' event and the same cluster ID
                                # Keeps ID of old cluster
                                # Overwrites cluster ID (so is possible for tag to be skipped but that's ok)
                                if cluster_ID in move_large_indexes:
                                    df_step.iloc[index_of_interest-1,5] = 'Move large'
                                    df_step.iloc[index_of_interest-1,6] = str([cluster_ID])
                                else:
                                    tag_number_current[no_same_locs_index[l]-1] = cluster_ID
                                    df_step.iloc[index_of_interest-1,5] = 'Move large'
                                    df_step.iloc[index_of_interest-1,6] = str([cluster_ID])
                                    move_large_indexes = np.append(move_large_indexes, cluster_ID)
                            elif old_cluster_size > new_cluster_size and percent_diff >= 20:
                                # The new cluster is significantly smaller than the old nearby cluster so has split                        
                                df_step.iloc[index_of_interest-1,5] = 'Splitting'
                                df_step.iloc[index_of_interest-1,6] = str([cluster_ID])

                            else:
                                # Cluster is significantly larger than nearby cluster so has just appeared
                                if x_cen < 10 or x_cen > 1015 or y_cen < 10 or y_cen > 1334:
                                # Cluster appeared at edge (type 2 is nearby non-assigned clusters)
                                    df_step.iloc[index_of_interest-1,5] = 'Edge Appearance type 2'                    
                                else:
                                    # Cluster has just appeared (type 2 is nearby non-assigned clusters)
                                    df_step.iloc[index_of_interest-1,5] = 'Appearance type 2'
                                # print('Error type 2 event')



                    else:
                        # No nearby non-assigned clusters but there are nearby clusters already assigned
                        # Pick random event with single cluster inversely proportional to distance
                        candidate_cluster_index_split_from = post_oper.pick_cluster_inverse_dist(near_clus, clus_distances)
                        # Get ID of selected cluster
                        candidate_cluster_ID = float(df_old.iloc[candidate_cluster_index_split_from - 1 , 0].item())

                        # Find and compare sizes of old and new cluster
                        old_cluster_size = float(df_old.iloc[candidate_cluster_index_split_from - 1 , 1].item())
                        new_cluster_size = area_2D_current[index_of_interest-1]
                        percent_diff = 100*(abs(new_cluster_size - old_cluster_size))/((old_cluster_size+new_cluster_size)/2)
        
                        if percent_diff >= 20 and new_cluster_size < old_cluster_size:
                            # Cluster significantly smaller than already assigned nearby cluster
                            # So we have splitting
                            df_step.iloc[index_of_interest-1,5] = 'Splitting'
                            df_step.iloc[index_of_interest-1,6] = str([candidate_cluster_ID])
                        
                        
                        else:
                            # Check if cluster at edge of field of view
                            if x_cen < 10 or x_cen > 1015 or y_cen < 10 or y_cen > 1334:
                                #cluster at edge (type 3 is nearby assigned cluster but no nearby unassigned clusters)
                                df_step.iloc[index_of_interest-1,5] = 'Edge Appearance type 3'
                            else:
                                df_step.iloc[index_of_interest-1,5] = 'Appearance type 3'


                        # else:
                        #     print('Error in event assignment')
                        #     # Search nearby clusters already assigned

                        #     # Can't only be splitting (cluster is too big)                       
                        #     # (can't move large as that can only happen if there is an unassigned cluster)
                        #     # Can't be appearance as that has been checked for at the very start
                        #     df_step.iloc[index_of_interest-1,5] = 'Appearance Error'
            
        ###   ------------------      Step 3: Save to dataframe file     ---------------  ###

            

        print('Number of cluster at timepoint', centres_2D_current.shape[0])

        # volume_3D_current = (4/3)*area_2D_current*np.sqrt(area_2D_current/math.pi)
        # volume_3D_current = np.round(np.multiply(np.multiply(area_2D_current,(4/3)),np.sqrt(np.divide(area_2D_current,math.pi))))
        volume_3D_current = (4/3)*math.pi*(((np.divide(np.sqrt(area_2D_current),math.sqrt(189*math.pi))))**3)

        # Save columns to df_step
        df_step.iloc[:,0] = tag_number_current
        df_step.iloc[:,1] = area_2D_current
        df_step.iloc[:,2] = volume_3D_current
        df_step.iloc[:,3] = centres_2D_current[:,0].tolist()
        df_step.iloc[:,4] = centres_2D_current[:,1].tolist()
        df_step.iloc[:,7] = current_timing
        df_step.iloc[:,8] = base_str
        df_step.iloc[:,9] = experiment_id


        # Update previous timestep variables to be current variables for use in next timepoint
        df_old = df_step
        non_assigned_old_tags_list = tag_number_current
        array_index_old = array_index_current_time
        # Update centres_2D_old for use in next timestep
        centres_2D_old = centres_2D_current

        print('Yay step', ticker, ' finished')

        # Save dataframe output for current timepoint to .csv file
        df_total_areas = pd.DataFrame(df_step)
        df_step_csv_name_list = basedir, 'specialised_pipeline/', 'post_processing_output/', base_str, '/', filename, '_post_processing', '.csv' 
        df_step_name_list_2  =''.join(df_step_csv_name_list)
        df_total_areas.to_csv(df_step_name_list_2, index=False, header=True)

        t_after_step = time.time()

        t_tot_step = t_after_step - t_before_step

        print('Time for step', ticker, t_tot_step)

        ticker += 1



t_after = time.time()

t_tot = t_after - t_before

print('Total time to run', t_tot)