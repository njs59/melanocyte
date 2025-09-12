import numpy as np
import pandas as pd
from ast import literal_eval

import matplotlib.pyplot as plt


basedir = '/Users/Nathan/Documents/Oxford/DPhil/'
exp_type = 'In_vitro_homogeneous_data/'
exp_date = '2017-02-03'
time_array = range(1,98)
num_times = len(time_array)
# Rename single digit values with 0 eg 1 to 01 for consistency
time_list = [str(x).zfill(2) for x in time_array]
well_loc = 's11'

cols = ["Tag number", "Cluster size", "Cluster Centre x", "Cluster Centre y", 
           "Event", "Clusters in event", "Timestep", "Date", "Well ID"]

df_end_now_csv_name_list = basedir, exp_type, 'post_processing_output/', exp_date, '/', well_loc, 't', '97', 'c2_post_processing', '.csv'
df_end_now_csv_name_list_2  =''.join(df_end_now_csv_name_list)
df_end_now = pd.read_csv(df_end_now_csv_name_list_2)
cluster_tags = df_end_now["Tag number"].to_numpy().astype(int)

# cluster_lineage = [56]
# cluster_lineage = [125]
cluster_lineage = [103]
cluster_lineage_partial = []
partial_ticker = 0
# for h in range(len(cluster_tags)):
#     cluster_lineage = [cluster_tags[h]]

for i in range(97, 50, -1) :
    print('i is', i)
    time_i = str(i).zfill(2)
    df_step_csv_name_list = basedir, exp_type, 'post_processing_output/', exp_date, '/', well_loc, 't', time_i, 'c2_post_processing', '.csv'
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

print(cluster_lineage)

# Find locations of clusters at time 30
df_step_csv_name_list = basedir, exp_type, 'post_processing_output/', exp_date, '/', well_loc, 't', '51', 'c2_post_processing', '.csv'
df_step_csv_name_list_2  =''.join(df_step_csv_name_list)
df_step_interest = pd.read_csv(df_step_csv_name_list_2)

rows_of_interest = pd.DataFrame(columns=cols)
for k in range(len(cluster_lineage)):
    new_row_of_interest = df_step_interest.loc[df_step['Tag number'] == cluster_lineage[k]]

    # to append df2 at the end of df1 dataframe
    rows_of_interest = pd.concat([rows_of_interest, new_row_of_interest])


# Store centres in array
df1 = rows_of_interest[['Cluster Centre x', 'Cluster Centre y']]
centres_2D_lineage = df1.to_numpy()

# Read in array for given timestep

# Locate indexes of clusters, print
index_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, 't', '51', 'c2', '_indexed', '.csv'
index_csv_name_list_2  =''.join(index_csv_name_list)
df_slice = pd.read_csv(index_csv_name_list_2, header=None)
current_array = df_slice.to_numpy()

indexes_lineage = []
for p in range(centres_2D_lineage.shape[0]):
    index_of_interest = current_array[int(centres_2D_lineage[p,0]),int(centres_2D_lineage[p,1])]
    indexes_lineage = np.append(indexes_lineage, index_of_interest)

descendents_arr = np.array([])

for q in range(len(indexes_lineage)):
    descendents_locs = np.where(current_array == int(indexes_lineage[q]))
    single_descendent_arr = np.asarray(descendents_locs)

    if q == 0:
        descendents_arr = single_descendent_arr
    else:
        descendents_arr = np.append(descendents_arr, single_descendent_arr, axis=1)



print('Yay')

bool_descendents = np.zeros(current_array.shape)

if descendents_arr.shape[0] > 0:
    for r in range(descendents_arr.shape[1]):
        bool_descendents[descendents_arr[0,r], descendents_arr[1,r]] = 1





    # Find locations of clusters at time 30
    df_end_csv_name_list = basedir, exp_type, 'post_processing_output/', exp_date, '/', well_loc, 't', '97', 'c2_post_processing', '.csv'
    df_end_csv_name_list_2  =''.join(df_end_csv_name_list)
    df_end_interest = pd.read_csv(df_end_csv_name_list_2)



    cluster_current_row_of_interest = df_end_interest.loc[df_end_interest['Tag number'] == cluster_lineage[0]]
    # Store centres in array
    df1_end = cluster_current_row_of_interest[['Cluster Centre x', 'Cluster Centre y']]
    centres_end_2D_lineage = df1_end.to_numpy()

    # Read in array for given timestep

    # Locate indexes of clusters, print
    end_index_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, 't', '97', 'c2', '_indexed', '.csv'
    end_index_csv_name_list_2  =''.join(end_index_csv_name_list)
    df_end_slice = pd.read_csv(end_index_csv_name_list_2, header=None)
    current_array_end = df_end_slice.to_numpy()

    end_index = current_array_end[int(centres_end_2D_lineage[0][0]),int(centres_end_2D_lineage[0][1])]

    descendents_arr = []


    index_locs = np.where(current_array_end == int(end_index))
    single_index_arr = np.asarray(index_locs)


    bool_index = np.zeros(current_array_end.shape)

    for r in range(single_index_arr.shape[1]):
        bool_index[single_index_arr[0,r], single_index_arr[1,r]] = 1


    plt.figure()

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2) 

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(bool_descendents)
    axarr[1].imshow(bool_index)
    axarr[0].axis([0, current_array.shape[1], 0, current_array.shape[0]])
    axarr[1].axis([0, current_array.shape[1], 0, current_array.shape[0]])
    plt.show()



    plt.imshow(bool_index)
    plt.axis([0, current_array.shape[1], 0, current_array.shape[0]])
    plt.show()




    plt.imshow(bool_descendents)
    plt.axis([0, current_array.shape[1], 0, current_array.shape[0]])
    plt.show()