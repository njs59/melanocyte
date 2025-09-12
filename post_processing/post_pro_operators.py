import math
import numpy as np

## Calculate centres of mass from arrays
def calc_clus_centre(labelled_arr):
    '''
    Calculate centres of clusters from a 2D index labelled array

    Input:
        labelled_arr: 2D indexed labelled array

    Output:
        centres_2D: 2D array of shape (n,2) storing the centres of the different clusters coordinates
                        in x,y directions to the nearest integer
    '''

    centres = np.array([])
    # Loop over clusters
    for i in range(1,labelled_arr.max()+1):
        # Find all locations of cluster with given index
        locs_of_size = np.transpose((labelled_arr==i).nonzero())
        # Calculate mean
        centre_of_mass = locs_of_size.mean(axis=0)
        # Round to nearest integer to store single location
        centre_of_mass = np.rint(centre_of_mass)
        centres = np.append(centres, centre_of_mass)

    # Convert array to shape (n,2)
    centres_2D = np.reshape(centres, (-1, 2))
    return centres_2D


def previous_clusters_at_loc(labelled_arr, centres_old, comparison_index):
    '''
    Compares current locations of a cluster with previous cluster centres to define 
        cluster lineage and events

    Inputs:
        labelled_arr: Labelled 2D array at current time
        centres_old: (n,2) array for cluster centres at previous timepoint
        comparison_index: Index of labelled array for cluster of interest

    Outputs:
        same_locs: Number of cluster centres from previous timestep in same location as cluster
        same_locs_centre: Cluster centres from previous timestep in same location as cluster
    '''

    same_locs = 0
    same_locs_store = np.array([])

    # Get list of coordinates corresponding to the comparison index
    d = np.nonzero(labelled_arr == comparison_index)
    d_x_min = d[0][0]
    d_x_max = d[0][-1]
    # Loop over previous timestep cluster centres
    for n in range(centres_old.shape[0]):
        if centres_old[n,0] > d_x_max:
            continue
        elif centres_old[n,0] < d_x_min:
            continue
        else:
        # Loop over all coordinates of the cluster
            for m in range(len(d[0])):
                # Compare coordinate to cluster centres
                if d[0][m] == centres_old[n,0] and d[1][m] == centres_old[n,1]:
                    same_locs += 1 # Centre and coordinate match

                    # If centre and coordinate match then store the location
                    if same_locs == 1:
                        same_locs_store = np.append(same_locs_store, centres_old[n,:])
                    else:
                        same_locs_store = np.vstack([same_locs_store, centres_old[n,:]])


    return same_locs, same_locs_store


def previous_clusters_at_loc_2(labelled_arr, centres_old, comparison_index):
    '''
    Compares current locations of a cluster with previous cluster centres to define 
        cluster lineage and events

    Uses vectorised version for updating array to only have the comparison index

    Inputs:
        labelled_arr: Labelled 2D array at current time
        centres_old: (n,2) array for cluster centres at previous timepoint
        comparison_index: Index of labelled array for cluster of interest

    Outputs:
        same_locs: Number of cluster centres from previous timestep in same location as cluster
        same_locs_centre: Cluster centres from previous timestep in same location as cluster
    '''

    same_locs = 0
    same_locs_store = np.array([])


    applyall = np.vectorize(single_index_bool)
    single_index_slice = applyall(labelled_arr, comparison_index)
    for n in range(centres_old.shape[0]):
        # Compare coordinate to cluster centres
        if single_index_slice[int(centres_old[n,0])][int(centres_old[n,1])] == 1:
            same_locs += 1 # Centre and coordinate match

            # If centre and coordinate match then store the location
            if same_locs == 1:
                same_locs_store = np.append(same_locs_store, [int(centres_old[n,0]), int(centres_old[n,1])])
            else:
                same_locs_store = np.vstack([same_locs_store, [int(centres_old[n,0]), int(centres_old[n,1])]])

    return same_locs, same_locs_store


# Function to give bool array for single index
def single_index_bool(arr, val):
    if arr == val:
        return 1
    else:
        return 0


def nearby_clusters(x_loc, y_loc, search_radius, labelled_arr):
    max_x, max_y = np.shape(labelled_arr)
    search_arr = labelled_arr[max(x_loc - search_radius, 0) : min(x_loc + search_radius + 1, max_x),
                              max(y_loc - search_radius, 0) : min(y_loc + search_radius + 1, max_y)]
    # search_arr[search_arr == index] = 0
    clusters_index_present = np.unique(search_arr)
    #Remove 0's from consideration
    clusters_index_output = clusters_index_present[1:]

    distances = []
    for i in range(1,len(clusters_index_present)):
        # if len(clusters_index_present) > 2:
        #     print('Hello')
        # Looping this way ignores the 0's present

        list_of_locs = np.where(labelled_arr == clusters_index_present[i])
        # list_of_locs = np.where(search_arr == clusters_index_present[i])
        min_dist = math.inf
        for k in range(len(list_of_locs[0])):
            # Loop over each element of cluster visible
            dist_x = abs(list_of_locs[0][k] - x_loc)
            dist_y = abs(list_of_locs[1][k] - y_loc)

            dist = dist_x + dist_y
            min_dist = min(min_dist, dist)

        distances = np.append(distances, min_dist)

    return clusters_index_output, distances


def pick_cluster_inverse_dist(clusters_index_output, distances):
    '''
    Select a cluster index based on chance based in inverse distances

    Input:
        clusters_index_output:  List of possible clusters' indexes to select from
        distances:              Distances from site of interest

    Output:
    cluster_selected: Cluster index randomly sampled with weight inverse to distance from site of interest
    '''
    if 0 in distances:
        # If a 0 appears in list select the corresponding index
        position = np.where(distances == 0)
        return clusters_index_output[position]
    else:
        weights = np.reciprocal(distances)     # Invert all distances
        weights = weights / np.sum(weights)         # Normalize
        cluster_selected = np.random.choice(clusters_index_output, p=weights) # Sample
        # Return the cluster index randomly sampled with weight inverse to distance from site of interest
        return int(cluster_selected)

# def calc_clus_centre(labelled_arr, index_keep):
# ## Returns a list of centres of clusters with desired indexes from a labelled array
#     centres = np.array([])
#     for i in range(len(index_keep)):
#         locs_of_size = np.transpose((labelled_arr==index_keep[i]).nonzero())
#         centre_of_mass = locs_of_size.mean(axis=0)
#         centre_of_mass = np.rint(centre_of_mass)
#         centres = np.append(centres, centre_of_mass)

#     centres_2D = np.reshape(centres, (-1, 2))
#     return centres_2D