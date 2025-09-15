import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re

from skimage import filters

def threshold_arr_supervised(tf_array, threshold, dim):
    '''
    Thresholds 3D array to give a boolean array of values above and below threshold

    Inputs:
      tf_array: 3D array to be thresholded
      threshold: Value above which position is assigned a 1 in boolean array

    Outputs:
      tf_array_bool: 3D boolean array after thresholdeing has occured
    '''
    tf_ad = tf_array

    vfunc = np.vectorize(bool_threshold_val)
    tf_array_bool = vfunc(tf_ad, threshold)

    if dim == 3:
      # reshaping the array from 3D
      # matrix to 2D matrix.
      arr_reshaped = tf_array_bool.reshape(tf_array_bool.shape[0], -1)
  
      # saving reshaped array to file.
      np.savetxt("/Users/Nathan/Documents/Oxford/DPhil/current_tiff.txt", arr_reshaped)



      # Checks:
      # retrieving data from file.
      loaded_arr = np.loadtxt("/Users/Nathan/Documents/Oxford/DPhil/current_tiff.txt")
  
      load_original_arr = loaded_arr.reshape(
          loaded_arr.shape[0], loaded_arr.shape[1] // tf_array_bool.shape[2], tf_array_bool.shape[2])
      
      # check the shapes:
      print("shape of arr: ", tf_array_bool.shape)
      print("shape of load_original_arr: ", load_original_arr.shape)
      
      # check if both arrays are same or not:
      if (load_original_arr == tf_array_bool).all():
          print("Yes, both the arrays are same")
      else:
          print("No, both the arrays are not same")
    else:
       print('Not 3D')


    return tf_array_bool

def bool_threshold_val(a, threshold):
    '''Return 1 if a > threshold, otherwise return 0'''
    if a > threshold:
        return 1
    else:
        return 0

def threshold_arr_unsupervised(tf_array):
    '''
    Thresholds 3D array to give a boolean array of values above and below threshold

    Inputs:
      tf_array: 3D array to be thresholded
      threshold: Value above which position is assigned a 1 in boolean array

    Outputs:
      tf_array_bool: 3D boolean array after thresholdeing has occured
    '''
    for i in range(tf_array.shape[2]):
      tf_ad = tf_array[:,:,i]

      text_threshold = filters.threshold_otsu  # Hit tab with the cursor after the underscore, try several methods
      thresh = text_threshold(tf_ad)
      array_i = tf_ad > thresh

      if i == 0:
         tf_array_bool = array_i
      else:
         tf_array_bool = np.dstack((tf_array_bool, array_i))

    
    # reshaping the array from 3D
    # matrix to 2D matrix.
    arr_reshaped = tf_array_bool.reshape(tf_array_bool.shape[0], -1)
 
    # saving reshaped array to file.
    np.savetxt("/Users/Nathan/Documents/Oxford/DPhil/current_tiff.txt", arr_reshaped)



    # Checks:
    # retrieving data from file.
    loaded_arr = np.loadtxt("/Users/Nathan/Documents/Oxford/DPhil/current_tiff.txt")
 
    load_original_arr = loaded_arr.reshape(
        loaded_arr.shape[0], loaded_arr.shape[1] // tf_array_bool.shape[2], tf_array_bool.shape[2])
    
    # check the shapes:
    print("shape of arr: ", tf_array_bool.shape)
    print("shape of load_original_arr: ", load_original_arr.shape)
    
    # check if both arrays are same or not:
    if (load_original_arr == tf_array_bool).all():
        print("Yes, both the arrays are same")
    else:
        print("No, both the arrays are not same")


    return tf_array_bool



def remove_fragments(area, num_clus, min_clus_size):
    '''
    Removes clusters from list of areas that fall below a minimum size.
    This is used to remove clusters that are smaller than a cell and so
      are comprised only of cell fragments.

    Inputs:
      area:           1D array of current candidate cluster areas
      num_clus:       Number of candidate clusters
      min_clus_size:  Minimum area needed for a candidate cluster to be accepted

    Outputs:
      area_new:   1D array of accepted cluster areas
      index_keep: List of indices to keep (by position in area input array)
    '''

    # Get a list of indexes for slice of array that are
    # not large enough to be considered a cluster
    index_to_del = np.argwhere(area < min_clus_size)
    # Delete the corresponding areas from area array
    area_new = np.delete(area, index_to_del)

    # Get list of possible indices
    index_keep = np.arange(num_clus+1)
    # Delete indices corresponding to too small clusters
    for i in range(len(index_to_del)):
        index_keep = np.delete(index_keep, np.where(index_keep == index_to_del[i]))


    return area_new, index_keep


def save_clus_areas(i, area_new, cluster_areas):
    '''
    Add a new row of cluster sizes to the 2D array storing all cluster sizes over time.
      Increasing the dimensions of the 2 arrays to match by adding 0's where necessary

    Inputs:
      i: timepoint
      area_new: 1D array of areas at current timepoint
      cluster_areas: 2D array of areas at previous timpoints

    Outputs:
      cluster_areas: 2D array updated with new areas at current time
    '''
    

    if i == 0:
      # If we're at the initial time just update the empty array with new areas
      cluster_areas = np.append(cluster_areas, area_new, axis=0)

    # Timestep 1 considered separately because we compare with 1D array for 
      # previous cluster areas rather than 2D array
    elif i == 1:

      # Print 2 sizes to compare (number of new clusters, number of old clusters)
      print('Compare', len(area_new), cluster_areas.shape[0])

      # If less clusters than previously
      if len(area_new) < cluster_areas.shape[0]: 
        # Get array of 0's of same length as previous clusters
        area_to_add = np.zeros((1,cluster_areas.shape[0]))
        # Assign new areas to first n elements of array
        for n in range(len(area_new)):
          area_to_add[0,n] = area_new[n]
        # Stack vertically into 2D array
        cluster_areas = np.vstack((cluster_areas, area_to_add))

      # If more clusters than previously  
      else:
        # Calculate how many more clusters are present
        extra_clusters = len(area_new) - cluster_areas.shape[0]
        for v in range(extra_clusters):
          # Add zeros to cluster_areas array to match array sizes
          cluster_areas = np.append(cluster_areas,0)
        # Now add the new data
        area_to_add = np.zeros((1,cluster_areas.shape[0]))
        for n in range(len(area_new)):
          area_to_add[0,n] = area_new[n]
        # Stack vertically into 2D array
        cluster_areas = np.vstack((cluster_areas, area_to_add))
      # Print current shape of 2D array
      print('Shape clus', cluster_areas.shape)

    # Case where i > 1
    else:
      # Print 2 sizes to compare (number of new clusters, max number of old clusters)
      print('Compare', len(area_new), cluster_areas.shape[1])

      # If less clusters than previous max number of clusters
      if len(area_new) < cluster_areas.shape[1]:
        # Get array of 0's of same length as previous clusters
        area_to_add = np.zeros((1,cluster_areas.shape[1]))
        # Assign new areas to first n elements of array
        for n in range(len(area_new)):
          area_to_add[0,n] = area_new[n]
        # Stack vertically in 2D array
        cluster_areas = np.vstack((cluster_areas, area_to_add))

      # If more clusters than previous max number of clusters
      else:
        # Calculate how many more clusters are present
        extra_clusters = len(area_new) - cluster_areas.shape[1]
        # Get array of zeros to add to current array to match array sizes
        extra_zeros = np.zeros((cluster_areas.shape[0], extra_clusters))
        # Add zeros to 2D array by horizontally stacking
        cluster_areas = np.hstack((cluster_areas, extra_zeros))
        # Now add the new data
        area_to_add = np.zeros((1,cluster_areas.shape[1]))
        for n in range(len(area_new)):
          area_to_add[0,n] = area_new[n]
        # Stack vertically into 2D array
        cluster_areas = np.vstack((cluster_areas, area_to_add))
      # Print current shape of 2D array
      print('Shape clus', cluster_areas.shape)

    return cluster_areas


def update_hist(num, data):
    '''
    Plots next histogram

    Input:
      num: row of data to plot
      data: 2D array of data
    '''
    plt.cla()
    plt.gca()
    # plt.set_ylim([0,60])
    plt.axis([150,25000,0,200])
    plt.hist(data[num,:], bins=[200, 600, 1000, 2000, 4000, 8000, 16000, 25000])



def update_heat_map(data):
  '''
  Plots next heat map figure

  Input:
  data: 2D array to be plotted in heat map form
  '''

  my_cmap = mpl.colormaps['spring']
  my_cmap.set_under('k')
  plt.imshow(data, cmap=my_cmap, vmin = 1)
  plt.axis([0, data.shape[1], 0, data.shape[0]])
  plt.colorbar()
  plt.show()



def threshold_arr_2D(tf_array, threshold):
    '''
    2D version of thresholding 

    Inputs:
      tf_array: 2D array
      threshold: Value for thresholding

    Outputs:
      tf_array_bool: Boolean array post-thresholding
    '''

    tf_ad = tf_array

    vfunc = np.vectorize(bool_threshold_val)
    tf_array_bool = vfunc(tf_ad, threshold)


    return tf_array_bool



def generate_filenames(experiment_id="B4", base="VID289"):
    filenames = []

    # Day 0: every 3 hours
    for hour in range(6, 24, 3):
        filename = f"{base}_{experiment_id}_1_00d{hour:02d}h00m"
        filenames.append(filename)

    # Days 1 to 4: only 00h
    for day in range(1, 6):
        filename = f"{base}_{experiment_id}_1_{day:02d}d00h00m"
        filenames.append(filename)

    return filenames


# Extract time in days from filenames using regular expressions
def extract_days(filenames):
    days = []
    for name in filenames:
        match = re.search(r"(\d{2})d(\d{2})h", name)
        if match:
            day = int(match.group(1))
            hour = int(match.group(2))
            day_fraction = day + hour / 24.0
            days.append(day_fraction)
    return days

# Use the same cropping function as before
def apply_rectangular_fov(arr, rect_width=1000, rect_height=800):
    center_x, center_y = arr.shape[1] // 2, arr.shape[0] // 2
    start_x = center_x - rect_width // 2
    start_y = center_y - rect_height // 2
    return arr[start_y:start_y + rect_height, start_x:start_x + rect_width]

