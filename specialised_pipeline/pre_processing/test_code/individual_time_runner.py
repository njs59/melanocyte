import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from skimage import filters
from scipy.ndimage import binary_fill_holes, label
from scipy.ndimage import sum as ndi_sum
from numpy import arange

import read_tif_file_operator as tif  # Custom module
import pre_pro_operators as pre_oper  # Custom module
# from pre_oper import calc_area_arr  # If calc_area_arr is defined there

min_clus_size = 150

# Function to update indexed array to array displaying areas
def calc_area_arr(arr):
    global area_new, index_keep
    index = np.where(index_keep == arr)
    if len(index[0]) != 0:        
        i = area_new[index[0]][0]
    else:
        i = 0
    return i

def apply_rectangular_fov(arr, rect_width=400, rect_height=300):
    """
    Crops a rectangular field of view from the center of a 2D array.

    Parameters:
    - arr (np.ndarray): Input 2D array (e.g., image or data grid).
    - rect_width (int): Width of the rectangular field of view.
    - rect_height (int): Height of the rectangular field of view.

    Returns:
    - np.ndarray: Cropped array containing only the rectangular region.
    """
    center_x, center_y = arr.shape[1] // 2, arr.shape[0] // 2
    start_x = center_x - rect_width // 2
    start_y = center_y - rect_height // 2

    return arr[start_y:start_y + rect_height, start_x:start_x + rect_width]




###    -----------   Input parameters   --------------     ###

basedir = '/Users/Nathan/Documents/Oxford/DPhil/melanocyte/'

data_folder = 'data/Still_Images_with_BF_for_Nathan/'
filename = 'VID289_A5_1_00d00h00m'

filnames = ['VID289_A5_1_01d00h00m','VID289_B4_1_01d00h00m','VID289_D5_1_01d00h00m','VID289_E2_1_01d00h00m']

fileID = '.tif'
# time_list = range(42,98,5)
# well_loc = 's073'

time_list = ''


# for i in range(97,98,1):
#     time = i
time = 0

raw_arr_2D_1, raw_arr_2D_2, raw_arr_2D_3 = tif.tif_to_arr(basedir, data_folder, filename, str(time), fileID)


# Convert to default integer type before summing to avoid overflow
arr1_normal = raw_arr_2D_1.astype(int)
arr2_normal = raw_arr_2D_2.astype(int)
arr3_normal = raw_arr_2D_3.astype(int)




# Assuming `arr` is a 2D NumPy array
arr1_normal = apply_rectangular_fov(arr1_normal, 1000, 800)
arr2_normal = apply_rectangular_fov(arr2_normal, 1000, 800)
arr3_normal = apply_rectangular_fov(arr3_normal, 1000, 800)


text_threshold = filters.threshold_otsu  # Hit tab with the cursor after the underscore, try several methods

thresh_otsu_1 = text_threshold(arr1_normal)
thresh_otsu_2 = text_threshold(arr2_normal)
thresh_otsu_3 = text_threshold(arr3_normal)

array_1 = arr1_normal < thresh_otsu_1
array_2 = arr2_normal < thresh_otsu_2
array_3 = arr3_normal < thresh_otsu_3

plt.imshow(array_1, cmap='gray')
plt.axis('off')
plt.show()
# plt.savefig('Tori_Multi_Otsu_Manipulated.png', dpi=300)

plt.imshow(array_2, cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(array_3, cmap='gray')
plt.axis('off')
plt.show()


# Fill any single pixel holes in clusters
current_array = binary_fill_holes(array_3).astype(int) 

# Label the clusters of the boolean array
label_arr, num_clus = label(current_array)

# plt.imshow(label_arr, interpolation=None)
# plt.show()



area_list = ndi_sum(current_array, labels=label_arr, index=np.arange(label_arr.max() + 1))


# # Get a 1D list of areas of the clusters
# area_list = sum(current_array, label_arr, index=arange(label_arr.max() + 1))

area_arr = label_arr

global area_new, index_keep
# Remove fragments, that is clusters smaller than
# min_clus_size so we only consider clusters large enough
# to be a cell rather than a cell fragment
area_new, index_keep = pre_oper.remove_fragments(area_list, num_clus, min_clus_size)

# Calculate and store total area now in clusters, number of clusters and mean cluster area 
total_curr_area = np.sum(area_new)
print('Total current area', total_curr_area)
mean_curr_area = np.mean(area_new)
print('Mean cluster area', mean_curr_area)


# Print total number of remaining clusters
print("Total number of clusters after removing fragments:", len(area_new))

# Plot histogram of cluster sizes
plt.hist(area_new, bins='auto', color='skyblue', edgecolor='black')
plt.title('Histogram of Cluster Sizes')
plt.xlabel('Cluster Area')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
# plt.savefig("cluster_size_histogram.png")
plt.show()



# Calculate 2D cluster array with cluster areas rather than indeces
applyall = np.vectorize(calc_area_arr)
area_slice = applyall(area_arr)

# Plot area array heatmap on a log scale and save figure
my_cmap = mpl.colormaps['spring']
my_cmap.set_under('k')
plt.imshow(area_slice, cmap=my_cmap, vmin=1)
plt.axis([0, area_slice.shape[1], 0, area_slice.shape[0]])
plt.colorbar()
# plt.savefig(f'{basedir}images/frame-{i:03d}.png', bbox_inches='tight', dpi=300)
plt.show()
plt.clf()




