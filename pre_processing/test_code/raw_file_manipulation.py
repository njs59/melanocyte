import matplotlib.pyplot as plt
import numpy as np

from skimage import filters

import read_tif_file_operator as tif

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
filename = 'VID289_D5_1_01d00h00m'

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


# Sum the arrays
raw_arr_2D_tot = arr1_normal + arr2_normal + arr3_normal


# raw_arr_2D_tot




plt.hist(raw_arr_2D_tot)
plt.show()

# plt.hist(raw_arr_2D_1)
# plt.show()
# plt.hist(raw_arr_2D_2)
# plt.show()
# plt.hist(raw_arr_2D_3)
# plt.show()

thresh = 690

# Find locations where the total array exceeds the threshold
mask = raw_arr_2D_tot > thresh

# mask = raw_arr_2D_1 == raw_arr_2D_1.max() and raw_arr_2D_2 == raw_arr_2D_2.max() and raw_arr_2D_3 == raw_arr_2D_3.max()

# Set those locations to 0 in all arrays
# raw_arr_2D_tot[mask] = 0
# # raw_arr_2D_1[mask] = 0
# # raw_arr_2D_2[mask] = 0
# # raw_arr_2D_3[mask] = 0
# arr1_normal[mask] = 0
# arr2_normal[mask] = 0
# arr3_normal[mask] = 0


plt.hist(raw_arr_2D_tot)
plt.show()

# plt.imshow(raw_arr_2D_tot)
# plt.show()

# plt.imshow(raw_arr_2D_1)
# plt.show()

# plt.imshow(raw_arr_2D_2)
# plt.show()

# plt.imshow(raw_arr_2D_3)
# plt.show()

text_threshold = filters.threshold_otsu  # Hit tab with the cursor after the underscore, try several methods
# text_threshold_otsu = filters.threshold_otsu
# thresh_otsu_1 = text_threshold(raw_arr_2D_1)
# thresh_otsu_2 = text_threshold(raw_arr_2D_2)
# thresh_otsu_3 = text_threshold(raw_arr_2D_3)

# array_1 = raw_arr_2D_1 > thresh_otsu_1
# array_2 = raw_arr_2D_2 > thresh_otsu_2
# array_3 = raw_arr_2D_3 > thresh_otsu_3

thresh_otsu_1 = text_threshold(arr1_normal)
thresh_otsu_2 = text_threshold(arr2_normal)
thresh_otsu_3 = text_threshold(arr3_normal)

array_1 = arr1_normal > thresh_otsu_1
array_2 = arr2_normal > thresh_otsu_2
array_3 = arr3_normal > thresh_otsu_3

array_otsu_tot = array_1.astype(int) + array_2.astype(int) + array_3.astype(int)


print("Thresholds are", thresh_otsu_1, thresh_otsu_2, thresh_otsu_3)

plt.imshow(array_otsu_tot, cmap='viridis')
plt.axis('off')
plt.savefig('Tori_Multi_Otsu_Manipulated.png', dpi=300)

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


