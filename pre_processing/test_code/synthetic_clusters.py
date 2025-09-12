import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from scipy.ndimage import *
from pylab import *

import pre_pro_operators as pre_oper

np.random.seed(2000)

min_clus_size = 3

x = np.random.rand(5, 5, 5)

x *= 400
x += 150

x[0,0] += 200
x[0,1] += 200
x[0,2] += 300
x[2,1] += 200
x[1,3] -= 200
x[2,4] -= 200
arr_for_thresholding = x.astype(int)

expected_boolean = [[1,1,1,1,0],[0,0,0,0,1],[0,1,1,1,0],[0,1,0,1,1],[1,1,1,1,0]]
expected_no_holes_boolean = [[1,1,1,1,0],[0,0,0,0,1],[0,1,1,1,0],[0,1,1,1,1],[1,1,1,1,0]]
expected_index = [[1,1,1,1,0],[0,0,0,0,2],[0,3,3,3,0],[0,3,3,3,3],[3,3,3,3,0]]
expected_areas = [0,4,1,11]
expected_kept_areas = [4,11]
expected_area_output = [[4,4,4,4,0],[0,0,0,0,0],[0,11,11,11,0],[0,11,11,11,11],[11,11,11,11,0]]
expected_index_output = [[1,1,1,1,0],[0,0,0,0,0],[0,2,2,2,0],[0,2,2,2,2],[2,2,2,2,0]]


# plt.imshow(expected_no_holes_boolean)
# plt.show()
print(arr_for_thresholding)

threshold = 440
# # Threshold 3D array to boolean array
tf_bool_3D = pre_oper.threshold_arr(arr_for_thresholding, threshold)


current_array_holes = tf_bool_3D[:,:,0]

plt.imshow(current_array_holes)
plt.show()


current_array_no_holes = binary_fill_holes(current_array_holes).astype(int)

label_arr, num_clus = label(current_array_no_holes)

area_list = ndimage.sum(current_array_no_holes, label_arr, index=arange(num_clus + 1))


area_arr = label_arr

global area_new, index_keep
area_new, index_keep = pre_oper.remove_fragments(area_list, num_clus, min_clus_size)

# Function to update indexed array to array displaying areas
def update_arr(arr):
    global area_new, index_keep
    index = np.where(index_keep == arr)
    if len(index[0]) != 0:        
        i = area_new[index[0]][0]
    else:
        i = 0
    return i


applyall = np.vectorize(update_arr)
area_slice = applyall(area_arr)

##### Re-binarize array
slice_binary = np.array(area_slice, dtype=bool).astype(int)

output_label_arr, nc = label(slice_binary)


# check if both arrays are same or not:
if (expected_boolean == current_array_holes).all():
    print("Yes, both the initial boolean arrays are the same")
else:
    print("No, both the initial boolean arrays are not the same")

# check if both arrays are same or not:
if (expected_no_holes_boolean == current_array_no_holes).all():
    print("Yes, both the no hole boolean arrays are the same")
else:
    print("No, both the no hole boolean arrays are not the same")

# check if both arrays are same or not:
if (expected_index == label_arr).all():
    print("Yes, both the initial labelled arrays are the same")
else:
    print("No, both the initial labelled arrays are not the same")

# check if both arrays are same or not:
if (expected_areas == area_list).all():
    print("Yes, both the initial area lists are the same")
else:
    print("No, both the initial area lists are not the same")

# check if both arrays are same or not:
if (expected_kept_areas == area_new).all():
    print("Yes, both the kept area lists are the same")
else:
    print("No, both the kept area lists are not the same")

# check if both arrays are same or not:
if (expected_area_output == area_slice).all():
    print("Yes, both the kept area outputs are the same")
else:
    print("No, both the kept area ousputs are not the same")

# check if both arrays are same or not:
if (expected_index_output == output_label_arr).all():
    print("Yes, both the kept area lists are the same")
else:
    print("No, both the kept area lists are not the same")


