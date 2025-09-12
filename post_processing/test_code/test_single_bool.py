import numpy as np

# Function to give bool array for single index
def single_index_bool(arr, val):
    if arr == val:
        return 1
    else:
        return 0

area_arr = np.array([[1, 2, 3], [4, 5, 6]])   

applyall = np.vectorize(single_index_bool)
area_slice = applyall(area_arr, 1)

print(area_slice)


att_2 = np.nonzero(area_arr == 1)

print(att_2)