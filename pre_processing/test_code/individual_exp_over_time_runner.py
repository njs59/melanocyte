import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from scipy.ndimage import binary_fill_holes, label
from scipy.ndimage import sum as ndi_sum

# Use existing imports from user's code
import read_tif_file_operator as tif  # Custom module
import pre_pro_operators as pre_oper  # Custom module

# Parameters
basedir = '/Users/Nathan/Documents/Oxford/DPhil/melanocyte/'
data_folder = 'data/Still_Images_with_BF_for_Nathan/'
fileID = '.tif'
time = 0
min_clus_size = 150

# Hardcoded filenames for 5 days of one experiment
filenames = [
    'VID289_A5_1_00d00h00m',
    'VID289_A5_1_01d00h00m',
    'VID289_A5_1_02d00h00m',
    'VID289_A5_1_03d00h00m',
    'VID289_A5_1_04d00h00m',
    'VID289_A5_1_05d00h00m'
]

total_areas = []
mean_areas = []
num_clusters = []

# Use the same cropping function as before
def apply_rectangular_fov(arr, rect_width=1000, rect_height=800):
    center_x, center_y = arr.shape[1] // 2, arr.shape[0] // 2
    start_x = center_x - rect_width // 2
    start_y = center_y - rect_height // 2
    return arr[start_y:start_y + rect_height, start_x:start_x + rect_width]

# Process each day
for filename in filenames:
    raw_arr_2D_1, raw_arr_2D_2, raw_arr_2D_3 = tif.tif_to_arr(basedir, data_folder, filename, str(time), fileID)
    arr2_normal = apply_rectangular_fov(raw_arr_2D_2.astype(int))

    thresh_otsu_2 = filters.threshold_otsu(arr2_normal)
    array_2 = arr2_normal < thresh_otsu_2
    current_array = binary_fill_holes(array_2).astype(int)
    label_arr, num_clus = label(current_array)

    area_list = ndi_sum(current_array, labels=label_arr, index=np.arange(label_arr.max() + 1))
    area_new, index_keep = pre_oper.remove_fragments(area_list, num_clus, min_clus_size)

    total_areas.append(np.sum(area_new))
    mean_areas.append(np.mean(area_new))
    num_clusters.append(len(area_new))

# Print summary statistics
print("Summary Statistics Over 5 Days:")
for i, filename in enumerate(filenames):
    print(f"Day {i+1} ({filename}):")
    print(f"  Total Area: {total_areas[i]}")
    print(f"  Mean Cluster Area: {mean_areas[i]:.2f}")
    print(f"  Number of Clusters: {num_clusters[i]}")

# Plot statistics over time
days = list(range(0, 6))



# Plot Total Area
plt.figure(figsize=(10, 6))
plt.plot(days, total_areas, marker='o', color='skyblue')
plt.title('Total Cluster Area Over 5 Days')
plt.xlabel('Day')
plt.ylabel('Total Area')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Mean Cluster Area
plt.figure(figsize=(10, 6))
plt.plot(days, mean_areas, marker='s', color='lightgreen')
plt.title('Mean Cluster Area Over 5 Days')
plt.xlabel('Day')
plt.ylabel('Mean Area')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Number of Clusters
plt.figure(figsize=(10, 6))
plt.plot(days, num_clusters, marker='^', color='salmon')
plt.title('Number of Clusters Over 5 Days')
plt.xlabel('Day')
plt.ylabel('Number of Clusters')
plt.grid(True)
plt.tight_layout()
plt.show()
