import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import filters
from scipy.ndimage import binary_fill_holes, label
from scipy.ndimage import sum as ndi_sum
import read_tif_file_operator as tif  # Custom module
import pre_pro_operators as pre_oper  # Custom module

# Parameters
basedir = '/Users/Nathan/Documents/Oxford/DPhil/melanocyte/'
data_folder = 'data/Still_Images_with_BF_for_Nathan/'
filenames = ['VID289_A5_1_01d00h00m','VID289_B4_1_01d00h00m','VID289_D5_1_01d00h00m','VID289_E2_1_01d00h00m']
fileID = '.tif'
time = 0
min_clus_size = 150

# Helper function to crop image
def apply_rectangular_fov(arr, rect_width=1000, rect_height=800):
    center_x, center_y = arr.shape[1] // 2, arr.shape[0] // 2
    start_x = center_x - rect_width // 2
    start_y = center_y - rect_height // 2
    return arr[start_y:start_y + rect_height, start_x:start_x + rect_width]

# Helper function to map cluster index to area
def calc_area_arr(arr, area_new, index_keep):
    area_dict = dict(zip(index_keep, area_new))
    return np.vectorize(lambda x: area_dict.get(x, 0))(arr)

# Store results for comparison
all_areas = {}
summary_stats = {}

# Loop over each filename
for filename in filenames:
    raw_arr_2D_1, raw_arr_2D_2, raw_arr_2D_3 = tif.tif_to_arr(basedir, data_folder, filename, str(time), fileID)

    arr2_normal = apply_rectangular_fov(raw_arr_2D_2.astype(int))
    
    # Thresholding and segmentation
    thresh_otsu_2 = filters.threshold_otsu(arr2_normal)
    array_2 = arr2_normal < thresh_otsu_2
    current_array = binary_fill_holes(array_2).astype(int)
    label_arr, num_clus = label(current_array)

    area_list = ndi_sum(current_array, labels=label_arr, index=np.arange(label_arr.max() + 1))
    area_arr = label_arr

    area_new, index_keep = pre_oper.remove_fragments(area_list, num_clus, min_clus_size)

    total_curr_area = np.sum(area_new)
    mean_curr_area = np.mean(area_new)
    num_clusters = len(area_new)

    # Store results
    all_areas[filename] = area_new
    summary_stats[filename] = {
        'Total Area': total_curr_area,
        'Mean Area': mean_curr_area,
        'Num Clusters': num_clusters
    }

# Plot combined histogram
plt.figure(figsize=(10, 6))
for filename, areas in all_areas.items():
    plt.hist(areas, bins='auto', alpha=0.6, label=filename)

plt.title('Cluster Size Comparison Across Images')
plt.xlabel('Cluster Area')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary statistics
for filename, stats in summary_stats.items():
    print(f"\nFilename: {filename}")
    print(f"  Total Area: {stats['Total Area']}")
    print(f"  Mean Cluster Area: {stats['Mean Area']:.2f}")
    print(f"  Number of Clusters: {stats['Num Clusters']}")

# --- Visualize Summary Statistics ---

filenames = list(summary_stats.keys())
total_areas = [summary_stats[f]['Total Area'] for f in filenames]
mean_areas = [summary_stats[f]['Mean Area'] for f in filenames]
num_clusters = [summary_stats[f]['Num Clusters'] for f in filenames]

# Plot Total Area
plt.figure(figsize=(10, 6))
plt.bar(filenames, total_areas, color='skyblue')
plt.title('Total Cluster Area per Image')
plt.ylabel('Total Area')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot Mean Cluster Area
plt.figure(figsize=(10, 6))
plt.bar(filenames, mean_areas, color='lightgreen')
plt.title('Mean Cluster Area per Image')
plt.ylabel('Mean Area')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot Number of Clusters
plt.figure(figsize=(10, 6))
plt.bar(filenames, num_clusters, color='salmon')
plt.title('Number of Clusters per Image')
plt.ylabel('Number of Clusters')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
