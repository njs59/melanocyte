import numpy as np
import matplotlib.pyplot as plt
import re
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

base_str = "VID289"




# experiment_id = "B4"  # Change this as needed
experiment_ids = ["A5","B4","D5","E2"]  # Change this as needed


# Initialize storage for all experiments
summary_data = {eid: {"days": [], "total": [], "mean": [], "count": [], "vol_total": [], "vol_mean": []} for eid in experiment_ids}


for experiment_id in experiment_ids:
    
    filenames = pre_oper.generate_filenames(experiment_id="B4", base="VID289",
                       start_day = 1, start_hour = 12, start_minute = 45, 
                       end_day = 1, end_hour = 18, end_minute = 15,
                       lowest_day = 1, lowest_hour = 0, lowest_minute = 0,
                       highest_day = 1, highest_hour = 21, highest_minute = 45, 
                       gap_days = 1, gap_hours = 3, gap_minutes = 15)
    for name in filenames:
        print(name)

    total_areas = []
    mean_areas = []
    num_clusters = []

    total_volumes = []
    mean_volumes = []


    # days = list(range(0, len(filenames)))
    days = pre_oper.extract_days(filenames)
    print("Days are", days)

    # Process each day
    for filename in filenames:
        raw_arr_2D_1, raw_arr_2D_2, raw_arr_2D_3 = tif.tif_to_arr(basedir, data_folder, filename, str(time), fileID)
        arr3_normal = pre_oper.apply_rectangular_fov(raw_arr_2D_3.astype(int))

        thresh_otsu_3 = filters.threshold_otsu(arr3_normal)
        array_3 = arr3_normal < thresh_otsu_3
        current_array = binary_fill_holes(array_3).astype(int)
        label_arr, num_clus = label(current_array)

        area_list = ndi_sum(current_array, labels=label_arr, index=np.arange(label_arr.max() + 1))
        area_new, index_keep = pre_oper.remove_fragments(area_list, num_clus, min_clus_size)


        total_area = np.sum(area_new)
        mean_area = np.mean(area_new)
        cluster_count = len(area_new)

        # Convert 2D area to spherical volume: V = (4/3) * pi * r^3, where r = sqrt(area / pi)
        volumes = [(4/3) * np.pi * (np.sqrt(a / np.pi))**3 for a in area_new]
        total_volume = np.sum(volumes)
        mean_volume = np.mean(volumes)

        total_areas.append(total_area)
        mean_areas.append(mean_area)
        num_clusters.append(cluster_count)
        total_volumes.append(total_volume)
        mean_volumes.append(mean_volume)


        # plt.imshow(current_array)
        # plt.show()

    # Print summary statistics
    print("Summary Statistics Over 5 Days:")
    for i, filename in enumerate(filenames):
        print(f"Day {i+1} ({filename}):")
        print(f"  Total Area: {total_areas[i]}")
        print(f"  Mean Cluster Area: {mean_areas[i]:.2f}")
        print(f"  Number of Clusters: {num_clusters[i]}")




    summary_data[experiment_id]["days"] = days
    summary_data[experiment_id]["total"] = total_areas
    summary_data[experiment_id]["mean"] = mean_areas
    summary_data[experiment_id]["count"] = num_clusters
    summary_data[experiment_id]["vol_total"] = total_volumes
    summary_data[experiment_id]["vol_mean"] = mean_volumes




# Plotting all experiments together

# Total Area
plt.figure(figsize=(10, 6))
for eid in experiment_ids:
    plt.plot(summary_data[eid]["days"], summary_data[eid]["total"], marker='o', label=eid)
plt.title('Total Cluster Area Over Time')
plt.xlabel('Day')
plt.ylabel('Total Area')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("total_area_summary.png")
plt.show()

# Mean Cluster Area
plt.figure(figsize=(10, 6))
for eid in experiment_ids:
    plt.plot(summary_data[eid]["days"], summary_data[eid]["mean"], marker='s', label=eid)
plt.title('Mean Cluster Area Over Time')
plt.xlabel('Day')
plt.ylabel('Mean Area')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mean_area_summary.png")
plt.show()

# Number of Clusters
plt.figure(figsize=(10, 6))
for eid in experiment_ids:
    plt.plot(summary_data[eid]["days"], summary_data[eid]["count"], marker='^', label=eid)
plt.title('Number of Clusters Over Time')
plt.xlabel('Day')
plt.ylabel('Number of Clusters')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cluster_count_summary.png")
plt.show()


# Plotting total and mean volume

# Total Volume
plt.figure(figsize=(10, 6))
for eid in experiment_ids:
    plt.plot(summary_data[eid]["days"], summary_data[eid]["vol_total"], marker='o', label=eid)
plt.title('Total Spherical Cluster Volume Over Time')
plt.xlabel('Day')
plt.ylabel('Total Volume (µm³)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("total_spherical_volume_summary.png")
plt.show()

# Mean Volume
plt.figure(figsize=(10, 6))
for eid in experiment_ids:
    plt.plot(summary_data[eid]["days"], summary_data[eid]["vol_mean"], marker='s', label=eid)
plt.title('Mean Spherical Cluster Volume Over Time')
plt.xlabel('Day')
plt.ylabel('Mean Volume (µm³)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mean_spherical_volume_summary.png")
plt.show()


print("Summary plots generated for all experiment IDs.")
