import numpy as np
import matplotlib.pyplot as plt
import re
from skimage import filters
from scipy.ndimage import binary_fill_holes, label
from scipy.ndimage import sum as ndi_sum

# Use existing imports from user's code
import read_tif_file_operator as tif  # Custom module
import pre_pro_operators as pre_oper  # Custom module


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



# Parameters
basedir = '/Users/Nathan/Documents/Oxford/DPhil/melanocyte/'
data_folder = 'data/Still_Images_with_BF_for_Nathan/'
fileID = '.tif'
time = 0
min_clus_size = 150




# experiment_id = "B4"  # Change this as needed
experiment_ids = ["A5","B4","D5","E2"]  # Change this as needed


# Initialize storage for all experiments
summary_data = {eid: {"days": [], "total": [], "mean": [], "count": [], "vol_total": [], "vol_mean": []} for eid in experiment_ids}


for experiment_id in experiment_ids:
    
    filenames = generate_filenames(experiment_id)
    # for name in filenames:
    #     print(name)

    total_areas = []
    mean_areas = []
    num_clusters = []

    total_volumes = []
    mean_volumes = []


    # days = list(range(0, len(filenames)))
    days = extract_days(filenames)
    print("Days are", days)

    # Process each day
    for filename in filenames:
        raw_arr_2D_1, raw_arr_2D_2, raw_arr_2D_3 = tif.tif_to_arr(basedir, data_folder, filename, str(time), fileID)
        arr3_normal = apply_rectangular_fov(raw_arr_2D_3.astype(int))

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
