import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    
    filenames = pre_oper.generate_filenames(experiment_id=experiment_id, base="VID289",
                       start_day = 0, start_hour = 0, start_minute = 0, 
                       end_day = 5, end_hour = 0, end_minute = 0,
                       lowest_day = 0, lowest_hour = 0, lowest_minute = 0,
                       highest_day = 5, highest_hour = 0, highest_minute = 0, 
                       gap_days = 1, gap_hours = 3, gap_minutes = 15)
    for name in filenames:
        print(name)

    total_areas = []
    mean_areas = []
    num_clusters = []

    total_volumes = []
    mean_volumes = []

    cluster_areas = np.array([])


    # days = list(range(0, len(filenames)))
    full_times_info = pre_oper.extract_time_components(filenames)
    times = full_times_info[0]
    print("Times in days are", times)

    # Process each timestep
    ticker = 0
    for filename in filenames:
        current_saving_dir_list = basedir, 'specialised_pipeline/', 'pre_processing_output/', base_str, '/'
        current_saving_dir  =''.join(current_saving_dir_list)


        raw_arr_2D_1, raw_arr_2D_2, raw_arr_2D_3 = tif.tif_to_arr(basedir, data_folder, filename, str(time), fileID)
        arr3_normal = pre_oper.apply_rectangular_fov(raw_arr_2D_3.astype(int))

        thresh_otsu_3 = filters.threshold_otsu(arr3_normal)
        array_3 = arr3_normal < thresh_otsu_3
        current_array = binary_fill_holes(array_3).astype(int)
        label_arr, num_clus = label(current_array)

        area_list = ndi_sum(current_array, labels=label_arr, index=np.arange(label_arr.max() + 1))
        area_new, index_keep = pre_oper.remove_fragments(area_list, num_clus, min_clus_size)
            

        update_area_arr = np.where(np.isin(label_arr, index_keep), label_arr, 0)
        update_boolean_arr = update_area_arr != 0
        update_label_arr, update_num_clus = label(update_boolean_arr)


        # Save area array to csv file
        df_area = pd.DataFrame(update_label_arr)
        area_csv_name_list = current_saving_dir, filename, '_area', '.csv'
        area_csv_name_list_2  =''.join(area_csv_name_list)
        df_area.to_csv(area_csv_name_list_2, index=False, header=False)

        # Save index array to csv file
        df_index = pd.DataFrame(update_area_arr)
        index_csv_name_list = current_saving_dir, filename, '_indexed', '.csv'
        index_csv_name_list_2  =''.join(index_csv_name_list)
        df_index.to_csv(index_csv_name_list_2, index=False, header=False)

        # Add the cluster areas to 2D array of cluster sizes over time (for histogram plotting)
        cluster_areas = pre_oper.save_clus_areas(ticker, area_new, cluster_areas)




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
        ticker += 1

    # Print summary statistics
    print("Summary Statistics Over 5 Days:")
    for i, filename in enumerate(filenames):
        print(f"Day {i+1} ({filename}):")
        print(f"  Total Area: {total_areas[i]}")
        print(f"  Mean Cluster Area: {mean_areas[i]:.2f}")
        print(f"  Number of Clusters: {num_clusters[i]}")



    summary_data[experiment_id]["days"] = times
    summary_data[experiment_id]["total"] = total_areas
    summary_data[experiment_id]["mean"] = mean_areas
    summary_data[experiment_id]["count"] = num_clusters
    summary_data[experiment_id]["vol_total"] = total_volumes
    summary_data[experiment_id]["vol_mean"] = mean_volumes


    ### -----------------   Outputs  ------------------------- ###

    # Save 2D cluster areas array to csv
    df_cluster_areas = pd.DataFrame(cluster_areas)
    cluster_areas_csv_name_list = current_saving_dir, 'csv_files/', base_str, '_', experiment_id, '_cluster_areas', '.csv'
    cluster_areas_csv_name_list_2  =''.join(cluster_areas_csv_name_list)
    df_cluster_areas.to_csv(cluster_areas_csv_name_list_2, index=False, header=False)

    # Save mean areas to csv
    df_mean_areas = pd.DataFrame(mean_areas)
    mean_areas_csv_name_list = current_saving_dir, 'csv_files/', base_str, '_', experiment_id, '_mean_areas', '.csv'
    mean_areas_csv_name_list_2  =''.join(mean_areas_csv_name_list)
    df_mean_areas.to_csv(mean_areas_csv_name_list_2, index=False, header=False)

    # Save total areas to csv
    df_total_areas = pd.DataFrame(total_areas)
    total_areas_csv_name_list = current_saving_dir, 'csv_files/', base_str, '_', experiment_id, '_total_areas', '.csv'
    total_areas_csv_name_list_2  =''.join(total_areas_csv_name_list)
    df_total_areas.to_csv(total_areas_csv_name_list_2, index=False, header=False)

    # Save number of clusters to csv
    df_number_clusters = pd.DataFrame(num_clusters)
    number_clusters_csv_name_list = current_saving_dir, 'csv_files/', base_str, '_', experiment_id, '_number_clusters', '.csv'
    number_clusters_csv_name_list_2  =''.join(number_clusters_csv_name_list)
    df_number_clusters.to_csv(number_clusters_csv_name_list_2, index=False, header=False)



# Export all summary data to CSV
for eid in experiment_ids:
    df = pd.DataFrame({
        "Day": summary_data[eid]["days"],
        "Total Area": summary_data[eid]["total"],
        "Mean Area": summary_data[eid]["mean"],
        "Cluster Count": summary_data[eid]["count"],
        "Total Volume": summary_data[eid]["vol_total"],
        "Mean Volume": summary_data[eid]["vol_mean"]
    })
    df_name_list = current_saving_dir, f"df_summary_{eid}.csv"
    df_name = ''.join(df_name_list)
    df.to_csv(df_name, index=False)




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
plot_name_list = current_saving_dir, 'plots/', base_str, 'total_area_summary.png'
plot_name = ''.join(plot_name_list)
plt.savefig(plot_name, dpi=300)
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
plot_name_list = current_saving_dir, 'plots/', base_str, 'mean_area_summary.png'
plot_name = ''.join(plot_name_list)
plt.savefig(plot_name, dpi=300)
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
plot_name_list = current_saving_dir, 'plots/', base_str, 'cluster_count_summary.png'
plot_name = ''.join(plot_name_list)
plt.savefig(plot_name, dpi=300)
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
plot_name_list = current_saving_dir, 'plots/', base_str, 'total_spherical_volume_summary.png'
plot_name = ''.join(plot_name_list)
plt.savefig(plot_name, dpi=300)
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
plot_name_list = current_saving_dir, 'plots/', base_str, 'mean_spherical_volume_summary.png'
plot_name = ''.join(plot_name_list)
plt.savefig(plot_name, dpi=300)
plt.show()


print("Summary plots generated for all experiment IDs.")
