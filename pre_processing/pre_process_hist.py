import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

import matplotlib.animation as animation

import pre_pro_operators as pre_oper


###   -----------   Input parameters   ----------   ###
basedir = '/Users/Nathan/Documents/Oxford/DPhil/'
exp_type = 'In_vitro_homogeneous_data/'
experiment = '2017-02-03_sphere_timelapse/'
exp_date = '2017-02-03'
folder = 'RAW/Timelapse/sphere_timelapse_useful_wells/'
folder_3 = 'sphere_timelapse/'
fileID = '.tif'

time_array = range(31,98)

time_list = [str(x).zfill(2) for x in time_array]
well_loc = 's11'


###   --------------   Plotting code   ---------   ###

# Read in cluster areas
cluster_2D_areas_csv_name_list = basedir, exp_type, 'pre_processing_output/', exp_date, '/', well_loc, '_cluster_areas', '.csv'
cluster_2D_areas_csv_name_list_2  =''.join(cluster_2D_areas_csv_name_list)
df_clus_areas = pd.read_csv(cluster_2D_areas_csv_name_list_2, header=None)
cluster_areas = df_clus_areas.to_numpy()

number_of_frames = len(time_list)
data = cluster_areas
# Get correct row index to start on
start_data = time_array[0] - 1

# Plot histograms at each time
data_times = data[start_data:,]
fig = plt.figure()
plt.axis([150,25000,0,200])
hist = plt.hist(data_times[0], bins=[200, 600, 1000, 2000, 4000, 8000, 16000, 25000])

# Plot as an animation object using the update_hist operator 
animation = animation.FuncAnimation(fig, pre_oper.update_hist, number_of_frames, interval=500, fargs=(data_times, ) )


# Save the animation to gif
animation.save(basedir + 'images/histograms/hist' + '.gif', fps=10)


# Show each frame in the gif
im = Image.open(basedir + 'images/histograms/hist' + '.gif')
im.show()
