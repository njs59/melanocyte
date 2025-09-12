import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import read_tif_file_operator as tif


###    -----------   Input parameters   --------------     ###
basedir = '/Users/Nathan/Documents/Oxford/DPhil/In_vitro_homogeneous_data/'
experiment = 'RAW_data/2017-02-03_sphere_timelapse/'
exp_date = '2017-02-03'

folder = 'RAW/Timelapse/sphere_timelapse_useful_wells/'
fileID = '.tif'
timestep = '67'

# thresh_1 = 250
# thresh_2 = 300
# thresh_3 = 350
# well_loc = 's09'

thresh_0 = 300
thresh_1 = 408
thresh_2 = 440
thresh_3 = 600
well_loc = 's11'

thresh_0 = 220
thresh_1 = 330
thresh_2 = 440
thresh_3 = 550
well_loc = 's11'



# Plot and store histogram images at each timepoint for use in a gif
raw_arr_2D = tif.tif_to_arr(basedir, experiment, folder, well_loc, timestep, fileID)

array_0 = raw_arr_2D > thresh_0
array_1 = raw_arr_2D > thresh_1
array_2 = raw_arr_2D > thresh_2
array_3 = raw_arr_2D > thresh_3

array = 1*array_0 + 1*array_1 + 1*array_2 + 1*array_3

top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')

colors = [(0,0,0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]  # R -> G -> B
n_bins = 5  # Discretizes the interpolation into bins
cmap_name = 'my_list'

    # Create the colormap
cmap_1 = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    # # Fewer bins will result in "coarser" colomap interpolation
    # im = ax.imshow(Z, origin='lower', cmap=cmap)
    # ax.set_title("N bins: %s" % n_bin)
    # fig.colorbar(im, ax=ax)
cmap_2 = cm.get_cmap('inferno',5)
cmap_2.colors[1,:]=[0.9,0.9,1,1]

plt.imshow(array, cmap=cmap_2)
plt.axis([0, array.shape[1], 0, array.shape[0]])
plt.colorbar()
plt.show()

