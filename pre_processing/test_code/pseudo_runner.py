import numpy as np
import matplotlib.pyplot as plt

from pylab import *
from scipy.ndimage import *


from skimage import filters



import read_tif_file_operator as tif
import pre_pro_operators as pre_oper


#Import the necessary libraries 
import matplotlib.pyplot as plt 
import numpy as np 

# from squidpy import ImageContainer, segment


def image_show_save(image):
    plt.imshow(image, cmap='gray')
    plt.axis([0, image.shape[1], 0, image.shape[0]])
    plt.axis('off')
    plt.savefig('test_un.jpg')
    plt.show()

def image_show(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

# Function to update indexed array to array displaying areas
def calc_area_arr(arr):
    global area_new, index_keep
    index = np.where(index_keep == arr)
    if len(index[0]) != 0:        
        i = area_new[index[0]][0]
    else:
        i = 0
    return i


###    -----------   Input parameters   --------------     ###
# basedir = '/Users/Nathan/Documents/Oxford/DPhil/In_vitro_homogeneous_data/'
# experiment = 'RAW_data/2017-02-03_sphere_timelapse/'
# exp_date = '2017-02-03'
#experiment = 'RAW_data/2017-02-13_sphere_timelapse_2/'
#exp_date = '2017-02-13'

basedir = '/Volumes/Elements/Nate thesis data/Thesis Data/'
experiment = '2017-03-16 sphere TL 6/'
exp_date = '2017-03-16'
folder = 'RAW/2017-03-16 sphere TL 6/2017-03-13 sphere TL 6-03/'

fileID = '.tif'
time_list = range(42,98,5)
well_loc = 's037'

num_clusters = []
total_area = []
mean_area = []

for i in range(97,98,1):
    time = i

    raw_arr_2D = tif.tif_to_arr(basedir, experiment, folder, well_loc, str(time), fileID)

    raw_arr_2D = raw_arr_2D[:,1:]
    # raw_arr_2D -= raw_arr_2D.min()
    # raw_arr_2D *= 10

    # text = data.page()
    image_show_save(raw_arr_2D)

    text_threshold = filters.threshold_yen  # Hit tab with the cursor after the underscore, try several methods
    thresh = text_threshold(raw_arr_2D)
    array = raw_arr_2D > thresh
    image_show(raw_arr_2D > thresh)
    print("Threshold is", thresh)


    # Fill any single pixel holes in clusters
    current_array = binary_fill_holes(array).astype(int) 

    # Label the clusters of the boolean array
    label_arr, num_clus = label(current_array)

    plt.imshow(label_arr, interpolation=None)
    plt.show()

    # Get a 1D list of areas of the clusters
    area_list = sum(current_array, label_arr, index=arange(label_arr.max() + 1))

    print(area_list)


    # Get a 1D list of areas of the clusters
    area_list = sum(current_array, label_arr, index=arange(label_arr.max() + 1))

    area_arr = label_arr

    global area_new, index_keep
    # Remove fragments, that is clusters smaller than
    # min_clus_size so we only consider clusters large enough
    # to be a cell rather than a cell fragment
    area_new, index_keep = pre_oper.remove_fragments(area_list, num_clus, 150)

    # Calculate and store total area now in clusters, number of clusters and mean cluster area 
    total_curr_area = np.sum(area_new)
    print('Total current area', total_curr_area)
    mean_curr_area = np.mean(area_new)
    print('Mean cluster area', mean_curr_area)
    percent_curr_area = (total_curr_area/(current_array.shape[0]*current_array.shape[1]))*100
    print('Confluence percentage', percent_curr_area)
    num_clusters = np.append(num_clusters,len(area_new))
    total_area = np.append(total_area, np.sum(area_new))
    mean_area = np.append(mean_area, np.mean(area_new))

    # Calculate 2D cluster array with cluster areas rather than indeces
    applyall = np.vectorize(calc_area_arr)
    area_slice = applyall(area_arr)

    # Plot area array heatmap on a log scale and save figure
    my_cmap = mpl.colormaps['spring']
    my_cmap.set_under('k')
    plt.imshow(area_slice, cmap=my_cmap, vmin=1)
    plt.axis([0, area_slice.shape[1], 0, area_slice.shape[0]])
    plt.colorbar()
    plt.savefig('Better_exp_PRO_Yen_heatmap.png', dpi=300)
    plt.show()

    plt.clf()

