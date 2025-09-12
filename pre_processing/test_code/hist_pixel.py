import glob
from PIL import Image
import time
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import read_tif_file_operator as tif

from skimage import filters


###    -----------   Input parameters   --------------     ###
basedir = '/Users/Nathan/Documents/Oxford/DPhil/In_vitro_homogeneous_data/'
experiment = 'RAW_data/2017-02-03_sphere_timelapse/'
exp_date = '2017-02-03'
# experiment = 'RAW_data/2017-02-13_sphere_timelapse_2/'
# exp_date = '2017-02-13'
# experiment = 'RAW_data/2017-03-16_sphere_timelapse/'
# exp_date = '2017-03-16'
folder = 'RAW/Timelapse/sphere_timelapse_useful_wells/'
fileID = '.tif'
time_list = range(67,88,5)
well_loc = 's11'
# well_loc = 's073'

# get the current time to use in the filename
timestr = time.strftime("%Y%m%d-%H%M%S")

thresh_0 = 220
thresh_1 = 330
thresh_2 = 440
thresh_3 = 550
cmap_2 = cm.get_cmap('inferno',5)
cmap_2.colors[1,:]=[0.6,0.6,1,1]

# Plot and store histogram images at each timepoint for use in a gif
for j in range(len(time_list)):
    raw_arr_2D = tif.tif_to_arr(basedir, experiment, folder, well_loc, str(time_list[j]), fileID)

    raw_arr_2D = raw_arr_2D[:,1:]
    # raw_arr_2D -= raw_arr_2D.min()
    # raw_arr_2D *= 10



    text_threshold = filters.threshold_otsu  # Hit tab with the cursor after the underscore, try several methods
    thresh_otsu = text_threshold(raw_arr_2D)
    print('Ostu thresh:', thresh_otsu)

    text_threshold = filters.threshold_yen  # Hit tab with the cursor after the underscore, try several methods
    thresh_yen = text_threshold(raw_arr_2D)
    print('Yen thresh:', thresh_yen)

    # plt.stairs(*np.histogram(raw_arr_2D, 1000), fill=True, color='skyblue')
    plt.hist(raw_arr_2D)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Number of pixels")
    # plt.axvline(thresh_otsu, color='b', linestyle='dashed', linewidth=1)
    # plt.axvline(thresh_yen, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(thresh_0, color=cmap_2.colors[1,:], linestyle='dashed', linewidth=1)
    plt.axvline(thresh_1, color=cmap_2.colors[2,:], linestyle='dashed', linewidth=1)
    plt.axvline(thresh_2, color=cmap_2.colors[3,:], linestyle='dashed', linewidth=1)
    plt.axvline(thresh_3, color=cmap_2.colors[4,:], linestyle='dashed', linewidth=1)
    # plt.xlim(150, 600) 
    plt.savefig(f'/Users/Nathan/Documents/Oxford/DPhil/clusters/data_analysis_3D/pre-processing/test_code/histogram/frame-{j:03d}.png', bbox_inches='tight', dpi=300)
    plt.clf()

images_hist = []
# get all the images in the 'images for gif' folder
for filename_hist in sorted(glob.glob('/Users/Nathan/Documents/Oxford/DPhil/clusters/data_analysis_3D/pre-processing/test_code/histogram/frame-*.png')): # loop through all png files in the folder
    im_hist = Image.open(filename_hist) # open the image
    images_hist.append(im_hist) # add the image to the list

# calculate the frame number of the last frame (ie the number of images)
last_frame = (len(images_hist)) 

# create 10 extra copies of the last frame (to make the gif spend longer on the most recent data)
for x in range(0, 9):
    im_hist = images_hist[last_frame-1]
    images_hist.append(im_hist)

# save as a gif   
images_hist[0].save('/Users/Nathan/Documents/Oxford/DPhil/clusters/data_analysis/pre-processing/test_code/histogram/' + timestr + '.gif',
            save_all=True, append_images=images_hist[1:], optimize=False, duration=500, loop=0)

