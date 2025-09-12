import numpy as np
import matplotlib.pyplot as plt

import skimage.data as data
import skimage.segmentation as seg
from skimage import filters
from skimage import draw
from skimage import color
from skimage import exposure
from scipy.ndimage.filters import median_filter

from PIL import Image, ImageFilter
import read_tif_file_operator as tif


#Import the necessary libraries 
import cv2 
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


###    -----------   Input parameters   --------------     ###

basedir = '/Users/Nathan/Documents/Oxford/DPhil/melanocyte/'

data_folder = 'data/Still_Images_with_BF_for_Nathan/'
filename = 'VID289_A5_1_01d00h00m'

fileID = '.tif'
# time_list = range(42,98,5)
# well_loc = 's073'

time_list = ''


# for i in range(97,98,1):
#     time = i
time = 0

raw_arr_2D_1, raw_arr_2D_2, raw_arr_2D_3 = tif.tif_to_arr(basedir, data_folder, filename, str(time), fileID)

raw_arr_2D_1 = raw_arr_2D_1[:,1:]
raw_arr_2D_2 = raw_arr_2D_2[:,1:]
raw_arr_2D_3 = raw_arr_2D_3[:,1:]



text_threshold_local_gaussian = filters.threshold_local(raw_arr_2D_2, block_size=31, method='gaussian')  # Hit tab with the cursor after the underscore, try several methods
array = raw_arr_2D_2 > text_threshold_local_gaussian
image_show(raw_arr_2D_2 > text_threshold_local_gaussian)
print("Threshold is", text_threshold_local_gaussian)

# thresh_1 = text_threshold(raw_arr_2D_1)
# thresh_2 = text_threshold(raw_arr_2D_2)
# thresh_3 = text_threshold(raw_arr_2D_3)

# print("Thresholds are", thresh_1, thresh_2, thresh_3)

plt.imshow(array, cmap='gray')
plt.axis('off')
plt.savefig('Tori_local_gaussian_35.png', dpi=300)

