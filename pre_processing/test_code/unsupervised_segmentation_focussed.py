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
filename = 'VID289_E2_1_04d00h00m'

# A5 B4 D5 E2

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


text_threshold = filters.threshold_yen  # Hit tab with the cursor after the underscore, try several methods
# text_threshold_yen = filters.threshold_yen
thresh_yen_1 = text_threshold(raw_arr_2D_1)
thresh_yen_2 = text_threshold(raw_arr_2D_2)
thresh_yen_3 = text_threshold(raw_arr_2D_3)

array_1 = raw_arr_2D_2 > thresh_yen_1
array_2 = raw_arr_2D_2 > thresh_yen_2
array_3 = raw_arr_2D_2 > thresh_yen_3

array_yen_tot = array_1.astype(int) + array_2.astype(int) + array_3.astype(int)


print("Thresholds are", thresh_yen_1, thresh_yen_2, thresh_yen_3)

plt.imshow(array_yen_tot, cmap='viridis')
plt.axis('off')
plt.savefig('Tori_Multi_Yen.png', dpi=300)

###############################################

text_threshold = filters.threshold_otsu  # Hit tab with the cursor after the underscore, try several methods
# text_threshold_otsu = filters.threshold_otsu
thresh_otsu_1 = text_threshold(raw_arr_2D_1)
thresh_otsu_2 = text_threshold(raw_arr_2D_2)
thresh_otsu_3 = text_threshold(raw_arr_2D_3)

array_1 = raw_arr_2D_1 > thresh_otsu_1
array_2 = raw_arr_2D_2 > thresh_otsu_2
array_3 = raw_arr_2D_3 > thresh_otsu_3

array_otsu_tot = array_1.astype(int) + array_2.astype(int) + array_3.astype(int)


print("Thresholds are", thresh_otsu_1, thresh_otsu_2, thresh_otsu_3)

plt.imshow(array_otsu_tot, cmap='viridis')
plt.axis('off')
plt.savefig('Tori_Multi_Otsu.png', dpi=300)

##############################################################

text_threshold = filters.threshold_li  # Hit tab with the cursor after the underscore, try several methods
# text_threshold_li = filters.threshold_li
thresh_li_1 = text_threshold(raw_arr_2D_1)
thresh_li_2 = text_threshold(raw_arr_2D_2)
thresh_li_3 = text_threshold(raw_arr_2D_3)

array_1 = raw_arr_2D_1 > thresh_li_1
array_2 = raw_arr_2D_2 > thresh_li_2
array_3 = raw_arr_2D_3 > thresh_li_3

array_li_tot = array_1.astype(int) + array_2.astype(int) + array_3.astype(int)


print("Thresholds are", thresh_li_1, thresh_li_2, thresh_li_3)

plt.imshow(array_li_tot, cmap='viridis')
plt.axis('off')
plt.savefig('Tori_Multi_Li.png', dpi=300)

# f = plt.subplots(2, 4)
plt.clf()
plt.subplot(2, 2, 1)
plt.imshow(raw_arr_2D_2, cmap='gray')
plt.title('Original image')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(array_yen_tot, cmap='summer')
plt.title('Yen')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(array_otsu_tot, cmap='summer')
plt.title('Otsu')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(array_li_tot, cmap='summer')
plt.title('Li')
plt.axis('off')
plt.show()
