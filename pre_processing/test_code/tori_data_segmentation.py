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

from osgeo import gdal as GD

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


basedir = '/Users/Nathan/Documents/Oxford/DPhil/Tori_data/'
experiment = 'SNC_No_cAMP/'
# exp_date = '2017-03-16'
exp_date = ''
# folder = 'RAW/2017-03-16 sphere TL 6/2017-03-13 sphere TL 6-03/'
folder = ''
fileID = '.tif'
time_list = range(42,98,5)
well_loc = 's073'


for i in range(97,98,1):
    time = i

    # raw_arr_2D = tif.tif_to_arr(basedir, experiment, folder, well_loc, str(time), fileID)

    # Spheroid_Formation_SNC_1_B4_16_09d20h00m.tif
    # name_list_b = basedir, experiment, folder, '2017-03-13 sphere TL 6-03_', well_loc, 't0', time_list, 'c2', '_ORG', fileID
    name_list_b = basedir, experiment, folder, 'Spheroid_Formation_SNC_1_B4_16_04d16h00m.tif'

    name_list_b_2  =''.join(name_list_b)
    data_set_b = GD.Open(name_list_b_2)
    # Only interested in green channel (R is band 0, G is band 1, B is band 2)
    band_2 = data_set_b.GetRasterBand(1) # green channel
    b2 = band_2.ReadAsArray()
  
    # Store normalised intensities in 3D array
    raw_arr_2D = b2

    # raw_arr_2D = raw_arr_2D[:,1:]

    image_show_save(raw_arr_2D)

    text_threshold = filters.threshold_yen  # Hit tab with the cursor after the underscore, try several methods
    text_threshold_yen = filters.threshold_yen
    thresh = text_threshold(raw_arr_2D)
    array = raw_arr_2D > thresh
    image_show(raw_arr_2D > thresh)
    print("Threshold is", thresh)

    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.savefig('Tori_exp_Yen.png', dpi=300)

    text_threshold = filters.threshold_otsu  # Hit tab with the cursor after the underscore, try several methods
    text_threshold_otsu = filters.threshold_otsu
    thresh = text_threshold(raw_arr_2D)
    array = raw_arr_2D > thresh
    image_show(raw_arr_2D > thresh)
    print("Threshold is", thresh)

    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.savefig('Tori_exp_Otsu.png', dpi=300)

    text_threshold = filters.threshold_li  # Hit tab with the cursor after the underscore, try several methods
    text_threshold_li = filters.threshold_li
    thresh = text_threshold(raw_arr_2D)
    array = raw_arr_2D > thresh
    image_show(raw_arr_2D > thresh)
    print("Threshold is", thresh)

    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.savefig('Tori_exp_Li.png', dpi=300)

    text_threshold = filters.threshold_isodata  # Hit tab with the cursor after the underscore, try several methods
    text_threshold_isodata = filters.threshold_isodata
    thresh = text_threshold(raw_arr_2D)
    array = raw_arr_2D > thresh
    image_show(raw_arr_2D > thresh)
    print("Threshold is", thresh)

    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.savefig('Tori_exp_Isodata.png', dpi=300)

    text_threshold = filters.threshold_minimum  # Hit tab with the cursor after the underscore, try several methods
    text_threshold_minimum = filters.threshold_minimum
    thresh = text_threshold(raw_arr_2D)
    array = raw_arr_2D > thresh
    image_show(raw_arr_2D > thresh)
    print("Threshold is", thresh)

    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.savefig('Tori_exp_Min.png', dpi=300)

    text_threshold = filters.threshold_mean  # Hit tab with the cursor after the underscore, try several methods
    text_threshold_mean = filters.threshold_mean
    thresh = text_threshold(raw_arr_2D)
    array = raw_arr_2D > thresh
    image_show(raw_arr_2D > thresh)
    print("Threshold is", thresh)

    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.savefig('Tori_exp_Mean.png', dpi=300)

    text_threshold = filters.threshold_triangle  # Hit tab with the cursor after the underscore, try several methods
    text_threshold_triangle = filters.threshold_triangle
    thresh = text_threshold(raw_arr_2D)
    array = raw_arr_2D > thresh
    image_show(raw_arr_2D > thresh)
    print("Threshold is", thresh)

    plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.savefig('Tori_exp_Triangle.png', dpi=300)
    # plt.hist(raw_arr_2D)
    # plt.show()


    thresh_yen = text_threshold_yen(raw_arr_2D)
    array_yen = raw_arr_2D > thresh_yen

    thresh_otsu = text_threshold_otsu(raw_arr_2D)
    array_otsu = raw_arr_2D > thresh_otsu

    thresh_li = text_threshold_li(raw_arr_2D)
    array_li = raw_arr_2D > thresh_li

    thresh_isodata = text_threshold_isodata(raw_arr_2D)
    array_isodata = raw_arr_2D > thresh_isodata

    thresh_min = text_threshold_minimum(raw_arr_2D)
    array_min = raw_arr_2D > thresh_min

    thresh_mean = text_threshold_mean(raw_arr_2D)
    array_mean = raw_arr_2D > thresh_mean

    thresh_triangle = text_threshold_triangle(raw_arr_2D)
    array_triangle = raw_arr_2D > thresh_triangle

    plt.clf()
    plt.hist(raw_arr_2D)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Number of pixels")
    plt.axvline(thresh_yen, linestyle='dashed', color = 'k', label='Yen', linewidth=1)
    plt.axvline(thresh_otsu, linestyle='solid', color = 'c', label='Otsu', linewidth=1)
    plt.axvline(thresh_triangle, linestyle='dashed', color = 'b', label='Triangle', linewidth=1)
    plt.axvline(thresh_li, linestyle='dashed', color = 'r', label='Li', linewidth=1)
    plt.axvline(thresh_isodata, linestyle='dashed', color = 'm', label='Isodata', linewidth=1)
    plt.axvline(thresh_mean, linestyle='dashed', color = 'g', label='Mean', linewidth=1)
    plt.axvline(thresh_min, linestyle='dashed', color = 'y', label='Minimum', linewidth=1)
    plt.legend()


    plt.savefig("Tori_exp_hist_2.png", dpi=300)


    # f = plt.subplots(2, 4)
    plt.clf()
    plt.subplot(2, 4, 1)
    plt.imshow(raw_arr_2D, cmap='gray')
    plt.title('Original image')
    plt.axis('off')
    plt.subplot(2, 4, 2)
    plt.imshow(array_yen, cmap='gray')
    plt.title('Yen')
    plt.axis('off')
    plt.subplot(2, 4, 3)
    plt.imshow(array_otsu, cmap='gray')
    plt.title('Otsu')
    plt.axis('off')
    plt.subplot(2, 4, 4)
    plt.imshow(array_li, cmap='gray')
    plt.title('Li')
    plt.axis('off')
    plt.subplot(2, 4, 5)
    plt.imshow(array_isodata, cmap='gray')
    plt.title('Isodata')
    plt.axis('off')
    plt.subplot(2, 4, 6)
    plt.imshow(array_min, cmap='gray')
    plt.title('Minimum')
    plt.axis('off')
    plt.subplot(2, 4, 7)
    plt.imshow(array_mean, cmap='gray')
    plt.title('Mean')
    plt.axis('off')
    plt.subplot(2, 4, 8)
    plt.imshow(array_triangle, cmap='gray')
    plt.title('Triangle')
    plt.axis('off')
    plt.savefig('Tori_exp_multi_thresh.png',bbox_inches='tight')
