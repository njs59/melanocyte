from skimage import filters, morphology, util
import matplotlib.pyplot as plt

import read_tif_file_operator as tif

###    -----------   Input parameters   --------------     ###

basedir = '/Users/Nathan/Documents/Oxford/DPhil/melanocyte/'

data_folder = 'data/Still_Images_without_BF_for_Nathan/'
filename = 'VID289_A5_1_00d00h00m'

fileID = '.tif'
# time_list = ''
time = 0

raw_arr_2D_1, raw_arr_2D_2, raw_arr_2D_3 = tif.tif_to_arr(basedir, data_folder, filename, str(time), fileID)

image_array = raw_arr_2D_2

# Ensure the image is in 8-bit format
image = util.img_as_ubyte(image_array)

# Define a disk-shaped structuring element
selem = morphology.disk(101)

# Apply local Otsu thresholding
local_otsu = filters.rank.otsu(image, selem)

# Create binary mask
binary_mask = image >= local_otsu

# Display results
plt.imshow(binary_mask, cmap='gray')
plt.title("Local Otsu Thresholding")
plt.axis('off')
plt.show()
