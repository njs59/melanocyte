import numpy as np
from osgeo import gdal as GD


def tif_to_arr(basedir, data_folder, filename, time_list, fileID):
  '''
  Reads in series of tif files and converts the pixel intensity to a single 3D array (3rd dimension is time)

  Inputs:
    Identify the experiment and where the tif files are stored
    basedir,
    experiment,
    folder,
    well_loc,

    time_list: list of deisred timepoints to read in
    fileID, this will be '.tif'


  '''
  # Loop over timepoints
  for i in range(len(time_list)):
    # Read in tif file
    name_list_b = basedir, data_folder, filename, time_list[i], fileID
    name_list_b_2  =''.join(name_list_b)
    data_set_b = GD.Open(name_list_b_2)
    # Only interested in green channel (R is band 0, G is band 1, B is band 2)
    band_0 = data_set_b.GetRasterBand(0) # Red channel
    b0 = band_0.ReadAsArray()

    band_1 = data_set_b.GetRasterBand(1) # Green channel
    b1 = band_1.ReadAsArray()

    band_2 = data_set_b.GetRasterBand(2) # Blue channel
    b2 = band_2.ReadAsArray()
    
    # Store normalised intensities in 3D array
    if i == 0:
      main_array_0 = b0
      main_array_1 = b1
      main_array_2 = b2

    else:
      print(i)
      main_array_0 = np.dstack((main_array_0, b0))
      main_array_1 = np.dstack((main_array_1, b0))
      main_array_2 = np.dstack((main_array_2, b0))
      print(main_array_0.shape)

  # Output 3D normalised arrays
  return main_array_0, main_array_1, main_array_2