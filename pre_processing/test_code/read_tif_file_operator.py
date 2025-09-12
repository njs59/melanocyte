import numpy as np
from osgeo import gdal as GD


def tif_to_arr(basedir, experiment, folder, well_loc, time_list, fileID):
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
  # for i in range(len(time_list)):
    # Read in tif file
  # name_list_b = basedir, experiment, folder, 'sphere_timelapse_', well_loc, 't', time_list, 'c2', '_ORG', fileID
  # name_list_b = basedir, experiment, folder, '2017-02-13 sphere timelapse 2_', well_loc, 't', str(time_list), 'c2', '_ORG', fileID
  name_list_b = basedir, experiment, folder, '2017-03-13 sphere TL 6-03_', well_loc, 't0', time_list, 'c2', '_ORG', fileID

  name_list_b_2  =''.join(name_list_b)
  data_set_b = GD.Open(name_list_b_2)
  # Only interested in green channel (R is band 0, G is band 1, B is band 2)
  band_2 = data_set_b.GetRasterBand(1) # green channel
  b2 = band_2.ReadAsArray()
  # img_1 = np.dstack((b2))
  
  # Store normalised intensities in 3D array
  main_array = b2

    # else:
    #   print(i)
    #   main_array = np.dstack((main_array, b2))
    #   print(main_array.shape)

  # Output 3D normalised array
  return main_array