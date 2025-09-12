import math
import numpy as np
import numpy.testing as npt

import unittest

import post_pro_operators as post_oper


def test_calc_clus_centre():

    test_labelled_array = np.array([[0,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,2,2,2],
                      [0,0,0,2,0,2],[0,0,0,0,0,0],[3,3,3,0,0,0]])
    
    test_centres = post_oper.calc_clus_centre(test_labelled_array)
    print(test_centres)
    return test_centres

test_calc_clus_centre()

print(test_calc_clus_centre() == np.array([[1,1],[2,4],[5,1]]))

print(np.all(test_calc_clus_centre() == np.array([[1,1],[2,4],[5,1]])))

def test_previous_clusters_at_loc(index):

    test_labelled_array = np.array([[0,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,2,2,2],
                    [0,0,0,2,0,2],[0,0,0,0,0,0],[3,3,3,0,0,0]])
    test_centres_old = np.array([[2,3],[3,5],[5,1]])

    same_locs, same_locs_store = post_oper.previous_clusters_at_loc(test_labelled_array, test_centres_old, index)

    return same_locs, same_locs_store



index_1_same_locs, index_1_same_locs_store = test_previous_clusters_at_loc(1)
index_2_same_locs, index_2_same_locs_store = test_previous_clusters_at_loc(2)
index_3_same_locs, index_3_same_locs_store = test_previous_clusters_at_loc(3)

print(index_1_same_locs == 0)
print(index_2_same_locs == 2)
print(index_3_same_locs == 1)
print(np.all(index_1_same_locs_store == np.array([])))
print(np.all(index_2_same_locs_store == np.array([[2,3],[3,5]])))
print(np.all(index_3_same_locs_store == np.array([5,1])))