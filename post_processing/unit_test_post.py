import math
import numpy as np
import numpy.testing as npt

import unittest

import post_pro_operators as post_oper

import random




# def test_calc_clus_centre():

#     test_labelled_array = np.array([[0,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,2,2,2],
#                       [0,0,0,2,0,2],[0,0,0,0,0,0],[3,3,3,0,0,0]])
    
#     test_centres = post_oper.calc_clus_centre(test_labelled_array)
#     print(test_centres)
#     return test_centres




# test_calc_clus_centre()

# print(test_calc_clus_centre() == np.array([[1,1],[2,4],[5,1]]))

# print(np.all(test_calc_clus_centre() == np.array([[1,1],[2,4],[5,1]])))





class TestPostProOperators(unittest.TestCase):

    def test_calc_clus_centre(self):
        test_labelled_array = np.array([[0,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,2,2,2],
                      [0,0,0,2,0,2],[0,0,0,0,0,0],[3,3,3,0,0,0]])
        # print(post_oper.calc_clus_centre(test_labelled_array))
        centres_array = post_oper.calc_clus_centre(test_labelled_array)


        self.assertTrue(np.all(centres_array == np.array([[1,1],[2,4],[5,1]])))

    def test_previous_clusters_at_loc(self):
        test_labelled_array = np.array([[0,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,2,2,2],
                      [0,0,0,2,0,2],[0,0,0,0,0,0],[3,3,3,0,0,0]])
        test_centres_old = np.array([[2,3],[3,5],[5,1]])

        index_1_same_locs, index_1_same_locs_store = post_oper.previous_clusters_at_loc(test_labelled_array, test_centres_old, 1)
        index_2_same_locs, index_2_same_locs_store = post_oper.previous_clusters_at_loc(test_labelled_array, test_centres_old, 2)
        index_3_same_locs, index_3_same_locs_store = post_oper.previous_clusters_at_loc(test_labelled_array, test_centres_old, 3)

        self.assertTrue(index_1_same_locs == 0)
        self.assertTrue(index_2_same_locs == 2)
        self.assertTrue(index_3_same_locs == 1)
        self.assertTrue(np.all(index_1_same_locs_store == np.array([])))
        self.assertTrue(np.all(index_2_same_locs_store == np.array([[2,3],[3,5]])))
        self.assertTrue(np.all(index_3_same_locs_store == np.array([5,1])))

    def test_nearby_clusters(self):

        ####################
        ## I Think THIS FN NEEDS WORK
        # All around search raidus, both how it deal with being at the edge 
        # and indexing hte search radius (I think it's slightly too small)
        
        test_labelled_array = np.array([[0,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,2,2,2],
                      [0,0,0,2,0,2],[0,0,0,0,0,0],[3,3,3,0,0,0]])
        
        # Parameters are (x_loc, y_loc, search_radius, labelled_arr)
        cluster_index, distances = post_oper.nearby_clusters(0,1,1,test_labelled_array)
        # print(cluster_index, distances)
        self.assertEqual(cluster_index,1)
        self.assertEqual(distances,1)

        cluster_index, distances = post_oper.nearby_clusters(4,2,1,test_labelled_array)
        # print(cluster_index, distances)
        self.assertTrue(np.all(cluster_index == np.array([2,3])))
        self.assertTrue(np.all(distances == np.array([2,1])))

        cluster_index, distances = post_oper.nearby_clusters(5,5,1,test_labelled_array)
        # print(cluster_index, distances)
        self.assertTrue(np.all(cluster_index == np.array([])))
        self.assertTrue(np.all(distances == np.array([])))

        cluster_index, distances = post_oper.nearby_clusters(5,5,2,test_labelled_array)
        # print(cluster_index, distances)
        self.assertTrue(np.all(cluster_index == np.array([2])))
        self.assertTrue(np.all(distances == np.array([2])))

        cluster_index, distances = post_oper.nearby_clusters(5,5,3,test_labelled_array)
        # print(cluster_index, distances)
        self.assertTrue(np.all(cluster_index == np.array([2,3])))
        self.assertTrue(np.all(distances == np.array([2,3])))



    def test_pick_cluster_inverse_dist(self):

        # Parameters are (clusters_index_output, distances)

        clus_select = post_oper.pick_cluster_inverse_dist(np.array([1,2,3]), np.array([0.0,2.0,3.0]))
        self.assertEqual(clus_select,np.array([1]))

        # Run multiple times with the random seed fixed
        # Shows both possibilities can be recovered
        np.random.seed(10)
        clus_select = post_oper.pick_cluster_inverse_dist([5,9], [2.0,4.0])
        self.assertEqual(clus_select,9)
        clus_select = post_oper.pick_cluster_inverse_dist([5,9], [2.0,4.0])
        self.assertEqual(clus_select,5)
        clus_select = post_oper.pick_cluster_inverse_dist([5,9], [2.0,4.0])
        self.assertEqual(clus_select,5)

        



if __name__ == '__main__':
    unittest.main()