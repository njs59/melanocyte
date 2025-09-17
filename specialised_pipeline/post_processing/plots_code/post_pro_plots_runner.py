# import lineage_tracer_function as ltf
# import cluster_tracker_function as ctf
# import size_and_loc_plotter as slp
import hist_plotter_gif as hsg

# These three parameters are needed for accessing data and saving to files
# Parameters
basedir = '/Users/Nathan/Documents/Oxford/DPhil/melanocyte/'
data_folder = 'data/Still_Images_with_BF_for_Nathan/'
fileID = '.tif'
# time = 0
min_clus_size = 150


base_str = "VID289"

well_loc = 'E2'


multi_loc = ["A1","A3","A5","B2","B4","B6","D1","D3","D5","E2","E4","E6"]

filename = 'VID289_E2_1_04d15h00m'

#cluster_lineage = ltf.lineage_tracer(51,97, basedir, exp_date, well_loc, plots = False)
#slp.size_and_loc_tracker(37, 97, 10, basedir, exp_date, well_loc, cluster_lineage)

hsg.hist_size_plotter(basedir, multi_loc, base_str, filename, timepoint=0)

# slp.size_and_loc_tracker(37, 97, 5, basedir, exp_type, exp_date, well_loc, [])

'''
Cluster tracker tracks an individually taggged cluster over time
Input arguments: 
    start_time, first timepoint to plot
    end_time, final timepoint to plot
    timejump, number of timesteps between each plot
    cluster_index_final_time, row in final time to select cluster ID tag from
    basedir,
    exp_date
    well_loc

Output:
    Series of plots 
'''
# ctf.cluster_tracker(37, 94, 5, 10, basedir, exp_type, exp_date, well_loc)



'''
Lineage tracer traces each cluster in turn back in time using event to find contributing clusters

Input arguments:
    start_time: Timepoint to be traced back to and plotted
    end_time: Timepoint to trace from and plot cluster 
    basedir,
    exp_date,
    well_loc,

Output:
    Series of subplots of start_time and end_time 
    for each cluster's lineage next to each other

'''
# cluster_lineage = ltf.lineage_tracer(51,94, basedir, exp_type, exp_date, well_loc, plots = True)


# Plot cluster size over time

# Plot cluster centre over time (spider plot)