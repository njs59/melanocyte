import lineage_tracer_function as ltf
import cluster_tracker_function as ctf
import size_and_loc_plotter as slp
import hist_size_gif as hsg

# These three parameters are needed for accessing data and saving to files
basedir = '/Users/Nathan/Documents/Oxford/DPhil/'
exp_type = 'In_vitro_homogeneous_data/'
exp_date = '2017-02-03'
# exp_date = '2017-02-13'
# exp_date = '2017-03-10'
# exp_date = '2017-03-16'

well_loc = 's11'
# well_loc = 's04'

# multi_loc = ['s11', 's12']
multi_loc = ['s073', 's074']

#cluster_lineage = ltf.lineage_tracer(51,97, basedir, exp_date, well_loc, plots = False)
#slp.size_and_loc_tracker(37, 97, 10, basedir, exp_date, well_loc, cluster_lineage)

# hsg.hist_size_plotter(basedir, exp_type, exp_date, multi_loc, 37, 142, 5)

slp.size_and_loc_tracker(37, 97, 1, basedir, exp_type, exp_date, well_loc, [])

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
ctf.cluster_tracker(37, 94, 5, 10, basedir, exp_type, exp_date, well_loc)



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