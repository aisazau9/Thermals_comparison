###############################################
A. Isaza - 2026

The tracking code was modified to also track descending thermals (with downdrafts = True in *run_tracking*, and up = False in *run_analysis* scripts). 

In this repository, each folder contains the results of the tracking algorithm for each event (subfolder: thermals_all), the results of the composite analysis (subfolder: composite_thermals.zip), and the Python codes used (subfolder: scripts), all run using the scripts in subfolder: jobs. 

The scripts in the jobs subfolder were run in the following order:

Just once (for either descending or ascending thermals):
	1. run_grid_slice0.sh (interpolates first subgrid - necessary due to memory restrictions)
	2. run_grid_sliceAll1.sh (interpolates remaining grid)
	3. merge_grids.sh (merge grids)
For thermals in both directions:
	4. run_tracking.sh (run tracking algorithm)
	5. run_analysis.sh (run analysis algorithm - with start_zero = 0 and compute_from_scratch = 0)


###############################################
D. Hernandez-Deckers - 2024

The purpose of the code in this repository is to identify, track and analyze
cumulus thermals in large eddy simulations of atmospheric convection. This is
done offline, so high resolution (~1min) output from a simulation must be 
stored and accessed by this code. It can be used with output from the WRF 
model, but also with output from the Goddard Cumulus Ensemble (GCE) model.

This code is provided "as is", with no warranty of any kind. Questions should
be addressed to dhernandezd@unal.edu.co.

If you use this code for any scientific publication, please cite the following
papers, where details about the tracking method are given:

Hernandez-Deckers, D., & Sherwood, S. C. (2016). A Numerical Investigation
of Cumulus Thermals, Journal of the Atmospheric Sciences, 73(10), 4117-4136.
https://doi.org/10.1175/JAS-D-15-0385.1 

Hernandez-Deckers, D., & Sherwood, S. C. (2018). On the Role of Entrainment
in the Fate of Cumulus Thermals, Journal of the Atmospheric Sciences, 75(11),
3911-3924. https://doi.org/10.1175/JAS-D-18-0077.1


