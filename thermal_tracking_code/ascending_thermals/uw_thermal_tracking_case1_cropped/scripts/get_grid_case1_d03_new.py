# Change the working directory to the directory containing the codes
import os
os.chdir('/thermal_tracking_case1_cropped/scripts/')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import WRF_3Dtracing_functions_new as trace
import pdb
import time
import datetime as dt
import sys

import glob
import re
import dill

idx_slice = int(sys.argv[1])
path_save = "/outputs_CASE1_budgets/grid_tracking_cropped_new/"

"""
*******************************************
PARAMETER SETTINGS:
*******************************************
"""

read_domain_from_cell_tracks = False              # if the domain for thermal tracking will be read from a file
csv_filename = 'cell_domains_tracking_data.csv'   # set csv file name here (only used if read_domain_from_cell_tracks)

path   = '/outputs_CASE1_budgets/splitted/'      # model output data location

header_fmt = 'wrfout_d03_YYYY-MM-DD_'             # format for output file names (YYYY, MM, and DD must be included!)
ending = ''                                       # in case the output file names have a specific ending (string)
GCE    = False


dx 	   = 200	# horizontal resolution of the data (m)
Dt	   = 60	    # time resolution of data (s)

nz     = 60     # number of vertical levels for tracking domain

downdrafts = True   # set to True if tracking downdrafts instead of rising thermals.
#*************************************************************
# if read_domain_from_cell_tracks is False, the following will be used:

# Use regex to extract the date and time components
files_all = sorted(glob.glob(f"{path}/{header_fmt[:10]}*"))
file_path = files_all[::10][idx_slice]
match = re.search(r'wrfout_d03_(\d{4})-(\d{2})-(\d{2})_(\d{2}):(\d{2})', file_path)

if match:
    YY0 = int(match.group(1))  # start year
    MM0 = int(match.group(2))   # start month
    DD0 = int(match.group(3))   # start day
    hr0 = int(match.group(4))    # start hour
    min0 = int(match.group(5))   # start minute

    print(f"YY0 = {YY0}")
    print(f"MM0 = {MM0}")
    print(f"DD0 = {DD0}")
    print(f"hr0 = {hr0}")
    print(f"min0 = {min0}")
else:
    print("No match found.")


nt   =  10      # number of timesteps
    
x0   =  50      # tracking domain start position in x direction (in km)
nxkm =  100      # tracking domain length in x direction (in km)
y0   =  50      # tracking domain start position in y direction (in km)
nykm =  100      # tracking domain length in y direction (in km)
#*************************************************************

w_thr			    = -0.8	    # minimum speed threshold for identifying w-max points (m/s)
qcloud_thr		    = 1e-5	    # minimum qcloud threshold for considering a wmax point (kg/kg) (including water, ice, snow, graupel and rain)
cluster_dist		= 5.*dx	    # minimum separation between different thermals (m)
min_thermal_duration= 3	        # minimum life time of a thermal in order for it to be considered (min)
avg_dist_R          = 10        # horizontal distance to consider when computing average density profile (for buoyancy) (in number of radii) 17-03-2023: this is not used for buoyancy! must check to see if it can be removed (DHD)
min_R               = 2.*dx     # minimum thermal radius to consider. Smaller radii will not be considered
s_factor            = 30000     # smoothing factor for the thermal centres based on wmax points (default is s=30000, the larger s, the smoother the trajectory)
W_min			    = -1.	    # minimum average thermal vertical velocity (m/s)
Rmax 			    = 4500	    # maximum possible radius to consider
compute_rh          = True      # compute relative humidity field
compute_theta       = True      # compute theta and theta_e

n_jobs_int          = 96        # number of jobs for parallelized interpolation
n_jobs_tracking     = 96        # number of jobs for tracking the thermals 
disc_r              = 0.8       # threshold for change in radius. If radius changes more than disc_r*100% of the smaller radius, this point is discarded.

def save_slice_grid(dx, YY0, MM0, DD0, hr0, min0, nt, x0, y0, i0, j0, nx, ny, nxkm, nykm,
        nz, path, Dt, n_jobs_int, ending, header_fmt, compute_rh,compute_theta,
        w_thr, qcloud_thr, cluster_dist, Rmax, W_min, min_thermal_duration,
        avg_dist_R, min_R, disc_r, s_factor, n_jobs_tracking, cell='', cell_mask=None, downdrafts=False):
    """
    This is the main function that calls the tracking scripts
    """
    start = time.time()
    if idx_slice == 0:
        domain = trace.Grid( dx=dx, YY0=YY0, MM0=MM0, DD0=DD0, hr0=hr0, min0=min0, nt=nt, x0=x0, y0=y0, i0=i0, j0=j0, nxi=nx, nyi=ny, nxkm=nxkm, nykm=nykm, nz=nz, path=path, dt=Dt, n_jobs=n_jobs_int, ending=ending, header_fmt=header_fmt,compute_rh=compute_rh, compute_theta = compute_theta, compute_extra_fields = True, GCE=GCE )
        
    else:
        # Read heights from first slice
        with open(f'{path_save}/grid_0.pkl', 'rb') as f:
            grid1 = dill.load(f)
        hgt000_grid1   = grid1.hgt
        hgt000_c_grid1 = grid1.hgt_c
        del grid1
        print ("Heights read from first slice")
            
        domain = trace.Grid( dx=dx, YY0=YY0, MM0=MM0, DD0=DD0, hr0=hr0, min0=min0, nt=nt, x0=x0, y0=y0, i0=i0, j0=j0, nxi=nx, nyi=ny, nxkm=nxkm, nykm=nykm, nz=nz, path=path, dt=Dt, n_jobs=n_jobs_int, ending=ending, header_fmt=header_fmt,compute_rh=compute_rh, compute_theta = compute_theta, compute_extra_fields = True, GCE=GCE, 
                           hgt000_fixed_vals   = hgt000_grid1,
                           hgt000_c_fixed_vals = hgt000_c_grid1,
                           hgt000_fixed = True)

    print('took %f minutes'%((time.time()-start)/60.))
    
    # Save it as a pickle
    with open(f'{path_save}/grid_{idx_slice}.pkl', 'wb') as f:
        dill.dump(domain, f)
    
i0 = int(x0*1e3/dx)
j0 = int(y0*1e3/dx)
i1 = int(np.rint(x0*1e3/dx + nxkm*1e3/dx))
j1 = int(np.rint(y0*1e3/dx + nykm*1e3/dx))
nx = i1 - i0
ny = j1 - j0
cell_mask = np.ones([nt,nx,ny,nz])
save_slice_grid(dx, YY0, MM0, DD0, hr0, min0, nt, x0, y0, None, None, None, None, nxkm, nykm,
         nz, path, Dt, n_jobs_int, ending, header_fmt, compute_rh,compute_theta,
         w_thr, qcloud_thr, cluster_dist, Rmax, W_min, min_thermal_duration,
         avg_dist_R, min_R, disc_r, s_factor, n_jobs_tracking, cell='', 
         cell_mask=cell_mask, downdrafts=downdrafts)
