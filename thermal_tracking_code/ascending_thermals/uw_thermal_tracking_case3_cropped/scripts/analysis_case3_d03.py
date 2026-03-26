# Change the working directory to the directory containing the codes
import os
case = "CASE3_new"

os.chdir('/uw_thermal_tracking_case3_cropped/scripts/')

    
import matplotlib
matplotlib.use('Agg')
from analysis_plots import Composite

import sys

start_zero = bool(int(sys.argv[1]))
compute_from_scratch = bool(int(sys.argv[2]))

"""
After running 3D_tracking_runscript.py (tracked all thermals), this code computes many properties, statistics, etc. and 
creates composites of the tracked thermals. Using the position and size of the tracked thermals, it needs to retrieve 
many additional variables from the simulation output files, and it will save additional information within each thermal's folder.
However, the important output of this analysis will be saved in a separate folder (folder_name).
"""

read_domain_from_cell_tracks = False                # if the thermal tracking was done based on a previous cell tracking
csv_filename = 'cell_domains_tracking_data.csv'     # change here

GCE = False


# ************************************
# Parameters of the simulation output: 
# ************************************
dx          = 200.                                  # horizontal gridspacing in the original data in m
nz          = 60                                    # number of vertical levels to read from original data
dt          = 60                                    # time interval between output data files in s
xff         = 150                                   # maximum x-value of domain size in km
yff         = 150                                   # maximum y-value of domain size in km
path   = '/g/data/w28/ai2733/outputs_CASE3_new_budgets/splitted/'      # model output data location
header_fmt  = 'wrfout_d03_YYYY-MM-DD_'             # format for output file names (YYYY, MM, and DD 
ending      = ''                                    # ending of simulation data file names

# ************************************
# Parameters for composite making: 
# ************************************
rescale     = True                                  # rescale the grid to R-coordinates (for composites). Default is "True"
R_range     = 2.4                                   # number of thermal radii to use for computing composites 
delta_R     = 0.2                                   # subdivision size for the radial-coordinates in the composites
plt_range   = 2.3                                   # range for composite plots, in radius coordinates (should be smaller or equal to R_range)


# ************************************
# Additional settings
# ************************************

n_jobs      = 20                                    # if read_domain_from_cell_tracks is False, this is used for parallelization of interpolation 
                                                    # of the variables to common grid. Since this is done by splitting the domain, it is not 
                                                    # recommended to use a large number of jobs (probably not larger than ~20, but it may depend on domain size)
                                                    # if read_domain_from_cell_tracks is True, this is used to paralellize by cells, so it could 
                                                    # be set to the maximum available cores, or the number of cells (test?)

exp_name    = f'{case.lower()}_d03'                 # name of experiment (used for header of output files)


#*************************************
# Domain for analysis (ignored if reading from cell tracks)
# (in principle it should match the tracking 
# domain, but could be smaller too)
# ************************************
xmin = 50                                            # domain start position in x direction (in km)
xmax = 150                                           # domain end position in x direction (must be equal or less than xff) (in km)
ymin = 50                                            # domain start position in y direction (in km)
ymax = 150                                           # domain end position in y direction (must be equal or less than yff) (in km)
# ************************************
# ************************************


# ************************************
# The following settings define which DD/UD to analyze
# ************************************

folder_name  = f'composite_thermals'   
up = True
thermal_list = './thermal_*'     
replot_only  = True if compute_from_scratch == False else False  

class Composite(Composite):
    def condition (self, ixl,ixr,iyl,iyr,izlow,izup,R,D,wmax,W):
        """
        Only cases that satisfy this condition will be used for this analysis. User can customize this function if needed (no guarantee!)
        """
        #if R>0:
        #    return True
        #else:
        #    return False
        return True

def run_analysis(thermal_list, folder_name, x0=0, y0=0, xmin=None, xmax=None, ymin=None, ymax=None, xff=None, yff=None, n_jobs=n_jobs, cell_nr='', up = False, start_zero = False):
    x = Composite(R_range=R_range, delta_R=delta_R, plt_range=plt_range, dx=dx, nz=nz, header_fmt=header_fmt, dt=dt, xff=xff, yff=yff, path=path, rescale=rescale, n_jobs=n_jobs, ending=ending, folder_name=folder_name, replot=replot_only, exp_name=exp_name, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, thermal_list=thermal_list, compute_from_scratch=compute_from_scratch, x0=x0, y0=y0, GCE=GCE, cell_nr=cell_nr, up = up, start_zero = start_zero )

    x.make_mean_composite_tseries()
    x.entrainment_analysis()
    x.height_profiles()
    x.plot_histograms(Rmax=1.41, Wmax=0.5, lifetimemax=1.4, z0max=0.5, delta_zmax=0.21, Fnhmax=1.1, buoymax=1.2, Fmixmax=2.5, accmax=1.4 )
    x.composite_fields()

run_analysis(thermal_list, folder_name, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xff=xff, yff=yff, up = up, start_zero = start_zero)
