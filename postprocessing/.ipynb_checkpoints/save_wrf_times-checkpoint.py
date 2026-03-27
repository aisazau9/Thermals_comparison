import glob
import wrf
import numpy as np
import netCDF4 as nc

"""""""""""""""
Save a numpy array with all WRF times (from 5-minute outputs) for each case

Output: Times_{case}.npy

"""""""""""""""

cases = ["CASE1", "CASE2_new", "CASE3_new"]
dom = "d03"

# Read all data
times_cases = {}
for case in cases:
    # Read original outputs
    files_all = sorted(glob.glob(f"/g/data/w28/ai2733/outputs_{case}_budgets/budget/wrfout_{dom}*"))
    
    # Get times
    times_dom = wrf.extract_times([nc.Dataset(f) for f in files_all],wrf.ALL_TIMES)
    times_all = np.array([np.datetime_as_string(t_, unit='s') for t_ in times_dom])

    times_cases[case] = times_all

# Save Time
path_arrays = "/results/wrf_arrays/"
#path_arrays = "/g/data/up6/ai2733/Thermals_comparison/results/wrf_arrays/"
for case in cases:
    np.save(f"{path_arrays}/Times_{case}.npy", times_cases[case])

    