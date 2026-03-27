import numpy as np
import os
import sys
import glob
import pickle
import dill

import xarray as xr
import netCDF4 as nc
import wrf
import xwrf

from matplotlib import rcParams

"""""""""""""""""""""""""""""""""""""""""""""

Similar to get_wrf_grid but just saves RAINNC

Extracts rainfall (RAINNC) from WRF outputs for a selected case and domain, crops it to the tracking grid, aligns simulation times with the grid time window, and saves the processed data as a pickle file 

Output: wrf_all_rain.pkl

"""""""""""""""""""""""""""""""""""""""""""""

case_idx = int(sys.argv[1]) # From 0 to 2
dom      = str(sys.argv[2]) 
case     = ["CASE1", "CASE2_new", "CASE3_new"][case_idx]
subdir = "" if dom == "d03" else "Alld02/"

rcParams['font.family'] = 'sans-serif'

date_cases    = {"CASE1": ("2015-12-16 07:00", "2015-12-16 07:59"),
                "CASE2_new": ("2009-01-20 04:30", "2009-01-20 05:29"),
                "CASE0": ("2016-01-14 03:31", "2016-01-14 04:00")}

lat_lon_cases = {"CASE0": (-33.95,151.17),
                 "CASE1": (-29.49, 149.85), 
                 "CASE2": (-35.16, 147.46), 
                 "CASE3": (-31.07,150.84),
                "CASE2_new": (-35.16, 147.46), 
                 "CASE3_new": (-31.07,150.84), 
                "CASE4": (-33.60,150.78), 
                "CASE5": (-33.90,150.73),
                "CASE4_new": (-33.60,150.78), 
                "CASE5_new": (-33.90,150.73)}

min_t_all_cases = {"CASE1": 420,
                  "CASE2_new": 270,
                  "CASE3_new": 450}

def latlon_dist(lat, lon, lats, lons):

        #Calculate great circle distance (Harversine) between a lat lon point (lat, lon) and a list of lat lon
        # points (lats, lons)

        R = 6373.0

        lat1 = np.deg2rad(lat)
        lon1 = np.deg2rad(lon)
        lat2 = np.deg2rad(lats)
        lon2 = np.deg2rad(lons)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return (R * c)
    

if dom == "d03": path_save = f"/g/data/w28/ai2733/outputs_{case}_budgets/grid_tracking_cropped_new/{subdir}"

elif dom == "d02": path_save = f"/g/data/w28/ai2733/outputs_{case}_budgets/grid_tracking_cropped_new_d02/{subdir}"

# Open data from grid
with open(f'{path_save}/data_grid.pkl', 'rb') as f: data_grid = dill.load(f)
x_grid = data_grid["x_grid"]
y_grid = data_grid["y_grid"]
hgt_c  = data_grid["hgt_c"]
nx = data_grid["nx"]
ny = data_grid["ny"]
nz = data_grid["nz"]
i0, i1 = data_grid["i0"], data_grid["i1"]
j0, j1 = data_grid["j0"], data_grid["j1"]

# Create an xarray Dataset with all the variables I need
x_coords = data_grid["x_coords"]
y_coords = data_grid["y_coords"]
z_coords = data_grid["z_coords"]
t_coords = data_grid["t_coords"]
    
# Read original outputs
files_all = sorted(glob.glob(f"/g/data/w28/ai2733/outputs_{case}_budgets/budget/wrfout_{dom}*"))

# Get times
times_dom = wrf.extract_times([nc.Dataset(f) for f in files_all],wrf.ALL_TIMES)
times_all = [np.datetime_as_string(t_, unit='s') for t_ in times_dom]

# Extract the initial and final date from GRID
files_grid = sorted(glob.glob(f"/g/data/w28/ai2733/outputs_{case}_budgets/splitted/wrfout_{dom}*")) 
date_time_ini = files_grid[0].split(f'wrfout_{dom}_')[1].replace('_', 'T', 1)
date_time_fin = files_grid[-1].split(f'wrfout_{dom}_')[1].replace('_', 'T', 1)
idx_ini = times_all.index(date_time_ini)
idx_fin = times_all.index(date_time_fin)

# Get just those times
times_dom = times_dom[idx_ini:idx_fin+1]
times_all = times_all[idx_ini:idx_fin+1]
if len(times_all) != len(t_coords):
    times_dom = times_dom[:-1]
    times_all = times_all[:-1]
assert (len(times_all) == len(t_coords))
assert(times_all[0] == date_time_ini)

if times_all[-1] != date_time_fin:
    idx_fin = idx_fin - 1

# Read WRF
wrf_all = xr.open_mfdataset(files_all, concat_dim = "Time", combine = "nested", chunks = "auto").sortby("Time")
wrf_all = wrf_all.xwrf.postprocess() 
wrf_all = wrf_all.isel(Time = slice(idx_ini, idx_fin+1),  x = slice(i0, i1), y = slice(j0, j1), z = slice(0, nz))

RAINNC_arr      = wrf_all['RAINNC'].transpose('x', 'y', 'Time').to_numpy()
del wrf_all

wrf_all = xr.Dataset(
{  'RAINNC':    (['west_east', 'south_north', 'Time'],  RAINNC_arr)},
coords={
    'west_east':   x_coords,  
    'south_north': y_coords,  
    'z':           z_coords,  
    'Time':        t_coords})

# Save as pickle 
with open(f'{path_save}/wrf_all_rain.pkl', 'wb') as f: pickle.dump(wrf_all, f)
