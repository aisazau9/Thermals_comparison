import sys
import glob
import numpy as np
import xarray as xr
import netCDF4 as nc
import wrf

"""""""""""""""""""""""""""""""""""""""""""""

# Save arrays with the spatial averages profiles of a certain variable, 80 km from AWS, during certain dates, and coarsen 200 m data to 1 km
# Inputs:
# case_idx (CASE, from 0 to 2)
# var_ (WRF variable)
#
# Outputs:
# TimeProf_{case}_d03_{var}_80kmAWS_originalgrid.npy
# TimeProf_{case}_d03_{var}_80kmAWS_coarsen.npy
# TimeProf_{case}_d02_{var}_80kmAWS_coarsen.npy
#

"""""""""""""""""""""""""""""""""""""""""""""
#path_arrays = "/g/data/up6/ai2733/Thermals_comparison/results/wrf_arrays/"
path_arrays = "/results/wrf_arrays/"

case_idx = int(sys.argv[1]) # From 0 to 2
var_     = str(sys.argv[2]) 
case     = ["CASE1", "CASE2_new", "CASE3_new"][case_idx]

def latlon_dist(lat, lon, lats, lons):

    #Function from mr: Calculate great circle distance (Harversine) between a lat lon point (lat, lon) and a list of lat lon
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

# General info
path_save = "/g/data/up6/ai2733/WRF_cases/"
path_figs = "/g/data/up6/ai2733/Codes/Figures/"

lat_lon_cases = {"CASE1": (-29.49, 149.85), 
                "CASE2_new": (-35.16, 147.46), 
                 "CASE3_new": (-31.07,150.84) }

dates_filter_cases = {"CASE1":{"d01":("2015-12-16_00:00:00", "2015-12-16_12:00:00"),
                              "d02":("2015-12-16_00:05:00", "2015-12-16_11:35:00"),
                              "d03":("2015-12-16_00:05:00", "2015-12-16_11:35:00")},
                     "CASE2_new":{"d01":("2009-01-20_00:00:00", "2009-01-20_12:00:00"),
                              "d02":("2009-01-20_00:05:00", "2009-01-20_12:05:00"),
                              "d03":("2009-01-20_00:05:00", "2009-01-20_12:05:00")},
                     "CASE3_new":{"d01":("2017-02-17_00:00:00", "2017-02-17_12:00:00"),
                              "d02":("2017-02-17_00:05:00", "2017-02-17_12:35:00"),
                              "d03":("2017-02-17_00:05:00", "2017-02-17_12:35:00")}}

# To save arrays: 80 km from station
def save_arrays(case, var):
    domains = ["d02", "d03"]
    path_wrf = f"/g/data/w28/ai2733/outputs_{case}/"
    wrflist = {}
    for dom in domains: wrflist[dom] = [nc.Dataset(f) for f in np.sort(glob.glob(f"{path_wrf}wrfout_{dom}_*"))]
    
    # Find the indexes of the files between dates: to save memory when reading the files
    def find_idx_file(dom, date_i, date_f): # Dates have to be exact
        list_files = np.sort(glob.glob(f"{path_wrf}wrfout_{dom}_*"))
        idx_ti = list(list_files).index(f'{path_wrf}wrfout_{dom}_{date_i}') 
        try:
            idx_tf = list(list_files).index(f'{path_wrf}wrfout_{dom}_{date_f}') 
            print (len(list_files), "files in total, ", (idx_tf - idx_ti) + 1, "files between dates")
        except: 
            idx_tf = -1
            print (len(list_files), "files in total, last date not found ")
        return (int(idx_ti), int(idx_tf))
    
    idx_dom = {}
    for dom in domains:
        di, df = dates_filter_cases[case][dom]
        idx_dom[dom] = find_idx_file(dom, di, df)
    
    # Get variables between times of interest
    wrf_all = {}
    wrf_all[var]  = {}
    wrf_temp = {}
    for dom in domains:
        ti, tf = idx_dom[dom]
        aux_ = wrf.getvar(wrflist[dom][ti:tf] if tf == -1 else wrflist[dom][ti:tf+1], var, timeidx=wrf.ALL_TIMES, method="cat")
        # Remove duplicates
        wrf_all[var][dom] = aux_.drop_duplicates("Time", keep='last')
        del aux_
        print (var, dom, "read")

    # Get pixels within distance from AWS
    min_w = {}
    for dom in domains:
        var_all  = wrf_all[var][dom]
        rad = 80
        lat, lon = lat_lon_cases[case]
        dist = latlon_dist(lat, lon, 
                    wrf.getvar(wrflist[dom], "XLAT", timeidx=0).values, 
                    wrf.getvar(wrflist[dom], "XLONG", timeidx=0).values)
        
        var_all  = xr.where((dist <= rad), var_all, np.nan)
        min_w[dom]    = var_all

        # Coarsen 200m to 1 km
        if dom == "d03":
            min_w[f"{dom}_coarsen"]    = min_w[dom].coarsen(south_north=5, west_east=5, boundary='trim').mean()
    
    # d03
    aux_d03      = min_w["d03"].dropna(dim = "south_north", how = "all").dropna(dim = "west_east", how = "all")
    # d02
    aux_d02      = min_w["d02"].dropna(dim = "south_north", how = "all").dropna(dim = "west_east", how = "all")
    # d03 coarsen
    aux_d03_coarsen  = min_w["d03_coarsen"].dropna(dim = "south_north", how = "all").dropna(dim = "west_east", how = "all")

    if var == "RAINNC":
        aux_d03 = aux_d03.diff("Time")
        aux_d02 = aux_d02.diff("Time")
        aux_d03_coarsen = aux_d03_coarsen.diff("Time")
                
    # Save spatial averages profiles

    # Save d03 original grid
    arr_d03 = aux_d03.mean(["south_north", "west_east"]).to_numpy()
    np.save(f'{path_arrays}/arrays/TimeProf_{case}_d03_{var}_80kmAWS_originalgrid.npy', arr_d03)

    # Save coarsen: exactly same grid as d02!
    if case == "CASE1":
        aux_d03_coarsen = aux_d03_coarsen.isel(south_north = slice(1,161)) # I am making sure lat and lons are the same!
    if case == "CASE2_new":
        aux_d03_coarsen = aux_d03_coarsen.isel(south_north = slice(1,160), west_east = slice(1,160))
    if case == "CASE3_new":
        aux_d03_coarsen = aux_d03_coarsen.isel(west_east = slice(0, 160))

    assert (aux_d03_coarsen.shape == aux_d02.shape)
        
    arr_d02         = aux_d02.mean(["south_north", "west_east"]).to_numpy()
    arr_d03_coarsen = aux_d03_coarsen.mean(["south_north", "west_east"]).to_numpy()

    np.save(f'{path_arrays}/TimeProf_{case}_d02_{var}_80kmAWS_coarsen.npy', arr_d02)
    np.save(f'{path_arrays}/TimeProf_{case}_d03_{var}_80kmAWS_coarsen.npy', arr_d03_coarsen)

    del aux_d02
    del aux_d03
    del aux_d03_coarsen
    del arr_d02
    del arr_d03
    del arr_d03_coarsen
    print (f"Arrays saved")

save_arrays(case, var_)


