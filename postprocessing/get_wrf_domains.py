import os
import pickle
import netCDF4 as nc
import wrf

"""
Plot to get WRF domains extents
"""

cases = ["CASE1", "CASE2_new", "CASE3_new"]

# Get WRF domains
path_domains = "/g/data/up6/ai2733/WRF_real/WRF/"

color_cases = {"CASE1":"k", 
              "CASE2_new": "blue", 
              "CASE3_new":"crimson"}

name_cases_maps    = {"CASE1": "Event 1: Moree",
                 "CASE2_new": "Event 2 : Wagga Wagga", 
                 "CASE3_new": "Event 3: Tamworth"}

lat_lon_cases = {"CASE1": (-29.49, 149.85), 
                "CASE2_new": (-35.16, 147.46), 
                 "CASE3_new": (-31.07,150.84)}

# Get extent of WRF domains
extent_d3 = {}
extent_d2 = {}
extent_d1 = {}

for case in cases:
    extent_dx = {}
    for dom_ in ["d03", "d02", "d01"]:
        ncfile_dx = nc.Dataset(f"{path_domains}/WPS_{case}/geo_em.{dom_}.nc", "r")
        var_dx =  wrf.getvar(ncfile_dx, "HGT_M", 0)
        lats_dx, lons_dx = wrf.latlon_coords(var_dx)
        lats_final_dx, lons_final_dx = wrf.to_np(lats_dx)[:,0], wrf.to_np(lons_dx)[0,:]
        lon1_dx, lon2_dx, lat1_dx, lat2_dx =  (lons_final_dx[0], lons_final_dx[-1], lats_final_dx[0], lats_final_dx[-1])
        ncfile_dx.close()
        extent_dx[dom_] = [lon1_dx, lon2_dx, lat1_dx, lat2_dx]
    extent_d3[case] = extent_dx["d03"]
    extent_d2[case] = extent_dx["d02"]
    extent_d1[case] = extent_dx["d01"]

# Save
path_save = "/g/data/up6/ai2733/Thermals_comparison/results/wps_domains/"
with open(f'{path_save}/extent_d3.pkl', 'wb') as f: pickle.dump(extent_d3, f)
with open(f'{path_save}/extent_d2.pkl', 'wb') as f: pickle.dump(extent_d2, f)
with open(f'{path_save}/extent_d1.pkl', 'wb') as f: pickle.dump(extent_d1, f)
