#### Post-processing scripts #### 

This folder contains the code used to post-process the WRF outputs and the results from the thermal tracking algorithm.

------------------------------------------------------------------------------------------------------

The following data is not included in this repository due to storage constraints, but access can be obtained upon reasonable request:

-> WRF outputs: wrfout_* files
These are used to run save_wrf_times.py, get_wrf_grid.py and get_wrf_rain.py, and save_profilestime.py

-> WPS outputs: geo_em files
Used to run get_wrf_domains.py

-> Interpolated WRF grid from tracking code: All_grid.pkl files
These are used to run get_grid_data.py

-> thermals_all/*/*.npy : the data of each tracked thermal
Used in save_tracking_results.py

-> wrf_all.pkl and wrf_all_rain.pkl: outputs from get_wrf_grid.py and get_wrf_rain.py
------------------------------------------------------------------------------------------------------

The following data is included in this repository but needs to be unziped before loading it:

-> thermal_tracking_code/{ascending/descending}_thermals/{subfolder_case}/Composite_thermals/*.npy : the data of the composite of the thermals
Used in save_tracking_results.py

Read thermal_tracking_code/README_zipfiles.txt for more information