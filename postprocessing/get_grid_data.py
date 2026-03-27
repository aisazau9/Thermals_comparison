import dill
import pickle
import numpy as np
import os

"""
Loads the thermal tracking grid, extracts grid coordinates and dimensions, and saves them in a simplified pickle file for later use in WRF data processing.
"""

cases = ["CASE1", "CASE2_new", "CASE3_new"]
dom = "d03"

for case in cases:
    
    if dom == "d03":
        path_grid = f"/g/data/w28/ai2733/outputs_{case}_budgets/grid_tracking_cropped_new/{subdir}"
    
    elif dom == "d02":
        path_grid = f"/g/data/w28/ai2733/outputs_{case}_budgets/grid_tracking_cropped_new_d02/{subdir}"
    
    # Open Grid
    with open(f'{path_grid}/All_grid.pkl', 'rb') as f:
        grid_object = dill.load(f)
    
    # Data from the grid
    x_grid = grid_object.x_grid
    y_grid = grid_object.y_grid
    hgt_c  = grid_object.hgt_c
    nx = grid_object.nx
    ny = grid_object.ny
    nz = grid_object.nz
    i0, i1 = grid_object.i0, grid_object.i1
    j0, j1 = grid_object.j0, grid_object.j1
    
    # Create an xarray Dataset with all the variables I need
    x_coords = np.arange(len(x_grid))
    y_coords = np.arange(len(y_grid))
    z_coords = np.arange(len(hgt_c))
    t_coords = np.arange(grid_object.nt)
    
    data_grid = {"x_grid":x_grid,
                 "y_grid":y_grid,
                 "hgt_c":hgt_c,
                 "nx":nx,
                 "ny":ny,
                 "nz":nz,
                 "i0": i0, "i1": i1,
                 "j0":j0,  "j1":j1,
                 "x_coords":x_coords,
                 "y_coords":y_coords,
                 "z_coords":z_coords,
                 "t_coords":t_coords}
    
    del grid_object
    # Save as pickle 
    #path_save = "/g/data/up6/ai2733/Thermals_comparison/results/tracking_grid/"
    path_save = "/results/tracking_grid/"
    with open(f'{path_save}/{case}/data_grid.pkl', 'wb') as f: pickle.dump(data_grid, f)
