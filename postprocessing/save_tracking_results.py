import glob
import pickle
import dill
import numpy as np

"""
Thermal and Downdraft Composite Data Extraction

This script reads composite data produced by the thermal tracking algorithm for
three case studies (CASE1, CASE2_new, CASE3_new) and stores the processed data
into pickle files for later analysis.

The script loads:
- thermal and downdraft composite fields
- time series of radius and vertical velocity
- momentum budget terms
- entrainment and detrainment
- vertical profiles
- histograms of shape parameters
- hydrometeor fields
- thermal characteristics (mass, lifetime, wmax, etc.)
- grid location and centre positions

For each case and each type (ascending thermals and descending downdrafts),
the function `get_data()` reads all composite files and returns the relevant
variables, which are then saved into a dictionary and exported as pickle files.

Output:
    /results/{CASE}_{d03}_{dd/ud}.pkl

where:
    CASE = CASE1, CASE2_new, CASE3_new
    d03  = domain 3
    dd   = downdrafts
    ud   = updrafts 

"""

name_cases = {"CASE1": "Event 1", 
             "CASE2_new": "Event 2",
             "CASE3_new": "Event 3"}

min_t_all_cases = {"CASE1": 420,
                  "CASE2_new": 270,
                  "CASE3_new": 450}

init_t_all_cases = {"CASE1": "2015-12-16 07:00",
                  "CASE2_new": "2009-01-20 04:30",
                  "CASE3_new": "2017-02-17 07:30"}

def get_dirs(case, downdrafts):
    """
    Returns paths and file lists for d03 thermal tracking cases.

    Parameters:
        case (str): Case name, e.g., "CASE1" or "CASE2_new"
        downdrafts (bool): True for descending thermals, False for ascending thermals

    Returns:
        path_track (str): Base folder of tracking data
        path_grid (str): Folder where grids are saved
        dd_all (list): Sorted list of npy files for each thermal
        index_all (list): Index list corresponding to dd_all
        dd_all_entr (list): Sorted list of net entrainment files
    """

    # Determine folder prefix
    prefix  = "thermal_tracking" if downdrafts else "uw_thermal_tracking"
    prefix2 = "descending_thermals" if downdrafts else "ascending_thermals"

    # Handle CASE1 vs others
    base_case = case.lower() if case == "CASE1" else case.split("_new")[0].lower()

    # Paths
    #path_track = f"/g/data/up6/ai2733/Thermals_comparison/thermal_tracking_code/{prefix2}/{prefix}_{base_case}_cropped/thermals_all/"
    path_track = f"/thermal_tracking_code/{prefix2}/{prefix}_{base_case}_cropped/thermals_all/"
    #path_grid = f"/g/data/up6/ai2733/Thermals_comparison/results/tracking_grid/{case}/"
    path_grid = f"/results/tracking_grid/{case}/"

    # File patterns
    file_type = "downdraft_*" if downdrafts else "thermal_*"
    dd_all      = sorted(glob.glob(f"{path_track}/{file_type}/{file_type}_data.npy"))
    dd_all_entr = sorted(glob.glob(f"{path_track}/{file_type}/{file_type}_net_entr_term.npy"))

    index_all = list(range(len(dd_all)))

    return path_track, path_grid, dd_all, index_all, dd_all_entr


# Function to read all tracking data
def get_data(case, dom, downdrafts = True, weighted = False, start_zero = False):
    # Select path
    path_case, path_save, dd_all, index_all, dd_all_entr = get_dirs(case, downdrafts)

    # Select which subfolders I will read the composite from
    folder_comp = "composite_thermals" # ONCE IT HAS BEEN UNZIPED!

    # ------------------------------
    # Location of thermals
    # ------------------------------

    path_file = glob.glob(f"{path_case}/{folder_comp}/x_centre_c.npy")[0]
    x_centre       = np.load(path_file, allow_pickle=True, encoding='latin1'  )    
    x_centre = x_centre*1e3
    
    path_file = glob.glob(f"{path_case}/{folder_comp}/y_centre_c.npy")[0]
    y_centre       = np.load(path_file, allow_pickle=True, encoding='latin1'  )   
    y_centre = y_centre*1e3
    
    path_file = glob.glob(f"{path_case}/{folder_comp}/z_centre_c.npy")[0]
    z_centre       = np.load(path_file, allow_pickle=True, encoding='latin1'  )   

    # Interpolate to get positions
    with open(f'{path_save}/data_grid.pkl', 'rb') as f: data_grid = dill.load(f)
    x_grid = data_grid["x_grid"]
    y_grid = data_grid["y_grid"]
    nx = data_grid["nx"]
    ny = data_grid["ny"]

    x_centre_idx, y_centre_idx = [], []
    for thermal_idx in index_all:
        # Positions of w centre
        ix_centr = np.interp(x_centre[thermal_idx], x_grid, np.arange(nx))
        iy_centr = np.interp(y_centre[thermal_idx], y_grid, np.arange(ny))
        
        x_centre_idx.append(ix_centr)
        y_centre_idx.append(iy_centr)

    x_centre_idx = np.round(x_centre_idx)
    y_centre_idx = np.round(y_centre_idx)
    
    # ------------------------------
    # Composite fields 
    # ------------------------------
    path_file = glob.glob(f"{path_case}/{folder_comp}/uvwp_grid.npy")[0]
    data       = np.load(path_file, allow_pickle=True, encoding='latin1'  ) 
    x_new = data[0]
    y_new = data[1]
    z_new = data[2]

    path_file = glob.glob(f"{path_case}/{folder_comp}/w_dev_grossmean.npy")[0]
    w_dev_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )    
    
    path_file = glob.glob(f"{path_case}/{folder_comp}/u_dev_grossmean.npy")[0]
    u_dev_grossmean = np.load(path_file, allow_pickle=True, encoding='latin1'  )    

    path_file = glob.glob(f"{path_case}/{folder_comp}/u_dev_grossmean.npy")[0]
    v_dev_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )    

    # Other variables
    path_file = glob.glob(f"{path_case}/{folder_comp}/rh_grossmean.npy")[0]
    rh_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )    
    
    path_file = glob.glob(f"{path_case}/{folder_comp}/qvapor_grossmean.npy")[0]
    qvapor_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )  

    path_file = glob.glob(f"{path_case}/{folder_comp}/qrain_grossmean.npy")[0]
    qrain_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )

    path_file = glob.glob(f"{path_case}/{folder_comp}/qcloud_grossmean.npy")[0]
    qcloud_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )
    
    path_file = glob.glob(f"{path_case}/{folder_comp}/qicesnow_grossmean.npy")[0]
    qicesnow_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )

    path_file = glob.glob(f"{path_case}/{folder_comp}/qghail_grossmean.npy")[0]
    qghail_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )

    path_file = glob.glob(f"{path_case}/{folder_comp}/buoy_mean.npy")[0]
    buoy_mean_t = np.load(path_file, allow_pickle=True, encoding='latin1'  )

    # Others
    path_file = glob.glob(f"{path_case}/{folder_comp}/detr_mean.npy")[0]
    detrainment_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )

    path_file = glob.glob(f"{path_case}/{folder_comp}/entr_mean.npy")[0]
    entrainment_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )

    path_file = glob.glob(f"{path_case}/{folder_comp}/latheat_grossmean.npy")[0]
    latheat_grossmean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )

    
    #path_file = glob.glob(f"{path_case}/{folder_comp}/buoy_mean_z.npy")[0]
    #buoy_mean       = np.load(path_file, allow_pickle=True, encoding='latin1'  )

    # ----------------------------------
    # Profile: other vars (NOT weighted)
    # ---------------------------------
    path_file = glob.glob(f"{path_case}/{folder_comp}/weighted_profile_OtherVars_all.npy")[0]
    data       = np.load(path_file, allow_pickle=True, encoding='latin1'  )    
    Z             = data[0]
    N_avg         =  data[1]
    lifetime_avg  =  data[2]
    wmax_avg      =  data[3]
    massflux_avg  =  data[4]
    
    # ------------------------------
    # Times series Number of Cases
    # ------------------------------
    path_file = glob.glob(f"{path_case}/{folder_comp}/mean_composite_NumberCases.npy")[0]
    #print (path_file)
    data  = np.load(path_file, allow_pickle=True, encoding='latin1'  )
    #t_range = data[0]
    N_t     = data[1]

    # ------------------------------
    # Times series W and R
    # ------------------------------
    if weighted: path_file = glob.glob(f"{path_case}/{folder_comp}/weighted_mean_composite_RW.npy")[0]
    else: path_file = glob.glob(f"{path_case}/{folder_comp}/mean_composite_RW.npy")[0]
    #print (path_file)
    data  = np.load(path_file, allow_pickle=True, encoding='latin1'  )
    t_range = data[0]
    R_mean = data[1]
    W_mean = data[2]
    R_std = data[3]
    R_10  = data[4]
    R_90  = data[5]
    W_std = data[6]
    W_10  = data[7]
    W_90  = data[8]
    
    # ------------------------------
    # Times series budget terms
    # ------------------------------
    if weighted: path_file = glob.glob(f"{path_case}/{folder_comp}/weighted_mean_composite_mom_budget.npy")[0]
    else: path_file = glob.glob(f"{path_case}/{folder_comp}/mean_composite_mom_budget.npy")[0]

    folder = ""
    data  = np.load(path_file, allow_pickle=True, encoding='latin1'  )
    t_range   = data[0]
    acc_mean  = data[1]
    Fres_mean = data[2]
    Fnh_mean  = data[3]
    buoy_mean = data[4]
    acc_10    = data[5]
    acc_90    = data[6]
    Fres_10   = data[7]
    Fres_90   = data[8]
    Fnh_10    = data[9]
    Fnh_90    = data[10]
    buoy_10   = data[11]
    buoy_90   = data[12]

    # ------------------------------
    # Times series entrainment term
    # ------------------------------
    if weighted: path_file = glob.glob(f"{path_case}/{folder_comp}/weighted_mean_composite_net_entr_term.npy")[0]
    else: path_file = glob.glob(f"{path_case}/{folder_comp}/mean_composite_net_entr_term.npy")[0]
    data  = np.load(path_file, allow_pickle=True, encoding='latin1'  )
    t_range2            = data[0] + 0.5 #data[0] = t_range[:-1]-0.5
    net_entr_term_mean = data[1]
    net_entr_term_std  = data[2]

    # ------------------------------
    # Profile W and R
    # ------------------------------
    
    if weighted: path_file = glob.glob(f"{path_case}/{folder_comp}/weighted_profile_RW_all.npy")[0]
    else: path_file = glob.glob(f"{path_case}/{folder_comp}/profile_RW_all.npy")[0]
    data  = np.load(path_file, allow_pickle=True, encoding='latin1'  )
    Z          = data[0]
    R_avg      =  data[1]*1e3
    W_avg      =  data[2]
    net_entr_avg =  data[3]
    Z2         =  data[4]
    net_entr_l =  data[5]
    net_entr_r =  data[6]

    # ------------------------------        
    # Profile budget
    # ------------------------------
    if weighted: path_file = glob.glob(f"{path_case}/{folder_comp}/weighted_profile_mom_budget_all.npy")[0]
    else: path_file = glob.glob(f"{path_case}/{folder_comp}/profile_mom_budget_all.npy")[0]
    data  = np.load(path_file, allow_pickle=True, encoding='latin1'  )
    Z             = data[0]
    acc_avg       =  data[1]
    Fnh_avg       =  data[2]
    Fres_avg      =  data[3]
    buoy_avg      =  data[4]
    Fentr_avg     =  data[5]
    Z2            =  data[6]

    # ------------------------------        
    # Histograms of shape parameters
    # ------------------------------  
    
    path_file1 = glob.glob(f"{path_case}/{folder_comp}/iz_straight_histogram.npz")[0]        
    data  = np.load(path_file1, allow_pickle=True, encoding='latin1'  )
    ix_left      =  data["arr_0"]
    ix_right     =  data["arr_1"]
    bins_x         =  data["arr_2"]
    weights_x      =  data["arr_3"]

    path_file2 = glob.glob(f"{path_case}/{folder_comp}/iz_straight_histogram.npz")[0]
    data  = np.load(path_file1, allow_pickle=True, encoding='latin1'  )
    iz_up      =  data["arr_0"]
    iz_low     =  data["arr_1"]
    bins_z     =  data["arr_2"]
    weights_z  =  data["arr_3"]

    # ------------------------------        
    # Budget terms all thermals
    # ------------------------------

    # Get budget terms
    acc_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/acc_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    Fres_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/Fres_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    buoy_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/buoy_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    fnh_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/Fnh_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    fmix_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/net_entr_term_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    t_range  = np.load( glob.glob(f"{path_case}/{folder_comp}/t_range.npy")[0], allow_pickle=True, encoding='latin1'  )

    # Other vars
    r_c = np.load( glob.glob(f"{path_case}/{folder_comp}/R_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    r_c_mean = np.nanmean(r_c, axis = 0)
    r_c_std  =  np.nanstd(r_c, axis = 0)
    
    # Mean
    acc_c_mean = np.nanmean(acc_c, axis = 0)
    Fres_c_mean = np.nanmean(Fres_c, axis = 0)
    buoy_c_mean = np.nanmean(buoy_c, axis = 0)
    fnh_c_mean = np.nanmean(fnh_c, axis = 0)
    fmix_c_mean = np.nanmean(fmix_c, axis = 0)
    
    # std
    acc_c_std = np.nanstd(acc_c, axis = 0)
    Fres_c_std = np.nanstd(Fres_c, axis = 0)
    buoy_c_std = np.nanstd(buoy_c, axis = 0)
    fnh_c_std = np.nanstd(fnh_c, axis = 0)
    fmix_c_std = np.nanstd(fmix_c, axis = 0)

    # ------------------------------        
    # Characteristics all thermals
    # ------------------------------

    wmax_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/wmax_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    W_c     = np.load(glob.glob(f"{path_case}/{folder_comp}/W_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    time_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/time_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    R_c     = np.load(glob.glob(f"{path_case}/{folder_comp}/R_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    z_centre_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/z_centre_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    mass_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/mass_c.npy")[0], allow_pickle=True, encoding='latin1'  )

    # Hydrometeors
    qghail_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/qghail_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    qcloud_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/qcloud_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    qicesnow_c  = np.load(glob.glob(f"{path_case}/{folder_comp}/qicesnow_c.npy")[0], allow_pickle=True, encoding='latin1'  )
    qrain_c     = np.load(glob.glob(f"{path_case}/{folder_comp}/qrain_c.npy")[0], allow_pickle=True, encoding='latin1'  )    

    
    return x_centre_idx, y_centre_idx, z_centre, wmax_c, W_c, time_c, R_c, z_centre_c, mass_c, path_case, N_t, t_range, R_mean, W_mean, R_std, R_10, R_90, W_std, W_10, W_90, acc_mean,  Fres_mean,Fnh_mean,buoy_mean,acc_10,acc_90, Fres_10, Fres_90,Fnh_10, Fnh_90, buoy_10, buoy_90, t_range2, net_entr_term_mean, net_entr_term_std, Z, R_avg, W_avg, net_entr_avg, Z2, net_entr_l, net_entr_r, acc_avg, Fnh_avg, Fres_avg, buoy_avg, Fentr_avg, ix_left, ix_right, bins_x, weights_x, iz_up, iz_low, bins_z, weights_z,acc_c, Fres_c, buoy_c, fnh_c, fmix_c, r_c, r_c_mean, r_c_std, acc_c_mean, Fres_c_mean,buoy_c_mean, fnh_c_mean, fmix_c_mean,  acc_c_std, Fres_c_std, buoy_c_std, fnh_c_std, fmix_c_std,  N_avg, lifetime_avg, wmax_avg, massflux_avg,     x_new, y_new, z_new, w_dev_grossmean, u_dev_grossmean, v_dev_grossmean, rh_grossmean, qvapor_grossmean, qrain_grossmean, qcloud_grossmean, qicesnow_grossmean, qghail_grossmean, buoy_mean_t, detrainment_grossmean, entrainment_grossmean, latheat_grossmean, qghail_c, qcloud_c, qicesnow_c, qrain_c

# Read all the tracked data
weighted   = False
start_zero = False
dom = "d03"

data_all_ = {}
for downdrafts in [True, False]:
    for case in ["CASE1", "CASE2_new", "CASE3_new"]: 
            data_all_[f"{case}_{dom}_{'dd' if downdrafts else 'ud'}"] = get_data(case, dom, downdrafts = downdrafts, weighted = weighted, start_zero = start_zero)

# Save tracking data
for key in data_all_.keys():
    #path_results = "/g/data/up6/ai2733/Thermals_comparison/results/tracking_results/"
    path_results = "/results/tracking_results/"
    with open(f'{path_results}/{key}.pkl', 'wb') as f:
        pickle.dump(data_all_[key], f)

