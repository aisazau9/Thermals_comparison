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

import metpy.calc as mpcalc
from metpy.units import units

"""""""""""""""""""""""""""""""""""""""""""""

Extracts variables from WRF outputs for a selected case and domain, crops it to the tracking grid, aligns simulation times with the grid time window, and saves the processed data as a pickle file 

Output: wrf_all.pkl

"""""""""""""""""""""""""""""""""""""""""""""

case_idx = int(sys.argv[1]) # From 0 to 2
dom      = str(sys.argv[2]) 
case     = ["CASE1", "CASE2_new", "CASE3_new"][case_idx]
subdir = "" if dom == "d03" else "Alld02_new/"

if dom == "d03": path_save = f"/g/data/w28/ai2733/outputs_{case}_budgets/grid_tracking_cropped_new/{subdir}"
elif dom == "d02": path_save = f"/g/data/w28/ai2733/outputs_{case}_budgets/grid_tracking_cropped_new_d02/{subdir}"

# Open data from grid
#path_save2 = f"/g/data/up6/ai2733/Thermals_comparison/results/tracking_grid/{case}/"
path_save2 = f"/results/tracking_grid/{case}/"

with open(f'{path_save2}/data_grid.pkl', 'rb') as f: data_grid = dill.load(f)
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

# Constants
Rd   = 287.                         # Gas constant for dry air, J/(kg K)
Rv   = 461.6                        # Gas constant for water vapor, J/(kg K)
cp   = 7.*0.5*Rd                    # Specific heat of dry air at constant pressure, J/(kg K)
cv   = cp - Rd                      # Specific heat of dry air at constant volume, J/(kg K)
pref = 100000.0                     # reference sea level pressure, Pa
g    = 9.81                         # gravitational constant (m/s^2)
LV = 2500*1000                      # Latent heat of vaporization of water (J/kg)

# Read original WRF outputs
all_vars = ['Times','XLAT','XLONG','LU_INDEX','ZNU','ZNW','ZS','DZS','VAR_SSO','BATHYMETRY_FLAG','U','V','W','PH','PHB','T','THM','HFX_FORCE','LH_FORCE','TSK_FORCE','HFX_FORCE_TEND','LH_FORCE_TEND','TSK_FORCE_TEND','MU','MUB','NEST_POS','P','PB','FNM','FNP','RDNW','RDN','DNW','DN','CFN','CFN1','THIS_IS_AN_IDEAL_RUN','P_HYD','Q2','T2','TH2','PSFC','U10','V10','RDX','RDY','AREA2D','DX2D','RESM','ZETATOP','CF1','CF2','CF3','ITIMESTEP','XTIME','QVAPOR','QCLOUD','QRAIN','QICE','QSNOW','QGRAUP','QHAIL','QNDROP','QNRAIN','QNICE','QNSNOW','QNGRAUPEL','QNHAIL','QVGRAUPEL','QVHAIL','SHDMAX','SHDMIN','SNOALB','TSLB','SMOIS','SH2O','SEAICE','XICEM','SFROFF','UDROFF','IVGTYP','ISLTYP','VEGFRA','GRDFLX','ACGRDFLX','ACSNOM','SNOW','SNOWH','CANWAT','SSTSK','WATER_DEPTH','COSZEN','LAI','VAR','TKE_PBL','EL_PBL','O3_GFS_DU','H_DIABATIC','QV_DIABATIC','MAPFAC_M','MAPFAC_U','MAPFAC_V','MAPFAC_MX','MAPFAC_MY','MAPFAC_UX','MAPFAC_UY','MAPFAC_VX','MF_VX_INV','MAPFAC_VY','F','E','SINALPHA','COSALPHA','HGT','TSK','P_TOP','GOT_VAR_SSO','T00','P00','TLP','TISO','TLP_STRAT','P_STRAT','MAX_MSFTX','MAX_MSFTY','RAINC','RAINSH','RAINNC','SNOWNC','GRAUPELNC','HAILNC','REFL_10CM','CLDFRA','SWDOWN','GLW','SWNORM','ACSWUPT','ACSWUPTC','ACSWDNT','ACSWDNTC','ACSWUPB','ACSWUPBC','ACSWDNB','ACSWDNBC','ACLWUPT','ACLWUPTC','ACLWDNT','ACLWDNTC','ACLWUPB','ACLWUPBC','ACLWDNB','ACLWDNBC','SWUPT','SWUPTC','SWDNT','SWDNTC','SWUPB','SWUPBC','SWDNB','SWDNBC','LWUPT','LWUPTC','LWDNT','LWDNTC','LWUPB','LWUPBC','LWDNB','LWDNBC','OLR','XLAT_U','XLONG_U','XLAT_V','XLONG_V','ALBEDO','CLAT','ALBBCK','EMISS','NOAHRES','TMN','XLAND','UST','PBLH','HFX','QFX','LH','ACHFX','ACLHF','SNOWC','SR','SAVE_TOPO_FROM_REAL','WSPD10MAX','W_UP_MAX','W_DN_MAX','REFD_MAX','UP_HELI_MAX','W_MEAN','GRPL_MAX','HAIL_MAXK1','HAIL_MAX2D','ISEEDARR_SPPT','ISEEDARR_SKEBS','ISEEDARR_RAND_PERTURB','ISEEDARRAY_SPP_CONV','ISEEDARRAY_SPP_PBL','ISEEDARRAY_SPP_LSM','ISNOW','TV','TG','CANICE','CANLIQ','EAH','TAH','CM','CH','FWET','SNEQVO','ALBOLD','QSNOWXY','QRAINXY','WSLAKE','ZWT','WA','WT','TSNO','ZSNSO','SNICE','SNLIQ','LFMASS','RTMASS','STMASS','WOOD','STBLCP','FASTCP','XSAI','TAUSS','T2V','T2B','Q2V','Q2B','TRAD','NEE','GPP','NPP','FVEG','QIN','RUNSF','RUNSB','ECAN','EDIR','ETRAN','FSA','FIRA','APAR','PSN','SAV','SAG','RSSUN','RSSHA','BGAP','WGAP','TGV','TGB','CHV','CHB','SHG','SHC','SHB','EVG','EVB','GHV','GHB','IRG','IRC','IRB','TR','EVC','CHLEAF','CHUC','CHV2','CHB2','CHSTAR','SMCWTD','RECH','QRFS','QSPRINGS','QSLAT','ACINTS','ACINTR','ACDRIPR','ACTHROR','ACEVAC','ACDEWC','FORCTLSM','FORCQLSM','FORCPLSM','FORCZLSM','FORCWLSM','ACRAINLSM','ACRUNSB','ACRUNSF','ACECAN','ACETRAN','ACEDIR','ACQLAT','ACQRF','ACETLSM','ACSNOWLSM','ACSUBC','ACFROC','ACFRZC','ACMELTC','ACSNBOT','ACSNMELT','ACPONDING','ACSNSUB','ACSNFRO','ACRAINSNOW','ACDRIPS','ACTHROS','ACSAGB','ACIRB','ACSHB','ACEVB','ACGHB','ACPAHB','ACSAGV','ACIRG','ACSHG','ACEVG','ACGHV','ACPAHG','ACSAV','ACIRC','ACSHC','ACEVC','ACTR','ACPAHV','ACSWDNLSM','ACSWUPLSM','ACLWDNLSM','ACLWUPLSM','ACSHFLSM','ACLHFLSM','ACGHFLSM','ACPAHLSM','ACCANHS','SOILENERGY','SNOWENERGY','ACEFLXB','GRAIN','GDD','CROPCAT','PGS','QTDRAIN','IRNUMSI','IRNUMMI','IRNUMFI','IRSIVOL','IRMIVOL','IRFIVOL','IRELOSS','IRRSPLH','C1H','C2H','C1F','C2F','C3H','C4H','C3F','C4F','PCB','PC','LANDMASK','LAKEMASK','SST','SST_INPUT']
vars_include = ["Times", 'XLAT',  'XLONG',  'XLAT_U', 'XLAT_V', 'XLONG_U', 'XLONG_V', 'XTIME', "PH", "PHB", "PB", "P",  "U", "V", 'W','WSPD10MAX', "QVAPOR", "QRAIN", "QHAIL", "QCLOUD","QICE","QGRAUP","QSNOW", "T", "tk","REFL_10CM","P_HYD" ]
vars_exclude = [i for i in all_vars if i not in vars_include]

wrf_all = xr.open_mfdataset(files_all, concat_dim = "Time", combine = "nested", chunks = "auto").sortby("Time").drop_vars(vars_exclude)
wrf_all = wrf_all.xwrf.destagger()
wrf_all = wrf_all.isel(Time = slice(idx_ini, idx_fin+1),  west_east = slice(i0, i1), south_north = slice(j0, j1), bottom_top = slice(0, nz))
wrf_all.load()

# Get height (geopotential): time-varying
hgt   = (wrf_all["PH"] + wrf_all["PHB"])/g 
# Total pressure
ptot = wrf_all["PB"] + wrf_all["P"]

# Get variables of interest
QRAIN_arr  = wrf_all['QRAIN'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
QVAPOR_arr = wrf_all['QVAPOR'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
QCLOUD_arr = wrf_all['QCLOUD'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
QGRAUP_arr = wrf_all['QGRAUP'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
QICE_arr   = wrf_all['QICE'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
QHAIL_arr  = wrf_all['QHAIL'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
QSNOW_arr  = wrf_all['QSNOW'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
T_arr      = wrf_all['T'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy() # Pert theta
theta_arr  = 300. + T_arr
u_arr      = wrf_all['U'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
v_arr      = wrf_all['V'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
P_HYD_arr  = wrf_all['P_HYD'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
WSMAX_arr  = wrf_all['WSPD10MAX'].transpose('west_east', 'south_north', 'Time').to_numpy()
p_arr      = ptot.transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
height_arr = hgt.transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
W_arr      = wrf_all['W'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
refl_arr   = wrf_all['REFL_10CM'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
xlat_arr   = wrf_all['XLAT'].transpose('west_east', 'south_north', 'Time').to_numpy()
xlong_arr  = wrf_all['XLONG'].transpose('west_east', 'south_north', 'Time').to_numpy()

# Calculate other variables with metpy
tk_arr                = mpcalc.temperature_from_potential_temperature (pressure = p_arr * units.pascal , potential_temperature = theta_arr * units.kelvin).magnitude
rh_arr                = mpcalc.relative_humidity_from_mixing_ratio(pressure = p_arr * units.pascal, temperature = tk_arr * units.kelvin , mixing_ratio = QVAPOR_arr * units("kg/kg")).to('percent').magnitude
rh_arr[rh_arr>100] = 100
dewpoint              = mpcalc.dewpoint_from_relative_humidity(temperature = tk_arr * units.kelvin , relative_humidity = rh_arr * units.percent).magnitude
eth_arr               = mpcalc.equivalent_potential_temperature(pressure = p_arr * units.pascal, temperature = tk_arr * units.kelvin, dewpoint = dewpoint * units.kelvin).magnitude

rho_m_arr             = mpcalc.density(pressure = p_arr * units.pascal, temperature = tk_arr * units.kelvin, mixing_ratio          = QVAPOR_arr * units("kg/kg")).magnitude
sh_arr                = mpcalc.specific_humidity_from_mixing_ratio(mixing_ratio = QVAPOR_arr * units("kg/kg")).to("kg/kg").magnitude
mse_arr               = mpcalc.moist_static_energy(height = height_arr * units.meter , temperature =  tk_arr * units.kelvin, specific_humidity = sh_arr * units("kg/kg")).magnitude
thetav_arr            = mpcalc.virtual_potential_temperature(pressure = p_arr * units.pascal, temperature = tk_arr * units.kelvin, mixing_ratio = QVAPOR_arr * units("kg/kg")).magnitude

# Base state variables from WRF
hgt_base     = wrf_all["PHB"]/g #base-state geopotential
hgt_base_arr = hgt_base.transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
pbase_arr     = wrf_all['PB'].transpose('west_east', 'south_north', 'bottom_top', 'Time').to_numpy()
thetav_bs_arr = tk_arr*0. + 300# Assuming the base state is dry, virtual potential temperature = potential temperature
tk_bs_arr     = mpcalc.temperature_from_potential_temperature (pressure = pbase_arr * units.pascal , potential_temperature = thetav_bs_arr * units.kelvin).magnitude
qv_bs_arr     = tk_arr*0 # mixing ratio 0 
rho_bs_arr    = mpcalc.density(pressure = pbase_arr * units.pascal, temperature = tk_bs_arr * units.kelvin, mixing_ratio  = qv_bs_arr * units("kg/kg")).magnitude

del wrf_all

# Save just variables of interest
wrf_all = xr.Dataset(
{   'REFL_10CM': (['west_east', 'south_north', 'z', 'Time'], refl_arr),
    'wa':       (['west_east', 'south_north', 'z', 'Time'],  W_arr),
    'QRAIN':    (['west_east', 'south_north', 'z', 'Time'], QRAIN_arr),
    'QVAPOR':   (['west_east', 'south_north', 'z', 'Time'], QVAPOR_arr), #kg kg-1
    'QCLOUD':   (['west_east', 'south_north', 'z', 'Time'], QCLOUD_arr),  
    'QGRAUP':   (['west_east', 'south_north', 'z', 'Time'], QGRAUP_arr),
    'QICE':     (['west_east', 'south_north', 'z', 'Time'], QICE_arr),
    'QHAIL':    (['west_east', 'south_north', 'z', 'Time'], QHAIL_arr),  
    'QSNOW':    (['west_east', 'south_north', 'z', 'Time'], QSNOW_arr),
    'tk':       (['west_east', 'south_north', 'z', 'Time'], tk_arr),
    'theta':    (['west_east', 'south_north', 'z', 'Time'], theta_arr),
    'eth':      (['west_east', 'south_north', 'z', 'Time'], eth_arr),  
    'rh':       (['west_east', 'south_north', 'z', 'Time'], rh_arr), # %
    'p':        (['west_east', 'south_north', 'z', 'Time'], p_arr), # Pascals? pb + p #base pressure + perturbation pressure
    'rho_m':    (['west_east', 'south_north', 'z', 'Time'], rho_m_arr), # Density of the dry + moist air rho_d*(1 + self.qvapor), where rho_d = (pref/(Rd*theta_m))*np.power((self.ptot/pref),(cv/cp)), pref = 100000.0 reference sea level pressure, Pa
    'u':        (['west_east', 'south_north', 'z', 'Time'], u_arr),  # u wind component
    'v':        (['west_east', 'south_north', 'z', 'Time'], v_arr),  # v wind component,
    'SH':       (['west_east', 'south_north', 'z', 'Time'], sh_arr),  
    'MSE':      (['west_east', 'south_north', 'z', 'Time'], mse_arr),  
    'theta_v':  (['west_east', 'south_north', 'z', 'Time'], thetav_arr),  
    'WSMAX':    (['west_east', 'south_north', 'Time'],      WSMAX_arr),
    'XLAT':     (['west_east', 'south_north', 'Time'],      xlat_arr) ,
    'XLONG':    (['west_east', 'south_north', 'Time'],      xlong_arr),
    'height':   (['west_east', 'south_north', 'z', 'Time'], height_arr),
    'p_base':    (['west_east', 'south_north', 'z', 'Time'], pbase_arr),
    'thetav_base': (['west_east', 'south_north', 'z', 'Time'], thetav_bs_arr),
    'rho_base':    (['west_east', 'south_north', 'z', 'Time'], rho_bs_arr),
    'height_base': (['west_east', 'south_north', 'z', 'Time'], hgt_base_arr),

             },
coords={
    'west_east':   x_coords,  
    'south_north': y_coords,  
    'z':           z_coords,  
    'Time':        t_coords
}
)

# Save as pickle 
with open(f'{path_save}/wrf_all.pkl', 'wb') as f: pickle.dump(wrf_all, f)

