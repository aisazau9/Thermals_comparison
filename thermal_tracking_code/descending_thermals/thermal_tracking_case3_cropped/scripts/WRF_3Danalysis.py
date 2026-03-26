import numpy as np
import pdb
import scipy.interpolate as pol
import WRF_3Dgrid as grid3D
import aux_functions as aux
import os
from WRF_3Dthermal import read_data as read
from joblib import Parallel, delayed
import scipy.ndimage.filters as filters
import time as tm


Rd   = 287.                         # Gas constant for dry air, J/(kg K)
cp   = 7.*0.5*Rd                    # Specific heat of dry air at constant pressure, J/(kg K)
Lv   = 2.260e6
g    = 9.81                         # gravitatioxnal constant (m/s^2)
min_z_qnice = 0.                    # minimum altitude for thermals to make qnice composites  #changed from 6000 to 0 (DHD 06-10-2023)

#*************************************************
# GLOBAL PARAMETERS: PLOT LABELS FOR EACH VARIABLE
#*************************************************
wmaxl       = '$\omega_{min}$ (m$\,$s$^{-1}$)' #ALEJANDRA
Rl          = 'R (km)'
Fresl       = 'Fmix (m$\,$s$^{-2}$)'
buoyl       = 'buoyancy (m$\,$s$^{-2}$)'
Fnhl        = 'Fnh (m$\,$s$^{-2}$)'
accl        = 'dW/dt (m$\,$s$^{-2}$)'
Dl          = 'D (km)'
D2l         = 'D2 (km)'
massl       = 'mass (1e$^5$ kg)'
Wl          = 'W (m$\,$s$^{-1}$)'
cond_ml     = 'M$_{cond}$ (1e$^5$ kg)'
z_centrel   = 'height (km)'
Pnzl        = 'Pnz (m$\,$s$^{-2}$)'
logel       = '$\log_{10}(\epsilon)$ (m$^{-1}$)'
logdl       = '$\log_{10}(\delta)$ (m$^{-1}$)'
el          = '$\epsilon$ (m$^{-1}$)'


class Composite(object):
    def __init__( self, R_range=3., delta_R = 0.05, plt_range=2.5, dx=100., nz=230, header_fmt='wrfout_d01_YYYY-MM-DD_', dt=60, xff=100, yff=15, path='/home/nfs/z3392395/DATA_ISLAND_003/', rescale=True, min_ratio=0.1, n_jobs=4, prefix='', ending='', folder_name='composite_thermal',replot=False, exp_name=None, xmin=None, xmax=None, ymin=None, ymax=None, thermal_list='./thermal*', path_NW=None, compute_from_scratch=False, compute_rh=True, gunzip=False, up=True, x0=0, y0=0, GCE=False, cell_nr='' , 
                start_zero = False): #ALEJANDRA: start_zero
        # x0 and y0 should provide the values of x and y at indices i=0 and j=0.
        self.GCE        = GCE
        self.X0         = x0
        self.Y0         = y0
        self.dx         = dx        # horizontal gridspacing in the original data
        self.dy         = dx        
        self.nz         = nz        # number of levels to read from original data
        self.header_fmt = header_fmt# header format of original data file names
        self.dt         = dt        # time interval between output data files
        self.xff        = xff       # maximum x-value in km
        self.yff        = yff       # maximum y-value in km
        self.R_range    = R_range   # number of radii to use for computing the composite (both horizontal and vertical)
        self.delta_R    = delta_R   # subdivision size for the radial-coordinates in the composite
        self.plt_range  = plt_range # range for plots, in radius coordinates (should be smaller or equal to R_range)
        self.path       = path      # path where simulation data is
        self.min_ratio  = min_ratio # minimum number of thermals required for averaging in composites at different stages of thermal lifetime, as fraction of total number of thermals (e.g., 0.1 means at least 10% of the thermals must be "alive" at a particular stage in order to compute composites.
        self.rescale    = rescale   # rescale to radius-coordinates before averaging all thermals.
        self.n_jobs     = n_jobs
        self.ending     = ending
        self.exp_name   = exp_name
        self.thermal_list= thermal_list
        self.path_NW    = path_NW
        self.compute_rh = compute_rh
        self.gunzip     = gunzip
        self.up         = up
        self.cell_nr    = cell_nr
        self.start_zero = start_zero # If ref = 0 instead of the time of max(wmax), for the time composites
        
        print ("-----MAKING SURE THEY ARE DOWNDRAFTS/UPDRAFTS---------------") #ALEJANDRA
        print ("self.up", self.up, "self.start_zero", self.start_zero)
        print ("----------------------------------------------------------------")

        if rescale:
            self.x_new      = np.arange(-R_range,R_range+delta_R,delta_R)
            self.x_new[np.where((self.x_new<1e-10)*(self.x_new>-1e-10))]=0.
            self.y_new      = np.copy(self.x_new)
            self.z_new      = np.copy(self.x_new)
            self.folder     = prefix + folder_name 
        else:
            self.x_new      = np.arange(-2000.,2000.+self.dx,self.dx)
            self.y_new      = np.copy(self.x_new)
            self.z_new      = np.copy(self.x_new)
            self.plt_range  = 1500. 
            self.folder     = prefix + folder_name
        self.prefix = prefix # in case folder containing tracked thermal data starts with something before "thermal_". (Usually not the case)
        self._load(replot, xmin,xmax,ymin,ymax, compute_from_scratch)

    def _load( self,replot=False, xmin=None, xmax=None, ymin=None, ymax=None, compute_from_scratch=False):
        """
        Reads from the data file the relevant information of each thermal found in the current directory.
        """
        if replot:
            self.tmax_ref               = np.load( self.folder+'/tmax_ref.npy', allow_pickle=True, encoding='latin1'  )
            self.min_N                  = np.load( self.folder+'/min_N.npy', allow_pickle=True, encoding='latin1'  )
            self.R_count                = np.load( self.folder+'/R_count.npy', allow_pickle=True, encoding='latin1'  )
            
            #*******************************************************************************************************************
            # variables for vertical cross sections throughout thermal lifetime stages:
            self.u_mean                 = np.load( self.folder+'/u_mean.npy', allow_pickle=True, encoding='latin1'  )                
            self.v_mean                 = np.load( self.folder+'/v_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.w_mean                 = np.load( self.folder+'/w_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.u_dev_mean             = np.load( self.folder+'/u_dev_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.v_dev_mean             = np.load( self.folder+'/v_dev_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.w_dev_mean             = np.load( self.folder+'/w_dev_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.buoy_mean              = np.load( self.folder+'/buoy_mean.npy', allow_pickle=True, encoding='latin1'  )               
            self.pdev_mean              = np.load( self.folder+'/pdev_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.sctot_mean             = np.load( self.folder+'/sctot_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.latheat_mean           = np.load( self.folder+'/latheat_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.qnice_mean             = np.load( self.folder+'/qnice_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.qncloud_mean           = np.load( self.folder+'/qncloud_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.qnrain_mean            = np.load( self.folder+'/qnrain_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.qice_mean              = np.load( self.folder+'/qice_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.qcloud_mean            = np.load( self.folder+'/qcloud_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.qrain_mean             = np.load( self.folder+'/qrain_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.qvapor_mean            = np.load( self.folder+'/qvapor_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.noninduc_mean          = np.load( self.folder+'/noninduc_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.cldnuc_mean            = np.load( self.folder+'/cldnuc_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.rh_mean                = np.load( self.folder+'/rh_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.epotential_mean        = np.load( self.folder+'/epotential_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.qngraupel_mean         = np.load( self.folder+'/qngraupel_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.qicesnow_mean          = np.load( self.folder+'/qicesnow_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.qghail_mean            = np.load( self.folder+'/qghail_mean.npy', allow_pickle=True, encoding='latin1'  )
            #*******************************************************************************************************************
            # same as above but for thermals that initiate below 7km:
            try:
                self.u_mean_7km                 = np.load( self.folder+'/u_mean_7km.npy', allow_pickle=True, encoding='latin1'  )                
                self.v_mean_7km                 = np.load( self.folder+'/v_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.w_mean_7km                 = np.load( self.folder+'/w_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.u_dev_mean_7km             = np.load( self.folder+'/u_dev_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.v_dev_mean_7km             = np.load( self.folder+'/v_dev_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.w_dev_mean_7km             = np.load( self.folder+'/w_dev_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.buoy_mean_7km              = np.load( self.folder+'/buoy_mean_7km.npy', allow_pickle=True, encoding='latin1'  )               
                self.pdev_mean_7km              = np.load( self.folder+'/pdev_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.sctot_mean_7km             = np.load( self.folder+'/sctot_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.latheat_mean_7km           = np.load( self.folder+'/latheat_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.qnice_mean_7km             = np.load( self.folder+'/qnice_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.qncloud_mean_7km           = np.load( self.folder+'/qncloud_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.qnrain_mean_7km            = np.load( self.folder+'/qnrain_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.qice_mean_7km              = np.load( self.folder+'/qice_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.qcloud_mean_7km            = np.load( self.folder+'/qcloud_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.qrain_mean_7km             = np.load( self.folder+'/qrain_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.qvapor_mean_7km            = np.load( self.folder+'/qvapor_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.noninduc_mean_7km          = np.load( self.folder+'/noninduc_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.cldnuc_mean_7km            = np.load( self.folder+'/cldnuc_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.rh_mean_7km                = np.load( self.folder+'/rh_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.epotential_mean_7km        = np.load( self.folder+'/epotential_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.qngraupel_mean_7km         = np.load( self.folder+'/qngraupel_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.qicesnow_mean_7km          = np.load( self.folder+'/qicesnow_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
                self.qghail_mean_7km            = np.load( self.folder+'/qghail_mean_7km.npy', allow_pickle=True, encoding='latin1'  )
 
            except:
                print('no cross-section fields for thermals initated below 7km. Rerun with replot=False and compute_from_scratch=True to create these.')
            #*******************************************************************************************************************

            self.entrainment_mean       = np.load( self.folder+'/entr_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.detrainment_mean       = np.load( self.folder+'/detr_mean.npy', allow_pickle=True, encoding='latin1'  )
            self.angles                 = np.load( self.folder+'/angles.npy', allow_pickle=True, encoding='latin1'  )
            self.N_angles               = len( self.angles )
            self.entrainment_grossmean  = np.load( self.folder+'/entrainment_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.detrainment_grossmean  = np.load( self.folder+'/detrainment_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.u_dev_grossmean        = np.load( self.folder+'/u_dev_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.v_dev_grossmean        = np.load( self.folder+'/v_dev_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.w_dev_grossmean        = np.load( self.folder+'/w_dev_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.sctot_grossmean        = np.load( self.folder+'/sctot_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.latheat_grossmean      = np.load( self.folder+'/latheat_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qnice_grossmean        = np.load( self.folder+'/qnice_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qncloud_grossmean      = np.load( self.folder+'/qncloud_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qnrain_grossmean       = np.load( self.folder+'/qnrain_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qice_grossmean         = np.load( self.folder+'/qice_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qcloud_grossmean       = np.load( self.folder+'/qcloud_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qrain_grossmean        = np.load( self.folder+'/qrain_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qvapor_grossmean       = np.load( self.folder+'/qvapor_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.noninduc_grossmean     = np.load( self.folder+'/noninduc_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.cldnuc_grossmean       = np.load( self.folder+'/cldnuc_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.rh_grossmean           = np.load( self.folder+'/rh_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.epotential_grossmean   = np.load( self.folder+'/epotential_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qngraupel_grossmean    = np.load( self.folder+'/qngraupel_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qicesnow_grossmean      = np.load( self.folder+'/qicesnow_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qghail_grossmean       = np.load( self.folder+'/qghail_grossmean.npy', allow_pickle=True, encoding='latin1'  )
            self.u_dev_lastmean        = np.load( self.folder+'/u_dev_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.v_dev_lastmean        = np.load( self.folder+'/v_dev_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.w_dev_lastmean        = np.load( self.folder+'/w_dev_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.sctot_lastmean        = np.load( self.folder+'/sctot_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.latheat_lastmean      = np.load( self.folder+'/latheat_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qnice_lastmean        = np.load( self.folder+'/qnice_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qncloud_lastmean      = np.load( self.folder+'/qncloud_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qnrain_lastmean       = np.load( self.folder+'/qnrain_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qice_lastmean         = np.load( self.folder+'/qice_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qcloud_lastmean       = np.load( self.folder+'/qcloud_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qrain_lastmean        = np.load( self.folder+'/qrain_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qvapor_lastmean       = np.load( self.folder+'/qvapor_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.cldnuc_lastmean       = np.load( self.folder+'/cldnuc_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.noninduc_lastmean     = np.load( self.folder+'/noninduc_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.rh_lastmean           = np.load( self.folder+'/rh_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.epotential_lastmean   = np.load( self.folder+'/epotential_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qngraupel_lastmean    = np.load( self.folder+'/qngraupel_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qicesnow_lastmean     = np.load( self.folder+'/qicesnow_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.qghail_lastmean       = np.load( self.folder+'/qghail_lastmean.npy', allow_pickle=True, encoding='latin1'  )
            self.iz_low                 = np.load( self.folder+'/iz_low.npy', allow_pickle=True, encoding='latin1' )
            self.iz_up                  = np.load( self.folder+'/iz_up.npy', allow_pickle=True, encoding='latin1' )
            self.ix_right               = np.load( self.folder+'/ix_right.npy', allow_pickle=True, encoding='latin1' )
            self.ix_left                = np.load( self.folder+'/ix_left.npy', allow_pickle=True, encoding='latin1' )
            self.iy_right               = np.load( self.folder+'/iy_right.npy', allow_pickle=True, encoding='latin1' )
            self.iy_left                = np.load( self.folder+'/iy_left.npy', allow_pickle=True, encoding='latin1' )

            self.z_centre    = np.load( self.folder+'/z_centre.npy', allow_pickle=True, encoding='latin1'     ) 
            self.x_centre    = np.load( self.folder+'/x_centre.npy', allow_pickle=True, encoding='latin1'     ) 
            self.y_centre    = np.load( self.folder+'/y_centre.npy', allow_pickle=True, encoding='latin1'     ) 
            self.Fres        = np.load( self.folder+'/Fres.npy', allow_pickle=True, encoding='latin1'  )
            self.buoy        = np.load( self.folder+'/buoy.npy', allow_pickle=True, encoding='latin1'  )
            self.Fnh         = np.load( self.folder+'/Fnh.npy', allow_pickle=True, encoding='latin1'  )
            self.acc         = np.load( self.folder+'/acc.npy', allow_pickle=True, encoding='latin1'  )
            self.D           = np.load( self.folder+'/D.npy', allow_pickle=True, encoding='latin1'  )
            self.Pnz         = np.load( self.folder+'/Pnz.npy', allow_pickle=True, encoding='latin1'  )
            self.time_or     = np.load( self.folder+'/time_or.npy', allow_pickle=True, encoding='latin1'  )
            self.time        = np.load( self.folder+'/time.npy', allow_pickle=True, encoding='latin1'  )
            self.W           = np.load( self.folder+'/W.npy', allow_pickle=True, encoding='latin1'  )
            self.wmax        = np.load( self.folder+'/wmax.npy', allow_pickle=True, encoding='latin1'  )
            self.R_or        = np.load( self.folder+'/R_or.npy', allow_pickle=True, encoding='latin1'  )
            self.R           = np.load( self.folder+'/R.npy', allow_pickle=True, encoding='latin1'  )
            self.fract_entr  = np.load( self.folder+'/fract_entr.npy', allow_pickle=True, encoding='latin1'  )
            self.mse_thermal = np.load( self.folder+'/mse_thermal.npy', allow_pickle=True, encoding='latin1'  )
            self.mse_thermal_or = np.load( self.folder+'/mse_thermal_or.npy', allow_pickle=True, encoding='latin1'  )
            self.mse_env     = np.load( self.folder+'/mse_env.npy', allow_pickle=True, encoding='latin1'  )
            self.mse_env_or  = np.load( self.folder+'/mse_env_or.npy', allow_pickle=True, encoding='latin1'  )
            self.mse_diff_init= np.load( self.folder+'/mse_diff_init.npy', allow_pickle=True, encoding='latin1'  )
            self.mixing_mse  = np.load( self.folder+'/mixing_mse.npy', allow_pickle=True, encoding='latin1'  )
            self.mass        = np.load( self.folder+'/mass.npy', allow_pickle=True, encoding='latin1'  )
            #self.mass_cond   = np.load( self.folder+'/mass_cond.npy', allow_pickle=True, encoding='latin1'  )
            self.wmax_c      = np.load( self.folder+'/wmax_c.npy', allow_pickle=True, encoding='latin1'  )
            self.R_c         = np.load( self.folder+'/R_c.npy', allow_pickle=True, encoding='latin1'  )
            self.Fres_c      = np.load( self.folder+'/Fres_c.npy', allow_pickle=True, encoding='latin1'  )
            self.buoy_c      = np.load( self.folder+'/buoy_c.npy', allow_pickle=True, encoding='latin1'  )
            self.Fnh_c       = np.load( self.folder+'/Fnh_c.npy', allow_pickle=True, encoding='latin1'  )
            self.acc_c       = np.load( self.folder+'/acc_c.npy', allow_pickle=True, encoding='latin1'  )
            self.D_c         = np.load( self.folder+'/D_c.npy', allow_pickle=True, encoding='latin1'  )
            self.mass_c      = np.load( self.folder+'/mass_c.npy', allow_pickle=True, encoding='latin1'  )
            #self.mass_cond_c = np.load( self.folder+'/mass_cond_c.npy', allow_pickle=True, encoding='latin1'  )
            self.W_c         = np.load( self.folder+'/W_c.npy', allow_pickle=True, encoding='latin1'  )
            self.Pnz_c       = np.load( self.folder+'/Pnz_c.npy', allow_pickle=True, encoding='latin1'  )
            self.z_centre_c  = np.load( self.folder+'/z_centre_c.npy', allow_pickle=True, encoding='latin1'  )
            self.y_centre_c  = np.load( self.folder+'/y_centre_c.npy', allow_pickle=True, encoding='latin1'  )
            self.x_centre_c  = np.load( self.folder+'/x_centre_c.npy', allow_pickle=True, encoding='latin1'  )
            self.qcloud_c    = np.load( self.folder+'/qcloud_c.npy', allow_pickle=True, encoding='latin1'  )
            self.qncloud_c   = np.load( self.folder+'/qncloud_c.npy', allow_pickle=True, encoding='latin1'  )
            self.qrain_c     = np.load( self.folder+'/qrain_c.npy', allow_pickle=True, encoding='latin1'  )
            self.qnrain_c    = np.load( self.folder+'/qnrain_c.npy', allow_pickle=True, encoding='latin1'  )
            self.cldnuc_c    = np.load( self.folder+'/cldnuc_c.npy', allow_pickle=True, encoding='latin1'  )
            self.noninduc_c    = np.load( self.folder+'/noninduc_c.npy', allow_pickle=True, encoding='latin1'  )
            self.sctot_c   = np.load( self.folder+'/sctot_c.npy', allow_pickle=True, encoding='latin1' )
            self.latheat_c   = np.load( self.folder+'/latheat_c.npy', allow_pickle=True, encoding='latin1' )
            self.epotential_c = np.load( self.folder+'/epotential_c.npy', allow_pickle=True, encoding='latin1' )
            self.qngraupel_c  = np.load( self.folder+'/qngraupel_c.npy', allow_pickle=True, encoding='latin1'  )
            self.qicesnow_c   = np.load( self.folder+'/qicesnow_c.npy', allow_pickle=True, encoding='latin1'  )
            self.qghail_c    = np.load( self.folder+'/qghail_c.npy', allow_pickle=True, encoding='latin1'  )
            self.it          = np.load( self.folder+'/it.npy', allow_pickle=True, encoding='latin1'  )
            self.loge_c      = np.load( self.folder+'/loge_c.npy' , allow_pickle=True, encoding='latin1' ) 
            self.time_c      = np.load( self.folder+'/time_c.npy', allow_pickle=True, encoding='latin1'  )
            self.area        = np.load( self.folder+'/area.npy', allow_pickle=True, encoding='latin1'  )
            self.simtime     = np.load( self.folder+'/simtime.npy', allow_pickle=True, encoding='latin1'  )
            self.t_range     = np.load( self.folder+'/t_range.npy', allow_pickle=True, encoding='latin1'  )
            self.tmin        = np.load( self.folder+'/tmin.npy', allow_pickle=True, encoding='latin1'  )
            self.tmax        = np.load( self.folder+'/tmax.npy', allow_pickle=True, encoding='latin1'  )
            #self.tracer_entr = np.load( self.folder+'/tracer_entr.npy' )
            #self.tracer_detr = np.load( self.folder+'/tracer_detr.npy' )
            self.delta_z     = np.load( self.folder+'/delta_z.npy', allow_pickle=True, encoding='latin1'  )
            self.z0          = np.load( self.folder+'/z0.npy', allow_pickle=True, encoding='latin1'  )
            self.deltazR     = np.load( self.folder+'/deltazR.npy', allow_pickle=True, encoding='latin1'  )
            self.entr_rate   = np.load( self.folder+'/entr_rate.npy', allow_pickle=True, encoding='latin1'  )
            
            self.volume      = 4.*np.pi*np.power(self.R*1e-3,3)/3.  # in km^3 (no need to save it)
            self.volume_c    = np.load( self.folder+'/volume_c.npy', allow_pickle=True, encoding='latin1'  )

            if os.path.isfile(self.folder+'/net_entr_term.npy'):
                self.net_entr_term = np.load( self.folder+'/net_entr_term.npy', allow_pickle=True, encoding='latin1'  )
                self.net_entr_term_c = np.load( self.folder+'/net_entr_term_c.npy', allow_pickle=True, encoding='latin1'  )
    
            self.N_thermals = len(self.wmax_c)
            self.min_N      = int(self.N_thermals*self.min_ratio)
            self._compute_net_entrainment()
        
            self.R_avg = np.load( self.folder+'/R_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.W_avg = np.load( self.folder+'/W_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.B_avg = np.load( self.folder+'/B_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.wmax_avg    = np.load( self.folder+'/wmax_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.qcloud_avg  = np.load( self.folder+'/qcloud_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.qncloud_avg = np.load( self.folder+'/qncloud_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.qrain_avg   = np.load( self.folder+'/qrain_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.qnrain_avg  = np.load( self.folder+'/qnrain_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.cldnuc_avg  = np.load( self.folder+'/cldnuc_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.latheat_avg = np.load( self.folder+'/latheat_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.noninduc_avg  = np.load( self.folder+'/noninduc_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.sctot_avg = np.load( self.folder+'/sctot_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.qngraupel_avg  = np.load( self.folder+'/qngraupel_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.epotential_avg = np.load( self.folder+'/epotential_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.qicesnows_avg = np.load( self.folder+'/qicesnow_avg.npy', allow_pickle=True, encoding='latin1'  )
            self.qghail_avg = np.load( self.folder+'/qghail_avg.npy', allow_pickle=True, encoding='latin1'  )

            ####ALEJANDRA: compute this again
            #self._get_centered_vars_at_wmax()
            #self._compute_mf_weights_by_altitude()
            #self._make_composite(compute_from_scratch=compute_from_scratch)
            #self._compute_average_values()
        
        else:
            os.system('ls -d '+ self.prefix + self.thermal_list + ' > ls'+self.cell_nr+'.txt')
            file = open('ls'+self.cell_nr+'.txt', 'r')
            ofile = file.readlines()
            os.system('rm -rf ls'+self.cell_nr+'.txt')
            for i in range(len(ofile)):
                print( ofile[i][:-1])

            wmax            = []
            R	            = []
            W	            = []
            U 	            = []
            V               = []
            z_centre        = []
            x_centre        = []
            y_centre        = []
            hr 	            = []
            mn 	            = []
            sc              = []
            time            = []
            buoy_map        = []
            mass            = []
            #mass_cond       = []
            D               = []
            Fres            = [] 
            buoy            = []
            Fnh             = []
            acc             = []
            Pnz             = []
            mse_thermal     = []
            mse_env         = []
            mse_diff_init   = []    # mse difference between thermal and environment at first time step
            mixing_mse      = []   
            tracer_detr     = []
            delta_z         = []
            z0              = []
            deltazR         = []
            entrainment_map = []
            detrainment_map = []
            entr_rate       = []
            net_entr_term   = []
            dBdZ            = []
            dBdZmax         = []
            dBdZmin         = []
            dBdZinit        = []
            qncloud         = []
            qnrain          = []
            qcloud          = []
            qrain           = []
            cldnuc          = []
            latheat         = []
            noninduc        = []
            sctot           = []
            qngraupel       = []
            epotential      = []
            qicesnow        = []
            qghail          = []
            yy              = []
            mm              = []
            dd              = []
            
            new_points = points_grid( self.x_new, self.y_new, self.z_new )
            buoy_mask = np.where( np.sqrt(new_points.T[0]*new_points.T[0] + new_points.T[1]*new_points.T[1] + new_points.T[2]*new_points.T[2]) >= 2. )

            self.N_alpha = 40
            self.N_phi   = 20
            d_alpha = 2.*np.pi/self.N_alpha
            d_phi = np.pi/self.N_phi
            alpha = np.arange( 0, 2*np.pi, d_alpha )
            #alpha = np.arange( d_alpha*0.5, 2*np.pi, d_alpha )
            #phi = np.arange( -np.pi/2.+d_phi*0.5, np.pi/2., d_phi )
            phi = np.arange( -np.pi/2.+d_phi, np.pi/2., d_phi )
            angles = angles_grid(alpha, phi)
            self.N_angles = len(angles)
            xyz = angles_to_xyz( angles ) 
            casename=[]
            for case in range(len(ofile)):
                start=0
                if ofile[case][:8]!='thermal' and ofile[case][start:start+9]!='downdraft' and ofile[case][:7]!='shifted':
                    start=1
                    while ofile[case][start:start+7]!='thermal' and ofile[case][start:start+9]!='downdraft' and ofile[case][start:start+7]!='shifted':
                        start+=1
                thermal_data = read( ofile[case][:-1]+'/'+ofile[case][start:-1] )
                if thermal_data.shape[0]==42: # (older version of the code)
                    xc = thermal_data[8]
                    yc = thermal_data[9]
                else:
                    xc = thermal_data[7]
                    yc = thermal_data[8]
                Rc = thermal_data[2]
                if xmin==None or (np.min(xc-Rc/1e3)>xmin and np.max(xc+Rc/1e3)<xmax and np.min(yc-Rc/1e3)>ymin and np.max(yc+Rc/1e3)<ymax):
                    time.append(        thermal_data[0])
                    wmax.append(        thermal_data[1])
                    R.append(           thermal_data[2])
                    mass.append(        thermal_data[3])
                    x_centre.append(    xc )
                    y_centre.append(    yc )
                    if thermal_data.shape[0]==42: # (older version of the code)
                        z_centre.append(    thermal_data[10]*1e3)#9]*1e3)
                        U.append(           thermal_data[11])
                        V.append(           thermal_data[12])
                        W.append(           thermal_data[13])
                        Pnz.append(         thermal_data[14])
                        Fnh.append(         thermal_data[14]-thermal_data[15])
                        Fres.append(        thermal_data[15])
                        buoy.append(        thermal_data[16])
                        acc.append(         thermal_data[18])
                        entr_rate.append(   thermal_data[26]*thermal_data[2]*4./3.)
                        D.append(           thermal_data[25])
                        mse_thermal.append( thermal_data[29])
                        mse_env.append(     thermal_data[30])
                        mse_diff_init.append(thermal_data[29][0]-thermal_data[30][0])
                        latheat.append(     thermal_data[31] )
                        qncloud.append(     thermal_data[34] )
                        qnrain.append(      thermal_data[35] )
                        cldnuc.append(      thermal_data[36] )
                        qcloud.append(      thermal_data[38] )
                        qrain.append(       thermal_data[39] )
                        sctot.append(       np.ones_like(thermal_data[1])*np.nan )
                        noninduc.append(    np.ones_like(thermal_data[1])*np.nan )
                        epotential.append(  np.ones_like(thermal_data[1])*np.nan )
                        qngraupel.append(   np.ones_like(thermal_data[1])*np.nan )
                        qicesnow.append(    np.ones_like(thermal_data[1])*np.nan )
                        qghail.append(      np.ones_like(thermal_data[1])*np.nan )
    
                    elif len(thermal_data)>=49: # new version of the code
                        z_centre.append(        thermal_data[9]*1e3)
                        U.append(               thermal_data[10])
                        V.append(               thermal_data[11])
                        W.append(               thermal_data[12])
                        Pnz.append(             thermal_data[13])
                        Fnh.append(             thermal_data[13]-thermal_data[15])
                        Fres.append(            thermal_data[14])
                        buoy.append(            thermal_data[15])
                        acc.append(             thermal_data[17])
                        entr_rate.append(       thermal_data[25]*thermal_data[2]*4./3.)
                        D.append(               thermal_data[27])
                        mse_thermal.append(     thermal_data[28])
                        mse_env.append(         thermal_data[29])
                        mse_diff_init.append(   thermal_data[28][0]-thermal_data[29][0])
                        latheat.append(         thermal_data[30] )
                        qncloud.append(         thermal_data[33] )
                        qnrain.append(          thermal_data[34] )
                        cldnuc.append(          thermal_data[35] )
                        qcloud.append(          thermal_data[37] )
                        qrain.append(           thermal_data[38] )

                        sctot.append(           thermal_data[41] )
                        noninduc.append(        thermal_data[43] )
                        epotential.append(      thermal_data[44] )
                        qngraupel.append(       thermal_data[46] )
                        qicesnow.append(        thermal_data[47] )
                        qghail.append(          thermal_data[48] )
                        if thermal_data.shape[0]==50:
                            date                  = thermal_data[49] # this should be an integer number in the form YYYYMMDD
                            yy.append((date*1e-4).astype(int))
                            mm.append(((date-yy[-1]*1e4)*1e-2).astype(int))
                            dd.append((((date-yy[-1]*1e4)*1e-2)*1e2-((date-yy[-1]*1e4)*1e-2).astype(int)*1e2).astype(int))
                            hr_tmp = (time[-1]/60).astype(int)
                            hr_tmp[np.where(hr_tmp>23)[0]] = hr_tmp[np.where(hr_tmp>23)[0]]-24
                            hr.append(hr_tmp)
                            mn.append((np.mod(time[-1],60)).astype(int))
                            sc.append((np.mod(time[-1],1)*60.).astype(int))
                        else:
                            import datetime
                            yy.append(np.ones(len(R[-1]))*2000)
                            mm.append(np.ones(len(R[-1])))
                            dd.append(np.ones(len(R[-1])))
                            hr_tmp=[]
                            mn_tmp=[]
                            sc_tmp=[]
                            ichar=7
                            istart=0
                            while ichar<len(ofile[case]) and istart==0:
                                if ofile[case][ichar-7:ichar+1]=='thermal_':
                                    istart=ichar+1
                                ichar+=1
                            for itemp in range(len(R[-1])):
                                time_tmp = datetime.datetime(2000,1,1,int(ofile[case][istart:istart+2]),int(ofile[case][istart+3:istart+5]),int(ofile[case][istart+6:istart+8])) + datetime.timedelta(minutes=itemp)
                                hr_tmp.append(time_tmp.hour)
                                mn_tmp.append(time_tmp.minute)
                                sc_tmp.append(time_tmp.second)
                            hr.append(hr_tmp)
                            mn.append(mn_tmp)
                            sc.append(sc_tmp)
                           
                    mixing_mse.append( np.insert( np.load( ofile[case][:-1]+'/'+ofile[case][start:-1]+'_mse_mixing.npy', allow_pickle=True, encoding='latin1'  ), 0, np.nan ) )

                    ## NOTE: entrainment from tracers was computed tracing N (N=500 is enough-no difference with N=1000 was found) tracers en between each time step and averaging for each thermal.
                    #if os.path.isfile( ofile[case][:-1]+'/tracers/fract_entr_tracers_avg.npy'):
                    #    entr = np.load( ofile[case][:-1]+'/tracers/fract_entr_tracers_avg.npy' )
                    #    tracer_entr.append( (entr[0],0.5*(z_centre[-1][int(entr[1])]+z_centre[-1][int(entr[2])]), entr[4], entr[5]) )
                    #    # this is the average entrainment, its average altitude, the maximum value and the minimum value throughout the thermal's lifetime
                    #    detr = np.load( ofile[case][:-1]+'/tracers/fract_detr_tracers_avg.npy')
                    #    tracer_detr.append( (detr[0],0.5*(z_centre[-1][int(detr[1])]+z_centre[-1][int(detr[2])]), detr[4], detr[5]) )
                    #else:
                    #    tracer_entr.append( (np.nan, np.nan, np.nan, np.nan) )
                    #    tracer_detr.append( (np.nan, np.nan, np.nan, np.nan) )
                    if os.path.isfile( ofile[case][:-1]+'/'+ofile[case][start:-1]+'_net_entr_term.npy'):
                        net_entr_term.append( np.load( ofile[case][:-1]+'/'+ofile[case][start:-1]+'_net_entr_term.npy', allow_pickle=True, encoding='latin1'  ) )
                    else:
                        net_entr_term.append( np.ones(len(time[-1]))*np.nan )
                    delta_z.append ( (z_centre[-1][-1]-z_centre[-1][0]) )
                    z0.append( z_centre[-1][0] )
                    deltazR.append( delta_z[-1]/np.mean(R[-1]) )
                    for i in range(len(time[-1])):
                        casename.append( (ofile[case][:-1],i) )
                    if os.path.isfile( ofile[case][:-1]+'/'+ofile[case][start:-1]+'_buoy_map.npz' ):
                        buoy_all = np.load( ofile[case][:-1]+'/'+ofile[case][start:-1]+'_buoy_map.npz', allow_pickle=True, encoding='latin1'  )
                        buoym=[]
                        #for i in range(len(buoy_all.items())): # had to change with python3 DHD 10.03.2020
                        for i in range(len(buoy_all)):
                            #buoym.append(buoy_all[i][1])
                            buoym.append(buoy_all['arr_%d'%(i)])
                        buoy_gridded=[]
                        for it in range(len(buoym)):
                            x = (buoym[it][0]-x_centre[-1][it]*1e3)/R[-1][it]
                            y = (buoym[it][1]-y_centre[-1][it]*1e3)/R[-1][it]
                            z = (buoym[it][2]-z_centre[-1][it])/R[-1][it]
                            interp = pol.griddata( ((buoym[it][0]-x_centre[-1][it]*1e3)/R[-1][it],(buoym[it][1]-y_centre[-1][it]*1e3)/R[-1][it], (buoym[it][2]-z_centre[-1][it])/R[-1][it]), buoym[it][3], new_points, method='nearest' )
                            interp[buoy_mask]=np.nan
                            buoy_gridded.append( np.reshape( interp, (len(self.x_new), len(self.y_new), len(self.z_new)) ) )
                        buoy_map.append( np.asarray(buoy_gridded) )
                    if os.path.isfile( ofile[case][:-1]+'/'+ofile[case][start:-1]+'_entr_distr.npy' ):
                        angle_data = np.load( ofile[case][:-1]+'/'+ofile[case][start:-1]+'_angles.npy', allow_pickle=True, encoding='latin1' )
                        entr_distr = np.load( ofile[case][:-1]+'/'+ofile[case][start:-1]+'_entr_distr.npy', allow_pickle=True, encoding='latin1' )
                        entrainment_gridded = []
                        detrainment_gridded = []

                        for it in range(entr_distr.shape[0]):
                            xyz_thermal = angles_to_xyz( angle_data[it] )
                            dS = R[-1][it]*R[-1][it]*(np.sin(angles[:,1]+d_phi*0.5) - np.sin(angles[:,1]-d_phi*0.5))*d_alpha
                            # use 'nearest point' interpolation to get values on common points (angles)
                            interp = pol.griddata( xyz_thermal, entr_distr[it], xyz, method='nearest' )*dS
                            positive = np.where(interp > 0)[0]
                            negative = np.where(interp < 0)[0]
                            entrainment = np.zeros(len(angles))
                            detrainment = np.zeros(len(angles))
                            entrainment[positive] = interp[positive]
                            detrainment[negative] = interp[negative]

                            entrainment_gridded.append( np.asarray( entrainment ) )
                            detrainment_gridded.append( np.asarray( detrainment ) )
                        entrainment_map.append( np.asarray( entrainment_gridded ) )
                        detrainment_map.append( np.asarray( detrainment_gridded ) )
                    #compute the vertical gradient of buoyancy within each thermal by averaging buoyancy in the upper and lower hemispheres, subtracting them and dividing by the distance between centers of mass of each hemisphere (2*3R/8 = 3R/4):
                    #construct the meshgrid:
                    XX,YY,ZZ=np.meshgrid(self.x_new,self.y_new,self.z_new)
                    rdist2=XX*XX+YY*YY+ZZ*ZZ
                    upper=np.where((rdist2<=1.)*ZZ>0.)
                    lower=np.where((rdist2<=1.)*ZZ<0.)
                    dBdZ0=[]
                    for i in range(buoy_map[-1].shape[0]):
                        dBdZ0.append((np.nanmean(buoy_map[-1][i,upper[0],upper[1],upper[2]])-np.nanmean(buoy_map[-1][i,lower[0],lower[1],lower[2]]))/(3.*R[-1][i]/4.))
                    dBdZ.append(np.mean(dBdZ0))
                    dBdZmax.append(np.max(dBdZ0))
                    dBdZmin.append(np.min(dBdZ0))
                    dBdZinit.append(dBdZ0[0])
            #self.tracer_entr = np.asarray(tracer_entr)
            #self.tracer_detr = np.asarray(tracer_detr)
 
            self.casename = casename

            self.entrainment_map = np.asarray(entrainment_map, dtype=object)
            self.detrainment_map = np.asarray(detrainment_map, dtype=object)
            self.angles = angles
            
            self.buoy_map   = np.asarray(buoy_map, dtype=object)
            self.dBdZ       = np.asarray(dBdZ)
            self.dBdZmax    = np.asarray(dBdZmax)
            self.dBdZmin    = np.asarray(dBdZmin)
            self.dBdZinit   = np.asarray(dBdZinit)

            self.wmax       = np.asarray(wmax, dtype=object)
            self.R          = np.asarray(R, dtype=object   )
            self.W          = np.asarray(W, dtype=object   )
            self.U          = np.asarray(U, dtype=object   )
            self.V          = np.asarray(V, dtype=object   )
            self.x_centre   = np.asarray(x_centre, dtype=object)
            self.y_centre   = np.asarray(y_centre, dtype=object)
            self.z_centre   = np.asarray(z_centre, dtype=object)
            self.time       = np.asarray(time, dtype=object)
            self.yy         = np.asarray(yy, dtype=object)
            self.mm         = np.asarray(mm, dtype=object)
            self.dd         = np.asarray(dd, dtype=object)
            self.hr         = np.asarray(hr, dtype=object)
            self.mn         = np.asarray(mn, dtype=object)
            self.sc         = np.asarray(sc, dtype=object)
            self.D          = np.asarray(D, dtype=object)

            self.Fres       = np.asarray(Fres, dtype=object  )
            self.buoy       = np.asarray(buoy, dtype=object  )
            self.Fnh        = np.asarray(Fnh, dtype=object   )
            self.acc        = np.asarray(acc, dtype=object   )
            self.Pnz        = np.asarray(Pnz, dtype=object   )
            self.mass       = np.asarray(mass, dtype=object  )*1e-5
            #self.mass_cond  = np.asarray(mass_cond, dtype=object)*1e-5
            self.fract_entr = 1./(self.D*1e3)
            self.mixing_mse  = np.asarray(mixing_mse, dtype=object)
            self.mse_thermal = np.asarray(mse_thermal, dtype=object)
            self.mse_env     = np.asarray(mse_env, dtype=object)
            self.mse_diff_init=np.asarray(mse_diff_init)
            self.delta_z     = np.asarray(delta_z)
            self.z0          = np.asarray(z0)
            self.deltazR     = np.asarray(deltazR)

            self.entr_rate   = np.asarray(entr_rate, dtype=object)

            self.net_entr_term = np.asarray(net_entr_term, dtype=object)

            self.qncloud    = np.asarray(qncloud, dtype=object)
            self.qcloud     = np.asarray(qcloud, dtype=object)
            self.qnrain     = np.asarray(qnrain, dtype=object)
            self.qrain      = np.asarray(qrain, dtype=object)
            self.cldnuc     = np.asarray(cldnuc, dtype=object)
            self.latheat    = np.asarray(latheat, dtype=object)
            self.noninduc   = np.asarray(noninduc, dtype=object)
            self.sctot      = np.asarray(sctot, dtype=object)
            self.qngraupel  = np.asarray(qngraupel, dtype=object)
            self.epotential = np.asarray(epotential, dtype=object)
            self.qicesnow   = np.asarray(qicesnow, dtype=object)
            self.qghail     = np.asarray(qghail, dtype=object)

            # get the volume of each thermal (will be used for weighting averages)
            self.volume     = 4.*np.pi*np.power(self.R*1e-3,3)/3.  # in km^3

            x_centre    = aux.flatten_array( np.asarray(x_centre, dtype=object) )
            y_centre    = aux.flatten_array( np.asarray(y_centre, dtype=object) )
            R_flat      = aux.flatten_array( np.asarray(self.R) )
           
            self.area = (np.max(x_centre+R_flat*1e-3)-np.min(x_centre-R_flat*1e-3))*(np.max(y_centre+R_flat*1e-3)-np.min(y_centre-R_flat*1e-3))
            time_flat = aux.flatten_array( time )
            self.simtime = (np.max(time_flat)-np.min(time_flat))/60.
            self.N_thermals = len(self.wmax)
            self.min_N      = int(self.N_thermals*self.min_ratio)

            self._get_centered_vars_at_wmax()
            self._compute_mf_weights_by_altitude()
            self._compute_net_entrainment()
            self._make_composite(compute_from_scratch=compute_from_scratch)
            self._compute_average_values()


    def _compute_mf_weights_by_altitude( self ):
        # create the vertical discretization for composites averaging by height
        #**********************************************************
        # DHD 14.03.2024
        zmax = 15000
        zmin = 0
        deltaz = 1000 
        # ALEJANDRA
        zmax = 7500
        zmin = 0
        deltaz = 500 
        self.z_edges = np.arange(zmin,zmax+1,deltaz)
        self.z_centers = (self.z_edges[:-1]+self.z_edges[1:])*0.5
        self.Nlevs = len(self.z_centers)
        self.z_flat = aux.flatten_array(self.z_centre) # altitude of each tracked timestep of each thermal
        mflux_flat = aux.flatten_array(self.mflux)
        self.mf_weights_z = np.ones(len(self.z_flat))*np.nan
        self.ind_zlev = np.zeros(len(self.z_flat))*np.nan        # the vertical level index of each tracked timestep (for composites by height)
        for i in range(self.Nlevs):
            ind = np.where((self.z_flat>self.z_edges[i])*(self.z_flat<=self.z_edges[i+1]))[0]
            mf_tot = np.nansum(mflux_flat[ind])
            self.mf_weights_z[ind] = mflux_flat[ind]/mf_tot
            self.ind_zlev[ind] = i
        self.ind_zlev = self.ind_zlev.astype(int)


    def _get_centered_vars_at_wmax( self ):
        it = []
        if self.up:
            for i in range(len(self.wmax)):
                if self.start_zero: #ALEJANDRA
                    ref = np.array([0])
                else:
                    ref = np.where(self.wmax[i] == np.amax(self.wmax[i][:]))[0]
                it.append((np.arange(len(self.wmax[i]))-ref[0]))
        else:
            for i in range(len(self.wmax)):
                if self.start_zero: #ALEJANDRA
                    ref = np.array([0])
                else:
                    ref = np.where(self.wmax[i] == np.amin(self.wmax[i][:]))[0]
                it.append((np.arange(len(self.wmax[i]))-ref[0]))

        it   = np.asarray(it, dtype=object)
        self.tmin = np.amin(it[0])
        self.tmax = np.amax(it[0])
        
        for i in range(len(it)):
            self.tmin = np.amin([np.amin(it[i]), self.tmin])
            self.tmax = np.amax([np.amax(it[i]), self.tmax])
        
        # get each thermal's mass flux for mass flux-weighted averages:
        self.mflux = self.mass*self.W
        ind=np.where(aux.flatten_array(self.z_centre)>=min_z_qnice)
        if np.sum(aux.flatten_array(self.mflux)[ind])>0:
            self.mflux_weights_grmean_qnice = self.mflux/np.sum(aux.flatten_array(self.mflux)[ind])
        else:
            self.mflux_weights_grmean_qnice = self.mflux*0
        self.mflux_weights_grmean = self.mflux/np.sum(aux.flatten_array(self.mflux))
        self.mf_weights_grmean_flat = aux.flatten_array(self.mflux_weights_grmean)
        self.mf_weights_grmean_qnice_flat = aux.flatten_array(self.mflux_weights_grmean_qnice)

        # get the volume-weights for 'grossmeans' (means over the entire arrays, not only per stages)
        self.v_weights_grmean = self.volume/np.sum(aux.flatten_array(self.volume))
        self.v_weights_grmean_flat = aux.flatten_array(self.v_weights_grmean)
        
        Trange = self.tmax - self.tmin + 1
        self.t_range = np.arange(self.tmin,self.tmax+1,1)

        self.wmax_c	= np.ones([len(it),Trange])*np.nan
        self.R_c                = np.ones([len(it),Trange])*np.nan
        self.Fres_c 	        = np.ones([len(it),Trange])*np.nan
        self.buoy_c 	        = np.ones([len(it),Trange])*np.nan
        self.Fnh_c 	            = np.ones([len(it),Trange])*np.nan
        self.acc_c 	            = np.ones([len(it),Trange])*np.nan
        self.D_c 	            = np.ones([len(it),Trange])*np.nan
        self.mass_c 	        = np.ones([len(it),Trange])*np.nan
        #self.mass_cond_c        = np.ones([len(it),Trange])*np.nan
        self.W_c                = np.ones([len(it),Trange])*np.nan
        self.Pnz_c              = np.ones([len(it),Trange])*np.nan
        self.z_centre_c         = np.ones([len(it),Trange])*np.nan
        self.y_centre_c         = np.ones([len(it),Trange])*np.nan
        self.x_centre_c         = np.ones([len(it),Trange])*np.nan
        self.time_c             = np.ones([len(it),Trange])*np.nan
        self.net_entr_term_c    = np.ones([len(it),Trange-1])*np.nan
        self.volume_c           = np.ones([len(it),Trange])*np.nan
        self.mflux_c            = np.ones([len(it),Trange])*np.nan
        self.v_weights_grmean_c = np.ones([len(it),Trange])*np.nan
        self.v_weights_stages_c = np.ones([len(it),Trange])*np.nan
        self.mf_weights_grmean_c= np.ones([len(it),Trange])*np.nan
        self.mf_weights_stages_c= np.ones([len(it),Trange])*np.nan
        self.mf_weights_stages_c_qnice = np.ones([len(it),Trange])*np.nan
        self.mf_weights_stages_c_7km = np.ones([len(it),Trange])*np.nan
        self.mf_weights_perthermal = np.ones([len(it),Trange])*np.nan
        self.mf_weights_perthermal_qnice = np.ones([len(it),Trange])*np.nan
        
        self.qcloud_c       = np.ones([len(it),Trange])*np.nan
        self.qncloud_c      = np.ones([len(it),Trange])*np.nan
        self.qrain_c        = np.ones([len(it),Trange])*np.nan
        self.qnrain_c       = np.ones([len(it),Trange])*np.nan
        self.cldnuc_c       = np.ones([len(it),Trange])*np.nan
        self.latheat_c      = np.ones([len(it),Trange])*np.nan
        self.noninduc_c     = np.ones([len(it),Trange])*np.nan
        self.sctot_c        = np.ones([len(it),Trange])*np.nan
        self.qngraupel_c    = np.ones([len(it),Trange])*np.nan
        self.epotential_c   = np.ones([len(it),Trange])*np.nan
        self.qicesnow_c     = np.ones([len(it),Trange])*np.nan
        self.qghail_c       = np.ones([len(it),Trange])*np.nan
      
        for i in range(len(it)):
            j = self.tmin
            while it[i][0] > j:
                j+=1
            j = j-self.tmin
            self.wmax_c      [i,j:len(self.wmax[i])+j] = self.wmax    [i][:]
            self.R_c	     [i,j:len(self.wmax[i])+j] = self.R       [i][:]
            self.Fres_c      [i,j:len(self.wmax[i])+j] = self.Fres    [i][:]  
            self.buoy_c      [i,j:len(self.wmax[i])+j] = self.buoy    [i][:]  
            self.Fnh_c       [i,j:len(self.wmax[i])+j] = self.Fnh     [i][:]  
            self.acc_c       [i,j:len(self.wmax[i])+j] = self.acc     [i][:]  
            self.D_c         [i,j:len(self.wmax[i])+j] = self.D       [i][:]
            self.mass_c      [i,j:len(self.wmax[i])+j] = self.mass    [i][:]  
            #self.mass_cond_c [i,j:len(self.wmax[i])+j] = self.mass_cond[i][:]  
            self.W_c	     [i,j:len(self.wmax[i])+j] = self.W       [i][:]  
            self.Pnz_c       [i,j:len(self.wmax[i])+j] = self.Pnz     [i][:]
            self.z_centre_c  [i,j:len(self.wmax[i])+j] = self.z_centre[i][:]
            self.y_centre_c  [i,j:len(self.wmax[i])+j] = self.y_centre[i][:]
            self.x_centre_c  [i,j:len(self.wmax[i])+j] = self.x_centre[i][:]
            self.time_c      [i,j:len(self.wmax[i])+j] = self.time    [i][:]
            self.net_entr_term_c[i,j:len(self.wmax[i])+j-1] = self.net_entr_term[i][:]
            self.volume_c    [i,j:len(self.wmax[i])+j] = self.volume  [i][:]
            self.v_weights_grmean_c[i,j:len(self.wmax[i])+j] = self.v_weights_grmean[i][:]
            self.mflux_c     [i,j:len(self.wmax[i])+j] = self.mflux  [i][:]
            self.mf_weights_grmean_c[i,j:len(self.wmax[i])+j] = self.mflux_weights_grmean[i][:]

            self.qcloud_c    [i,j:len(self.wmax[i])+j] = self.qcloud [i][:]
            self.qncloud_c   [i,j:len(self.wmax[i])+j] = self.qncloud[i][:]
            self.qrain_c     [i,j:len(self.wmax[i])+j] = self.qrain  [i][:]
            self.qnrain_c    [i,j:len(self.wmax[i])+j] = self.qnrain [i][:]
            self.cldnuc_c    [i,j:len(self.wmax[i])+j] = self.cldnuc [i][:]
            self.latheat_c   [i,j:len(self.wmax[i])+j] = self.latheat[i][:]
            self.noninduc_c  [i,j:len(self.wmax[i])+j] = self.noninduc [i][:]
            self.sctot_c     [i,j:len(self.wmax[i])+j] = self.sctot[i][:]
            self.qngraupel_c [i,j:len(self.wmax[i])+j] = self.qngraupel [i][:]
            self.epotential_c[i,j:len(self.wmax[i])+j] = self.epotential[i][:]
            self.qicesnow_c  [i,j:len(self.wmax[i])+j] = self.qicesnow[i][:]
            self.qghail_c    [i,j:len(self.wmax[i])+j] = self.qghail [i][:]


        self.it = it

        # Now the weights for stage averages:
        sum_vol = np.zeros(Trange)
        sum_mf = np.zeros(Trange)
        sum_mf_qnice = np.zeros(Trange)
        # thermals that initiate below 7km:
        sum_mf_7km = np.zeros(Trange)
        ind_below7km = np.where(self.z0<=7000)
        for j in range(Trange):
            sum_vol[j] = np.nansum(self.volume_c[:,j])
            sum_mf[j] = np.nansum(self.mflux_c[:,j])
            ind=np.where(self.z_centre_c[:,j]>=min_z_qnice)
            sum_mf_qnice[j] = np.nansum(self.mflux_c[:,j][ind])
            sum_mf_7km[j] = np.nansum(self.mflux_c[ind_below7km,j])
            self.v_weights_stages_c[:,j] = self.volume_c[:,j]/sum_vol[j]
            self.mf_weights_stages_c[:,j] = self.mflux_c[:,j]/sum_mf[j]
            self.mf_weights_stages_c[:,j] = self.mflux_c[:,j]/sum_mf[j]
            self.mf_weights_stages_c_qnice[:,j][ind] = self.mflux_c[:,j][ind]/sum_mf_qnice[j]
            self.mf_weights_stages_c_7km[ind_below7km,j] = self.mflux_c[ind_below7km,j]/sum_mf_7km[j]
        self.v_weights_stages = self.v_weights_stages_c[np.where(~np.isnan(self.v_weights_stages_c))]
        self.mf_weights_stages = self.mf_weights_stages_c[np.where(~np.isnan(self.mf_weights_stages_c))]
        self.mf_weights_stages_qnice = self.mf_weights_stages_c_qnice[np.where(~np.isnan(self.mf_weights_stages_c))]
        self.mf_weights_stages_7km = self.mf_weights_stages_c_7km[np.where(~np.isnan(self.mf_weights_stages_c))]

        self.loge_c = np.log10(1./(self.D_c*1e3))
        ind=np.where(np.nanmean(self.z_centre_c,axis=1)>=min_z_qnice)
        for i in range(Trange):
            self.mf_weights_perthermal[:,i] = np.nanmean(self.mflux_c,axis=1)/np.nansum(np.nanmean(self.mflux_c,axis=1))
            self.mf_weights_perthermal_qnice[:,i][ind]=np.nanmean(self.mflux_c,axis=1)[ind]/np.nansum(np.nanmean(self.mflux_c,axis=1)[ind])
        self.mf_weights_perthermal = self.mf_weights_perthermal[np.where(~np.isnan(self.mf_weights_stages_c))] # these weights are all equal for each timestep of the same thermal, but have the same shape as those for the stages
        self.mf_weights_perthermal_qnice  = self.mf_weights_perthermal_qnice[np.where(~np.isnan(self.mf_weights_stages_c))]
    
    def condition( self,ixl,ixr,iyl,iyr,izlow,izup,R,D,wmax,W ):
        """
        This is for subsampling. Should be defined in the running script if needed.
        """
        return True


    def _make_composite(self, res_fac=1, iz_sample=None, compute_from_scratch=False):
        """
        makes a composite of all thermals, taking as reference timestep the one in which each thermal reaches its peak wmax.
        If only certain cases want to be used for the composite based on their vertical elongation, then iz_sample should indicate
        the extreme values to be considered, for example iz_sample=[-2.,-1] in order to count only thermals that extend downward from 
        1 times its radius to twice its radius (iz_low>=-2 and iz_low<=-1).
        Also, makes composites based on altitude (z) DHD 14.03.2024
        """
        N_thermals = self.N_thermals 
        tmax_ref = []	# timestep where wmax peaks (considering first and last) this will be t=0 in plots
        length = []     # length of time that each thermal lives
        if self.up:
            for i in range(N_thermals):
                if self.start_zero: #ALEJANDRA: to include start_zero option
                    tmax_ref.append(0)
                else:
                    tmax_ref.append( np.where(self.wmax[i]==np.amax(self.wmax[i][:]))[0][0] )
                length.append( len(self.wmax[i]) )
        else:
            for i in range(N_thermals):
                if self.start_zero: #ALEJANDRA: to include start_zero option
                    tmax_ref.append(0)
                else:
                    tmax_ref.append( np.where(self.wmax[i]==np.amin(self.wmax[i][:]))[0][0] )    
                length.append( len(self.wmax[i]) )
        tmax_ref = np.asarray(tmax_ref)
        length = np.asarray(length)
        
        print ("--------------Times of reference-----------------") #Alejandra
        print ("tmax_ref", tmax_ref)
        print ("--------------------------------------------------")

        
        ind = []        # common index for all thermals. Covers from the earliest to the latest thermal, so has the maximum possible stages
        for i in range(N_thermals):
            ind.append(np.arange(len(self.wmax[i]))-tmax_ref[i] + np.max(tmax_ref) )
        # nx, ny and nz in R-coordinates
        nxR = len(self.x_new)
        nyR = len(self.y_new)
        nzR = np.copy(nxR)

        max_left = np.max(tmax_ref)         # maximum number of timesteps after tref
        max_right = np.max(length-tmax_ref) # maximum number of timesteps before tref
        stages = max_right + max_left    # total (maximum) number of stages of the thermals
        R_count                 = np.zeros(stages)
        R_mean                  = np.zeros(stages)
        D_mean                  = np.zeros(stages)
        #**********************************************************
        # variables used for cross sections throughout altitude:
        u_mean_z                = np.zeros([nxR,nyR,nzR,self.Nlevs])
        v_mean_z                = np.zeros([nxR,nyR,nzR,self.Nlevs])
        w_mean_z                = np.zeros([nxR,nyR,nzR,self.Nlevs])
        u_dev_mean_z            = np.zeros([nxR,nyR,nzR,self.Nlevs])
        v_dev_mean_z            = np.zeros([nxR,nyR,nzR,self.Nlevs])
        w_dev_mean_z            = np.zeros([nxR,nyR,nzR,self.Nlevs])
        buoy_mean_z             = np.zeros([nxR,nyR,nzR,self.Nlevs])
        pdev_mean_z             = np.zeros([nxR,nyR,nzR,self.Nlevs])
        sctot_mean_z            = np.zeros([nxR,nyR,nzR,self.Nlevs])
        latheat_mean_z          = np.zeros([nxR,nyR,nzR,self.Nlevs])
        qnice_mean_z            = np.zeros([nxR,nyR,nzR,self.Nlevs])
        qncloud_mean_z          = np.zeros([nxR,nyR,nzR,self.Nlevs])
        qnrain_mean_z           = np.zeros([nxR,nyR,nzR,self.Nlevs])
        qice_mean_z             = np.zeros([nxR,nyR,nzR,self.Nlevs])
        qcloud_mean_z           = np.zeros([nxR,nyR,nzR,self.Nlevs])
        qrain_mean_z            = np.zeros([nxR,nyR,nzR,self.Nlevs])
        qvapor_mean_z           = np.zeros([nxR,nyR,nzR,self.Nlevs])
        noninduc_mean_z         = np.zeros([nxR,nyR,nzR,self.Nlevs])
        cldnuc_mean_z           = np.zeros([nxR,nyR,nzR,self.Nlevs])
        rh_mean_z               = np.zeros([nxR,nyR,nzR,self.Nlevs])
        epotential_mean_z       = np.zeros([nxR,nyR,nzR,self.Nlevs])
        qngraupel_mean_z        = np.zeros([nxR,nyR,nzR,self.Nlevs])
        qicesnow_mean_z         = np.zeros([nxR,nyR,nzR,self.Nlevs])
        qghail_mean_z           = np.zeros([nxR,nyR,nzR,self.Nlevs])
        #**********************************************************
        # variables used for cross sections throughout thermal lifetime stages:
        u_mean                  = np.zeros([nxR,nyR,nzR,stages])
        v_mean                  = np.zeros([nxR,nyR,nzR,stages])
        w_mean                  = np.zeros([nxR,nyR,nzR,stages])
        u_dev_mean              = np.zeros([nxR,nyR,nzR,stages])
        v_dev_mean              = np.zeros([nxR,nyR,nzR,stages])
        w_dev_mean              = np.zeros([nxR,nyR,nzR,stages])
        buoy_mean               = np.zeros([nxR,nyR,nzR,stages])
        pdev_mean               = np.zeros([nxR,nyR,nzR,stages])
        sctot_mean              = np.zeros([nxR,nyR,nzR,stages])
        latheat_mean            = np.zeros([nxR,nyR,nzR,stages])
        qnice_mean              = np.zeros([nxR,nyR,nzR,stages])
        qncloud_mean            = np.zeros([nxR,nyR,nzR,stages])
        qnrain_mean             = np.zeros([nxR,nyR,nzR,stages])
        qice_mean               = np.zeros([nxR,nyR,nzR,stages]) #DHD 22.20.2020
        qcloud_mean             = np.zeros([nxR,nyR,nzR,stages]) #DHD 22.20.2020
        qrain_mean              = np.zeros([nxR,nyR,nzR,stages]) #DHD 22.20.2020
        qvapor_mean             = np.zeros([nxR,nyR,nzR,stages]) #DHD 22.20.2020
        noninduc_mean           = np.zeros([nxR,nyR,nzR,stages])
        cldnuc_mean             = np.zeros([nxR,nyR,nzR,stages])
        rh_mean                 = np.zeros([nxR,nyR,nzR,stages])
        epotential_mean         = np.zeros([nxR,nyR,nzR,stages])
        qngraupel_mean          = np.zeros([nxR,nyR,nzR,stages])
        qicesnow_mean           = np.zeros([nxR,nyR,nzR,stages])
        qghail_mean             = np.zeros([nxR,nyR,nzR,stages])
        #***********************************************************       
        # same as above, but for thermals that start below 7km
        u_mean_7km              = np.zeros([nxR,nyR,nzR,stages])
        v_mean_7km              = np.zeros([nxR,nyR,nzR,stages])
        w_mean_7km              = np.zeros([nxR,nyR,nzR,stages])
        u_dev_mean_7km          = np.zeros([nxR,nyR,nzR,stages])
        v_dev_mean_7km          = np.zeros([nxR,nyR,nzR,stages])
        w_dev_mean_7km          = np.zeros([nxR,nyR,nzR,stages])
        buoy_mean_7km           = np.zeros([nxR,nyR,nzR,stages])
        pdev_mean_7km           = np.zeros([nxR,nyR,nzR,stages])
        sctot_mean_7km          = np.zeros([nxR,nyR,nzR,stages])
        latheat_mean_7km        = np.zeros([nxR,nyR,nzR,stages])
        qnice_mean_7km          = np.zeros([nxR,nyR,nzR,stages])
        qncloud_mean_7km        = np.zeros([nxR,nyR,nzR,stages])
        qnrain_mean_7km         = np.zeros([nxR,nyR,nzR,stages])
        qice_mean_7km           = np.zeros([nxR,nyR,nzR,stages]) 
        qcloud_mean_7km         = np.zeros([nxR,nyR,nzR,stages]) 
        qrain_mean_7km          = np.zeros([nxR,nyR,nzR,stages]) 
        qvapor_mean_7km         = np.zeros([nxR,nyR,nzR,stages]) 
        noninduc_mean_7km       = np.zeros([nxR,nyR,nzR,stages])
        noninduc_counter_7km    = np.zeros([nxR,nyR,nzR,stages])
        cldnuc_mean_7km         = np.zeros([nxR,nyR,nzR,stages])
        rh_mean_7km             = np.zeros([nxR,nyR,nzR,stages])
        epotential_mean_7km     = np.zeros([nxR,nyR,nzR,stages])
        qngraupel_mean_7km      = np.zeros([nxR,nyR,nzR,stages])
        qicesnow_mean_7km       = np.zeros([nxR,nyR,nzR,stages])
        qghail_mean_7km         = np.zeros([nxR,nyR,nzR,stages])
        #***********************************************************       
        entrainment_mean        = np.zeros([len(self.angles),stages])
        detrainment_mean        = np.zeros([len(self.angles),stages])
        #u_counter               = np.zeros([nxR,nyR,nzR,stages])
        #v_counter               = np.zeros([nxR,nyR,nzR,stages])
        #w_counter               = np.zeros([nxR,nyR,nzR,stages])
        #buoy_counter            = np.zeros([nxR,nyR,nzR,stages])
        #pdev_counter            = np.zeros([nxR,nyR,nzR,stages])
        #sctot_counter           = np.zeros([nxR,nyR,nzR,stages])
        #latheat_counter         = np.zeros([nxR,nyR,nzR,stages])
        #qnice_counter           = np.zeros([nxR,nyR,nzR,stages])
        #qncloud_counter         = np.zeros([nxR,nyR,nzR,stages])
        #qnrain_counter          = np.zeros([nxR,nyR,nzR,stages])
        #qice_counter            = np.zeros([nxR,nyR,nzR,stages]) #DHD 22.20.2020
        #qcloud_counter          = np.zeros([nxR,nyR,nzR,stages]) #DHD 22.20.2020
        #qrain_counter           = np.zeros([nxR,nyR,nzR,stages]) #DHD 22.20.2020
        #noninduc_counter        = np.zeros([nxR,nyR,nzR,stages])
        #cldnuc_counter          = np.zeros([nxR,nyR,nzR,stages])
        #rh_counter              = np.zeros([nxR,nyR,nzR,stages])
        #epotential_counter      = np.zeros([nxR,nyR,nzR,stages])
        #qngraupel_counter       = np.zeros([nxR,nyR,nzR,stages])
        entrainment_grossmean   = np.zeros(len(self.angles))
        detrainment_grossmean   = np.zeros(len(self.angles))
        u_dev_grossmean         = np.zeros([nxR, nyR, nzR]) 
        v_dev_grossmean         = np.zeros([nxR, nyR, nzR]) 
        w_dev_grossmean         = np.zeros([nxR, nyR, nzR])
        u_dev_lastmean          = np.zeros([nxR, nyR, nzR]) 
        v_dev_lastmean          = np.zeros([nxR, nyR, nzR]) 
        w_dev_lastmean          = np.zeros([nxR, nyR, nzR])
        sctot_grossmean         = np.zeros([nxR, nyR, nzR]) 
        latheat_grossmean       = np.zeros([nxR, nyR, nzR]) 
        qnice_grossmean         = np.zeros([nxR, nyR, nzR]) 
        qncloud_grossmean       = np.zeros([nxR, nyR, nzR]) 
        qnrain_grossmean        = np.zeros([nxR, nyR, nzR]) 
        qice_grossmean          = np.zeros([nxR, nyR, nzR]) 
        qcloud_grossmean        = np.zeros([nxR, nyR, nzR]) 
        qrain_grossmean         = np.zeros([nxR, nyR, nzR]) 
        qvapor_grossmean        = np.zeros([nxR, nyR, nzR]) 
        noninduc_grossmean      = np.zeros([nxR, nyR, nzR])
        cldnuc_grossmean        = np.zeros([nxR, nyR, nzR]) 
        rh_grossmean            = np.zeros([nxR, nyR, nzR]) 
        epotential_grossmean    = np.zeros([nxR, nyR, nzR]) 
        qngraupel_grossmean     = np.zeros([nxR, nyR, nzR]) 
        qicesnow_grossmean      = np.zeros([nxR, nyR, nzR]) 
        qghail_grossmean        = np.zeros([nxR, nyR, nzR]) 
        sctot_lastmean          = np.zeros([nxR, nyR, nzR]) 
        latheat_lastmean        = np.zeros([nxR, nyR, nzR]) 
        qnice_lastmean          = np.zeros([nxR, nyR, nzR]) 
        qncloud_lastmean        = np.zeros([nxR, nyR, nzR]) 
        qnrain_lastmean         = np.zeros([nxR, nyR, nzR]) 
        qice_lastmean           = np.zeros([nxR, nyR, nzR]) 
        qcloud_lastmean         = np.zeros([nxR, nyR, nzR]) 
        qrain_lastmean          = np.zeros([nxR, nyR, nzR]) 
        qvapor_lastmean         = np.zeros([nxR, nyR, nzR]) 
        noninduc_lastmean       = np.zeros([nxR, nyR, nzR]) 
        cldnuc_lastmean         = np.zeros([nxR, nyR, nzR]) 
        rh_lastmean             = np.zeros([nxR, nyR, nzR]) 
        epotential_lastmean     = np.zeros([nxR, nyR, nzR]) 
        qngraupel_lastmean      = np.zeros([nxR, nyR, nzR]) 
        qicesnow_lastmean       = np.zeros([nxR, nyR, nzR]) 
        qghail_lastmean         = np.zeros([nxR, nyR, nzR]) 
        R_grossmean             = 0.
        gross_counter           = 0.
        R_lastmean              = 0.
        last_counter            = 0.
        iz_low, iz_up, ix_right, ix_left, iy_right, iy_left = [],[],[],[],[],[]

        ind_flat        = aux.flatten_array(ind).astype(int)
        time_flat       = aux.flatten_array(self.time)
        yy_flat         = aux.flatten_array(self.yy)
        mm_flat         = aux.flatten_array(self.mm)
        dd_flat         = aux.flatten_array(self.dd)
        hr_flat         = aux.flatten_array(self.hr)
        mn_flat         = aux.flatten_array(self.mn)
        sc_flat         = aux.flatten_array(self.sc)
        x_centre_flat   = aux.flatten_array(self.x_centre)
        y_centre_flat   = aux.flatten_array(self.y_centre)
        z_centre_flat   = aux.flatten_array(self.z_centre)
        R_flat          = aux.flatten_array(self.R)
        D_flat          = aux.flatten_array(self.D)
        wmax_flat       = aux.flatten_array(self.wmax)
        W_flat          = aux.flatten_array(self.W)

        include = np.zeros_like(ind_flat)

        shape = self.buoy_map.shape
        buoy_map_flat = []
        entrainment_map_flat = []
        detrainment_map_flat = []
        for i in range(self.buoy_map.shape[0]):
            for j in range(self.buoy_map[i].shape[0]):
                buoy_map_flat.append( self.buoy_map[i][j]  )
                entrainment_map_flat.append( self.entrainment_map[i][j] )
                detrainment_map_flat.append( self.detrainment_map[i][j] )
                
        buoy_map_flat = np.asarray(buoy_map_flat)
        entrainment_map_flat = np.asarray(entrainment_map_flat)
        detrainment_map_flat = np.asarray(detrainment_map_flat)

        i_ordered = np.argsort(time_flat)
        data = None
        n=0
        if iz_sample==None:
            iz0=-self.R_range-self.delta_R
            iz1=0.
        else:
            iz0 = iz_sample[0]
            iz1 = iz_sample[1]
        #find optimal factor for the weights (so that weights are not too big not too small...)
        factor = 1
        while np.around(np.log10(factor*np.max(self.mf_weights_stages)))+np.around(np.log10(factor*np.min(self.mf_weights_stages)))<0:
            factor=factor*10
        #print( '** factor for weights: %d'%(factor))
        #ice_counter=0

        # identify indices of the last tracked timestep of each thermal (to make composites of this instant):
        last_indices = []
        iii=0
        for thermal_ind in ind:
            if iii!=0:
                last_indices.append(len(thermal_ind)+last_indices[-1])
            else:
                last_indices.append(len(thermal_ind)-1)
                iii+=1

        for i in i_ordered:
            time = time_flat[i]
            yy = yy_flat[i]
            mm = mm_flat[i]
            dd = dd_flat[i]
            hr = hr_flat[i]
            mn = mn_flat[i]
            sc = sc_flat[i]
            if (not compute_from_scratch) and os.path.isfile( self.casename[i][0] + '/uvwp_%02d.npy'%(self.casename[i][1])) and np.all( np.load(self.casename[i][0]+'/uvwp_grid.npy', allow_pickle=True, encoding='latin1' )==np.array([self.R_range, self.delta_R]) ):
                # this will read in the previously computed 3D fields for this thermal
                uvwp = np.load(self.casename[i][0] + '/uvwp_%02d.npy'%(self.casename[i][1]), allow_pickle=True, encoding='latin1' )
                size = uvwp.shape[1]
                u_new           = uvwp[:size,:,:]
                v_new           = uvwp[size:2*size,:,:]  
                w_new           = uvwp[2*size:3*size,:,:]
                pdev_new        = uvwp[3*size:4*size,:,:]
                latheat_new     = uvwp[4*size:5*size,:,:]
                qnice_new       = uvwp[5*size:6*size,:,:]
                qncloud_new     = uvwp[6*size:7*size,:,:]
                qnrain_new      = uvwp[7*size:8*size,:,:]
                cldnuc_new      = uvwp[8*size:9*size,:,:]
                rh_new          = uvwp[9*size:10*size,:,:]
                qice_new        = uvwp[10*size:11*size,:,:]
                qcloud_new      = uvwp[11*size:12*size,:,:]
                qrain_new       = uvwp[12*size:13*size,:,:]
                qvapor_new      = uvwp[13*size:14*size,:,:]
                if uvwp.shape[0]>14*size:
                    sctot_new       = uvwp[14*size:15*size,:,:]
                    noninduc_new    = uvwp[15*size:16*size,:,:]
                    epotential_new  = uvwp[16*size:17*size,:,:]
                    qngraupel_new   = uvwp[17*size:18*size,:,:]
                    qicesnow_new    = uvwp[18*size:19*size,:,:]
                    qghail_new      = uvwp[19*size:20*size,:,:]
                else: # in case the previous run did not have these variables
                    sctot_new       = np.ones_like(uvwp[size:2*size,:,:])*np.nan
                    noninduc_new    = np.ones_like(uvwp[size:2*size,:,:])*np.nan
                    epotential_new  = np.ones_like(uvwp[size:2*size,:,:])*np.nan
                    qngraupel_new   = np.ones_like(uvwp[size:2*size,:,:])*np.nan
                    qicesnow_new    = np.ones_like(uvwp[size:2*size,:,:])*np.nan
                    qghail_new      = np.ones_like(uvwp[size:2*size,:,:])*np.nan

                u_dev = u_new - aux.flatten_array(self.U)[i]
                v_dev = v_new - aux.flatten_array(self.V)[i]
                w_dev = w_new - aux.flatten_array(self.W)[i]
            else:
                # this will compute the 3D fields for this thermal (either because it has not been done previously, or because the user wants to do it again (compute_from_scratch=True)
                if data == None or data.hr0*60.+data.min0+data.sec0/60. != time:
                    same_tstep=np.where(time_flat==time)[0]

                    x0 = np.max( [np.min(x_centre_flat[same_tstep] - R_flat[same_tstep]/1e3 - 0.5), 0] )
                    y0 = np.max( [np.min(y_centre_flat[same_tstep] - R_flat[same_tstep]/1e3 - 0.5), 0] )
                    nxkm = np.min([np.max(x_centre_flat[same_tstep] + R_flat[same_tstep]/1e3 + 0.5) - x0, self.xff-x0])
                    nykm = np.min([np.max(y_centre_flat[same_tstep] + R_flat[same_tstep]/1e3 + 0.5) - y0, self.yff-y0])
                    if self.X0!=0 or self.Y0!=0:
                        x0 = np.min(x_centre_flat[same_tstep] - R_flat[same_tstep]/1e3 - 0.5)
                        i0 = int((x0-self.X0)/(self.dx*1e-3))
                        y0 = np.min(y_centre_flat[same_tstep] - R_flat[same_tstep]/1e3 - 0.5)
                        j0 = int((y0-self.Y0)/(self.dx*1e-3))
                        x1 = np.max(x_centre_flat[same_tstep] + R_flat[same_tstep]/1e3 + 0.5)
                        y1 = np.max(y_centre_flat[same_tstep] + R_flat[same_tstep]/1e3 + 0.5)
                        nxi = int((x1-x0)/(self.dx*1e-3)) + 1
                        nyi = int((y1-y0)/(self.dy*1e-3)) + 1
                        data = grid3D.Grid( dx=self.dx, YY0=yy, MM0=mm, DD0=dd, hr0=hr, min0=mn, sec0=sc, nt=1, x0=self.X0, y0=self.Y0, i0=i0, j0=j0, nxi=nxi, nyi=nyi, nz=self.nz, path=self.path, dt=self.dt, header_fmt=self.header_fmt, n_jobs=self.n_jobs, ending=self.ending, compute_rh=self.compute_rh, gunzip=self.gunzip, GCE=self.GCE )
                    else:
                        data = grid3D.Grid( dx=self.dx, YY0=yy, MM0=mm, DD0=dd, hr0=hr, min0=mn, sec0=sc, nt=1, x0=x0, y0=y0, nxkm=nxkm, nykm=nykm, nz=self.nz, path=self.path, dt=self.dt, header_fmt=self.header_fmt, n_jobs=self.n_jobs, ending=self.ending, compute_rh=self.compute_rh, gunzip=self.gunzip, GCE=self.GCE ) #seems like interpolation with more than 23 jobs (n_jobs>23) is not efficient (but this may depend on the size of the domain)
                    data.ptot = np.ma.masked_array(data.ptot, mask=np.isnan(data.ptot))
                    pdev_orig = np.zeros_like(data.ptot)
                    for iz in range(pdev_orig.shape[2]):
                        pdev_orig[:,:,iz] = data.ptot[:,:,iz]-np.ma.mean(data.ptot[:,:,iz])                   # pressure deviation (from horizontal mean)
                xcentre = x_centre_flat[i]
                ycentre = y_centre_flat[i]
                zcentre = z_centre_flat[i]
                radius  = R_flat[i]
                if self.X0!=0 or self.Y0!=0:
                    x_grid_scaled = ((data.x_grid-self.X0*1e3+x0*1e3) - xcentre*1e3)/radius
                    y_grid_scaled = ((data.y_grid-self.Y0*1e3+y0*1e3) - ycentre*1e3)/radius
                else:
                    x_grid_scaled = (data.x_grid-xcentre*1e3)/radius
                    y_grid_scaled = (data.y_grid-ycentre*1e3)/radius
                hgt_scaled = (data.hgt_c-zcentre)/radius
                x_slice = np.where((x_grid_scaled<=self.R_range+self.delta_R)*(x_grid_scaled>=-self.R_range-self.delta_R))[0]
                y_slice = np.where((y_grid_scaled<=self.R_range+self.delta_R)*(y_grid_scaled>=-self.R_range-self.delta_R))[0]
                z_slice = np.where((hgt_scaled<=self.R_range+self.delta_R)*(hgt_scaled>=-self.R_range-self.delta_R))[0]
                points = points_grid( x_grid_scaled[x_slice], y_grid_scaled[y_slice], hgt_scaled[z_slice] )
                new_points = points_grid( self.x_new, self.y_new, self.z_new )
                #coordinates of the points where we want to interpolate to, in terms of the original grids (thermal.x_grid...):
                ix_new   = np.interp(self.x_new, x_grid_scaled[x_slice], np.arange(len(x_grid_scaled[x_slice])))
                iy_new   = np.interp(self.y_new, y_grid_scaled[y_slice], np.arange(len(y_grid_scaled[y_slice])))
                iz_new   = np.interp(self.z_new, hgt_scaled[z_slice], np.arange(len(hgt_scaled[z_slice])))
                or_grid  = points_grid( np.arange(len(x_grid_scaled[x_slice])), np.arange(len(y_grid_scaled[y_slice])), np.arange(len(hgt_scaled[z_slice])) )
                new_grid = points_grid( ix_new, iy_new, iz_new )

                ints  = len(ix_new)*len(iy_new)*len(iz_new)
                u_new           = np.zeros( ints )
                v_new           = np.zeros( ints )
                w_new           = np.zeros( ints )
                pdev_new        = np.zeros( ints )
                sctot_new       = np.zeros( ints )
                latheat_new     = np.zeros( ints )
                qnice_new       = np.zeros( ints )
                qncloud_new     = np.zeros( ints )
                qnrain_new      = np.zeros( ints )
                qice_new        = np.zeros( ints )
                qcloud_new      = np.zeros( ints )
                qrain_new       = np.zeros( ints )
                qvapor_new      = np.zeros( ints )
                cldnuc_new      = np.zeros( ints )
                noninduc_new    = np.zeros( ints )
                rh_new          = np.zeros( ints )
                epotential_new  = np.zeros( ints )
                qngraupel_new   = np.zeros( ints )
                qicesnow_new    = np.zeros( ints )
                qghail_new      = np.zeros( ints )
                
                #start=tm.time()
                # divide the work by splitting the spatial grid
                subjob = []
                l0=0
                n_jobs=self.n_jobs
                dl = int(ints/n_jobs)
                jobs = []
                for j in range(n_jobs-1):
                    l1 = l0 + dl
                    subjob.append([l0,l1])
                    l0 = l1
                subjob.append([l0,ints])
                for j in range(n_jobs):
                    jobs.append( (subjob[j], x_grid_scaled[x_slice], y_grid_scaled[y_slice], hgt_scaled[z_slice], new_grid, data.u_c[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.v_c[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.w_c[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], pdev_orig[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.latheat[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.qnice[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.qncloud[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.qnrain[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.cldnuc[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.rh[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.qice[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.qcloud[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.qrain[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.qvapor[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.sctot[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.noninduc[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.epotential[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.qngraupel[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.qicesnow[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0], data.qghail[x_slice[0]:x_slice[-1]+1,y_slice[0]:y_slice[-1]+1,z_slice[0]:z_slice[-1]+1,0] ) )
                #print 'interp_points start'
                #( vars0 ) = Parallel(n_jobs=n_jobs)(delayed(interp_points)(*jobs[j]) for j in range(len(jobs)))
                ( vars0 ) = Parallel(n_jobs=n_jobs)(delayed(interp_points_RGI)(*jobs[j]) for j in range(len(jobs)))
                for j in range(n_jobs):
                    u_new[subjob[j][0]:subjob[j][1]]          = vars0[j][0]
                    v_new[subjob[j][0]:subjob[j][1]]          = vars0[j][1]
                    w_new[subjob[j][0]:subjob[j][1]]          = vars0[j][2]
                    pdev_new[subjob[j][0]:subjob[j][1]]       = vars0[j][3]
                    latheat_new[subjob[j][0]:subjob[j][1]]    = vars0[j][4]
                    qnice_new[subjob[j][0]:subjob[j][1]]      = vars0[j][5]
                    qncloud_new[subjob[j][0]:subjob[j][1]]    = vars0[j][6]
                    qnrain_new[subjob[j][0]:subjob[j][1]]     = vars0[j][7]
                    cldnuc_new[subjob[j][0]:subjob[j][1]]     = vars0[j][8]
                    rh_new[subjob[j][0]:subjob[j][1]]         = vars0[j][9]
                    qice_new[subjob[j][0]:subjob[j][1]]       = vars0[j][10]
                    qcloud_new[subjob[j][0]:subjob[j][1]]     = vars0[j][11]
                    qrain_new[subjob[j][0]:subjob[j][1]]      = vars0[j][12]
                    qvapor_new[subjob[j][0]:subjob[j][1]]     = vars0[j][13]
                    sctot_new[subjob[j][0]:subjob[j][1]]      = vars0[j][14]
                    noninduc_new[subjob[j][0]:subjob[j][1]]   = vars0[j][15]
                    epotential_new[subjob[j][0]:subjob[j][1]] = vars0[j][16]
                    qngraupel_new[subjob[j][0]:subjob[j][1]]  = vars0[j][17]
                    qicesnow_new[subjob[j][0]:subjob[j][1]]   = vars0[j][18]
                    qghail_new[subjob[j][0]:subjob[j][1]]     = vars0[j][19]

                u_new         = np.reshape( u_new, (len(ix_new), len(iy_new), len(iz_new)) )
                v_new         = np.reshape( v_new, (len(ix_new), len(iy_new), len(iz_new)) )
                w_new         = np.reshape( w_new, (len(ix_new), len(iy_new), len(iz_new)) )
                pdev_new      = np.reshape( pdev_new, (len(ix_new), len(iy_new), len(iz_new)) )
                latheat_new   = np.reshape( latheat_new, (len(ix_new), len(iy_new), len(iz_new)) )
                qnice_new     = np.reshape( qnice_new, (len(ix_new), len(iy_new), len(iz_new)) )
                qncloud_new   = np.reshape( qncloud_new, (len(ix_new), len(iy_new), len(iz_new)) )
                qnrain_new    = np.reshape( qnrain_new, (len(ix_new), len(iy_new), len(iz_new)) )
                cldnuc_new    = np.reshape( cldnuc_new, (len(ix_new), len(iy_new), len(iz_new)) )
                rh_new        = np.reshape( rh_new, (len(ix_new), len(iy_new), len(iz_new)) )
                qice_new      = np.reshape( qice_new, (len(ix_new), len(iy_new), len(iz_new)) )
                qcloud_new    = np.reshape( qcloud_new, (len(ix_new), len(iy_new), len(iz_new)) )
                qrain_new     = np.reshape( qrain_new, (len(ix_new), len(iy_new), len(iz_new)) )
                qvapor_new    = np.reshape( qvapor_new, (len(ix_new), len(iy_new), len(iz_new)) )
                sctot_new     = np.reshape( sctot_new, (len(ix_new), len(iy_new), len(iz_new)) )
                noninduc_new  = np.reshape( noninduc_new, (len(ix_new), len(iy_new), len(iz_new)) )
                epotential_new= np.reshape( epotential_new, (len(ix_new), len(iy_new), len(iz_new)) )
                qngraupel_new = np.reshape( qngraupel_new, (len(ix_new), len(iy_new), len(iz_new)) )
                qicesnow_new  = np.reshape( qicesnow_new, (len(ix_new), len(iy_new), len(iz_new)) )
                qghail_new    = np.reshape( qghail_new, (len(ix_new), len(iy_new), len(iz_new)) )
                #print( 'interp_points took %f seconds'%(tm.time()-start))

                
                np.save( self.casename[i][0] + '/' + 'uvwp_%02d.npy'%(self.casename[i][1]), np.vstack((u_new,
                    v_new,
                    w_new,
                    pdev_new,
                    latheat_new,
                    qnice_new,
                    qncloud_new,
                    qnrain_new,
                    cldnuc_new,
                    rh_new,
                    qice_new,
                    qcloud_new,
                    qrain_new,
                    qvapor_new, 
                    sctot_new, 
                    noninduc_new, 
                    epotential_new, 
                    qngraupel_new, 
                    qicesnow_new, 
                    qghail_new)) )
                if self.casename[i][1]==0:
                    np.save( self.casename[i][0] + '/' + 'uvwp_grid.npy', np.array([self.R_range,self.delta_R]) )

                u_dev = u_new - aux.flatten_array(self.U)[i]
                v_dev = v_new - aux.flatten_array(self.V)[i] 
                w_dev = w_new - aux.flatten_array(self.W)[i]

            # ALEJANDRA: to make sure this are calculated again!
            if os.path.isfile( self.casename[i][0] + '/' + 'ixiyiz_%02d.npy'%(self.casename[i][1]) ):
                ixiyiz = np.load( self.casename[i][0] + '/' + 'ixiyiz_%02d.npy'%(self.casename[i][1]), allow_pickle=True, encoding='latin1'  )
                ix_left.append(ixiyiz[0])
                ix_right.append(ixiyiz[1])
                iy_left.append(ixiyiz[2])
                iy_right.append(ixiyiz[3])
                iz_low.append(ixiyiz[4])
                iz_up.append(ixiyiz[5])
            else:
                ixl, ixr, iyl, iyr, izl, izh, w_plume = self._find_thermal_shape_parameters( w_dev )
                np.save( self.casename[i][0] + '/' + 'ixiyiz_%02d.npy'%(self.casename[i][1]), np.array([ixl,ixr,iyl,iyr,izl,izh]) )
                if np.any(w_plume!=None):
                    np.save( self.casename[i][0] + '/' + 'w_plume_%02d.npy'%(self.casename[i][1]), w_plume )
                ix_left.append(ixl)
                ix_right.append(ixr)
                iy_left.append(iyl)
                iy_right.append(iyr)
                iz_low.append(izl)
                iz_up.append(izh)
          
            if self.condition(ix_left[-1],ix_right[-1],iy_left[-1],iy_right[-1],iz_low[-1],iz_up[-1],R_flat[i],D_flat[i],wmax_flat[i],W_flat[i]):
                # here the composites are calculated:
                u_new[np.where(np.isnan(u_new))]                =0.
                v_new[np.where(np.isnan(v_new))]                =0.
                w_new[np.where(np.isnan(w_new))]                =0.
                u_dev[np.where(np.isnan(u_dev))]                =0.
                v_dev[np.where(np.isnan(v_dev))]                =0.
                w_dev[np.where(np.isnan(w_dev))]                =0.
                pdev_new[np.where(np.isnan(pdev_new))]          =0.
                sctot_new[np.where(np.isnan(sctot_new))]        =0.
                latheat_new[np.where(np.isnan(latheat_new))]    =0.
                qnice_new[np.where(np.isnan(qnice_new))]        =0.
                qncloud_new[np.where(np.isnan(qncloud_new))]    =0.
                qnrain_new[np.where(np.isnan(qnrain_new))]      =0.
                noninduc_new[np.where(np.isnan(noninduc_new))]  =0.
                cldnuc_new[np.where(np.isnan(cldnuc_new))]      =0.
                rh_new[np.where(np.isnan(rh_new))]              =0.
                qice_new[np.where(np.isnan(qice_new))]          =0.
                qcloud_new[np.where(np.isnan(qcloud_new))]      =0.
                qrain_new[np.where(np.isnan(qrain_new))]        =0.
                qvapor_new[np.where(np.isnan(qvapor_new))]      =0.
                epotential_new[np.where(np.isnan(epotential_new))]=0.
                qngraupel_new[np.where(np.isnan(qngraupel_new))]=0.
                qicesnow_new[np.where(np.isnan(qicesnow_new))]  =0.
                qghail_new[np.where(np.isnan(qghail_new))]      =0.
                #*********************************************************************************
                #composites by height:
                u_mean_z[:,:,:,self.ind_zlev[i]] = u_mean_z[:,:,:,self.ind_zlev[i]] + u_new*self.mf_weights_z[i]*factor
                v_mean_z[:,:,:,self.ind_zlev[i]] = v_mean_z[:,:,:,self.ind_zlev[i]] + v_new*self.mf_weights_z[i]*factor
                w_mean_z[:,:,:,self.ind_zlev[i]] = w_mean_z[:,:,:,self.ind_zlev[i]] + w_new*self.mf_weights_z[i]*factor
                u_dev_mean_z[:,:,:,self.ind_zlev[i]] = u_dev_mean_z[:,:,:,self.ind_zlev[i]] + u_dev*self.mf_weights_z[i]*factor 
                v_dev_mean_z[:,:,:,self.ind_zlev[i]] = v_dev_mean_z[:,:,:,self.ind_zlev[i]] + v_dev*self.mf_weights_z[i]*factor
                w_dev_mean_z[:,:,:,self.ind_zlev[i]] = w_dev_mean_z[:,:,:,self.ind_zlev[i]] + w_dev*self.mf_weights_z[i]*factor
                #buoy_counter[:,:,:,self.ind_zlev[i]] += -np.isnan(buoy_map_flat[i])*1.
                buoy_mean_z[:,:,:,self.ind_zlev[i]] = buoy_mean_z[:,:,:,self.ind_zlev[i]] + buoy_map_flat[i]*self.mf_weights_z[i]*factor
                pdev_mean_z[:,:,:,self.ind_zlev[i]] += pdev_new*self.mf_weights_z[i]*factor
                sctot_mean_z[:,:,:,self.ind_zlev[i]] += sctot_new*self.mf_weights_z[i]*factor
                latheat_mean_z[:,:,:,self.ind_zlev[i]] += latheat_new*self.mf_weights_z[i]*factor
                #if z_centre_flat[i]>=min_z_qnice:  #only take into account levels where ice may form  (???) testing DHD 03.09.2020
                qnice_mean_z[:,:,:,self.ind_zlev[i]]   += qnice_new*self.mf_weights_z[i]*factor
                qice_mean_z[:,:,:,self.ind_zlev[i]]   += qice_new*self.mf_weights_z[i]*factor # (use same weights for qice as for qnice)
                #ice_counter=1
                qncloud_mean_z[:,:,:,self.ind_zlev[i]]    += qncloud_new*self.mf_weights_z[i]*factor
                qnrain_mean_z[:,:,:,self.ind_zlev[i]]     += qnrain_new*self.mf_weights_z[i]*factor
                noninduc_mean_z[:,:,:,self.ind_zlev[i]]   += noninduc_new*self.mf_weights_z[i]*factor
                cldnuc_mean_z[:,:,:,self.ind_zlev[i]]     += cldnuc_new*self.mf_weights_z[i]*factor
                rh_mean_z[:,:,:,self.ind_zlev[i]]         += rh_new*self.mf_weights_z[i]*factor
                qcloud_mean_z[:,:,:,self.ind_zlev[i]]     += qcloud_new*self.mf_weights_z[i]*factor
                qrain_mean_z[:,:,:,self.ind_zlev[i]]      += qrain_new*self.mf_weights_z[i]*factor
                qvapor_mean_z[:,:,:,self.ind_zlev[i]]     += qvapor_new*self.mf_weights_z[i]*factor
                epotential_mean_z[:,:,:,self.ind_zlev[i]] += epotential_new*self.mf_weights_z[i]*factor
                qngraupel_mean_z[:,:,:,self.ind_zlev[i]]  += qngraupel_new*self.mf_weights_z[i]*factor
                qicesnow_mean_z[:,:,:,self.ind_zlev[i]]   += qicesnow_new*self.mf_weights_z[i]*factor
                qghail_mean_z[:,:,:,self.ind_zlev[i]]     += qghail_new*self.mf_weights_z[i]*factor
                
                #*********************************************************************************

                u_mean[:,:,:,ind_flat[i]] = u_mean[:,:,:,ind_flat[i]] + u_new*self.mf_weights_stages[i]*factor  # multiply by factor to avoid errors due to small values (divide by it at the end!)
                v_mean[:,:,:,ind_flat[i]] = v_mean[:,:,:,ind_flat[i]] + v_new*self.mf_weights_stages[i]*factor
                w_mean[:,:,:,ind_flat[i]] = w_mean[:,:,:,ind_flat[i]] + w_new*self.mf_weights_stages[i]*factor
                u_dev_mean[:,:,:,ind_flat[i]] = u_dev_mean[:,:,:,ind_flat[i]] + u_dev*self.mf_weights_stages[i]*factor 
                v_dev_mean[:,:,:,ind_flat[i]] = v_dev_mean[:,:,:,ind_flat[i]] + v_dev*self.mf_weights_stages[i]*factor
                w_dev_mean[:,:,:,ind_flat[i]] = w_dev_mean[:,:,:,ind_flat[i]] + w_dev*self.mf_weights_stages[i]*factor
                #buoy_counter[:,:,:,ind_flat[i]] += -np.isnan(buoy_map_flat[i])*1.
                buoy_mean[:,:,:,ind_flat[i]] = buoy_mean[:,:,:,ind_flat[i]] + buoy_map_flat[i]*self.mf_weights_stages[i]*factor
                pdev_mean[:,:,:,ind_flat[i]] += pdev_new*self.mf_weights_stages[i]*factor
                sctot_mean[:,:,:,ind_flat[i]] += sctot_new*self.mf_weights_stages[i]*factor
                latheat_mean[:,:,:,ind_flat[i]] += latheat_new*self.mf_weights_stages[i]*factor
                if z_centre_flat[i]>=min_z_qnice:  #only take into account levels where ice may form  (???) testing DHD 03.09.2020
                    qnice_mean[:,:,:,ind_flat[i]]   += qnice_new*self.mf_weights_stages_qnice[i]*factor
                    qice_mean[:,:,:,ind_flat[i]]   += qice_new*self.mf_weights_stages_qnice[i]*factor # (use same weights for qice as for qnice)
                    #ice_counter+=1
                qncloud_mean[:,:,:,ind_flat[i]]    += qncloud_new*self.mf_weights_stages[i]*factor
                qnrain_mean[:,:,:,ind_flat[i]]     += qnrain_new*self.mf_weights_stages[i]*factor
                noninduc_mean[:,:,:,ind_flat[i]]   += noninduc_new*self.mf_weights_stages[i]*factor
                cldnuc_mean[:,:,:,ind_flat[i]]     += cldnuc_new*self.mf_weights_stages[i]*factor
                rh_mean[:,:,:,ind_flat[i]]         += rh_new*self.mf_weights_stages[i]*factor
                qcloud_mean[:,:,:,ind_flat[i]]     += qcloud_new*self.mf_weights_stages[i]*factor
                qrain_mean[:,:,:,ind_flat[i]]      += qrain_new*self.mf_weights_stages[i]*factor
                qvapor_mean[:,:,:,ind_flat[i]]     += qvapor_new*self.mf_weights_stages[i]*factor
                epotential_mean[:,:,:,ind_flat[i]] += epotential_new*self.mf_weights_stages[i]*factor
                qngraupel_mean[:,:,:,ind_flat[i]]  += qngraupel_new*self.mf_weights_stages[i]*factor
                qicesnow_mean[:,:,:,ind_flat[i]]   += qicesnow_new*self.mf_weights_stages[i]*factor
                qghail_mean[:,:,:,ind_flat[i]]     += qghail_new*self.mf_weights_stages[i]*factor
                #*******************************************************************************************************************
                i_end=len(self.casename[i][0])-1
                while self.casename[i][0][i_end]!='z' and i_end>0:
                    i_end+=-1
                if int(self.casename[i][0][i_end+1:])<=7000:
                    u_mean_7km[:,:,:,ind_flat[i]] = u_mean_7km[:,:,:,ind_flat[i]] + u_new*self.mf_weights_stages[i]*factor  # multiply by factor to avoid errors due to small values (divide by it at the end!)
                    v_mean_7km[:,:,:,ind_flat[i]]           = v_mean_7km[:,:,:,ind_flat[i]] + v_new*self.mf_weights_stages_7km[i]*factor
                    w_mean_7km[:,:,:,ind_flat[i]]           = w_mean_7km[:,:,:,ind_flat[i]] + w_new*self.mf_weights_stages_7km[i]*factor
                    u_dev_mean_7km[:,:,:,ind_flat[i]]       = u_dev_mean_7km[:,:,:,ind_flat[i]] + u_dev*self.mf_weights_stages_7km[i]*factor 
                    v_dev_mean_7km[:,:,:,ind_flat[i]]       = v_dev_mean_7km[:,:,:,ind_flat[i]] + v_dev*self.mf_weights_stages_7km[i]*factor
                    w_dev_mean_7km[:,:,:,ind_flat[i]]       = w_dev_mean_7km[:,:,:,ind_flat[i]] + w_dev*self.mf_weights_stages_7km[i]*factor
                    buoy_mean_7km[:,:,:,ind_flat[i]]        = buoy_mean_7km[:,:,:,ind_flat[i]] + buoy_map_flat[i]*self.mf_weights_stages_7km[i]*factor
                    pdev_mean_7km[:,:,:,ind_flat[i]]       += pdev_new*self.mf_weights_stages_7km[i]*factor
                    sctot_mean_7km[:,:,:,ind_flat[i]]      += sctot_new*self.mf_weights_stages_7km[i]*factor
                    latheat_mean_7km[:,:,:,ind_flat[i]]    += latheat_new*self.mf_weights_stages_7km[i]*factor
                    #if z_centre_flat[i]>=min_z_qnice:  #only take into account levels where ice may form  (???) testing DHD 03.09.2020
                    qnice_mean_7km[:,:,:,ind_flat[i]]      += qnice_new*self.mf_weights_stages_7km[i]*factor
                    qice_mean_7km[:,:,:,ind_flat[i]]       += qice_new*self.mf_weights_stages_7km[i]*factor # (use same weights for qice as for qnice)
                    qncloud_mean_7km[:,:,:,ind_flat[i]]    += qncloud_new*self.mf_weights_stages_7km[i]*factor
                    qnrain_mean_7km[:,:,:,ind_flat[i]]     += qnrain_new*self.mf_weights_stages_7km[i]*factor
                    noninduc_mean_7km[:,:,:,ind_flat[i]]   += noninduc_new*self.mf_weights_stages_7km[i]*factor
                    cldnuc_mean_7km[:,:,:,ind_flat[i]]     += cldnuc_new*self.mf_weights_stages_7km[i]*factor
                    rh_mean_7km[:,:,:,ind_flat[i]]         += rh_new*self.mf_weights_stages_7km[i]*factor
                    qcloud_mean_7km[:,:,:,ind_flat[i]]     += qcloud_new*self.mf_weights_stages_7km[i]*factor
                    qrain_mean_7km[:,:,:,ind_flat[i]]      += qrain_new*self.mf_weights_stages_7km[i]*factor
                    qvapor_mean_7km[:,:,:,ind_flat[i]]     += qvapor_new*self.mf_weights_stages_7km[i]*factor
                    epotential_mean_7km[:,:,:,ind_flat[i]] += epotential_new*self.mf_weights_stages_7km[i]*factor
                    qngraupel_mean_7km[:,:,:,ind_flat[i]]  += qngraupel_new*self.mf_weights_stages_7km[i]*factor
                    qicesnow_mean_7km[:,:,:,ind_flat[i]]   += qicesnow_new*self.mf_weights_stages_7km[i]*factor
                    qghail_mean_7km[:,:,:,ind_flat[i]]     += qghail_new*self.mf_weights_stages_7km[i]*factor

                R_count[ind_flat[i]] += 1
                R_mean[ind_flat[i]] += R_flat[i]*self.mf_weights_stages[i]*factor
                R_grossmean += R_flat[i]*self.mf_weights_grmean_flat[i]*factor
                D_mean[ind_flat[i]] += D_flat[i]*self.mf_weights_stages[i]*factor
                entrainment_mean[:,ind_flat[i]] = entrainment_mean[:,ind_flat[i]] + entrainment_map_flat[i]*self.mf_weights_stages[i]*factor
                detrainment_mean[:,ind_flat[i]] = detrainment_mean[:,ind_flat[i]] + detrainment_map_flat[i]*self.mf_weights_stages[i]*factor
                entrainment_grossmean = entrainment_grossmean + entrainment_map_flat[i]*self.mf_weights_grmean_flat[i]*factor 
                detrainment_grossmean = detrainment_grossmean + detrainment_map_flat[i]*self.mf_weights_grmean_flat[i]*factor
                #gross_counter+=1.
                u_dev_grossmean = u_dev_grossmean + u_dev*self.mf_weights_grmean_flat[i]*factor 
                v_dev_grossmean = v_dev_grossmean + v_dev*self.mf_weights_grmean_flat[i]*factor
                w_dev_grossmean = w_dev_grossmean + w_dev*self.mf_weights_grmean_flat[i]*factor
                sctot_grossmean   = sctot_grossmean + sctot_new*self.mf_weights_grmean_flat[i]*factor
                latheat_grossmean   = latheat_grossmean + latheat_new*self.mf_weights_grmean_flat[i]*factor
                if z_centre_flat[i]>=min_z_qnice:  #only take into account levels where ice may form  (???) testing DHD 03.09.2020
                    qnice_grossmean     = qnice_grossmean + qnice_new*self.mf_weights_grmean_qnice_flat[i]*factor
                    qice_grossmean      = qice_grossmean + qice_new*self.mf_weights_grmean_qnice_flat[i]*factor #(same weights for qnice and qice)
                qncloud_grossmean   = qncloud_grossmean + qncloud_new*self.mf_weights_grmean_flat[i]*factor
                qnrain_grossmean    = qnrain_grossmean + qnrain_new*self.mf_weights_grmean_flat[i]*factor
                noninduc_grossmean    = noninduc_grossmean + noninduc_new*self.mf_weights_grmean_flat[i]*factor
                cldnuc_grossmean    = cldnuc_grossmean + cldnuc_new*self.mf_weights_grmean_flat[i]*factor
                rh_grossmean        = rh_grossmean + rh_new*self.mf_weights_grmean_flat[i]*factor
                qcloud_grossmean    = qcloud_grossmean + qcloud_new*self.mf_weights_grmean_flat[i]*factor
                qrain_grossmean     = qrain_grossmean + qrain_new*self.mf_weights_grmean_flat[i]*factor
                qvapor_grossmean    = qvapor_grossmean + qvapor_new*self.mf_weights_grmean_flat[i]*factor
                epotential_grossmean= epotential_grossmean + epotential_new*self.mf_weights_grmean_flat[i]*factor
                qngraupel_grossmean = qngraupel_grossmean + qngraupel_new*self.mf_weights_grmean_flat[i]*factor
                qicesnow_grossmean  = qicesnow_grossmean + qicesnow_new*self.mf_weights_grmean_flat[i]*factor
                qghail_grossmean    = qghail_grossmean + qghail_new*self.mf_weights_grmean_flat[i]*factor
                #vel_gross_counter += -np.isnan(u_new)*1.
                include[i] = 1
                if i in last_indices: # composite of last tracked timestep of each thermal
                    u_dev_lastmean = u_dev_lastmean + u_dev*self.mf_weights_perthermal[i]*factor 
                    v_dev_lastmean = v_dev_lastmean + v_dev*self.mf_weights_perthermal[i]*factor
                    w_dev_lastmean = w_dev_lastmean + w_dev*self.mf_weights_perthermal[i]*factor
                    sctot_lastmean = sctot_lastmean + sctot_new*self.mf_weights_perthermal[i]*factor
                    latheat_lastmean = latheat_lastmean + latheat_new*self.mf_weights_perthermal[i]*factor
                    epotential_lastmean = epotential_lastmean + epotential_new*self.mf_weights_perthermal[i]*factor
                    qngraupel_lastmean    = qngraupel_lastmean + qngraupel_new*self.mf_weights_perthermal[i]*factor
                    if z_centre_flat[i]>=min_z_qnice:  #only take into account levels where ice may form
                        qnice_lastmean     = qnice_lastmean + qnice_new*self.mf_weights_perthermal_qnice[i]*factor
                        qice_lastmean      = qice_lastmean + qice_new*self.mf_weights_perthermal_qnice[i]*factor #(same weights for qnice and qice)
                    qncloud_lastmean   = qncloud_lastmean + qncloud_new*self.mf_weights_perthermal[i]*factor
                    qnrain_lastmean    = qnrain_lastmean + qnrain_new*self.mf_weights_perthermal[i]*factor
                    noninduc_lastmean    = noninduc_lastmean + noninduc_new*self.mf_weights_perthermal[i]*factor
                    cldnuc_lastmean    = cldnuc_lastmean + cldnuc_new*self.mf_weights_perthermal[i]*factor
                    rh_lastmean        = rh_lastmean + rh_new*self.mf_weights_perthermal[i]*factor
                    qcloud_lastmean    = qcloud_lastmean + qcloud_new*self.mf_weights_perthermal[i]*factor
                    qrain_lastmean     = qrain_lastmean + qrain_new*self.mf_weights_perthermal[i]*factor
                    qvapor_lastmean    = qvapor_lastmean + qvapor_new*self.mf_weights_perthermal[i]*factor
                    qicesnow_lastmean  = qicesnow_lastmean + qicesnow_new*self.mf_weights_perthermal[i]*factor
                    qghail_lastmean    = qghail_lastmean + qghail_new*self.mf_weights_perthermal[i]*factor

            else:
                del iz_low[-1]
                del iz_up[-1]
                del ix_left[-1]
                del ix_right[-1]
                del iy_left[-1]
                del iy_right[-1]
                gc.collect()
            n+=1
            print( 'processed %d / %d time steps'%(n, len(i_ordered)),end='\r')
        #print( '\n ice counter = %d\n'%(ice_counter))
        self.iz_low     = np.asarray(iz_low)
        self.iz_up      = np.asarray(iz_up)
        self.ix_right   = np.asarray(ix_right)
        self.ix_left    = np.asarray(ix_left)
        self.iy_right   = np.asarray(iy_right)
        self.iy_left    = np.asarray(iy_left)
        self.entrainment_grossmean = entrainment_grossmean/factor#/gross_counter
        self.detrainment_grossmean = detrainment_grossmean/factor#/gross_counter
        R_grossmean = R_grossmean/factor#/gross_counter
        dx = self.delta_R*R_grossmean 
        self.u_dev_grossmean = u_dev_grossmean/factor#/gross_counter
        self.v_dev_grossmean = v_dev_grossmean/factor#/gross_counter
        self.w_dev_grossmean = w_dev_grossmean/factor#/gross_counter
        self.sctot_grossmean  = sctot_grossmean/factor#/gross_counter
        self.latheat_grossmean  = latheat_grossmean/factor#/gross_counter
        self.qnice_grossmean    = qnice_grossmean/factor#/gross_counter
        self.qncloud_grossmean  = qncloud_grossmean/factor#/gross_counter
        self.qnrain_grossmean   = qnrain_grossmean/factor#/gross_counter
        self.qice_grossmean     = qice_grossmean/factor#/gross_counter
        self.qcloud_grossmean   = qcloud_grossmean/factor#/gross_counter
        self.qrain_grossmean    = qrain_grossmean/factor#/gross_counter
        self.qvapor_grossmean   = qvapor_grossmean/factor#/gross_counter
        self.noninduc_grossmean = noninduc_grossmean/factor#/gross_counter
        self.cldnuc_grossmean   = cldnuc_grossmean/factor#/gross_counter
        self.rh_grossmean       = rh_grossmean/factor#/gross_counter
        self.epotential_grossmean= epotential_grossmean/factor#/gross_counter
        self.qngraupel_grossmean= qngraupel_grossmean/factor#/gross_counter
        self.qicesnow_grossmean = qicesnow_grossmean/factor#/gross_counter
        self.qghail_grossmean   = qghail_grossmean/factor#/gross_counter
        self.u_dev_lastmean     = u_dev_lastmean/factor
        self.v_dev_lastmean     = v_dev_lastmean/factor
        self.w_dev_lastmean     = w_dev_lastmean/factor
        self.sctot_lastmean     = sctot_lastmean/factor
        self.latheat_lastmean   = latheat_lastmean/factor
        self.qnice_lastmean     = qnice_lastmean/factor
        self.qncloud_lastmean   = qncloud_lastmean/factor
        self.qnrain_lastmean    = qnrain_lastmean/factor
        self.qice_lastmean      = qice_lastmean/factor
        self.qcloud_lastmean    = qcloud_lastmean/factor
        self.qrain_lastmean     = qrain_lastmean/factor
        self.qvapor_lastmean    = qvapor_lastmean/factor
        self.cldnuc_lastmean    = cldnuc_lastmean/factor
        self.noninduc_lastmean  = noninduc_lastmean/factor
        self.rh_lastmean        = rh_lastmean/factor
        self.epotential_lastmean= epotential_lastmean/factor
        self.qngraupel_lastmean = qngraupel_lastmean/factor
        self.qicesnow_lastmean  = qicesnow_lastmean/factor
        self.qghail_lastmean    = qghail_lastmean/factor
    
        for i in range(len(R_count)):
            if R_count[i]==0:
                entrainment_mean[:,i] = np.nan
                detrainment_mean[:,i] = np.nan
            else:
                entrainment_mean[:,i] = entrainment_mean[:,i]/factor#R_count[i]
                detrainment_mean[:,i] = detrainment_mean[:,i]/factor#R_count[i]
        
        self.entrainment_mean = entrainment_mean
        self.detrainment_mean = detrainment_mean

        self.R_mean = R_mean/factor#/R_count
        self.D_mean = D_mean/factor#/R_count
        self.R_count = R_count
        self.tmax_ref = tmax_ref
        # for composites by height:
        self.u_mean_z = u_mean_z/factor#/u_counter # divide by factor because we multiplied it earlier by it to avoid small values!
        self.v_mean_z = v_mean_z/factor#/v_counter
        self.w_mean_z = w_mean_z/factor#/w_counter
        self.u_dev_mean_z = u_dev_mean_z/factor#u_counter
        self.v_dev_mean_z = v_dev_mean_z/factor#v_counter
        self.w_dev_mean_z = w_dev_mean_z/factor#/w_counter
        self.buoy_mean_z = buoy_mean_z/factor#/buoy_counter
        self.pdev_mean_z = pdev_mean_z/factor#/pdev_counter
        self.sctot_mean_z   = sctot_mean_z/factor#/pdev_counter
        self.latheat_mean_z   = latheat_mean_z/factor#/pdev_counter
        self.qnice_mean_z     = qnice_mean_z/factor#/pdev_counter
        self.qncloud_mean_z   = qncloud_mean_z/factor#/pdev_counter
        self.qnrain_mean_z    = qnrain_mean_z/factor#/pdev_counter
        self.qice_mean_z      = qice_mean_z/factor#/pdev_counter
        self.qcloud_mean_z    = qcloud_mean_z/factor#/pdev_counter
        self.qrain_mean_z     = qrain_mean_z/factor#/pdev_counter
        self.qvapor_mean_z    = qvapor_mean_z/factor#/pdev_counter
        self.noninduc_mean_z  = noninduc_mean_z/factor#/pdev_counter
        self.cldnuc_mean_z    = cldnuc_mean_z/factor#/pdev_counter
        self.rh_mean_z        = rh_mean_z/factor#/pdev_counter
        self.epotential_mean_z= epotential_mean_z/factor#/pdev_counter
        self.qngraupel_mean_z = qngraupel_mean_z/factor#/pdev_counter
        self.qicesnow_mean_z  = qicesnow_mean_z/factor#/pdev_counter
        self.qghail_mean_z    = qghail_mean_z/factor#/pdev_counter



        self.u_mean = u_mean/factor#/u_counter # divide by factor because we multiplied it earlier by it to avoid small values!
        self.v_mean = v_mean/factor#/v_counter
        self.w_mean = w_mean/factor#/w_counter
        self.u_dev_mean = u_dev_mean/factor#u_counter
        self.v_dev_mean = v_dev_mean/factor#v_counter
        self.w_dev_mean = w_dev_mean/factor#/w_counter
        self.buoy_mean = buoy_mean/factor#/buoy_counter
        self.pdev_mean = pdev_mean/factor#/pdev_counter
        self.sctot_mean   = sctot_mean/factor#/pdev_counter
        self.latheat_mean   = latheat_mean/factor#/pdev_counter
        self.qnice_mean     = qnice_mean/factor#/pdev_counter
        self.qncloud_mean   = qncloud_mean/factor#/pdev_counter
        self.qnrain_mean    = qnrain_mean/factor#/pdev_counter
        self.qice_mean      = qice_mean/factor#/pdev_counter
        self.qcloud_mean    = qcloud_mean/factor#/pdev_counter
        self.qrain_mean     = qrain_mean/factor#/pdev_counter
        self.qvapor_mean    = qvapor_mean/factor#/pdev_counter
        self.noninduc_mean  = noninduc_mean/factor#/pdev_counter
        self.cldnuc_mean    = cldnuc_mean/factor#/pdev_counter
        self.rh_mean        = rh_mean/factor#/pdev_counter
        self.epotential_mean= epotential_mean/factor#/pdev_counter
        self.qngraupel_mean = qngraupel_mean/factor#/pdev_counter
        self.qicesnow_mean  = qicesnow_mean/factor#/pdev_counter
        self.qghail_mean    = qghail_mean/factor#/pdev_counter

        self.u_mean_7km         = u_mean_7km/factor#/u_counter # divide by factor because we multiplied it earlier by it to avoid small values!
        self.v_mean_7km         = v_mean_7km/factor#/v_counter
        self.w_mean_7km         = w_mean_7km/factor#/w_counter
        self.u_dev_mean_7km     = u_dev_mean_7km/factor#u_counter
        self.v_dev_mean_7km     = v_dev_mean_7km/factor#v_counter
        self.w_dev_mean_7km     = w_dev_mean_7km/factor#/w_counter
        self.buoy_mean_7km      = buoy_mean_7km/factor#/buoy_counter
        self.pdev_mean_7km      = pdev_mean_7km/factor#/pdev_counter
        self.sctot_mean_7km     = sctot_mean_7km/factor#/pdev_counter
        self.latheat_mean_7km   = latheat_mean_7km/factor#/pdev_counter
        self.qnice_mean_7km     = qnice_mean_7km/factor#/pdev_counter
        self.qncloud_mean_7km   = qncloud_mean_7km/factor#/pdev_counter
        self.qnrain_mean_7km    = qnrain_mean_7km/factor#/pdev_counter
        self.qice_mean_7km      = qice_mean_7km/factor#/pdev_counter
        self.qcloud_mean_7km    = qcloud_mean_7km/factor#/pdev_counter
        self.qrain_mean_7km     = qrain_mean_7km/factor#/pdev_counter
        self.qvapor_mean_7km    = qvapor_mean_7km/factor#/pdev_counter
        self.noninduc_mean_7km  = noninduc_mean_7km/factor#/pdev_counter
        self.cldnuc_mean_7km    = cldnuc_mean_7km/factor#/pdev_counter
        self.rh_mean_7km        = rh_mean_7km/factor#/pdev_counter
        self.epotential_mean_7km= epotential_mean_7km/factor#/pdev_counter
        self.qngraupel_mean_7km = qngraupel_mean_7km/factor#/pdev_counter
        self.qicesnow_mean_7km  = qicesnow_mean_7km/factor
        self.qghail_mean_7km    = qghail_mean_7km/factor
        
        include = np.where(include)

        self.z_centre   = z_centre_flat[include]
        self.x_centre   = x_centre_flat[include]
        self.y_centre   = y_centre_flat[include]
        self.Fres       = aux.flatten_array( self.Fres )[include]
        self.buoy       = aux.flatten_array( self.buoy )[include]
        self.Fnh        = aux.flatten_array( self.Fnh )[include]
        self.acc        = aux.flatten_array( self.acc )[include]
        self.D          = aux.flatten_array( self.D )[include]
        self.Pnz        = aux.flatten_array( self.Pnz  )[include]
        self.time_or    = np.copy(self.time)
        self.time       = aux.flatten_array( self.time )[include]
        self.W          = aux.flatten_array( self.W    )[include]
        self.wmax       = aux.flatten_array( self.wmax )[include]
        self.R_or       = np.copy(self.R    )
        self.R          = aux.flatten_array( self.R    )[include]
        self.fract_entr = aux.flatten_array( self.fract_entr )[include]
        self.mse_thermal_or = np.copy(self.mse_thermal)
        self.mse_thermal= aux.flatten_array( self.mse_thermal )[include]
        self.mse_env_or = np.copy(self.mse_env)
        self.mse_env    = aux.flatten_array( self.mse_env )[include]
        #self.mse_diff_init= aux.flatten_array( self.mse_diff_init )[include]
        self.mixing_mse = aux.flatten_array( self.mixing_mse )[include]
        self.mass       = aux.flatten_array( self.mass )[include]
        #self.mass_cond  = aux.flatten_array( self.mass_cond )[include]

        self.include    = include
        self._mask_centered_vars()
        if not os.path.isdir( self.folder ):
            #os.system( 'rm -r ' + self.folder )
            os.mkdir( self.folder )
        self._save_data()

    def _save_data(self):
        np.save( self.folder + '/uvwp_grid.npy', np.vstack( (self.x_new,self.y_new,self.z_new) ) )
        np.save( self.folder+'/dBdZ.npy',self.dBdZ)
        np.save( self.folder+'/dBdZmax.npy',self.dBdZmax)
        np.save( self.folder+'/dBdZmin.npy',self.dBdZmin)
        np.save( self.folder+'/dBdZinit.npy',self.dBdZinit)
        np.save( self.folder+'/R_count.npy', self.R_count )
        np.save( self.folder+'/tmax_ref.npy', self.tmax_ref )
        np.save( self.folder+'/min_N.npy', self.min_N )

        #*******************************************************************
        # variables used for cross sections of thermals by height
        np.save( self.folder+'/z_centers_z.npy',self.z_centers )
        np.save( self.folder+'/w_mean_z.npy', self.w_mean_z )
        np.save( self.folder+'/u_mean_z.npy', self.u_mean_z )
        np.save( self.folder+'/v_mean_z.npy', self.v_mean_z )
        np.save( self.folder+'/w_dev_mean_z.npy', self.w_dev_mean_z )
        np.save( self.folder+'/u_dev_mean_z.npy', self.u_dev_mean_z )
        np.save( self.folder+'/v_dev_mean_z.npy', self.v_dev_mean_z )
        np.save( self.folder+'/buoy_mean_z.npy', self.buoy_mean_z )
        np.save( self.folder+'/pdev_mean_z.npy', self.pdev_mean_z )
        np.save( self.folder+'/sctot_mean_z.npy', self.sctot_mean_z )
        np.save( self.folder+'/latheat_mean_z.npy', self.latheat_mean_z )
        np.save( self.folder+'/qnice_mean_z.npy', self.qnice_mean_z )
        np.save( self.folder+'/qncloud_mean_z.npy', self.qncloud_mean_z )
        np.save( self.folder+'/qnrain_mean_z.npy', self.qnrain_mean_z )
        np.save( self.folder+'/qice_mean_z.npy', self.qice_mean_z )
        np.save( self.folder+'/qcloud_mean_z.npy', self.qcloud_mean_z )
        np.save( self.folder+'/qrain_mean_z.npy', self.qrain_mean_z )
        np.save( self.folder+'/qvapor_mean_z.npy', self.qvapor_mean_z )
        np.save( self.folder+'/noninduc_mean_z.npy', self.noninduc_mean_z )
        np.save( self.folder+'/cldnuc_mean_z.npy', self.cldnuc_mean_z )
        np.save( self.folder+'/rh_mean_z.npy', self.rh_mean_z )
        np.save( self.folder+'/epotential_mean_z.npy', self.epotential_mean_z )
        np.save( self.folder+'/qngraupel_mean_z.npy', self.qngraupel_mean_z )
        np.save( self.folder+'/qicesnow_mean_z.npy', self.qicesnow_mean_z )
        np.save( self.folder+'/qghail_mean_z.npy', self.qghail_mean_z )
        #*******************************************************************
        # variables used for cross sections of thermals throughout lifetime stages
        np.save( self.folder+'/w_mean.npy', self.w_mean )
        np.save( self.folder+'/u_mean.npy', self.u_mean )
        np.save( self.folder+'/v_mean.npy', self.v_mean )
        np.save( self.folder+'/w_dev_mean.npy', self.w_dev_mean )
        np.save( self.folder+'/u_dev_mean.npy', self.u_dev_mean )
        np.save( self.folder+'/v_dev_mean.npy', self.v_dev_mean )
        np.save( self.folder+'/buoy_mean.npy', self.buoy_mean )
        np.save( self.folder+'/pdev_mean.npy', self.pdev_mean )
        np.save( self.folder+'/sctot_mean.npy', self.sctot_mean )
        np.save( self.folder+'/latheat_mean.npy', self.latheat_mean )
        np.save( self.folder+'/qnice_mean.npy', self.qnice_mean )
        np.save( self.folder+'/qncloud_mean.npy', self.qncloud_mean )
        np.save( self.folder+'/qnrain_mean.npy', self.qnrain_mean )
        np.save( self.folder+'/qice_mean.npy', self.qice_mean )
        np.save( self.folder+'/qcloud_mean.npy', self.qcloud_mean )
        np.save( self.folder+'/qrain_mean.npy', self.qrain_mean )
        np.save( self.folder+'/qvapor_mean.npy', self.qvapor_mean )
        np.save( self.folder+'/noninduc_mean.npy', self.noninduc_mean )
        np.save( self.folder+'/cldnuc_mean.npy', self.cldnuc_mean )
        np.save( self.folder+'/rh_mean.npy', self.rh_mean )
        np.save( self.folder+'/epotential_mean.npy', self.epotential_mean )
        np.save( self.folder+'/qngraupel_mean.npy', self.qngraupel_mean )
        np.save( self.folder+'/qicesnow_mean.npy', self.qicesnow_mean )
        np.save( self.folder+'/qghail_mean.npy', self.qghail_mean )
        #*******************************************************************
        # same as above, but for thermals that initiate below 7km
        np.save( self.folder+'/w_mean_7km.npy', self.w_mean_7km )
        np.save( self.folder+'/u_mean_7km.npy', self.u_mean_7km )
        np.save( self.folder+'/v_mean_7km.npy', self.v_mean_7km )
        np.save( self.folder+'/w_dev_mean_7km.npy', self.w_dev_mean_7km )
        np.save( self.folder+'/u_dev_mean_7km.npy', self.u_dev_mean_7km )
        np.save( self.folder+'/v_dev_mean_7km.npy', self.v_dev_mean_7km )
        np.save( self.folder+'/buoy_mean_7km.npy', self.buoy_mean_7km )
        np.save( self.folder+'/pdev_mean_7km.npy', self.pdev_mean_7km )
        np.save( self.folder+'/sctot_mean_7km.npy', self.sctot_mean_7km )
        np.save( self.folder+'/latheat_mean_7km.npy', self.latheat_mean_7km )
        np.save( self.folder+'/qnice_mean_7km.npy', self.qnice_mean_7km )
        np.save( self.folder+'/qncloud_mean_7km.npy', self.qncloud_mean_7km )
        np.save( self.folder+'/qnrain_mean_7km.npy', self.qnrain_mean_7km )
        np.save( self.folder+'/qice_mean_7km.npy', self.qice_mean_7km )
        np.save( self.folder+'/qcloud_mean_7km.npy', self.qcloud_mean_7km )
        np.save( self.folder+'/qrain_mean_7km.npy', self.qrain_mean_7km )
        np.save( self.folder+'/qvapor_mean_7km.npy', self.qvapor_mean_7km )
        np.save( self.folder+'/noninduc_mean_7km.npy', self.noninduc_mean_7km )
        np.save( self.folder+'/cldnuc_mean_7km.npy', self.cldnuc_mean_7km )
        np.save( self.folder+'/rh_mean_7km.npy', self.rh_mean_7km )
        np.save( self.folder+'/epotential_mean_7km.npy', self.epotential_mean_7km )
        np.save( self.folder+'/qngraupel_mean_7km.npy', self.qngraupel_mean_7km )
        np.save( self.folder+'/qicesnow_mean_7km.npy', self.qicesnow_mean_7km )
        np.save( self.folder+'/qghail_mean_7km.npy', self.qghail_mean_7km )
        #*******************************************************************
        np.save( self.folder+'/entr_mean.npy', self.entrainment_mean )
        np.save( self.folder+'/detr_mean.npy', self.detrainment_mean )
        np.save( self.folder+'/angles.npy', self.angles )
        np.save( self.folder+'/entrainment_grossmean.npy', self.entrainment_grossmean )
        np.save( self.folder+'/detrainment_grossmean.npy', self.detrainment_grossmean )
        np.save( self.folder+'/u_dev_grossmean.npy', self.u_dev_grossmean )
        np.save( self.folder+'/v_dev_grossmean.npy', self.v_dev_grossmean )
        np.save( self.folder+'/w_dev_grossmean.npy', self.w_dev_grossmean )
        np.save( self.folder+'/sctot_grossmean.npy', self.sctot_grossmean )
        np.save( self.folder+'/latheat_grossmean.npy', self.latheat_grossmean )
        np.save( self.folder+'/qnice_grossmean.npy', self.qnice_grossmean )
        np.save( self.folder+'/qncloud_grossmean.npy', self.qncloud_grossmean )
        np.save( self.folder+'/qnrain_grossmean.npy', self.qnrain_grossmean )
        np.save( self.folder+'/qice_grossmean.npy', self.qice_grossmean )
        np.save( self.folder+'/qcloud_grossmean.npy', self.qcloud_grossmean )
        np.save( self.folder+'/qrain_grossmean.npy', self.qrain_grossmean )
        np.save( self.folder+'/qvapor_grossmean.npy', self.qvapor_grossmean )
        np.save( self.folder+'/noninduc_grossmean.npy', self.noninduc_grossmean )
        np.save( self.folder+'/cldnuc_grossmean.npy', self.cldnuc_grossmean )
        np.save( self.folder+'/rh_grossmean.npy', self.rh_grossmean )
        np.save( self.folder+'/epotential_grossmean.npy', self.epotential_grossmean )
        np.save( self.folder+'/qngraupel_grossmean.npy', self.qngraupel_grossmean )
        np.save( self.folder+'/qicesnow_grossmean.npy', self.qicesnow_grossmean )
        np.save( self.folder+'/qghail_grossmean.npy', self.qghail_grossmean )

        np.save( self.folder+'/u_dev_lastmean.npy', self.u_dev_lastmean )
        np.save( self.folder+'/v_dev_lastmean.npy', self.v_dev_lastmean )
        np.save( self.folder+'/w_dev_lastmean.npy', self.w_dev_lastmean )
        np.save( self.folder+'/sctot_lastmean.npy', self.sctot_lastmean )
        np.save( self.folder+'/latheat_lastmean.npy', self.latheat_lastmean )
        np.save( self.folder+'/qnice_lastmean.npy', self.qnice_lastmean )
        np.save( self.folder+'/qncloud_lastmean.npy', self.qncloud_lastmean )
        np.save( self.folder+'/qnrain_lastmean.npy', self.qnrain_lastmean )
        np.save( self.folder+'/qice_lastmean.npy', self.qice_lastmean )
        np.save( self.folder+'/qcloud_lastmean.npy', self.qcloud_lastmean )
        np.save( self.folder+'/qrain_lastmean.npy', self.qrain_lastmean )
        np.save( self.folder+'/qvapor_lastmean.npy', self.qvapor_lastmean )
        np.save( self.folder+'/noninduc_lastmean.npy', self.noninduc_lastmean )
        np.save( self.folder+'/cldnuc_lastmean.npy', self.cldnuc_lastmean )
        np.save( self.folder+'/rh_lastmean.npy', self.rh_lastmean )
        np.save( self.folder+'/epotential_lastmean.npy', self.epotential_lastmean )
        np.save( self.folder+'/qngraupel_lastmean.npy', self.qngraupel_lastmean )
        np.save( self.folder+'/qicesnow_lastmean.npy', self.qicesnow_lastmean )
        np.save( self.folder+'/qghail_lastmean.npy', self.qghail_lastmean )
        np.save( self.folder+'/iz_low.npy', self.iz_low )
        np.save( self.folder+'/iz_up.npy', self.iz_up )
        np.save( self.folder+'/ix_right.npy', self.ix_right )
        np.save( self.folder+'/ix_left.npy', self.ix_left )
        np.save( self.folder+'/iy_right.npy', self.iy_right )
        np.save( self.folder+'/iy_left.npy', self.iy_left )

        np.save( self.folder+'/z_centre.npy', self.z_centre   ) 
        np.save( self.folder+'/x_centre.npy', self.x_centre   ) 
        np.save( self.folder+'/y_centre.npy', self.y_centre   ) 
        np.save( self.folder+'/Fres.npy', self.Fres       )
        np.save( self.folder+'/buoy.npy', self.buoy       )
        np.save( self.folder+'/Fnh.npy', self.Fnh        )
        np.save( self.folder+'/acc.npy', self.acc        )
        np.save( self.folder+'/D.npy', self.D          )
        np.save( self.folder+'/Pnz.npy', self.Pnz        )
        np.save( self.folder+'/time_or.npy', self.time_or    )
        np.save( self.folder+'/time.npy', self.time       )
        np.save( self.folder+'/W.npy', self.W          )
        np.save( self.folder+'/wmax.npy', self.wmax       )
        np.save( self.folder+'/R_or.npy', self.R_or       )
        np.save( self.folder+'/R.npy', self.R          )
        np.save( self.folder+'/fract_entr.npy', self.fract_entr )
        np.save( self.folder+'/mse_thermal_or.npy', self.mse_thermal_or) 
        np.save( self.folder+'/mse_thermal.npy', self.mse_thermal) 
        np.save( self.folder+'/mse_env.npy', self.mse_env    )
        np.save( self.folder+'/mse_env_or.npy', self.mse_env_or    )
        np.save( self.folder+'/mse_diff_init.npy', self.mse_diff_init)
        np.save( self.folder+'/mixing_mse.npy', self.mixing_mse )
        np.save( self.folder+'/mass.npy', self.mass       )
        #np.save( self.folder+'/mass_cond.npy', self.mass_cond )
        np.save( self.folder+'/wmax_c.npy', self.wmax_c     ) 
        np.save( self.folder+'/R_c.npy', self.R_c	 )   
        np.save( self.folder+'/Fres_c.npy', self.Fres_c     ) 
        np.save( self.folder+'/buoy_c.npy', self.buoy_c     ) 
        np.save( self.folder+'/Fnh_c.npy', self.Fnh_c      ) 
        np.save( self.folder+'/acc_c.npy', self.acc_c      ) 
        np.save( self.folder+'/D_c.npy', self.D_c        ) 
        np.save( self.folder+'/mass_c.npy', self.mass_c     ) 
        #np.save( self.folder+'/mass_cond_c.npy', self.mass_cond_c ) 
        np.save( self.folder+'/W_c.npy', self.W_c	 )   
        np.save( self.folder+'/Pnz_c.npy', self.Pnz_c      ) 
        np.save( self.folder+'/z_centre_c.npy', self.z_centre_c ) 
        np.save( self.folder+'/y_centre_c.npy', self.y_centre_c ) 
        np.save( self.folder+'/x_centre_c.npy', self.x_centre_c ) 

        np.save( self.folder+'/qcloud_c.npy', self.qcloud_c     ) 
        np.save( self.folder+'/qncloud_c.npy',self.qncloud_c    ) 
        np.save( self.folder+'/qrain_c.npy',  self.qrain_c      ) 
        np.save( self.folder+'/qnrain_c.npy', self.qnrain_c     ) 
        np.save( self.folder+'/cldnuc_c.npy', self.cldnuc_c     ) 
        np.save( self.folder+'/latheat_c.npy', self.latheat_c   ) 
        np.save( self.folder+'/noninduc_c.npy', self.noninduc_c     ) 
        np.save( self.folder+'/sctot_c.npy', self.sctot_c   ) 
        np.save( self.folder+'/epotential_c.npy', self.epotential_c   ) 
        np.save( self.folder+'/qngraupel_c.npy', self.qngraupel_c     ) 
        np.save( self.folder+'/qicesnow_c.npy',self.qicesnow_c    )
        np.save( self.folder+'/qghail_c.npy', self.qghail_c     )
 
        np.save( self.folder+'/it.npy', self.it         )        
        np.save( self.folder+'/loge_c.npy', self.loge_c     )
        np.save( self.folder+'/area.npy', self.area     )
        np.save( self.folder+'/simtime.npy', self.simtime )
        np.save( self.folder+'/t_range.npy', self.t_range )
        np.save( self.folder+'/tmin.npy', self.tmin )
        np.save( self.folder+'/tmax.npy', self.tmax )
        #np.save( self.folder+'/tracer_entr.npy', self.tracer_entr )
        #np.save( self.folder+'/tracer_detr.npy', self.tracer_detr )
        np.save( self.folder+'/delta_z.npy', self.delta_z )
        np.save( self.folder+'/z0.npy', self.z0 )
        np.save( self.folder+'/deltazR.npy', self.deltazR )
        np.save( self.folder+'/entr_rate.npy', self.entr_rate )
        np.save( self.folder+'/net_entr_tsteps.npy', self.net_entr_tsteps )
        np.save( self.folder+'/net_gross_entr.npy', self.net_gross_entr )
        np.save( self.folder+'/net_entr_term.npy', self.net_entr_term )
        np.save( self.folder+'/net_entr_term_c.npy', self.net_entr_term_c )
        np.save( self.folder+'/time_c.npy', self.time_c )
        np.save( self.folder+'/volume_c.npy', self.volume_c )

    def _mask_centered_vars(self):
        """
        mask the centered variables in order to use only the 'include' ones also for the mean composites
        """
        self.it_flat = aux.flatten_array(self.it)
        include_centered = np.zeros(len(self.it_flat))
        count = 0
        include_centered[0] = (self.it_flat[0]-self.tmin)
        l = len(self.t_range)
        for i in range(1,len(self.it_flat)):
            if self.it_flat[i]!=self.it_flat[i-1] + 1:
                count+=1
            include_centered[i] = l*count + (self.it_flat[i]-self.tmin)
        include_c = include_centered.astype(int)[self.include]
        mask = np.ones_like(self.R_c)
        mask[np.unravel_index(include_c,shape=self.wmax_c.shape)]=0
        self.wmax_c      [np.where(mask)] = np.nan 
        self.R_c         [np.where(mask)] = np.nan        
        self.Fres_c      [np.where(mask)] = np.nan
        self.buoy_c      [np.where(mask)] = np.nan
        self.Fnh_c       [np.where(mask)] = np.nan
        self.acc_c       [np.where(mask)] = np.nan
        self.D_c         [np.where(mask)] = np.nan
        self.mass_c      [np.where(mask)] = np.nan
        self.W_c         [np.where(mask)] = np.nan    
        self.Pnz_c       [np.where(mask)] = np.nan
        self.z_centre_c  [np.where(mask)] = np.nan
        self.y_centre_c  [np.where(mask)] = np.nan
        self.x_centre_c  [np.where(mask)] = np.nan
        self.loge_c      [np.where(mask)] = np.nan
        self.time_c      [np.where(mask)] = np.nan
        self.qcloud_c    [np.where(mask)] = np.nan
        self.qncloud_c   [np.where(mask)] = np.nan
        self.qrain_c     [np.where(mask)] = np.nan
        self.qnrain_c    [np.where(mask)] = np.nan
        self.cldnuc_c    [np.where(mask)] = np.nan
        self.latheat_c   [np.where(mask)] = np.nan
        self.noninduc_c  [np.where(mask)] = np.nan
        self.sctot_c     [np.where(mask)] = np.nan
        self.epotential_c[np.where(mask)] = np.nan
        self.noninduc_c  [np.where(mask)] = np.nan
        self.qicesnow_c  [np.where(mask)] = np.nan
        self.qghail_c    [np.where(mask)] = np.nan
        #self.net_entr_term_c[np.where(mask)] = np.nan


    def composite_fields( self ):
        min_N = self.min_N
        stages = self.u_mean.shape[-1]
        max_w= np.nanmax(self.w_mean)
        max_w_dev = np.nanmax(self.w_dev_mean)
        max_w_dev = 2.0
        max_w_dev = None # Alejandra I.

        min_sctot=0. #e+11
        max_sctot=4.  #e+12
        max_latheat=0.06
        max_qnice=2 #e5
        min_qnice=0.
        max_qncloud=1 #e8
        min_qncloud=0 #e8
        max_qnrain=1 #e5
        min_qnrain=0
        min_noninduc= -0.6 #e+9
        max_noninduc=  0.6 #e+9
        max_cldnuc= 1#e-8
        min_cldnuc=0
        max_rh=1.1
        min_rh=0.8
        max_epotential=  6
        min_epotential= -6
        max_qngraupel= 1
        min_qngraupel= 0
        max_qicesnow=1.
        min_qicesnow=0
        factor_qicesnow = 1.e3
        qicesnow_label="Qi+Qs(g kg$^{-1}$)"
        max_qghail=1.
        min_qghail=0
        factor_qghail = 1.e3
        qghail_label="Qg+Qh(g kg$^{-1}$)"

        self.entrainment_mean = self.entrainment_mean*self.N_angles         # this is to 'normalize' the contribution of each arc to the fractional entrainment distribution
        self.detrainment_mean = self.detrainment_mean*self.N_angles
        self.entrainment_grossmean = self.entrainment_grossmean*self.N_angles
        self.detrainment_grossmean = self.detrainment_grossmean*self.N_angles
        max_entr = 2.5e-3
        max_detr = max_entr
        min_pdev = -4.5
        max_pdev = 4.5
        mx=self.plt_range
        if self.rescale:
            circ_r = 1.
        else:
            circ_r = self.R_mean[istage]
        yl='Y (R)'
        xl='X (R)'
        fname = [self.folder + '/sctot_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.sctot_grossmean*1e+11, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_sctot, vmax=max_sctot, zero_contour=True, cblabel="SCTOT (x10$^{-11}$C m$^{-3}$)", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.sctot_grossmean*1e+11, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_sctot, vmax=max_sctot, zero_contour=True, cblabel="SCTOT (x10$^{-11}$C m$^{-3}$)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/latheat_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.latheat_grossmean, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmax=max_latheat, zero_contour=True, cblabel="LH (K s$^{-1}$)", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.latheat_grossmean, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmax=max_latheat, zero_contour=True, cblabel="LH (K s$^{-1}$)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/qnice_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qnice_grossmean*1e-5, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qnice, vmax=max_qnice, zero_contour=True, cblabel="QNICE (x10$^{5}$kg$^{-1}$)", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qnice_grossmean*1e-5, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qnice, vmax=max_qnice, zero_contour=True, cblabel="QNICE (x10$^5$kg$^{-1}$)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/qncloud_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qncloud_grossmean*1e-8, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qncloud, vmax=max_qncloud, zero_contour=True, cblabel="QNCLOUD (x10$^8$kg$^{-1}$)", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qncloud_grossmean*1e-8, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qncloud, vmax=max_qncloud, zero_contour=True, cblabel="QNCLOUD (x10$^8$kg$^{-1}$)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/qnrain_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qnrain_grossmean*1e-5, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qnrain, vmax=max_qnrain, zero_contour=True, cblabel="QNRAIN (x10$^5$kg$^{-1}$)", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qnrain_grossmean*1e-5, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qnrain, vmax=max_qnrain, zero_contour=True, cblabel="QNRAIN (x10$^5$kg$^{-1}$)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/noninduc_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.noninduc_grossmean*1e+9, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_noninduc, vmax=max_noninduc, zero_contour=True, cblabel="NONINDUC (nC m$^{-3}$)", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.noninduc_grossmean*1e+9, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_noninduc, vmax=max_noninduc, zero_contour=True, cblabel="NONINDUC (nC m$^{-3}$)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/cldnuc_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.cldnuc_grossmean*1e8, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_cldnuc, vmax=max_cldnuc, zero_contour=True, cblabel="CLDNUC (x10$^{-8}$kg kg$^{-1}$s$^{-1}$)", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.cldnuc_grossmean*1e8, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_cldnuc, vmax=max_cldnuc, zero_contour=True, cblabel="CLDNUC (x10$^{-8}$kg kg$^{-1}$s$^{-1}$)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/rh_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.rh_grossmean, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_rh, vmax=max_rh, zero_contour=False, cblabel="RH", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f',x_contour=1 )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.rh_grossmean, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_rh, vmax=max_rh, zero_contour=False, cblabel="RH", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name,x_contour=1 )
        fname = [self.folder + '/epotential_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.epotential_grossmean*1e-7, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_epotential, vmax=max_epotential, zero_contour=True, cblabel="EPOT (x10$^{+8}$V)", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.epotential_grossmean*1e-7, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_epotential, vmax=max_epotential, zero_contour=True, cblabel="EPOT (x10$^{+8}$V)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/qngraupel_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qngraupel_grossmean*1e-5, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qngraupel, vmax=max_qngraupel, zero_contour=True, cblabel="QNGRAUPEL (x10$^5$kg$^{-1}$)", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qngraupel_grossmean*1e-5, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qngraupel, vmax=max_qngraupel, zero_contour=True, cblabel="QNGRAUPEL (x10$^5$kg$^{-1}$)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/qicesnow_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qicesnow_grossmean*factor_qicesnow, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qicesnow, vmax=max_qicesnow, zero_contour=True, cblabel=qicesnow_label, title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qicesnow_grossmean*factor_qicesnow, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qicesnow, vmax=max_qicesnow, zero_contour=True, cblabel=qicesnow_label, ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/qghail_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qghail_grossmean*1e+3, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qghail, vmax=max_qghail, zero_contour=True, cblabel=qghail_label, title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f' )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qghail_grossmean*1e+3, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qghail, vmax=max_qghail, zero_contour=True, cblabel=qghail_label, ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title=self.exp_name )
        fname = [self.folder + '/w_dev_composite.png']
        aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.w_dev_grossmean, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmax=max_w_dev, zero_contour=True, cblabel="w' (m s$^{-1}$)", title=self.exp_name, ylabel=yl, xlabel=xl, ticks_fmt='%.1f', cmap = "jet_r" if self.up else "jet_r" ) #ALEJANDRA
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.w_dev_grossmean, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmax=max_w_dev, zero_contour=True, cblabel="w' (m s$^{-1}$)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title='' )#self.exp_name  )
        aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.w_dev_grossmean, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmax=max_w_dev, zero_contour=True, cblabel="w' (m s$^{-1}$)", ylabel=yl, xlabel=xl, ticks_fmt='%.1f', title='', axis='yz' )#self.exp_name  )
        fname = [self.folder + '/mean_entrainment_distribution.png',self.folder + '/mean_entrainment_distribution.png']
        aux.plot_mixing( self.angles, self.entrainment_grossmean, self.x_new, self.y_new, self.z_new, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, cblabel='$\epsilon$ (m$^{-1}$)', scale=1e3, vmax=max_entr, fname=fname, title='mean fractional entrainment', ylabel=yl, xlabel=xl )
        aux.plot_mixing_single( self.angles, self.entrainment_grossmean, self.x_new, self.y_new, self.z_new, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, cblabel='$\epsilon$ (m$^{-1}$)', scale=1e3, vmax=max_entr, fname=fname, title='mean fractional entrainment', ylabel=yl, xlabel=xl )
        fname = [self.folder + '/mean_detrainment_distribution.png', self.folder + '/mean_detrainment_distribution.png']
        aux.plot_mixing_single( self.angles, -self.detrainment_grossmean, self.x_new, self.y_new, self.z_new, self.u_dev_grossmean, self.v_dev_grossmean, self.w_dev_grossmean, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, cblabel='$\delta$ (m$^{-1}$)', scale=1e3, vmax=max_entr, fname=fname, title='mean fractional detrainment', ylabel=yl, xlabel=xl )
        for istage in range(stages):
            stage_number = istage-np.max(self.tmax_ref)
            print( 't=%2d: %.1f (out of approx. %.1f) thermals can be used. min. N of thermals is %d'%(stage_number, self.R_count[istage], min_N/self.min_ratio, min_N))
            if self.R_count[istage] >= min_N:
                fname = [self.folder + '/sctot_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/sctot_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.sctot_mean[:,:,:,istage]*1e+9, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=0., vmax=0.1, cblabel='SCTOT (nC m$^{-3}$)', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/sctot_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.sctot_mean[:,:,:,istage]*1e+9, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=0., vmax=0.1, cblabel='SCTOT (nC m$^{-3}$)', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )

                fname = [self.folder + '/latheat_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/latheat_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.latheat_mean[:,:,:,istage], self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=0., vmax=0.1, cblabel='LH (K s$^{-1}$)', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/latheat_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.latheat_mean[:,:,:,istage], self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=0., vmax=0.1, cblabel='LH (K s$^{-1}$)', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )

                fname = [self.folder + '/qnice_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/qnice_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qnice_mean[:,:,:,istage]*1e-5, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=0., vmax=max_qnice, cblabel='QNICE (x10$^5$kg$^{-1}$)', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/qnice_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qnice_mean[:,:,:,istage]*1e-5, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=0., vmax=max_qnice, cblabel='QNICE (x10$^5$kg$^{-1}$)', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )

                fname = [self.folder + '/qncloud_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/qncloud_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qncloud_mean[:,:,:,istage]*1e-8, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qncloud, vmax=max_qncloud, cblabel='QNCLOUD (x10$^8$kg$^{-1}$)', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/qncloud_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qncloud_mean[:,:,:,istage]*1e-8, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qncloud, vmax=max_qncloud, cblabel='QNCLOUD (x10$^8$kg$^{-1}$)', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )

                fname = [self.folder + '/qnrain_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/qnrain_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qnrain_mean[:,:,:,istage]*1e-5, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=0., vmax=max_qnrain, cblabel='QNRAIN (x10$^5$kg$^{-1}$)', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/qnrain_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qnrain_mean[:,:,:,istage]*1e-5, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=0., vmax=max_qnrain, cblabel='QNRAIN (x10$^5$kg$^{-1}$)', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )

                fname = [self.folder + '/noninduc_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/noninduc_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.noninduc_mean[:,:,:,istage]*1e8, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_noninduc, vmax=max_noninduc, cblabel='NONINDUC (nC m$^{-3}$)', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/noninduc_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.noninduc_mean[:,:,:,istage]*1e8, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_noninduc, vmax=max_noninduc, cblabel='NONINDUC (nC m$^{-3}$)', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )
                fname = [self.folder + '/cldnuc_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/cldnuc_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.cldnuc_mean[:,:,:,istage]*1e8, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_cldnuc, vmax=max_cldnuc, cblabel='CLDNUC (x10$^{-8}$kg kg$^{-1}$s$^{-1}$)', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/cldnuc_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.cldnuc_mean[:,:,:,istage]*1e8, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_cldnuc, vmax=max_cldnuc, cblabel='CLDNUC (x10$^{-8}$kg kg$^{-1}$s$^{-1}$)', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )

                fname = [self.folder + '/rh_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/rh_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.rh_mean[:,:,:,istage], self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_rh, vmax=max_rh, cblabel='RH', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl,x_contour=1 )
                fname = [self.folder + '/rh_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.rh_mean[:,:,:,istage], self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_rh, vmax=max_rh, cblabel='RH', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl,x_contour=1 )
                fname = [self.folder + '/epotential_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/epotential_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.epotential_mean[:,:,:,istage]*1e-7, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_epotential, vmax=max_epotential, cblabel='EPOT (x10$^{+8}$V)', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/epotential_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.epotential_mean[:,:,:,istage]*1e-7, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_epotential, vmax=max_epotential, cblabel='EPOT (x10$^{+8}$V)', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )
                fname = [self.folder + '/qngraupel_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/qngraupel_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qngraupel_mean[:,:,:,istage]*1e-5, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qngraupel, vmax=max_qngraupel, cblabel='QNGRAUPEL (x10$^5$kg$^{-1}$)', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/qngraupel_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qngraupel_mean[:,:,:,istage]*1e-5, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qngraupel, vmax=max_qngraupel, cblabel='QNGRAUPEL (x10$^5$kg$^{-1}$)', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )
                fname = [self.folder + '/qicesnow_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/qicesnow_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qicesnow_mean[:,:,:,istage]*factor_qicesnow, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qicesnow, vmax=max_qicesnow, cblabel=qicesnow_label, title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/qicesnow_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qicesnow_mean[:,:,:,istage]*factor_qicesnow, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=min_qicesnow, vmax=max_qicesnow, cblabel=qicesnow_label, title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )

                fname = [self.folder + '/qghail_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/qghail_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.qghail_mean[:,:,:,istage]*factor_qghail, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=0., vmax=max_qghail, cblabel=qghail_label, title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/qghail_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.qghail_mean[:,:,:,istage]*factor_qghail, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=0., vmax=max_qghail, cblabel=qghail_label, title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )

                fname = [self.folder + '/buoyancy_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]#, self.folder + '/buoyancy_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.buoy_mean[:,:,:,istage], self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=-0.03, vmax=0.03, cblabel='buoy (m s$^{-2}$)', title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/buoyancy_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.buoy_mean[:,:,:,istage], self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmin=-0.03, vmax=0.03, cblabel='buoy (m s$^{-2}$)', title='t = '+str(int(stage_number)), ticks_fmt='%.2f', ylabel=yl, xlabel=xl )

                fname = [self.folder + '/w_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/w_frame_%05d.png'%int(istage) ]#, self.folder + '/w_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.w_mean[:,:,:,istage], self.u_mean[:,:,:,istage], self.v_mean[:,:,:,istage], self.w_mean[:,:,:,istage], fname=fname, xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmax=max_w, zero_contour=True, cblabel='w (m s$^{-1}$)', title='t= '+str(int(stage_number)), ylabel=yl, xlabel=xl )

                fname = [self.folder + '/w_dev_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/w_dev_frame_%05d.png'%int(istage) ]#, self.folder + '/w_dev_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.w_dev_mean[:,:,:,istage], self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmax=max_w_dev, zero_contour=True, cblabel="w' (m s$^{-1}$)", title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl , cmap = "jet_r" if self.up else "jet_r" ) #ALEJANDRA
                fname = [self.folder + '/w_dev_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.w_dev_mean[:,:,:,istage], self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, vmax=max_w_dev, zero_contour=True, cblabel="w' (m s$^{-1}$)", title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl  ,cmap = "jet_r" if self.up else "jet_r") #ALEJANDRA

                fname = [self.folder + '/pdev_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/pdev_frame_%05d.png'%int(istage) ]#, self.folder + '/pdev_frame_%05d.png'%int(istage) ]
                aux.plot_field_streamlines( self.x_new, self.y_new, self.z_new, self.pdev_mean[:,:,:,istage], self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, scale=1., vmax=max_pdev, vmin=min_pdev, zero_contour=True, cblabel="P' (Pa)", title='t = '+str(int(stage_number)), ticks_fmt='%d', ylabel=yl, xlabel=xl  )
                fname = [self.folder + '/pdev_composite_t_' + str(int(stage_number)) + '.png']
                aux.plot_field_streamlines_single( self.x_new, self.y_new, self.z_new, self.pdev_mean[:,:,:,istage], self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], fname=fname, xmin=-mx,xmax=mx, ymin=-mx, ymax=mx, centered_circle=circ_r, scale=1., vmax=max_pdev, vmin=min_pdev, zero_contour=True, cblabel="P' (Pa)", title='t = '+str(int(stage_number)), ticks_fmt='%d', ylabel=yl, xlabel=xl )

                fname = [self.folder + '/entr_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/entr_frame_%05d.png'%int(istage) ]#, self.folder + '/entr_frame_%05d.png'%int(istage)]
                aux.plot_mixing( self.angles, self.entrainment_mean[:,istage], self.x_new, self.y_new, self.z_new, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, cblabel='$\epsilon$ (m$^{-1}$)', scale=1e3, vmax=max_entr, fname=fname, title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/entr_composite_t_' + str(int(stage_number)) + '_xz.png']
                aux.plot_mixing_single( self.angles, self.entrainment_mean[:,istage], self.x_new, self.y_new, self.z_new, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, cblabel='$\epsilon$ (m$^{-1}$)', scale=1e3, vmax=max_entr, fname=fname, title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )

                fname = [self.folder + '/detr_composite_t_' + str(int(stage_number)) + '.png', self.folder + '/detr_frame_%05d.png'%int(istage) ]#, self.folder + '/detr_frame_%05d.png'%int(istage)]
                aux.plot_mixing( self.angles, -self.detrainment_mean[:,istage], self.x_new, self.y_new, self.z_new, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, cblabel='$\delta$ (m$^{-1}$)', scale=1e3, vmax=max_detr, fname=fname, title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )
                fname = [self.folder + '/detr_composite_t_' + str(int(stage_number)) + '_xz.png']
                aux.plot_mixing_single( self.angles, -self.detrainment_mean[:,istage], self.x_new, self.y_new, self.z_new, self.u_dev_mean[:,:,:,istage], self.v_dev_mean[:,:,:,istage], self.w_dev_mean[:,:,:,istage], xmin=-mx, xmax=mx, ymin=-mx, ymax=mx, cblabel='$\delta$ (m$^{-1}$)', scale=1e3, vmax=max_detr, fname=fname, title='t = '+str(int(stage_number)), ylabel=yl, xlabel=xl )



    def _make_pdf_from_latex( self, fname='composite_thermal.tex' ):
        """
        create a pdf file from a latex file, that includes the plots of the composite thermal at the different stages
        """
        header=r"'\documentclass[12pt,a4paper]{article}'"[1:-1]+'\n'+r"'\usepackage[dvips]{graphicx}'"[1:-1]+'\n'+r"'\setlength{\textwidth}{19cm}'"[1:-1]+'\n' + r"'\setlength{\textheight}{27cm}'"[1:-1]+'\n'+ r"'\hoffset-1in'"[1:-1]+'\n' +r"'\voffset-1.4in'"[1:-1]+'\n'+r"'\begin{document}'"[1:-1]+'\n'+r"'\parindent0mm'"[1:-1]+'\n'+r"'\parskip0mm'"[1:-1]+'\n'

        tex_file = open( fname, 'w+' )
        tex_file.write( header )
        lines = []
        #lines = lines + include_plots( folder=self.folder, prefix='vorticity' )
        #lines.append( repr('\newpage')[1:-1] )
       
        lines = lines + include_plots( folder=self.folder, prefix='buoyancy' )
        lines.append( repr('\newpage')[1:-1] )
        lines = lines + include_plots( folder=self.folder, prefix='w' )
        lines.append( repr('\newpage')[1:-1] )
        lines = lines + include_plots( folder=self.folder, prefix='w_dev' )
        lines.append( repr('\newpage')[1:-1] )
        lines = lines + include_plots( folder=self.folder, prefix='entr' )
        lines.append( repr('\newpage')[1:-1] )
        lines = lines + include_plots( folder=self.folder, prefix='detr' )
        lines.append(  repr('\end{document}')[2:-1] + '\n' )

        for line in lines:
            tex_file.write(line)
        tex_file.close()
        
        os.system( 'latex  '+fname )
        os.system( 'latex  '+fname )
        os.system( 'dvipdf '+fname[:-4] +'.dvi')
        os.system( 'rm -f  '+fname[:-4] +'.dvi' )
        os.system( 'rm -f  '+fname[:-4] +'.aux' )
        os.system( 'rm -f  '+fname[:-4] +'.log' )


    def _find_thermal_shape_parameters(self, w_dev):
        
        print ("-------Finding shape parameters---------")
        
        if self.up:
            c = np.where(self.x_new==0)[0][0]
            end = self.x_new.shape[0] -1
            ix,iy,iz,iz_c = c,c,c,c
            # find the lowest point where w_dev>0, looking straight down:
            while w_dev[ix,iy,iz_c]>0 and iz_c>0:
                iz_c+=-1
            if iz_c==0 and w_dev[ix,iy,iz_c]>0: # simply to make the histogram bins consistent
                iz_int = self.z_new[iz_c]-0.5*self.delta_R
            else:
                iz_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz_c],w_dev[ix,iy,iz_c+1]],fp=[self.z_new[iz_c],self.z_new[iz_c+1]])[0]
                if iz_int>=0.:
                    iz_int = np.nan
            # now do the same, but not looking straight down but rather following the highest w_dev path:
            next_w = []
            next_w.append((w_dev[ix,iy,iz], self.x_new[ix], self.y_new[iy], self.z_new[iz]))
            reach_end=False
            while w_dev[ix,iy,iz]>0 and iz>0 and (not reach_end):# and (w_low<=w_low0 or count<1):
                if np.all(np.isnan(w_dev[ix-1:ix+2,iy-1:iy+2,iz-1])):
                    print ('Warning: subgrid for composite is probably not large enough... try increasing R_range! (for thermal shape parameters)\n\n')
                    reach_end=True
                else:
                    ix_new,iy_new = np.where( w_dev[ix-1:ix+2,iy-1:iy+2,iz-1]==np.nanmax(w_dev[ix-1:ix+2,iy-1:iy+2,iz-1]) )
                    ix = ix_new[0]+ix-1
                    iy = iy_new[0]+iy-1
                    iz = iz-1
                next_w.append( (w_dev[ix,iy,iz], self.x_new[ix], self.y_new[iy], self.z_new[iz]) )
            if iz==0 and w_dev[ix,iy,iz]>0:
                iz2_int = self.z_new[iz]-0.5*self.delta_R
                w_plume = np.asarray(next_w)
                iz2_int = _correct_for_nearby_thermal( w_plume, iz2_int )
            else:
                iz2_int = np.interp(x=[0], xp=[w_dev[ix,iy,iz],w_dev[ix,iy,iz+1]],fp=[self.z_new[iz],self.z_new[iz+1]])[0]
                if iz2_int>=0.:
                    iz2_int=np.nan
                w_plume = np.asarray(next_w)
            iz_low = (self.x_new[ix],self.y_new[iy],iz2_int,iz_int)

            #do the same in the other 5 directions (z+,x+-,y+-):
            ix,iy,iz,iz_c = c,c,c,c
            while w_dev[ix,iy,iz_c]>0 and iz_c<end:
                iz_c+=1
            if iz_c==end and w_dev[ix,iy,iz_c]>0:
                iz_int = self.z_new[iz_c]+0.5*self.delta_R
            else:
                iz_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz_c],w_dev[ix,iy,iz_c-1]],fp=[self.z_new[iz_c],self.z_new[iz_c-1]])[0]
                if iz_int<0.:
                    iz_int=np.nan
            while w_dev[ix,iy,iz]>0 and iz<end:# and (w_up<=w_up0 or count<1):
                next_w = np.nanmax(w_dev[ix-1:ix+2,iy-1:iy+2,iz+1])
                ix_new,iy_new = np.where( w_dev[ix-1:ix+2,iy-1:iy+2,iz+1]==next_w )
                ix = ix_new[0]+ix-1
                iy = iy_new[0]+iy-1
                iz = iz+1
            if iz==end and w_dev[ix,iy,iz]>0:
                iz2_int = self.z_new[iz]+0.5*self.delta_R
            else:
                iz2_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz],w_dev[ix,iy,iz-1]],fp=[self.z_new[iz],self.z_new[iz-1]])[0]
                if iz2_int<0.:
                    iz2_int=np.nan
            iz_up = (self.x_new[ix],self.y_new[iy],iz2_int, iz_int)

            ix,iy,iz,ix_c = c,c,c,c
            while w_dev[ix_c,iy,iz]>0 and ix_c<end:
                ix_c+=1
            if ix_c==end and w_dev[ix_c,iy,iz]>0:
                ix_int = self.x_new[ix_c]+0.5*self.delta_R
            else:
                ix_int = np.interp(x=[0],xp=[w_dev[ix_c,iy,iz],w_dev[ix_c-1,iy,iz]],fp=[self.x_new[ix_c],self.x_new[ix_c-1]])[0]
                if ix_int<0.:
                    ix_int = np.nan
            while w_dev[ix,iy,iz]>0 and ix<end:# and (w_right<=w_right0 or count<1):
                next_w = np.nanmax(w_dev[ix+1,iy-1:iy+2,iz-1:iz+2])
                iy_new,iz_new = np.where( w_dev[ix+1,iy-1:iy+2,iz-1:iz+2]==next_w )
                ix = ix+1
                iy = iy_new[0]+iy-1
                iz = iz_new[0]+iz-1
            if ix==end and w_dev[ix,iy,iz]>0:
                ix2_int = self.x_new[ix]+0.5*self.delta_R
            else:
                ix2_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz],w_dev[ix-1,iy,iz]],fp=[self.x_new[ix],self.x_new[ix-1]])[0]
                if ix2_int<0.:
                    ix2_int=np.nan

            ix_right = (ix2_int,self.y_new[iy],self.z_new[iz],ix_int)

            ix,iy,iz,ix_c = c,c,c,c
            while w_dev[ix_c,iy,iz]>0 and ix_c>0:
                ix_c+=-1
            if ix_c==0 and w_dev[ix_c,iy,iz]>0:
                ix_int = self.x_new[ix_c]-0.5*self.delta_R
            else:
                ix_int = np.interp(x=[0],xp=[w_dev[ix_c,iy,iz],w_dev[ix_c+1,iy,iz]],fp=[self.x_new[ix_c],self.x_new[ix_c+1]])[0]
                if ix_int>=0.:
                    ix_int=np.nan
            while w_dev[ix,iy,iz]>0 and ix>0:# and (w_left<=w_left0 or count<1):
                next_w = np.nanmax(w_dev[ix-1,iy-1:iy+2,iz-1:iz+2])
                iy_new,iz_new = np.where( w_dev[ix-1,iy-1:iy+2,iz-1:iz+2]==next_w )
                ix = ix-1
                iy = iy_new[0]+iy-1
                iz = iz_new[0]+iz-1
            if ix==0 and w_dev[ix,iy,iz]>0:
                ix2_int = self.x_new[ix]-0.5*self.delta_R
            else:
                ix2_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz],w_dev[ix+1,iy,iz]],fp=[self.x_new[ix],self.x_new[ix+1]])[0]
                if ix2_int>=0.:
                    ix2_int=np.nan
            ix_left = (ix2_int,self.y_new[iy],self.z_new[iz],ix_int)

            ix,iy,iz,iy_c = c,c,c,c
            while w_dev[ix,iy_c,iz]>0 and iy_c>0:
                iy_c+=-1
            if iy_c==0 and w_dev[ix,iy_c,iz]>0:
                iy_int = self.y_new[iy_c]-0.5*self.delta_R
            else:
                iy_int = np.interp(x=[0],xp=[w_dev[ix,iy_c,iz],w_dev[ix,iy_c+1,iz]],fp=[self.y_new[iy_c],self.y_new[iy_c+1]])[0]
                if iy_int>=0:
                    iy_int=np.nan
            while w_dev[ix,iy,iz]>0 and iy>0:# and (w_lefty<=w_lefty0 or count<1):
                next_w = np.nanmax(w_dev[ix-1:ix+2,iy-1,iz-1:iz+2])
                ix_new,iz_new = np.where( w_dev[ix-1:ix+2,iy-1,iz-1:iz+2]==next_w )
                ix = ix_new[0]+ix-1
                iy = iy-1
                iz = iz_new[0]+iz-1
            if iy==0 and w_dev[ix,iy,iz]>0:
                iy2_int = self.y_new[iy]-0.5*self.delta_R
            else:
                iy2_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz],w_dev[ix,iy+1,iz]],fp=[self.y_new[iy],self.y_new[iy+1]])[0]
                if iy2_int>=0:
                    iy2_int=np.nan
            iy_left = (self.x_new[ix],iy2_int,self.z_new[iz],iy_int)

            ix,iy,iz,iy_c = c,c,c,c
            while w_dev[ix,iy_c,iz]>0 and iy_c<end:
                iy_c+=1
            if iy_c==end and w_dev[ix,iy_c,iz]>0:
                iy_int = self.y_new[iy_c]+0.5*self.delta_R
            else:
                iy_int = np.interp(x=[0],xp=[w_dev[ix,iy_c,iz],w_dev[ix,iy_c-1,iz]],fp=[self.y_new[iy_c],self.y_new[iy_c-1]])[0]
                if iy_int<0:
                    iy_int=np.nan
            while w_dev[ix,iy,iz]>0 and iy<end:# and (w_righty<=w_righty0 or count<1):
                next_w = np.nanmax(w_dev[ix-1:ix+2,iy+1,iz-1:iz+2])
                ix_new,iz_new = np.where( w_dev[ix-1:ix+2,iy+1,iz-1:iz+2]==next_w )
                ix = ix_new[0]+ix-1
                iy = iy+1
                iz = iz_new[0]+iz-1
            if iy==end and w_dev[ix,iy,iz]>0:
                iy2_int = self.y_new[iy]+0.5*self.delta_R
            else:
                iy2_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz],w_dev[ix,iy-1,iz]],fp=[self.y_new[iy],self.y_new[iy-1]])[0]
                if iy2_int<0:
                    iy2_int=np.nan
            iy_right = (self.x_new[ix],iy2_int,self.z_new[iz],iy_int)
        
        else: #ALEJANDRA: FOR DD (same but looking at w_dev < 0)   
            c = np.where(self.x_new==0)[0][0]
            end = self.x_new.shape[0] -1
            ix,iy,iz,iz_c = c,c,c,c
            # find the lowest point where w_dev<0, looking straight down:
            while w_dev[ix,iy,iz_c]<0 and iz_c>0:
                iz_c+=-1
            if iz_c==0 and w_dev[ix,iy,iz_c]<0: # simply to make the histogram bins consistent
                iz_int = self.z_new[iz_c]-0.5*self.delta_R
            else:
                iz_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz_c],w_dev[ix,iy,iz_c+1]],fp=[self.z_new[iz_c],self.z_new[iz_c+1]])[0]
                if iz_int>=0.:
                    iz_int = np.nan
            # now do the same, but not looking straight down but rather following the highest w_dev path:
            next_w = []
            next_w.append((w_dev[ix,iy,iz], self.x_new[ix], self.y_new[iy], self.z_new[iz]))
            reach_end=False
            while w_dev[ix,iy,iz]<0 and iz>0 and (not reach_end):# and (w_low<=w_low0 or count<1):
                if np.all(np.isnan(w_dev[ix-1:ix+2,iy-1:iy+2,iz-1])):
                    print ('Warning: subgrid for composite is probably not large enough... try increasing R_range! (for thermal shape parameters)\n\n')
                    reach_end=True
                else:
                    ix_new,iy_new = np.where( w_dev[ix-1:ix+2,iy-1:iy+2,iz-1]==np.nanmin(w_dev[ix-1:ix+2,iy-1:iy+2,iz-1]) )
                    ix = ix_new[0]+ix-1
                    iy = iy_new[0]+iy-1
                    iz = iz-1
                next_w.append( (w_dev[ix,iy,iz], self.x_new[ix], self.y_new[iy], self.z_new[iz]) )
            if iz==0 and w_dev[ix,iy,iz]<0:
                iz2_int = self.z_new[iz]-0.5*self.delta_R
                w_plume = np.asarray(next_w)
                iz2_int = _correct_for_nearby_thermal( w_plume, iz2_int )
            else:
                iz2_int = np.interp(x=[0], xp=[w_dev[ix,iy,iz],w_dev[ix,iy,iz+1]],fp=[self.z_new[iz],self.z_new[iz+1]])[0]
                if iz2_int>=0.:
                    iz2_int=np.nan
                w_plume = np.asarray(next_w)
            iz_low = (self.x_new[ix],self.y_new[iy],iz2_int,iz_int)

            #do the same in the other 5 directions (z+,x+-,y+-):
            ix,iy,iz,iz_c = c,c,c,c
            while w_dev[ix,iy,iz_c]<0 and iz_c<end:
                iz_c+=1
            if iz_c==end and w_dev[ix,iy,iz_c]<0:
                iz_int = self.z_new[iz_c]+0.5*self.delta_R
            else:
                iz_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz_c],w_dev[ix,iy,iz_c-1]],fp=[self.z_new[iz_c],self.z_new[iz_c-1]])[0]
                if iz_int<0.:
                    iz_int=np.nan
            while w_dev[ix,iy,iz]<0 and iz<end:# and (w_up<=w_up0 or count<1):
                next_w = np.nanmin(w_dev[ix-1:ix+2,iy-1:iy+2,iz+1])
                ix_new,iy_new = np.where( w_dev[ix-1:ix+2,iy-1:iy+2,iz+1]==next_w )
                ix = ix_new[0]+ix-1
                iy = iy_new[0]+iy-1
                iz = iz+1
            if iz==end and w_dev[ix,iy,iz]<0:
                iz2_int = self.z_new[iz]+0.5*self.delta_R
            else:
                iz2_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz],w_dev[ix,iy,iz-1]],fp=[self.z_new[iz],self.z_new[iz-1]])[0]
                if iz2_int<0.:
                    iz2_int=np.nan
            iz_up = (self.x_new[ix],self.y_new[iy],iz2_int, iz_int)

            ix,iy,iz,ix_c = c,c,c,c
            while w_dev[ix_c,iy,iz]<0 and ix_c<end:
                ix_c+=1
            if ix_c==end and w_dev[ix_c,iy,iz]<0:
                ix_int = self.x_new[ix_c]+0.5*self.delta_R
            else:
                ix_int = np.interp(x=[0],xp=[w_dev[ix_c,iy,iz],w_dev[ix_c-1,iy,iz]],fp=[self.x_new[ix_c],self.x_new[ix_c-1]])[0]
                if ix_int<0.:
                    ix_int = np.nan
            while w_dev[ix,iy,iz]<0 and ix<end:# and (w_right<=w_right0 or count<1):
                next_w = np.nanmin(w_dev[ix+1,iy-1:iy+2,iz-1:iz+2])
                iy_new,iz_new = np.where( w_dev[ix+1,iy-1:iy+2,iz-1:iz+2]==next_w )
                ix = ix+1
                iy = iy_new[0]+iy-1
                iz = iz_new[0]+iz-1
            if ix==end and w_dev[ix,iy,iz]<0:
                ix2_int = self.x_new[ix]+0.5*self.delta_R
            else:
                ix2_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz],w_dev[ix-1,iy,iz]],fp=[self.x_new[ix],self.x_new[ix-1]])[0]
                if ix2_int<0.:
                    ix2_int=np.nan

            ix_right = (ix2_int,self.y_new[iy],self.z_new[iz],ix_int)

            ix,iy,iz,ix_c = c,c,c,c
            while w_dev[ix_c,iy,iz]<0 and ix_c>0:
                ix_c+=-1
            if ix_c==0 and w_dev[ix_c,iy,iz]<0:
                ix_int = self.x_new[ix_c]-0.5*self.delta_R
            else:
                ix_int = np.interp(x=[0],xp=[w_dev[ix_c,iy,iz],w_dev[ix_c+1,iy,iz]],fp=[self.x_new[ix_c],self.x_new[ix_c+1]])[0]
                if ix_int>=0.:
                    ix_int=np.nan
            while w_dev[ix,iy,iz]<0 and ix>0:# and (w_left<=w_left0 or count<1):
                next_w = np.nanmin(w_dev[ix-1,iy-1:iy+2,iz-1:iz+2])
                iy_new,iz_new = np.where( w_dev[ix-1,iy-1:iy+2,iz-1:iz+2]==next_w )
                ix = ix-1
                iy = iy_new[0]+iy-1
                iz = iz_new[0]+iz-1
            if ix==0 and w_dev[ix,iy,iz]<0:
                ix2_int = self.x_new[ix]-0.5*self.delta_R
            else:
                ix2_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz],w_dev[ix+1,iy,iz]],fp=[self.x_new[ix],self.x_new[ix+1]])[0]
                if ix2_int>=0.:
                    ix2_int=np.nan
            ix_left = (ix2_int,self.y_new[iy],self.z_new[iz],ix_int)

            ix,iy,iz,iy_c = c,c,c,c
            while w_dev[ix,iy_c,iz]<0 and iy_c>0:
                iy_c+=-1
            if iy_c==0 and w_dev[ix,iy_c,iz]<0:
                iy_int = self.y_new[iy_c]-0.5*self.delta_R
            else:
                iy_int = np.interp(x=[0],xp=[w_dev[ix,iy_c,iz],w_dev[ix,iy_c+1,iz]],fp=[self.y_new[iy_c],self.y_new[iy_c+1]])[0]
                if iy_int>=0:
                    iy_int=np.nan
            while w_dev[ix,iy,iz]<0 and iy>0:# and (w_lefty<=w_lefty0 or count<1):
                next_w = np.nanmin(w_dev[ix-1:ix+2,iy-1,iz-1:iz+2])
                ix_new,iz_new = np.where( w_dev[ix-1:ix+2,iy-1,iz-1:iz+2]==next_w )
                ix = ix_new[0]+ix-1
                iy = iy-1
                iz = iz_new[0]+iz-1
            if iy==0 and w_dev[ix,iy,iz]<0:
                iy2_int = self.y_new[iy]-0.5*self.delta_R
            else:
                iy2_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz],w_dev[ix,iy+1,iz]],fp=[self.y_new[iy],self.y_new[iy+1]])[0]
                if iy2_int>=0:
                    iy2_int=np.nan
            iy_left = (self.x_new[ix],iy2_int,self.z_new[iz],iy_int)

            ix,iy,iz,iy_c = c,c,c,c
            while w_dev[ix,iy_c,iz]<0 and iy_c<end:
                iy_c+=1
            if iy_c==end and w_dev[ix,iy_c,iz]<0:
                iy_int = self.y_new[iy_c]+0.5*self.delta_R
            else:
                iy_int = np.interp(x=[0],xp=[w_dev[ix,iy_c,iz],w_dev[ix,iy_c-1,iz]],fp=[self.y_new[iy_c],self.y_new[iy_c-1]])[0]
                if iy_int<0:
                    iy_int=np.nan
            while w_dev[ix,iy,iz]<0 and iy<end:# and (w_righty<=w_righty0 or count<1):
                next_w = np.nanmin(w_dev[ix-1:ix+2,iy+1,iz-1:iz+2])
                ix_new,iz_new = np.where( w_dev[ix-1:ix+2,iy+1,iz-1:iz+2]==next_w )
                ix = ix_new[0]+ix-1
                iy = iy+1
                iz = iz_new[0]+iz-1
            if iy==end and w_dev[ix,iy,iz]<0:
                iy2_int = self.y_new[iy]+0.5*self.delta_R
            else:
                iy2_int = np.interp(x=[0],xp=[w_dev[ix,iy,iz],w_dev[ix,iy-1,iz]],fp=[self.y_new[iy],self.y_new[iy-1]])[0]
                if iy2_int<0:
                    iy2_int=np.nan
            iy_right = (self.x_new[ix],iy2_int,self.z_new[iz],iy_int)

        return ix_left, ix_right, iy_left, iy_right, iz_low, iz_up, w_plume


    def _compute_net_entrainment( self ):
        net_entr_c = np.ones_like(self.z_centre_c)*np.nan           # net entrainment in between time steps, retaining the thermal stage information (centered on peak wmax)
        net_gross_entr = np.ones(self.z_centre_c.shape[0])*np.nan   # net entrainment for each thermal (last time step - first timestep, one value per thermal)
        net_entr_tsteps = []                                        # same as net_entr_c, but keeping just a 1d array with each time step of every thermal, all together.
        dRdz = []                                                   # instantaneous opening angle of thermal (dR/dz)
        net_entr_time = []                                          # time array corresponding to net_entr_tsteps
        net_dRdz = np.ones(self.z_centre_c.shape[0])*np.nan
        z_for_net_entr_tsteps = []
        for i in range(self.z_centre_c.shape[0]): # i is the index for each thermal
            j=0
            while np.isnan(self.z_centre_c[i,j]):
                j+=1
            z0=self.z_centre_c[i,j]
            m0=self.mass_c[i,j]
            R0=self.R_c[i,j]
            while j<self.z_centre_c.shape[1]-1 and ~np.isnan(self.z_centre_c[i,j+1]):
                net_entr_c[i,j+1]=(self.mass_c[i,j+1]-self.mass_c[i,j])/(0.5*(self.mass_c[i,j]+self.mass_c[i,j+1])*(self.z_centre_c[i,j+1]-self.z_centre_c[i,j]))
                # this is (1/M)*dM/dz
                net_entr_tsteps.append(net_entr_c[i,j+1])
                net_entr_time.append(0.5*(self.time_c[i,j+1]+self.time_c[i,j]))
                z_for_net_entr_tsteps.append( 0.5*(self.z_centre_c[i,j+1]+self.z_centre_c[i,j]) )
                dRdz.append((self.R_c[i,j+1]-self.R_c[i,j])/(0.5*(self.R_c[i,j]+self.R_c[i,j+1])*(self.z_centre_c[i,j+1]-self.z_centre_c[i,j])))
                # this is (1/R)*dR/dz
                j+=1
            zf=self.z_centre_c[i,j]
            mf=self.mass_c[i,j]
            Rf=self.R_c[i,j]
            net_gross_entr[i] = (mf-m0)/(0.5*(m0+mf)*(zf-z0))       # this should be divided by the largest between m0 and mf, not the average... 04.10.2016
            net_dRdz[i] = (Rf-R0)/(0.5*(R0+Rf)*(zf-z0))
        self.net_entr_c = net_entr_c
        self.net_gross_entr = net_gross_entr
        self.net_entr_tsteps = np.asarray(net_entr_tsteps)
        self.net_entr_time = np.asarray(net_entr_time)
        self.z_for_net_entr_tsteps = np.asarray(z_for_net_entr_tsteps)
        self.dRdz = np.asarray(dRdz)
        self.net_dRdz = net_dRdz
        if not os.path.isdir( self.folder ):
            os.mkdir( self.folder )
        np.save( self.folder+'/net_entr_tsteps.npy', self.net_entr_tsteps )
        np.save( self.folder+'/net_entr_time.npy', self.net_entr_time )
        np.save( self.folder+'/net_gross_entr.npy', self.net_gross_entr )
        np.save( self.folder+'/net_entr_c.npy', self.net_entr_c )
        np.save( self.folder+'/dRdz.npy', self.dRdz )
        np.save( self.folder+'/net_dRdz.npy', self.net_dRdz )

    def _compute_average_values( self ):
        # compute average values per thermal (average individual timesteps equally weighted)
        self.R_avg=np.nanmean(self.R_c,axis=1)
        self.W_avg=np.nanmean(self.W_c,axis=1)
        self.B_avg=np.nanmean(self.buoy_c,axis=1)
        self.wmax_avg=np.nanmean(self.wmax_c,axis=1)
        self.qcloud_avg=np.nanmean(self.qcloud_c,axis=1)
        self.qncloud_avg=np.nanmean(self.qncloud_c,axis=1)
        self.qrain_avg=np.nanmean(self.qrain_c,axis=1)
        self.qnrain_avg=np.nanmean(self.qnrain_c,axis=1)
        self.cldnuc_avg=np.nanmean(self.cldnuc_c,axis=1)
        self.latheat_avg=np.nanmean(self.latheat_c,axis=1)
        self.sctot_avg=np.nanmean(self.sctot_c,axis=1)
        self.noninduc_avg=np.nanmean(self.noninduc_c,axis=1)
        self.qngraupel_avg=np.nanmean(self.qngraupel_c,axis=1)
        self.epotential_avg=np.nanmean(self.epotential_c,axis=1)
        self.qicesnow_avg=np.nanmean(self.qicesnow_c,axis=1)
        self.qghail_avg=np.nanmean(self.qghail_c,axis=1)

        np.save( self.folder+'/R_avg.npy', self.R_avg )
        np.save( self.folder+'/W_avg.npy', self.W_avg )
        np.save( self.folder+'/B_avg.npy', self.B_avg )
        np.save( self.folder+'/wmax_avg.npy', self.wmax_avg )
        np.save( self.folder+'/qcloud_avg.npy', self.qcloud_avg )
        np.save( self.folder+'/qncloud_avg.npy', self.qncloud_avg )
        np.save( self.folder+'/qrain_avg.npy', self.qrain_avg )
        np.save( self.folder+'/qnrain_avg.npy', self.qnrain_avg )
        np.save( self.folder+'/cldnuc_avg.npy', self.cldnuc_avg )
        np.save( self.folder+'/latheat_avg.npy', self.latheat_avg )
        np.save( self.folder+'/noninduc_avg.npy', self.noninduc_avg )
        np.save( self.folder+'/sctot_avg.npy', self.sctot_avg )
        np.save( self.folder+'/qngraupel_avg.npy', self.qngraupel_avg )
        np.save( self.folder+'/epotential_avg.npy', self.epotential_avg )
        np.save( self.folder+'/qicesnow_avg.npy', self.qicesnow_avg )
        np.save( self.folder+'/qghail_avg.npy', self.qghail_avg )

    def make_mean_composite_tseries( self, tminplot=-4, tmaxplot=4, ymaxplot=0.05 ):
        wmaxtemp2 = np.ma.masked_array(self.wmax_c    ,np.isnan(self.wmax_c   ))
        Rtemp2    = np.ma.masked_array(self.R_c*1e-3  ,np.isnan(self.R_c      ))
        Frestemp2 = np.ma.masked_array(self.Fres_c    ,np.isnan(self.Fres_c   ))
        buoytemp2 = np.ma.masked_array(self.buoy_c    ,np.isnan(self.buoy_c   ))
        Fnhtemp2  = np.ma.masked_array(self.Fnh_c     ,np.isnan(self.Fnh_c    ))
        acctemp2  = np.ma.masked_array(self.acc_c     ,np.isnan(self.acc_c    ))
        Dtemp2    = np.ma.masked_array(self.D_c       ,np.isnan(self.D_c      ))
        masstemp2 = np.ma.masked_array(self.mass_c    ,np.isnan(self.mass_c   ))
        Wtemp2    = np.ma.masked_array(self.W_c       ,np.isnan(self.W_c      ))
        Pnztemp2  = np.ma.masked_array(self.Pnz_c     ,np.isnan(self.Pnz_c    ))
        logetemp2 = np.ma.masked_array(self.loge_c    ,np.isnan(self.loge_c   ))
        mf_temp2  = masstemp2*Wtemp2
        
        # massflux weights for averages (for each stage)
        weights=mf_temp2/np.nansum(mf_temp2, axis=0)
        
        if hasattr(self,'net_entr_c'):
            weights_net_e = np.ones_like(weights.data)*np.nan
            for i in range(weights_net_e.shape[0]):
                weights_net_e[i,1:]=(weights.data[i,1:]+weights.data[i,:-1])*0.5
            net_entrtemp2   = np.ma.masked_array(self.net_entr_c, np.isnan(self.net_entr_c))
            net_entr_mean   = np.ma.average(net_entrtemp2, axis=0 )
            net_entr_wmean  = np.ma.average(net_entrtemp2, axis=0, weights=weights_net_e )
            net_entr_std    = np.std( net_entrtemp2, axis=0)
            aux.composite_plot( self.t_range-0.5, net_entr_mean*1e3, net_entr_std*1e3, ylabel='net entrainment (m$^{-1}$)', fname='net_entr', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=6 )
            np.save(self.folder +'/mean_composite_net_entr.npy', np.vstack((self.t_range-0.5, net_entr_mean, net_entr_std)).data)
            np.save(self.folder +'/weighted_mean_composite_net_entr.npy', np.vstack((self.t_range-0.5, net_entr_wmean, net_entr_std)).data)
        if hasattr(self,'net_entr_term_c'):
            net_entr_termtemp2 = np.ma.masked_array(self.net_entr_term_c, np.isnan(self.net_entr_term_c))
            weights_net_e_term = np.ones_like(net_entr_termtemp2.data)*np.nan
            for i in range(weights_net_e_term.shape[0]):
                weights_net_e_term[i,:]=(weights.data[i,1:]+weights.data[i,:-1])*0.5
            net_entr_term_mean  = np.ma.average(net_entr_termtemp2, axis=0)
            net_entr_term_wmean = np.ma.average(net_entr_termtemp2, axis=0, weights=weights_net_e_term)
            net_entr_term_std   = np.std( net_entr_termtemp2, axis=0)
            aux.composite_plot( self.t_range[:-1]-0.5, net_entr_term_mean, net_entr_term_std, ylabel='net entr. term (m s$^{-2}$)', fname='net_entr_term', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot )
            np.save(self.folder +'/mean_composite_net_entr_term.npy', np.vstack((self.t_range[:-1]-0.5, net_entr_term_mean, net_entr_term_std)).data)
            np.save(self.folder +'/weighted_mean_composite_net_entr_term.npy', np.vstack((self.t_range[:-1]-0.5, net_entr_term_wmean, net_entr_term_std)).data)

        wmax_mean   = np.ma.average( wmaxtemp2  , axis=0, weights=weights )
        R_mean      = np.ma.average( Rtemp2     , axis=0 )
        R_wmean     = np.ma.average( Rtemp2     , axis=0, weights=weights )
        Fres_mean   = np.ma.average( Frestemp2  , axis=0 )
        Fres_wmean  = np.ma.average( Frestemp2  , axis=0, weights=weights )
        buoy_mean   = np.ma.average( buoytemp2  , axis=0 )
        buoy_wmean  = np.ma.average( buoytemp2  , axis=0, weights=weights )
        Fnh_mean    = np.ma.average( Fnhtemp2   , axis=0 )
        Fnh_wmean   = np.ma.average( Fnhtemp2   , axis=0, weights=weights )
        acc_mean    = np.ma.average( acctemp2   , axis=0 )
        acc_wmean   = np.ma.average( acctemp2   , axis=0, weights=weights )
        D_mean      = np.ma.average( Dtemp2     , axis=0, weights=weights )
        mass_mean   = np.ma.average( masstemp2  , axis=0, weights=weights )
        W_mean      = np.ma.average( Wtemp2     , axis=0 )  
        W_wmean     = np.ma.average( Wtemp2     , axis=0, weights=weights )  
        Pnz_mean    = np.ma.average( Pnztemp2   , axis=0, weights=weights )    
        loge_mean   = np.ma.average( logetemp2  , axis=0, weights=weights )
        
        #overall mass flux weights:
        weights_flat = self.mass*self.W/np.sum(self.mass*self.W)
        #print('overal mean entr. distance = %f'%(np.ma.mean(D_mean))
        D_overal_mean = np.average(self.D, weights=weights_flat)
        print ('overal mean entr. distance = %f'%(D_overal_mean))
        np.save(self.folder + '/mean_entr_distance.npy', D_overal_mean)
        #print('overal (log10) mean fract. entr. = %f'%(np.ma.mean(loge_mean))
        #np.save(self.folder + '/mean_loge.npy', np.ma.mean(loge_mean))
        print( '********************')
        peak = np.where(self.t_range==0)[0]
        print( 'values at wmax peak:')
        print( 'mean entr. distance = %f'%(D_mean[peak]))
        np.save(self.folder + '/mean_entr_dist_wpeak.npy',D_mean[peak].data)
        print( '(log10) mean fract. entr. = %f'%(loge_mean[peak]))
        np.save(self.folder + '/mean_loge_wpeak.npy', loge_mean[peak].data)

        
        Fres_std    = np.std( Frestemp2  , axis=0 )
        buoy_std    = np.std( buoytemp2  , axis=0 )
        Fnh_std     = np.std( Fnhtemp2   , axis=0 )
        acc_std     = np.std( acctemp2   , axis=0 )
        Pnz_std     = np.std( Pnztemp2   , axis=0 )
        D_std       = np.std( Dtemp2     , axis=0 )
        R_std       = np.std( Rtemp2     , axis=0 )
        loge_std    = np.std( logetemp2  , axis=0 )
        W_std       = np.std( Wtemp2     , axis=0 )
        wmax_std    = np.std( wmaxtemp2  , axis=0 )

        
        Fres_90 = np.ones_like(Fres_std.data)*np.nan 
        buoy_90 = np.ones_like(buoy_std.data)*np.nan 
        Fnh_90  = np.ones_like(Fnh_std.data)*np.nan 
        acc_90  = np.ones_like(acc_std.data)*np.nan
        Pnz_90  = np.ones_like(Pnz_std.data)*np.nan
        D_90    = np.ones_like(D_std.data)*np.nan
        Fres_10 = np.ones_like(Fres_std.data)*np.nan
        buoy_10 = np.ones_like(buoy_std.data)*np.nan 
        Fnh_10  = np.ones_like(Fnh_std.data)*np.nan 
        acc_10  = np.ones_like(acc_std.data)*np.nan
        Pnz_10  = np.ones_like(Pnz_std.data)*np.nan
        D_10    = np.ones_like(D_std.data)*np.nan 
        R_90    = np.ones_like(R_std.data)*np.nan
        R_10    = np.ones_like(R_std.data)*np.nan
        loge_90 = np.ones_like(loge_std.data)*np.nan
        loge_10 = np.ones_like(loge_std.data)*np.nan
        W_90    = np.ones_like(W_std.data)*np.nan
        wmax_90 = np.ones_like(wmax_std.data)*np.nan
        W_10    = np.ones_like(W_std.data)*np.nan
        wmax_10 = np.ones_like(wmax_std.data)*np.nan
        N       = np.zeros_like(R_mean.data)
        
        for i in range((self.tmax-self.tmin)+1):
            if len(Rtemp2.data.transpose()[i]     [np.isfinite(Rtemp2.data.transpose()[i])]) > 0:
                N[i]        = len(np.where(~np.isnan(Rtemp2[:,i]))[0])
                R_90[i]     = np.percentile( Rtemp2.data.transpose()[i]     [np.isfinite(Rtemp2.data.transpose()[i])]     , 90 )
                R_10[i]     = np.percentile( Rtemp2.data.transpose()[i]     [np.isfinite(Rtemp2.data.transpose()[i])]     , 10 )
                W_90[i]     = np.percentile( Wtemp2.data.transpose()[i]     [np.isfinite(Wtemp2.data.transpose()[i])]     , 90 )
                wmax_90[i]  = np.percentile( wmaxtemp2.data.transpose()[i]  [np.isfinite(wmaxtemp2.data.transpose()[i])]  , 90 )
                W_10[i]     = np.percentile( Wtemp2.data.transpose()[i]     [np.isfinite(Wtemp2.data.transpose()[i])]     , 10 )
                wmax_10[i]  = np.percentile( wmaxtemp2.data.transpose()[i]  [np.isfinite(wmaxtemp2.data.transpose()[i])]  , 10 )
                Fres_90[i]  = np.percentile( Frestemp2.data.transpose()[i]  [np.isfinite(Frestemp2.data.transpose()[i])]  , 90 )
                buoy_90[i]  = np.percentile( buoytemp2.data.transpose()[i]  [np.isfinite(buoytemp2.data.transpose()[i])]  , 90 )
                Fnh_90[i]   = np.percentile( Fnhtemp2.data.transpose()[i]   [np.isfinite(Fnhtemp2.data.transpose()[i])]   , 90 )
                acc_90[i]   = np.percentile( acctemp2.data.transpose()[i]   [np.isfinite(acctemp2.data.transpose()[i])]   , 90 )
                Pnz_90[i]   = np.percentile( Pnztemp2.data.transpose()[i]   [np.isfinite(Pnztemp2.data.transpose()[i])]   , 90 )
                D_90[i]     = np.percentile( Dtemp2.data.transpose()[i]     [np.isfinite(Dtemp2.data.transpose()[i])]     , 90 )
                try:
                    loge_90[i]  = np.percentile( logetemp2.data.transpose()[i]  [np.isfinite(logetemp2.data.transpose()[i])]  , 90 )
                    loge_10[i]  = np.percentile( logetemp2.data.transpose()[i]  [np.isfinite(logetemp2.data.transpose()[i])]  , 10 )
                except:
                    print('loge percentiles not computed... all nans!')
                Fres_10[i]  = np.percentile( Frestemp2.data.transpose()[i]  [np.isfinite(Frestemp2.data.transpose()[i])]  , 10 )
                buoy_10[i]  = np.percentile( buoytemp2.data.transpose()[i]  [np.isfinite(buoytemp2.data.transpose()[i])]  , 10 )
                Fnh_10[i]   = np.percentile( Fnhtemp2.data.transpose()[i]   [np.isfinite(Fnhtemp2.data.transpose()[i])]   , 10 )
                acc_10[i]   = np.percentile( acctemp2.data.transpose()[i]   [np.isfinite(acctemp2.data.transpose()[i])]   , 10 )
                Pnz_10[i]   = np.percentile( Pnztemp2.data.transpose()[i]   [np.isfinite(Pnztemp2.data.transpose()[i])]   , 10 )
                D_10[i]     = np.percentile( Dtemp2.data.transpose()[i]     [np.isfinite(Dtemp2.data.transpose()[i])]     , 10 )

        if tminplot==None:
            tminplot=self.tmin
        if tmaxplot==None:
            tmaxplot=self.tmax
        
        # plot a time series of the number of cases of each stage of the thermals:
        N_total = Rtemp2.shape[0] # total number of thermals
        Ni      = np.zeros(Rtemp2.shape[1])  # stages of the thermal
        for i in range(Rtemp2.shape[1]):
            Ni[i] = len(np.where(~np.isnan(Rtemp2[:,i]))[0])
        aux.composite_plot( self.t_range, Ni, error=None, ylabel='Number of cases', fname='N', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=N_total + 5, ymin0=0, grid=True )

        # ALEJANDRA
        np.save( self.folder+'/mean_composite_N.npy', np.vstack((self.t_range, Ni)).data)

        aux.composite_plot( self.t_range, buoy_mean, buoy_std, ylabel=buoyl, fname='buoy', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot )
        aux.composite_plot( self.t_range, Fres_mean, Fres_std, ylabel=Fresl, fname='Fres', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot )
        aux.composite_plot( self.t_range, Fnh_mean, Fnh_std, ylabel=Fnhl, fname='Fnh', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot )
        aux.composite_plot( self.t_range, acc_mean, acc_std, ylabel=accl, fname='acc', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot )
        aux.composite_plot( self.t_range, Pnz_mean, Pnz_std, ylabel=Pnzl, fname='Pnz', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot )
        aux.composite_plot( self.t_range, D_mean, D_std, ylabel=Dl, fname='D', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot )
        aux.composite_plot( self.t_range, R_mean, R_std, ylabel=Rl, fname='R', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=.6, ymin0=0. )
        
        aux.composite_plot( self.t_range, buoy_mean, buoy_std, ylabel=buoyl, fname='buoy_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot, pctl=[buoy_10, buoy_90] )
        aux.composite_plot( self.t_range, Fres_mean, Fres_std, ylabel=Fresl, fname='Fres_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot, pctl=[Fres_10, Fres_90] )
        aux.composite_plot( self.t_range, Fnh_mean, Fnh_std, ylabel=Fnhl, fname='Fnh_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot, pctl=[Fnh_10, Fnh_90] )
        aux.composite_plot( self.t_range, acc_mean, acc_std, ylabel=accl, fname='acc_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot, pctl=[acc_10, acc_90] )
        aux.composite_plot( self.t_range, Pnz_mean, Pnz_std, ylabel=Pnzl, fname='Pnz_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot, pctl=[Pnz_10, Pnz_90] )
        aux.composite_plot( self.t_range, D_mean, D_std, ylabel=Dl, fname='D_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot, pctl=[D_10, D_90] )

        aux.composite_plot( self.t_range, R_mean, R_std, ylabel=Rl, fname='R_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=.7, ymin0=0., pctl=[R_10, R_90] )
        aux.composite_plot( self.t_range, W_mean, W_std, ylabel=Wl, fname='W_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None, ymin0=None, pctl=[W_10, W_90] )
        aux.composite_plot( self.t_range, wmax_mean, wmax_std, ylabel=wmaxl, fname='wmax_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None, ymin0=None, pctl=[wmax_10, wmax_90] )

        aux.composite_plot( self.t_range, loge_mean, loge_std, ylabel=logel, fname='loge_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=-2., ymin0=-3.2, pctl=[loge_10, loge_90], zero_y=False )

        np.save( self.folder+'/mean_composite_mom_budget.npy', np.vstack((self.t_range, acc_mean, Fres_mean, Fnh_mean, buoy_mean, acc_10, acc_90, Fres_10, Fres_90, Fnh_10, Fnh_90, buoy_10, buoy_90)).data)
        np.save( self.folder+'/weighted_mean_composite_mom_budget.npy', np.vstack((self.t_range, acc_wmean, Fres_wmean, Fnh_wmean, buoy_wmean, acc_10, acc_90, Fres_10, Fres_90, Fnh_10, Fnh_90, buoy_10, buoy_90)).data)
        np.save( self.folder+'/mean_composite_RW.npy', np.vstack((self.t_range, R_mean, W_mean, R_std, R_10, R_90, W_std, W_10, W_90)).data )
        np.save( self.folder+'/weighted_mean_composite_RW.npy', np.vstack((self.t_range, R_wmean, W_wmean, R_std, R_10, R_90, W_std, W_10, W_90)).data )
        np.save( self.folder+'/mean_net_entr_term.npy', np.vstack((self.t_range[:-1]+0.5,net_entr_term_mean, net_entr_term_std)).data )
        np.save( self.folder+'/weighted_mean_net_entr_term.npy', np.vstack((self.t_range[:-1]+0.5,net_entr_term_wmean, net_entr_term_std)).data )

    def entrainment_analysis( self ):
        # compute change in buoyancy with height, to compare to entrainment:
        dBdZ=[]
        dZ=[]
        for i in range(self.buoy_c.shape[0]):
            b0=self.buoy_c[i][np.where(~np.isnan(self.buoy_c[i]))[0][0]]
            bf=self.buoy_c[i][np.where(~np.isnan(self.buoy_c[i]))[0][-1]]
            z0=self.z0[i]
            zf=self.z_centre_c[i][np.where(~np.isnan(self.z_centre_c[i]))[0][-1]]
            dZ.append(zf-z0)
            dBdZ.append((bf-b0)/(zf-z0))
        self.dBdZ = np.asarray(dBdZ)
        dZ = np.asarray(dZ)
        dMSEdZ = []
        
        # compute change in MSE with height:
        for i in range(self.mse_thermal_or.shape[0]):
            mse0 = self.mse_thermal_or[i][0]
            msef = self.mse_thermal_or[i][-1]
            dMSEdZ.append((msef-mse0)/dZ[i])
        self.dMSEdZ = np.asarray(dMSEdZ)
        
        # compute mixing from MSE:
        mse_entr=[]
        for i in range(self.mse_env_or.shape[0]):
            mse_out=np.nanmean(self.mse_env_or[i])
            mse_in=np.nanmean(self.mse_thermal_or[i])
            mse_entr.append((self.mse_thermal_or[i][-1]-self.mse_thermal_or[i][0])/(dZ[i]*(mse_out-mse_in)))

        self.mse_entr = np.asarray(mse_entr)
        self.mse_entr = np.ma.masked_array(self.mse_entr, mask=np.isnan(self.mse_entr))
        #self.tracer_entr = np.ma.masked_array(self.tracer_entr, mask=np.isnan(self.tracer_entr))

    def height_profiles( self ):
        
        z_centre= self.z_centre 
        Fres    = self.Fres 
        buoy    = self.buoy 
        Fnh     = self.Fnh  
        acc     = self.acc  
        D       = self.D    
        Pnz     = self.Pnz  
        time    = self.time 
        W       = self.W    
        wmax    = self.wmax 
        R       = self.R
        mass    = self.mass*1e5
        massflux = mass*self.W*1e-5
        fract_entr = self.fract_entr 
        mse_in  = self.mse_thermal 
        mse_out = self.mse_env 
        entr_rate = aux.flatten_array(self.entr_rate)
        net_entr = self.net_entr_tsteps
        z_net_entr = self.z_for_net_entr_tsteps
        Fentr = aux.flatten_array(self.net_entr_term)
        
        
        #these are values for each thermal (not each time step):
        mse_entr    = self.mse_entr
        z_thermal = np.nanmean(self.z_centre_c, axis=1)
        massflux_thermal = np.nanmean((self.W_c*self.mass_c),axis=1)
        
        delta_R = []
        length = np.zeros(len(self.R_or))
        dt = self.time_or[0][1]-self.time_or[0][0]
        self.lifetime = []
        for i in range(len(self.R_or)):
            length[i]=len(self.R_or[i])*dt
            for j in range(len(self.R_or[i])-1):
                delta_R.append((self.R_or[i][j+1]-self.R_or[i][j])/np.min([self.R_or[i][j+1],self.R_or[i][j]]))
            for j in range(len(self.R_or[i])):
                self.lifetime.append(length[i])
        self.lifetime=np.asarray(self.lifetime)
        self.length = length
        lifetime = self.lifetime

        Ntotal= self.z_centre.shape[0]
        Nz = int(Ntotal/nlevs)

        Z               = np.ones(int(nlevs))*np.nan
        N               = np.ones(int(nlevs))*np.nan
        Z2              = np.ones(int(nlevs))*np.nan
        N2              = np.ones(int(nlevs))*np.nan
        Z3              = np.ones(int(nlevs))*np.nan
        N3              = np.ones(int(nlevs))*np.nan

        #n        = np.ones_like(Z)*np.nan
        Z_w      = np.ones_like(Z)*np.nan 
        Z2_w      = np.ones_like(Z2)*np.nan 
        Fres_avg = np.ones_like(Z)*np.nan 
        Fres_wavg = np.ones_like(Z)*np.nan 
        Fres_l   = np.ones_like(Z)*np.nan
        Fres_r   = np.ones_like(Z)*np.nan
        Fnh_avg  = np.ones_like(Z)*np.nan
        Fnh_wavg  = np.ones_like(Z)*np.nan
        Fnh_l    = np.ones_like(Z)*np.nan
        Fnh_r    = np.ones_like(Z)*np.nan
        buoy_avg = np.ones_like(Z)*np.nan
        buoy_wavg = np.ones_like(Z)*np.nan
        buoy_l   = np.ones_like(Z)*np.nan
        buoy_r   = np.ones_like(Z)*np.nan
        acc_avg  = np.ones_like(Z)*np.nan
        acc_wavg  = np.ones_like(Z)*np.nan
        acc_l    = np.ones_like(Z)*np.nan
        acc_r    = np.ones_like(Z)*np.nan
        D_avg    = np.ones_like(Z)*np.nan
        D_l      = np.ones_like(Z)*np.nan
        D_r      = np.ones_like(Z)*np.nan
        Pnz_avg  = np.ones_like(Z)*np.nan
        Pnz_l    = np.ones_like(Z)*np.nan
        Pnz_r    = np.ones_like(Z)*np.nan
        W_avg    = np.ones_like(Z)*np.nan
        W_wavg    = np.ones_like(Z)*np.nan
        W_l      = np.ones_like(Z)*np.nan
        W_r      = np.ones_like(Z)*np.nan
        wmax_avg = np.ones_like(Z)*np.nan
        R_avg    = np.ones_like(Z)*np.nan
        R_wavg    = np.ones_like(Z)*np.nan
        R_l      = np.ones_like(Z)*np.nan
        R_r      = np.ones_like(Z)*np.nan
        e_avg    = np.ones_like(Z)*np.nan
        e_wavg    = np.ones_like(Z)*np.nan
        e_l      = np.ones_like(Z)*np.nan
        e_r      = np.ones_like(Z)*np.nan
        mse_entr_avg    = np.ones_like(Z)*np.nan
        mse_entr_wavg    = np.ones_like(Z)*np.nan
        mse_entr_l      = np.ones_like(Z)*np.nan
        mse_entr_r      = np.ones_like(Z)*np.nan
        entr_rate_avg   = np.ones_like(Z)*np.nan
        entr_rate_l     = np.ones_like(Z)*np.nan
        entr_rate_r     = np.ones_like(Z)*np.nan
        loge_avg        = np.ones_like(Z)*np.nan
        loge_l          = np.ones_like(Z)*np.nan
        loge_r          = np.ones_like(Z)*np.nan
        mse_in_avg      = np.ones_like(Z)*np.nan
        mse_in_l        = np.ones_like(Z)*np.nan 
        mse_in_r        = np.ones_like(Z)*np.nan 
        mse_out_avg     = np.ones_like(Z)*np.nan
        mse_out_l       = np.ones_like(Z)*np.nan 
        mse_out_r       = np.ones_like(Z)*np.nan 
        mass_avg        = np.ones_like(Z)*np.nan
        mass_l          = np.ones_like(Z)*np.nan
        mass_r          = np.ones_like(Z)*np.nan
        massflux_avg    = np.ones_like(Z)*np.nan
        massflux_l      = np.ones_like(Z)*np.nan
        massflux_r      = np.ones_like(Z)*np.nan
        #tot_mflux       = np.ones_like(Z)*np.nan
        loge = np.log10(fract_entr)
        lifetime_avg    = np.ones_like(Z)*np.nan
        lifetime_l      = np.ones_like(Z)*np.nan
        lifetime_r      = np.ones_like(Z)*np.nan
        net_entr_avg    = np.ones_like(Z)*np.nan
        net_entr_wavg    = np.ones_like(Z)*np.nan
        net_entr_l      = np.ones_like(Z)*np.nan
        net_entr_r      = np.ones_like(Z)*np.nan
        Fentr_avg       = np.ones_like(Z)*np.nan
        Fentr_wavg       = np.ones_like(Z)*np.nan

        z0 = np.min(self.z_centre) # in m
        z1 = z0
        for k in range(int(nlevs)):
            nz=0
            while nz<Nz and z1<20000.:
                z1+=1.
                i   = np.where((self.z_centre>=z0)*(self.z_centre<z1))[0]
                nz  = len(i)
            weights = massflux[i]/np.sum(massflux[i])
            N[k]            = nz
            Z[k]            = np.average( self.z_centre[i] )
            Z_w[k]            = np.average( self.z_centre[i], weights=weights )
            Fres_avg[k]     = np.average( Fres[i])
            Fres_wavg[k]     = np.average( Fres[i], weights=weights)
            Fres_l[k]       = np.percentile( Fres[i], 25 )
            Fres_r[k]       = np.percentile( Fres[i], 75 )
            Fnh_avg[k]      = np.average(Fnh[i])
            Fnh_wavg[k]      = np.average(Fnh[i], weights=weights)
            Fnh_l[k]        = np.percentile( Fnh[i], 25 )
            Fnh_r[k]        = np.percentile( Fnh[i], 75 )
            buoy_avg[k]     = np.average(buoy[i])
            buoy_wavg[k]     = np.average(buoy[i], weights=weights)
            buoy_l[k]       = np.percentile( buoy[i], 25 )
            buoy_r[k]       = np.percentile( buoy[i], 75 )
            acc_avg[k]      = np.average(acc[i])
            acc_wavg[k]      = np.average(acc[i], weights=weights)
            acc_l[k]        = np.percentile( acc[i], 25 )
            acc_r[k]        = np.percentile( acc[i], 75 )
            D_avg[k]        = np.average(D[i])
            D_l[k]          = np.percentile( D[i], 25 )
            D_r[k]          = np.percentile( D[i], 75 )
            Pnz_avg[k]      = np.average(Pnz[i])
            Pnz_l[k]        = np.percentile( Pnz[i], 25 )
            Pnz_r[k]        = np.percentile( Pnz[i], 75 )
            W_avg[k]        = np.average(W[i])
            W_wavg[k]        = np.average(W[i], weights=weights)
            W_l[k]          = np.percentile( W[i], 10 )
            W_r[k]          = np.percentile( W[i], 90 )
            wmax_avg[k]     = np.average(wmax[i])
            R_avg[k]        = np.ma.average(R[i])
            R_wavg[k]        = np.ma.average(R[i], weights=weights)
            R_l[k]          = np.percentile( R[i], 10 )
            R_r[k]          = np.percentile( R[i], 90 )
            e_avg[k]        = np.average( fract_entr[i] )
            e_wavg[k]        = np.average( fract_entr[i], weights=weights )
            e_l[k]          = np.nanpercentile( fract_entr[i], 10)
            e_r[k]          = np.nanpercentile( fract_entr[i], 90 )
            mass_avg[k]     = np.average( mass[i] )
            mass_l[k]       = np.percentile( mass[i], 10)
            mass_r[k]       = np.percentile( mass[i], 90 )
            massflux_avg[k] = np.average( massflux[i] )
            massflux_l[k]   = np.percentile( massflux[i], 10)
            massflux_r[k]   = np.percentile( massflux[i], 90 )
            entr_rate_avg[k]= np.average( entr_rate[i] )
            entr_rate_l[k]  = np.percentile( entr_rate[i], 10)
            entr_rate_r[k]  = np.percentile( entr_rate[i], 90 )
            loge_avg[k]     = np.average( np.log10(fract_entr[i]) )
            loge_l[k]       = np.percentile( np.log10(fract_entr[i]), 10 )
            loge_r[k]       = np.percentile( np.log10(fract_entr[i]), 90 )
            mse_in_avg[k]   = np.ma.average(mse_in[i])
            mse_out_avg[k]  = np.ma.average(np.ma.masked_array(mse_out[i], mask=np.isnan(mse_out[i])))
            mse_in_l[k]     = np.percentile( mse_in[i], 25 ) 
            mse_in_r[k]     = np.percentile( mse_in[i], 75 ) 
            mse_out_l[k]    = np.percentile( mse_out[i], 25 ) 
            mse_out_r[k]    = np.percentile( mse_out[i], 75 )
            #n[k]            = len(i)/(np.max([dz,(np.max(z_centre[i]) - np.min(z_centre[i]))])/dz)
            #tot_mflux[k]    = np.sum(massflux[i])/(np.max([dz,(np.max(z_centre[i]) - np.min(z_centre[i]))])/dz)
            lifetime_avg[k] = np.average(lifetime[i])
            lifetime_l[k]   = np.percentile( lifetime[i], 25 )
            lifetime_r[k]   = np.percentile( lifetime[i], 75 )
            z0=np.copy(z1)

        Ntotal= z_net_entr.shape[0]
        Nz = int(Ntotal/nlevs)
        z0 = np.min(z_net_entr) # in m
        z1 = z0
        warning=False
        for k in range(int(nlevs)):
            nz2=0
            while nz2<Nz and z1<20000.:
                z1+=1.
                i2  = np.where((z_net_entr>=z0)*(z_net_entr<z1))[0]
                nz2 = len(i2)
            if nz2<1 and warning==False:
                warning = True
                print('Warning: probably not enough thermals to construct vertical profile!')
            elif nz2>0:
                weights = massflux[i2]/np.sum(massflux[i2])
                N2[k]           = nz2
                Z2[k]           = np.average( z_net_entr[i2] )
                Z2_w[k]           = np.average( z_net_entr[i2], weights=weights )
                net_entr_avg[k] = np.average( net_entr[i2] )
                net_entr_wavg[k] = np.average( net_entr[i2], weights=weights )
                net_entr_l[k]   = np.percentile( net_entr[i2], 25 )
                net_entr_r[k]   = np.percentile( net_entr[i2], 75 )
                Fentr_avg[k]    = np.average( Fentr[i2] )
                Fentr_wavg[k]    = np.average( Fentr[i2], weights=weights )
                z0=np.copy(z1)

        # Now compute the profiles for thermal averages, not individual time steps:
        Ntotal= z_thermal.shape[0]
        Nz = int(Ntotal/nlevs)
        z0 = np.min(z_thermal) # in m
        z1 = z0
        for k in range(int(nlevs)):
            nz3=0
            while nz3<Nz and z1<20000:
                z1+=1.
                i3 = np.where((z_thermal>=z0)*(z_thermal<z1))[0]
                valid_mse = np.where(np.abs(mse_entr[i3]*1e3)<40)[0] # remove outliers
                nz3=len(i3[valid_mse])
            if nz3<1 and warning==False:
                warning=True
                print('Warning: probably not enough thermals to construct vertical profile!')
            elif nz3>0:
                weights= massflux_thermal[i3][valid_mse]/np.sum(massflux_thermal[i3][valid_mse])
                #mse_entr_avg[k]= np.nanmedian( mse_entr[i3] )
                mse_entr_avg[k]= np.nanmean( mse_entr[i3][valid_mse] )
                mse_entr_wavg[k]= np.ma.average( mse_entr[i3][valid_mse], weights=weights )
                mse_entr_l[k]  = np.nanpercentile( mse_entr[i3][valid_mse], 10)
                mse_entr_r[k]  = np.nanpercentile( mse_entr[i3][valid_mse], 90 )
                Z3[k]       = np.average(z_thermal[i3][valid_mse])
                N3[k]       = nz3
                z0=np.copy(z1)
        

        case = 'all'

        # ALEJANDRA
        print ("---------------- SAVING PROFILES --------------------")

        np.save( self.folder+'/profile_mom_budget_'+case+'.npy', np.vstack((Z, acc_avg, Fnh_avg, Fres_avg, buoy_avg, Fentr_avg, Z2)) )
        np.save( self.folder+'/weighted_profile_mom_budget_'+case+'.npy', np.vstack((Z_w, acc_wavg, Fnh_wavg, Fres_wavg, buoy_wavg, Fentr_wavg, Z2_w)) )
        np.save( self.folder+'/profile_RW_'+case+'.npy', np.vstack((Z, R_avg/1e3, W_avg, net_entr_avg, Z2, net_entr_l, net_entr_r)) )
        np.save( self.folder+'/weighted_profile_RW_'+case+'.npy', np.vstack((Z_w, R_wavg/1e3, W_wavg, net_entr_wavg, Z2_w, net_entr_l, net_entr_r)) )

        #ALEJANDRA
        print ("---------------- SAVING MASS FLUX --------------------")
        np.save( self.folder+'/profile_MassFlux_'+case+'.npy', np.vstack((Z, massflux_avg,massflux_l,massflux_r,mass_avg,mass_avg_l, mass_avg_r)) )
        
        #aux.height_profile( [self.tracer_e_avg,mse_entr_avg,e_avg],[Z/1e3,Z3/1e3,Z/1e3],label=['tracers','mse','direct'],fname=self.folder+'/profile_entrainment_new.pdf')

        #aux.height_profile( [tracer_entr_wavg,mse_entr_wavg,e_wavg],[Z3/1e3,Z3/1e3,Z/1e3],label=['tracers','mse','direct'],fname=None)

        #aux.height_profile( [acc_avg,Fnh_avg,Fres_avg,buoy_avg], [Z/1e3,Z/1e3,Z/1e3,Z/1e3], label=['dW/dt','Fnh','Fmix','buoy'], xticks=np.arange(-0.02,0.021,0.01), fname=self.folder+'/profile_mom_budget_'+case+'.pdf', title=self.exp_name, xmin=None, xmax=None, xlabel='m$\,$s$^{-2}$' )
        
        aux.height_profile( [acc_avg,Fnh_avg,Fres_avg,buoy_avg], [Z/1e3,Z/1e3,Z/1e3,Z/1e3], label=['dW/dt','Fnh','Fmix','buoy'], xticks=None, fname=self.folder+'/profile_mom_budget_'+case+'.pdf', title=self.exp_name, xmin=None, xmax=None, xlabel='m$\,$s$^{-2}$' ) #ALEJANDRA

            
        aux.height_profile( [lifetime_avg], Z/1e3, zero=False, fname=self.folder+'/profile_lifetime_'+case+'.pdf', title=self.exp_name, xlabel='lifetime (min)' )
        aux.height_profile( [D_avg], Z/1e3, zero=False, fname=self.folder+'/profile_D_'+case+'.pdf', title=self.exp_name, range_l=[D_l], range_r=[D_r], xlabel='D (km)' )
        #aux.height_profile( [n], Z/1e3, zero=False, fname=self.folder+'/profile_n_'+case+'.png',title=self.exp_name, xmin=None, xmax=None, xlabel='Number of thermals' )
        aux.height_profile( [W_avg], Z/1e3, zero=False, fname=self.folder+'/profile_W_'+case+'_talk.pdf', title=None, xlabel='W (m/s)', xmin=0, xmax=8, thin=True, ylabel=False )
        aux.height_profile( [W_avg], Z/1e3, zero=False, fname=self.folder+'/profile_W_'+case+'.pdf', title='W '+case, xlabel='W (m/s)', xmin=None, xmax=None, range_l=[W_l], range_r=[W_r] )
        aux.height_profile( [wmax_avg], Z/1e3, zero=False, fname=self.folder+'/profile_wmax_'+case+'.pdf', title='wmax '+case, xlabel='wmax (m/s)' )
        aux.height_profile( [R_avg*1e-3], Z/1e3, zero=False, fname=self.folder+'/profile_R_'+case+'.png', title=self.exp_name, xlabel='R (km)', xmin=0, xmax=1.2, range_l=[R_l*1e-3], range_r=[R_r*1e-3] )
        aux.height_profile( [loge_avg], Z/1e3, zero=False, fname=self.folder+'/profile_fract_entr_'+case+'.pdf', title=None, xlabel='$\log_{10}(\epsilon)$ $(m^{-1})$', xmin=-4, xmax=-2, range_l=[loge_l], range_r=[loge_r] )
        aux.tracer_mixing( loge, z_centre/1e3, xlabel='$\log_{10}(\epsilon)$ $(m^{-1})$', fname=self.folder+'/profile_fract_entr_'+case+'.pdf', xmin=-5, xmax=-1, mean=loge_avg, Z=Z/1e3, title='Direct method' )
        aux.tracer_mixing( loge, z_centre/1e3, xlabel='$\log_{10}(\epsilon)$ $(m^{-1})$', fname=self.folder+'/profile_fract_entr_'+case+'.png', xmin=-5, xmax=-1, mean=loge_avg, Z=Z/1e3, title='Direct method' )
        aux.height_profile( [massflux_avg], Z/1e3, zero=False, fname=self.folder+'/profile_massflux_'+case+'.pdf', title='Average mass flux '+self.exp_name, xlabel='10$^{5}$ kg m/s', xmin=None, xmax=None , zmin=1, zmax=7.5) #ALEJANDRA zmin/zmaz
        aux.height_profile( [mass_avg*1e-3], Z/1e3, zero=False, fname=self.folder+'/profile_mass_'+case+'.pdf', title='Mass '+self.exp_name, xlabel='10$^{3}$ kg', range_l=[mass_l*1e-3], range_r=[mass_r*1e-3], xmin=None, xmax=None ,zmin=1, zmax=7.5) #ALEJANDRA zmin, zmax)
        valid=np.where(~np.isnan(mse_in_l))[0]
        #aux.height_profile( [tot_mflux*1e-5,n*mass_avg*W_avg*1e-10], [Z/1e3,Z/1e3], label=['M. flux', 'n*$\overline{m}$*$\overline{W}$'], zero=False, fname=self.folder+'/profile_tot_mflux_'+case+'.png', title='Total mass flux '+self.exp_name, xlabel='x 10$^{10}$ kg m/s', xmin=None, xmax=None )
        aux.height_profile( [mse_in_avg[valid]*1e-5, mse_out_avg[valid]*1e-5], [Z[valid]/1e3,Z[valid]/1e3], label=['MSE$_{thermal}$', 'MSE$_{env}$'], zero=False, fname=self.folder+'/profile_mse_'+case+'.png', title=self.exp_name, xlabel='x 10$^{5}$ J/kg', xmin=3.35, xmax=3.55, range_l=[mse_in_l[valid]*1e-5, None], range_r=[mse_in_r[valid]*1e-5, None], filled=True )
        
        aux.height_profile( [entr_rate_avg], Z/1e3, zero=False, fname=self.folder+'/profile_entr_rate'+case+'.pdf', title=None, xlabel='E (kg s$^{-1}$ m$^{-2}$)', xmin=None, xmax=None, range_l=[entr_rate_l], range_r=[entr_rate_r] )

        # for mixing computation sample mse only every 5 steps, otherwise it is too noisy (we need the vertical gradient of mse, so the vertical steps should not be too small!)
        #mse_in_avg = mse_in_avg[::10]
        #mse_out_avg = mse_out_avg[::10]
        #Z0=Z[::10]
        Z0=Z
        mixing_mse = np.ones_like(Z0)*np.nan
        mixing_mse[1:] = (mse_in_avg[1:]-mse_in_avg[:-1])/((Z0[1:]-Z0[:-1])*((mse_out_avg[1:]-mse_in_avg[1:])+(mse_out_avg[:-1]-mse_in_avg[:-1]))*0.5)
        mixing_mse = np.ma.masked_array(mixing_mse, mask= np.isnan(mixing_mse))
        mixing_mse[np.where(np.ma.less_equal(mixing_mse,0))[0]]=np.nan
        mixing_mse[np.where(np.ma.greater(Z0,9000.))[0]]=np.nan
        mixing_mse_nan=np.copy(mixing_mse)
        print( 'entrainment from MSE = %.4f'%(np.nanmean(mixing_mse_nan)))
        mixing_mse[np.where(np.isnan(mixing_mse.data))]=-9999
        log_mixing_mse = np.ma.log10(mixing_mse.data)
        aux.height_profile( [mixing_mse_nan*1e3], Z0/1e3, label=None, zero=False, fname=self.folder+'/profile_mixing_mse_'+case+'.png', title=self.exp_name, xlabel='$\epsilon_{mse}$ (x10$^{-3}$ m$^{-1}$)', xmin=0, xmax=2.5 )
        aux.height_profile( [log_mixing_mse], Z0/1e3, fname=self.folder+'/profile_mixing_mse_'+case+'.png', title='MSE', xlabel='$\log_{10}(\epsilon)$ (m$^{-1}$)', xmin=-5, xmax=-1 )
        

        self.tracer_e_avg = np.ones_like(e_avg)*np.nan # old tracer estimate. Set to nan to avoid confusion!
        self.tracer_e_l = np.ones_like(e_avg)*np.nan 
        self.tracer_e_r = np.ones_like(e_avg)*np.nan 

        aux.corr_plot( R_avg*1e-3, e_avg*1e3, ylabel='$\epsilon$ (x 10$^{-3}$ m$^{-1}$)', xlabel=Rl, fname='profile_R_e', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=0.1, xmax=0.6, ymin=0.2, ymax=3.2, markersize=10, label_regr=True )
        
        aux.corr_plot( 1./(R_avg*1e-3), e_avg*1e3, ylabel='$\epsilon$ (x 10$^{-3}$ m$^{-1}$)', xlabel='R$^{-1}$ (x 10$^{-3}$ m$^{-1}$)', fname='profile_Rinv_e', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=0.6, xmax=5, ymin=0.6, ymax=5, markersize=10, label_regr=True, title=self.exp_name+' profile (direct)' )
        
        try:
            aux.corr_plot( np.log10(R_avg), np.log10(e_avg), ylabel=logel, xlabel='$\log_{10}$(R) (m)', fname='profile_logR_loge', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=2.1, xmax=3., ymin=-3.5, ymax=-2.2, markersize=10 )
        except:
            print('could not plot correlation plot of log10(R_avg) and log10(e_avg)')

        aux.corr_plot( W_avg, e_avg*1e3, ylabel='$\epsilon$ (x 10$^{-3}$ m$^{-1}$)', xlabel=Wl, fname='profile_W_e', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=1.5, xmax=6.5, ymin=0.2, ymax=3.2, markersize=10 )
        try:
            aux.corr_plot( np.log10(W_avg), np.log10(e_avg), ylabel=logel, xlabel='$\log_{10}$(W) (m s$^{-1}$)', fname='profile_logW_loge', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=None, xmax=None, ymin=-3.5, ymax=-2.2, markersize=10 )
        except:
            print('could not plot correlation plot of log10(W_avg) and log10(e_avg)')
       
        B = buoy_avg[np.where(buoy_avg>0)]
        W = W_avg[np.where(buoy_avg>0)]
        e = e_avg[np.where(buoy_avg>0)]
        w = wmax_avg[np.where(buoy_avg>0)]
        #e_tr = self.tracer_e_avg[np.where(buoy_avg>0)]
        aux.corr_plot( B*1e3/(W*W), e*1e3, ylabel='$\epsilon$ (x 10$^{-3}$ m$^{-1}$)', xlabel='B W$^{-2}$ (x 10$^{-3}$ m$^{-1}$)', fname='profile_BW2_e', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=None, xmax=None, ymin=0.2, ymax=3.2, markersize=10 )
        aux.corr_plot( B*1e3/(w*w), e*1e3, ylabel='$\epsilon$ (x 10$^{-3}$ m$^{-1}$)', xlabel='B w$_{max}$$^{-2}$ (x 10$^{-3}$ m$^{-1}$)', fname='profile_Bwmax2_e', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=None, xmax=None, ymin=0.2, ymax=3.2, markersize=10 )

        np.save( self.folder+'/mse_mixing_mean.npy', np.ma.mean(np.ma.masked_array(mixing_mse, mask=np.isnan(mixing_mse))) )


    def plot_histograms( self, vars=None, disc_r=80, Rmax=None, Wmax=None, lifetimemax=None, z0max=None, delta_zmax=None, Fnhmax=None, buoymax=None, Fmixmax=None, accmax=None, emax=None ):
        weights = np.ones_like( self.iz_up[:,3] )*100./len(self.iz_up[:,3])
        bins = np.arange(-self.R_range-self.delta_R,self.R_range+1.5*self.delta_R,self.delta_R)
        aux.histogram_plot( self.iz_up[:,3],     second_data=self.iz_low[:,3],   bins=bins, folder=self.folder, ylabel='Rz', fname='iz_straight', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights, weights_2=weights, xlabel='percent (%)', orientation='horizontal', title=self.exp_name )
        np.savez( self.folder+'/iz_straight_histogram.npz', self.iz_up[:,3], self.iz_low[:,3], bins, weights )
        
        aux.histogram_plot( self.iy_left[:,3],   second_data=self.iy_right[:,3], bins=bins, folder=self.folder, xlabel='Ry', fname='iy_straight', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights,weights_2=weights, ylabel='percent (%)', title=self.exp_name)
        aux.histogram_plot( self.ix_left[:,3],   second_data=self.ix_right[:,3], bins=bins, folder=self.folder, xlabel='Rx', fname='ix_straight', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights,weights_2=weights, ylabel='percent (%)', title=self.exp_name)
        np.savez( self.folder+'/ix_straight_histogram.npz', self.ix_left[:,3], self.ix_right[:,3], bins, weights )

        aux.histogram_plot( self.iz_up[:,2],     second_data=self.iz_low[:,2],   bins=bins, folder=self.folder, ylabel='Rz', fname='iz', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights,weights_2=weights, xlabel='percent (%)', orientation='horizontal' , title=self.exp_name)

        aux.histogram_plot( self.ix_left[:,0],   second_data=self.ix_right[:,0], bins=bins, folder=self.folder, xlabel='Rx', fname='ix', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights,weights_2=weights, ylabel='percent (%)' , title=self.exp_name)
        aux.histogram_plot( self.iy_left[:,1],   second_data=self.iy_right[:,1], bins=bins, folder=self.folder, xlabel='Ry', fname='iy', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights,weights_2=weights, ylabel='percent (%)', title=self.exp_name )

        ix = np.concatenate((self.ix_left[:,3],self.ix_right[:,3]))
        iz = np.concatenate((self.iz_low[:,3],self.iz_up[:,3]))
        ix0 = np.zeros_like(iz)
        iz0 = np.zeros_like(ix)
        izz = np.concatenate((iz0,iz))
        ixx = np.concatenate((ix,ix0))

        delta_R = []
        length = np.zeros(len(self.R_or))
        dt = self.time_or[0][1]-self.time_or[0][0]
        self.lifetime = []
        for i in range(len(self.R_or)):
            length[i]=len(self.R_or[i])*dt
            for j in range(len(self.R_or[i])-1):
                delta_R.append((self.R_or[i][j+1]-self.R_or[i][j])/np.min([self.R_or[i][j+1],self.R_or[i][j]]))
            for j in range(len(self.R_or[i])):
                self.lifetime.append(length[i])
        self.lifetime=np.asarray(self.lifetime)
        self.length = length
        mean_mf=np.nanmean(self.mass_c*self.W_c,axis=1)
        weights_lifetime= mean_mf/np.sum(mean_mf)

        delta_R = np.asarray(delta_R)*100.
        R_range=disc_r
        bins = np.arange(-(R_range+0.5),R_range+0.6,1)
        ticks=np.arange(-R_range,R_range+1,10)
        aux.histogram_plot( x=delta_R, bins=bins, fname='deltaR', xlabel='$\Delta$ R (\%)', folder=self.folder, xmin=-R_range, xmax=R_range, mean=True, xticks=ticks, title=self.exp_name )

        if vars==None:
            time    = self.time           
            wmax    = self.wmax    
            R       = self.R       
            Fmix    = self.Fres    
            buoy    = self.buoy    
            Fnh     = self.Fnh     
            acc     = self.acc     
            D       = self.D       
            mass    = self.mass    
            W        = self.W             
            z_centre= self.z_centre
            Pnz     = self.Pnz     
            #it      = self.it
            e       = self.fract_entr
            mix     = self.mixing_mse
            net_gross_entr = self.net_gross_entr
            net_entr_tsteps= self.net_entr_tsteps
            Fentr = aux.flatten_array(self.net_entr_term)
            fname = ''
            flatten=False
            wghts = W*mass/(np.sum(W*mass))
            wghts_Fentr = []
            for i in range(self.W_c.shape[0]):
                ind = np.where(~np.isnan(self.W_c[i]))[0]
                wghts_Fentr.append(0.5*((self.W_c[i]*self.mass_c[i])[ind][1:]+(self.W_c[i]*self.mass_c[i])[ind][:-1]))
            wghts_Fentr = aux.flatten_array(wghts_Fentr)
            wghts_Fentr = wghts_Fentr/np.sum(wghts_Fentr)
        else:
            time    = vars[0] 
            wmax    = vars[1]
            R       = vars[2]
            Fmix    = vars[3]
            buoy    = vars[4]
            Fnh     = vars[5]
            acc     = vars[6]
            D       = vars[7]
            mass    = vars[8]
            W        = vars[9]
            z_centre= vars[11]
            Pnz     = vars[12]
            #it      = vars[13]
            fname = 'subset_'
            flatten=False

        #radius = aux.flatten_array(R)
        radius  = R
        np.save( self.folder+'/R_mean_pctls.npy', [np.mean(radius), np.percentile(radius,10), np.percentile(radius,90)] )
        thermalW = W
        np.save( self.folder+'/W_mean_pctls.npy', [np.mean(thermalW), np.percentile(thermalW,10), np.percentile(thermalW,90)] )
        peakW = wmax
        np.save( self.folder+'/wmax_mean_pctls.npy', [np.mean(peakW), np.percentile(peakW,10), np.percentile(peakW,90)] )
        entr = np.ma.log10( np.ma.masked_array(e, np.isnan(e)) )
        np.save( self.folder+'/loge_direct_mean_pctls.npy', [np.mean(entr), np.percentile(entr,10), np.percentile(entr,90)] )
        weights = np.ones(len(R))/(self.area*self.simtime)
        if hasattr(self,'y_new'):
            y_units = 'km$^{-2}$ hr$^{-1}$'
        else:
            y_units = 'km$^{-1}$ hr$^{-1}$'
        weights2 = np.ones(len(self.delta_z))/(self.area*self.simtime) 
        weights3 = np.ones(len(self.delta_z))*100./len(self.delta_z)

        weights_Fentr = np.ones(len(Fentr))/(self.area*self.simtime)
       
        prefix = list(self.exp_name)
        for i in range(len(prefix)):
            if prefix[i]==' ':
                prefix[i]='_'
        prefix=''.join(prefix)
        
        np.save( self.folder + '/' + prefix + '_histograms_1.npy', np.vstack((weights,R,W,wmax,e)))
        np.save( self.folder + '/' + prefix + '_histograms_2.npy', np.vstack((weights2,length,self.z0*1e-3,self.delta_z*1e-3)))
        
        bins = np.arange(0.5,np.amax(length)+1.5,1)
        aux.histogram_plot( x=R,  bins=np.logspace(0,4,100), fname=fname+'R_log', xlabel=Rl, folder=self.folder, flatten=flatten, xmin=100, xmax=3000, mean=False, log=True, weights=weights, y_units=y_units, cumulative=False, ymax=Rmax, title='mean radius$\,=\,$%d m'%(int(np.ma.mean(R))), histtype='step' )
        #********************************************************* 'main characteristics of the thermals' *********************************************************
        aux.histogram_plot( x=R*1e-3,  N=100, fname=fname+'R', xlabel=Rl, folder=self.folder, flatten=flatten, xmin=0, xmax=3, mean=False, weights=None, y_units=y_units, cumulative=True, ymax=None, title='mean radius$\,=\,$%d m'%(int(np.ma.mean(R))) )
        print( 'mean radius = %.3f +/- %.3f km'%(np.ma.mean(R*1e-3), np.ma.std(R*1e-3)))
        print( 'mean weighted radius = %.3f km'%(np.ma.average(R*1e-3,weights=wghts)))
        aux.histogram_plot( x=W, bins=np.arange(0.05,14.,0.1), fname=fname+'W', xlabel=Wl, folder=self.folder, flatten=flatten, mean=False, weights=weights, xmin=0, xmax=14, y_units=y_units, cumulative=True, ymax=Wmax, deltay=0.2, title='mean ascent rate$\,=\,$%.1f m$\,$s$^{-1}$'%(np.ma.mean(W)))
        print ('mean ascent rate (W) = %.2f +/- %.2f m/s'%(np.ma.mean(W), np.ma.std(W)))
        print ('mean weighted ascent rate (W) = %.2f m/s'%(np.ma.average(W,weights=wghts)))
        aux.histogram_plot( x=length, bins=bins, xlabel='Thermal lifetime (min)', fname='lifetime', folder=self.folder, xmin=0, xmax=15, mean=False, weights=weights2, y_units=y_units, ymax=lifetimemax, cumulative=True, title='average lifetime = %.1f min'%(np.ma.mean(length)) )
        print ('mean lifetime = %.2f +/- %.2f min'%(np.ma.mean(length), np.ma.std(length)))
        print ('mean weighted lifetime = %.2f min'%(np.ma.average(length,weights=weights_lifetime)))
        aux.histogram_plot( x=self.z0*1e-3, bins=np.arange(0,15.01,0.3), fname=fname+'z0', xlabel='$Z_0 (km)$', folder=self.folder,flatten=False, xmin=0, xmax=15, mean=False, weights=weights2, y_units=y_units, ymax=z0max, cumulative=True, title='average $Z_0$ = %.2f km'%(np.ma.mean(self.z0*1e-3)) )
        aux.histogram_plot( x=self.delta_z*1e-3, bins=np.arange(0,3.61,0.04), fname=fname+'delta_z', xlabel='$\Delta Z (km)$', folder=self.folder,flatten=False, xmin=0, xmax=3.5, mean=False, weights=weights2, y_units=y_units, ymax=delta_zmax, title='average $\Delta Z$ = %d m'%(np.ma.mean(self.delta_z)), cumulative=True )
        print ('mean distance traveled = %.3f +/- %.3f km'%(np.ma.mean(self.delta_z*1e-3),np.ma.std(self.delta_z*1e-3)))
        print ('mean weighted distance traveled = %.3f km'%(np.ma.average(self.delta_z*1e-3,weights=weights_lifetime)))
        aux.histogram_plot( x=self.z0*1e-3, bins=np.arange(0,15.01,0.3), fname=fname+'z0_rotated', ylabel='$Z_0 (km)$', folder=self.folder,flatten=False, xmin=0, xmax=14.1, mean=False, weights=weights2, ymax=z0max, cumulative=False, orientation='horizontal', xlabel='counts '+ y_units, title=None)
        print ('mean starting level = %.2f +/- %.2f km'%(np.ma.mean(self.z0*1e-3),np.ma.std(self.z0*1e-3)))
        print ('mean weighted starting level = %.2f km'%(np.ma.average(self.z0*1e-3,weights=weights_lifetime)))

        #**********************************************************************************************************************************************************

        aux.histogram_plot( x=self.deltazR, N=100, fname=fname+'deltazR', xlabel='$\Delta Z/R$', folder=self.folder,flatten=False, xmin=0, xmax=6, mean=False, weights=weights2, y_units=y_units, title=self.exp_name)
        #limit = np.around( np.amax(np.concatenate((np.abs(Fres),np.abs(Fnh),np.abs(buoy),np.abs(acc)))), decimals=3 )
        limit = 0.2
        dl = 2.*limit/200.
        bins = np.arange(-limit, limit+0.5*dl, dl)
        #********************************************************* 'momentum budget' ******************************************************************************
        aux.histogram_plot( x=Fnh , bins=bins, fname=fname+'Fnh',  folder=self.folder, flatten=flatten, xmin=-0.07, xmax=0.07, zero=True, weights=weights, y_units=y_units, ymax=Fnhmax,  deltax=0.05, cumulative=True, title='Fnh' )
        print ('mean Fnh = %.3f +/- %.3f'%(np.ma.mean(Fnh),np.ma.std(Fnh, ddof=1)))
        print ('mean weighted Fnh = %.3f'%(np.ma.average(Fnh,weights=wghts)))
        aux.histogram_plot( x=buoy, bins=bins, fname=fname+'buoy', folder=self.folder, flatten=flatten, xmin=-0.07, xmax=0.07, zero=True, weights=weights, y_units=y_units, ymax=buoymax, deltax=0.05, cumulative=True, title='Buoyancy' )
        print ('mean buoy = %.3f +/- %.3f'%(np.ma.mean(buoy),np.ma.std(buoy, ddof=1)))
        print ('mean weighted buoy = %.3f'%(np.ma.average(buoy,weights=wghts)))
        aux.histogram_plot( x=Fentr, bins=bins, fname=fname+'Fentr', xlabel='m s$^{-2}$', folder=self.folder, flatten=flatten, xmin=-0.07, xmax=0.07, zero=True, weights=weights_Fentr, y_units=y_units, ymax=Fmixmax, deltax=0.05, cumulative=True, title='Fentr' )
        print ('mean Fentr = %.3f +/- %.3f'%(np.ma.mean(Fentr),np.ma.std(Fentr, ddof=1)))
        print ('mean weighted Fentr = %.4f'%(np.ma.average(Fentr,weights=wghts_Fentr)))
        aux.histogram_plot( x=Fmix, bins=bins, fname=fname+'Fmix', xlabel='m s$^{-2}$', folder=self.folder, flatten=flatten, xmin=-0.07, xmax=0.07, zero=True, weights=weights, y_units=y_units, ymax=Fmixmax, deltax=0.05, cumulative=True, title='Fmix' )
        print( 'mean Fmix = %.3f +/- %.3f'%(np.ma.mean(Fmix),np.ma.std(Fmix, ddof=1)))
        print( 'mean weighted Fmix = %.3f'%(np.ma.average(Fmix,weights=wghts)))
        aux.histogram_plot( x=acc , bins=bins, fname=fname+'acc',  xlabel='m s$^{-2}$', folder=self.folder, flatten=flatten, xmin=-0.07, xmax=0.07, zero=True, weights=weights, y_units=y_units, ymax=accmax,  deltax=0.05, cumulative=True, title='acc' )
        print( 'mean acc = %.3f +/- %.3f'%(np.ma.mean(acc),np.ma.std(acc, ddof=1)))
        print( 'mean weighted acc = %.3f'%(np.ma.average(acc,weights=wghts)))
        #**********************************************************************************************************************************************************

        np.save( self.folder + '/' + prefix + '_histograms_3.npy', np.vstack((weights, buoy, Fnh, Fmix, acc, wghts)) )
        np.save( self.folder + '/' + prefix + '_histograms_Fentr.npy', np.vstack((weights_Fentr, Fentr, wghts_Fentr)) )
        self.Fentr=Fentr
        if vars==None:
            #np.save( self.folder + '/' + prefix + '_histogram_net_entr_gross.npy', np.vstack((weights, net_gross_entr)) )
            bins=np.arange(-8,8.1,0.5)
            aux.histogram_plot( x=net_gross_entr*1e3 , bins=bins, fname=fname+'netgross_entr',  xlabel='x10$^{-3}$m$^{-1}$', folder=self.folder, flatten=flatten, xmin=-8, xmax=8, zero=True, weights=None, y_units='counts', ymax=None,  deltax=2, cumulative=True, title='net gross $\epsilon$', mean=True )
            aux.histogram_plot( x=net_entr_tsteps*1e3, bins=bins, fname=fname+'net_entr',  xlabel='x10$^{-3}$m$^{-1}$', folder=self.folder, flatten=flatten, xmin=-8, xmax=8, zero=True, weights=None, y_units='counts', ymax=None,  deltax=2, cumulative=False, title='net $\epsilon$', mean=True )
            np.save( self.folder + '/' + prefix + '_histogram_net_entr_tsteps.npy', net_entr_tsteps*1e3 )
            print( 'average of net entrainment (per time step): %f'%(np.mean(net_entr_tsteps)))
            print( 'weighted average of net entrainment (per time step): %f'%(np.average(net_entr_tsteps,weights=wghts_Fentr)))
            print( 'average of net entrainment (per thermal): %f'%(np.mean(net_gross_entr)))

        min_mixing = -0.01
        max_mixing = 0.01
        aux.histogram_plot( x=mix     , N=100, fname=fname+'mixing',    xlabel='mse mixing (m$^{-1}$)', folder=self.folder, flatten=flatten, xmin=min_mixing, xmax=max_mixing, zero=True, mean=True, weights=weights, y_units=y_units, title=self.exp_name )
        if vars!=None:
            n, bins = np.histogram( self.D, bins=100 )
            aux.histogram_plot( x=D       , bins=bins, fname=fname+'D',          xlabel=Dl       , folder=self.folder, flatten=flatten, mean=True, title=self.exp_name )
        else:
            loge = entr # np.log10(e)
            aux.histogram_plot( x=D       , N=100, fname=fname+'D',          xlabel=Dl       , folder=self.folder, flatten=flatten, mean=True, title=self.exp_name )
            bins = np.arange(0,7, 0.05)
            aux.histogram_plot( x=e*1e3, bins=bins, fname=fname+'e',  xlabel='$\epsilon$ (x 10$^{-3}$ m$^{-1})$', folder=self.folder, flatten=False, mean=False, xmin=0, xmax=7, weights=weights, y_units=y_units, title=self.exp_name+' (direct method)', cumulative=True, ymax=emax)
            print( 'mean epsilon (direct method) = %.4f'%(np.nanmean(e)))
            nonnan=np.where(~np.isnan(e))[0]
            print( 'weighted mean epsilon (direct method) = %.4f'%(np.average(e[nonnan],weights=wghts[nonnan])))
            #tracer_entr = np.ma.masked_array(self.tracer_entr[:,0], mask=np.isnan(self.tracer_entr[:,0]))
            #aux.histogram_plot( x=tracer_entr*1e3, bins=bins, fname=fname+'e_tracers',  xlabel='$\epsilon$ (x 10$^{-3}$ m$^{-1})$', folder=self.folder, flatten=False, mean=False, xmin=0, xmax=7, weights=weights2, y_units=y_units, title=self.exp_name+' (tracers)', cumulative=True )
            bins = np.arange(-4.0, -1.5, (2.5/100))
            aux.histogram_plot( x=loge, bins=bins, fname=fname+'fract_entr',  xlabel='$\log_{10}(\epsilon) (m^{-1})$', folder=self.folder, flatten=False, mean=True, xmin=-4., xmax=-2., weights=weights, y_units=y_units, title=self.exp_name)#, ymax=1.1 )#, ymax=120 )
            #aux.histogram_plot( x=aux.flatten_array(self.entr_rate), fname=fname+'entr_rate',  xlabel='E (kg s$^{-1}$ m$^{-2})$', folder=self.folder, flatten=False, mean=False, xmin=None, xmax=None, weights=weights, y_units=y_units, title=self.exp_name, cumulative=True, ymax=None)
        aux.histogram_plot( x=mass    , N=100, fname=fname+'mass',       xlabel=massl    , folder=self.folder, flatten=flatten, mean=True, title=self.exp_name )

        aux.histogram_plot( x=z_centre*1e-3, N=80, fname=fname+'z_centre',   xlabel=z_centrel, folder=self.folder, flatten=flatten, mean=False, title=self.exp_name, xmin=0, xmax=15 )
        aux.histogram_plot( x=Pnz     , N=100, fname=fname+'Pnz',        xlabel=Pnzl     , folder=self.folder, flatten=flatten, zero=True, mean=True, title=self.exp_name )
        
        if vars!=None:
            n, bins = np.histogram(self.wmax, bins=100)
            aux.histogram_plot( x=wmax, bins=bins, fname=fname+'wmax',       xlabel=wmaxl    , folder=self.folder, flatten=flatten, mean=False, xmax=25, xmin=0, weights=weights, y_units=y_units, title=self.exp_name, ymax=.6 )
        else:
            aux.histogram_plot( x=wmax    , N=100, fname=fname+'wmax',       xlabel=wmaxl    , folder=self.folder, flatten=flatten, mean=False, xmax=25, xmin=0, weights=weights, y_units=y_units, title=self.exp_name, ymax=.6 )
        aux.histogram_plot( x=time    , N=100, fname=fname+'time',       xlabel='time (min)', folder=self.folder, flatten=flatten, mean=True, title=self.exp_name )
        #bins = np.arange(-15.5,15.6,1)
        #aux.histogram_plot( x=it, bins=bins, fname=fname+'stages',       xlabel='stage (tsteps)', folder=self.folder, flatten=flatten, mean=True )















def include_plots( folder, prefix='buoyancy' ):
    os.system('ls -d '+folder+'/'+prefix+'_frame*.eps > ls.txt')
    file = open('ls.txt', 'r')
    ofile = file.readlines()
    file.close()
    os.system('rm -rf ls.txt')
    for i in range(len(ofile)):
        print (ofile[i][:-1])
    scale = '0.95'+r"'\textwidth'"[1:-1]
    new_lines = []
    for i in range(0,len(ofile)):
        new_lines.append(repr('\includegraphics[width=')[2:-1]+scale+']{'+ofile[i][:-1]+repr('}\\')[1:-1]+'\n')
    return new_lines


def points_grid( x, y, z ):
    """
    creates a matrix with the positions of x and y (useful for scipy griddata interpolations!)
    """
    size = int(len(x)*len(y)*len(z))
    grid = np.zeros([size,3])
    ind = np.unravel_index(np.arange(size), (len(x),len(y),len(z)))
    grid[:,0] = x[ind[0]]
    grid[:,1] = y[ind[1]]
    grid[:,2] = z[ind[2]]
    return grid

def interp_points_RGI_splitvars( var, x_grid, y_grid, hgt_c, new_grid ):
    nx=len(x_grid)
    ny=len(y_grid)
    nz=len(hgt_c)
    ints = new_grid.shape[0]
    new     = np.zeros(ints)
    for l in range(ints):
        i0, i1, j0, j1, k0, k1 = create_smallest_subgrid_for_interpolation_RGI( new_grid[l][0], new_grid[l][1], new_grid[l][2], nx, ny, nz )
        rgi     = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=var[i0:i1,j0:j1,k0:k1], fill_value=None, bounds_error=False )
        new[l]  = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
    return new

def interp_points_RGI( subjob, x_grid, y_grid, hgt_c, new_grid, u_c, v_c, w_c, pdev, latheat, qnice, qncloud, qnrain, cldnuc, rh, qice, qcloud, qrain, qvapor, sctot, noninduc, epotential, qngraupel, qicesnow, qghail):
    nx=len(x_grid)
    ny=len(y_grid)
    nz=len(hgt_c)
    u_new           = np.zeros((subjob[1]-subjob[0]))
    v_new           = np.zeros((subjob[1]-subjob[0]))
    w_new           = np.zeros((subjob[1]-subjob[0]))
    pdev_new        = np.zeros((subjob[1]-subjob[0]))
    latheat_new     = np.zeros((subjob[1]-subjob[0]))
    qnice_new       = np.zeros((subjob[1]-subjob[0]))
    qncloud_new     = np.zeros((subjob[1]-subjob[0]))
    qnrain_new      = np.zeros((subjob[1]-subjob[0]))
    cldnuc_new      = np.zeros((subjob[1]-subjob[0]))
    rh_new          = np.zeros((subjob[1]-subjob[0]))
    qice_new        = np.zeros((subjob[1]-subjob[0]))
    qcloud_new      = np.zeros((subjob[1]-subjob[0]))
    qrain_new       = np.zeros((subjob[1]-subjob[0]))
    qvapor_new      = np.zeros((subjob[1]-subjob[0]))
    sctot_new       = np.zeros((subjob[1]-subjob[0]))
    noninduc_new    = np.zeros((subjob[1]-subjob[0]))
    epotential_new  = np.zeros((subjob[1]-subjob[0]))
    qngraupel_new   = np.zeros((subjob[1]-subjob[0]))
    qicesnow_new    = np.zeros((subjob[1]-subjob[0]))
    qghail_new      = np.zeros((subjob[1]-subjob[0]))
    for l in range(subjob[0],subjob[1]):
        i0, i1, j0, j1, k0, k1 = create_smallest_subgrid_for_interpolation_RGI( new_grid[l][0], new_grid[l][1], new_grid[l][2], nx, ny, nz )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=u_c[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        u_new[l-subjob[0]]      = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=v_c[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        v_new[l-subjob[0]]      = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=w_c[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        w_new[l-subjob[0]]      = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=pdev[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        pdev_new[l-subjob[0]]   = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=latheat[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        latheat_new[l-subjob[0]]= rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=qnice[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        qnice_new[l-subjob[0]]  = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=qncloud[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        qncloud_new[l-subjob[0]]= rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=qnrain[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        qnrain_new[l-subjob[0]] = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=cldnuc[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        cldnuc_new[l-subjob[0]] = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=rh[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        rh_new[l-subjob[0]]     = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=qice[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        qice_new[l-subjob[0]]  = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=qcloud[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        qcloud_new[l-subjob[0]]= rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=qrain[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        qrain_new[l-subjob[0]] = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=qvapor[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        qvapor_new[l-subjob[0]] = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=sctot[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        sctot_new[l-subjob[0]]= rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=noninduc[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        noninduc_new[l-subjob[0]] = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=epotential[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        epotential_new[l-subjob[0]]= rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=qngraupel[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        qngraupel_new[l-subjob[0]] = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=qicesnow[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        qicesnow_new[l-subjob[0]] = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        rgi      = pol.RegularGridInterpolator( points=[np.arange(i0,i1),np.arange(j0,j1),np.arange(k0,k1)], values=qghail[i0:i1,j0:j1,k0:k1],fill_value=None,bounds_error=False )
        qghail_new[l-subjob[0]] = rgi( (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
    
    return u_new, v_new, w_new, pdev_new, latheat_new, qnice_new, qncloud_new, qnrain_new, cldnuc_new, rh_new, qice_new, qcloud_new, qrain_new, qvapor_new, sctot_new, noninduc_new, epotential_new, qngraupel_new, qicesnow_new, qghail_new

def interp_points( subjob, x_grid, y_grid, hgt_c, new_grid, u_c, v_c, w_c, pdev ):
    nx=len(x_grid)
    ny=len(y_grid)
    nz=len(hgt_c)
    u_new       = np.zeros((subjob[1]-subjob[0]))
    v_new       = np.zeros((subjob[1]-subjob[0]))
    w_new       = np.zeros((subjob[1]-subjob[0]))
    pdev_new    = np.zeros((subjob[1]-subjob[0]))
    for l in range(subjob[0],subjob[1]):
        subgrid, i0, i1, j0, j1, k0, k1 = create_smallest_subgrid_for_interpolation( new_grid[l][0], new_grid[l][1], new_grid[l][2], nx, ny, nz )
        u_new[l-subjob[0]]      = pol.griddata( subgrid, u_c[i0:i1,j0:j1,k0:k1].flatten(), (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        v_new[l-subjob[0]]      = pol.griddata( subgrid, v_c[i0:i1,j0:j1,k0:k1].flatten(), (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        w_new[l-subjob[0]]      = pol.griddata( subgrid, w_c[i0:i1,j0:j1,k0:k1].flatten(), (new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
        pdev_new[l-subjob[0]]   = pol.griddata( subgrid, pdev[i0:i1,j0:j1,k0:k1].flatten(),(new_grid[l][0], new_grid[l][1], new_grid[l][2]) )
    
    return u_new, v_new, w_new, pdev_new


def angles_grid( alpha, phi ):
   size = int(len(alpha)*len(phi))
   grid = np.zeros([size,2])
   ind = np.unravel_index(np.arange(size),(len(alpha), len(phi)))
   grid[:,0] = alpha[ind[0]]
   grid[:,1] = phi[ind[1]]
   return grid

def angles_to_xyz( angles, R=1. ):
    """
    convert a list of [alpha,phi] angles to [x,y,z] coordinates
    where alpha is 'longitude' and phi 'latitude' on a sphere.
    """
    xyz = np.zeros([len(angles),3])
    xyz[:,0] = np.cos(angles[:,1].astype(float))*np.cos(angles[:,0].astype(float))
    xyz[:,1] = np.cos(angles[:,1].astype(float))*np.sin(angles[:,0].astype(float))
    xyz[:,2] = np.sin(angles[:,1].astype(float))
    return xyz*R


def create_smallest_subgrid_for_interpolation_RGI( ix_c, iy_c, iz_c, nx, ny, nz ):
    ixlow = int(np.max([0,np.rint(ix_c)-1]))
    iylow = int(np.max([0,np.rint(iy_c)-1]))
    izlow = int(np.max([0,np.rint(iz_c)-1]))
    ixhigh = np.amin([ixlow+3, nx])
    iyhigh = np.amin([iylow+3, ny])
    izhigh = np.amin([izlow+3, nz])
    dimx=ixhigh-ixlow
    dimy=iyhigh-iylow
    dimz=izhigh-izlow
    return ixlow, ixhigh, iylow, iyhigh, izlow, izhigh

def create_smallest_subgrid_for_interpolation( ix_c, iy_c, iz_c, nx, ny, nz ):
    ixlow = int(np.max([0,np.rint(ix_c)-1]))
    iylow = int(np.max([0,np.rint(iy_c)-1]))
    izlow = int(np.max([0,np.rint(iz_c)-1]))
    ixhigh = np.amin([ixlow+3, nx])
    iyhigh = np.amin([iylow+3, ny])
    izhigh = np.amin([izlow+3, nz])
    dimx=ixhigh-ixlow
    dimy=iyhigh-iylow
    dimz=izhigh-izlow
    subgrid = index_grid(dimx,dimy,dimz,x0=ixlow,y0=iylow,z0=izlow)
    return subgrid, ixlow, ixhigh, iylow, iyhigh, izlow, izhigh

def index_grid( nx, ny, nz, x0=0, y0=0, z0=0, ndivx=1., ndivy=1., ndivz=1. ):
    """
    creates a matrix with the indices of a nx*ny*nz grid (useful for scipy griddata interpolations!)
    """
    size = int(nx*ny*nz)
    grid = np.zeros([size,3])
    ind = np.unravel_index(np.arange(size),(nx,ny,nz))
    grid[:,0] = ind[0]/ndivx + x0
    grid[:,1] = ind[1]/ndivy + y0
    grid[:,2] = ind[2]/ndivz + z0
    return grid

def _correct_for_nearby_thermal( w_plume, iz2_int ):
    w_plume_filtered = filters.maximum_filter( w_plume[:,0], 8 )
    maxima = w_plume_filtered==w_plume[:,0]
    max_ind = np.where(maxima)[0]
    w_plume_filtered = filters.minimum_filter( w_plume[:,0], 8 )
    minima = (w_plume_filtered==w_plume[:,0])*(w_plume[:,3]!=-4.)
    min_ind = np.where(minima)[0]
    edge=iz2_int
    return edge
