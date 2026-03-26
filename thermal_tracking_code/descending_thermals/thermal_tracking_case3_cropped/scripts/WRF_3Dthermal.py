import numpy as np
import scipy as sp
from scipy import stats
from scipy import interpolate
import scipy.io as io
import scipy.interpolate as pol
from joblib import Parallel, delayed
import os
import pdb
import time
import scipy.ndimage.filters as filters
from pylab import *
import gc

"""
*************************************
CONSTANTS USED (CONSISTENT WITH WRF)
*************************************
"""

Rd   = 287.                         # Gas constant for dry air, J/(kg K)
Rv   = 461.6                        # Gas constant for water vapor, J/(kg K)
cp   = 7.*0.5*Rd                    # Specific heat of dry air at constant pressure, J/(kg K)
cv   = cp - Rd                      # Specific heat of dry air at constant volume, J/(kg K)
pref = 100000.0                     # reference sea level pressure, Pa
g    = 9.81                         # gravitational constant (m/s^2)

prec = np.float64                   # precision of the arrays (np.float32 or np.float64)


int_method = 'linear'               # interpolation method for scipy.interpolate.griddata

class Thermal(object):

    def __init__( self, max_radius, grid, dx=100, dt=60, t00=0, prev_thermal=None, W_min=1., min_thermal_duration=4, avg_dist_R=5., min_R=200., max_steps=2, disc_r=0.6, n_jobs=1, split_time=True, shifted=0., parallel_thermals=False, up=False, cell=''):
        self.cell = cell
        self.parallel_thermals=parallel_thermals
        self.shifted    = shifted
        self.split_time = split_time            # for parallel surface interpolation (surface integrals). If True, splits time, if False, splits the sphere into n_jobs.
        if self.parallel_thermals:
            self.n_jobs     = 1
        else:
            self.n_jobs     = n_jobs
        self.disc_r     = disc_r
        self.max_steps  = max_steps
        self.min_t      = min_thermal_duration
        self.avg_dist_R = avg_dist_R
        self.split      = False            # true if the thermal originates from a split thermal
        self.grid       = grid
        self.nx         = self.grid.nx
        self.ny         = self.grid.ny
        self.nz         = self.grid.nz
        self.x0         = self.grid.x0
        self.y0         = self.grid.y0
        self.W_min      = W_min
        self.dt         = dt
        self.dx         = dx
        self.dy         = dx
        self.max_radius = max_radius
        self.min_R      = min_R
    
        self.hgt        = self.grid.hgt
        self.hgt_c      = self.grid.hgt_c
        self.x_grid     = self.grid.x_grid
        self.y_grid     = self.grid.y_grid
        self.dh         = self.grid.dh
        self.dh_c       = self.grid.dh_c
        self.tsteps     = self.grid.tsteps[t00:].astype(int)
        self.nt         = len(self.tsteps)
        self.time       = (self.tsteps*self.dt)/60. + self.grid.hr0*60 + self.grid.min0    + self.grid.sec0/60.# time in minutes where the thermal is tracked
        self.YY0        = self.grid.YY0 # initial date of the tracking
        self.MM0        = self.grid.MM0
        self.DD0        = self.grid.DD0
        
        self.xmax         = self.grid.xmax[t00:]
        self.ymax         = self.grid.ymax[t00:]
        self.hmax         = self.grid.hmax[t00:]
        self.x_centre     = self.grid.x_centre[t00:]
        self.y_centre     = self.grid.y_centre[t00:]
        self.z_centre     = self.grid.z_centre[t00:]
        self.ix_centre    = self.grid.ix_centre[t00:]
        self.iy_centre    = self.grid.iy_centre[t00:]
        self.iz_centre    = self.grid.iz_centre[t00:]
        self.u_thermal    = self.grid.u_thermal[t00:]
        self.v_thermal    = self.grid.v_thermal[t00:]
        self.w_thermal    = self.grid.w_thermal[t00:]
        self.x_centregrid = self.x_centre[:] - self.grid.x0*1e3
        self.y_centregrid = self.y_centre[:] - self.grid.y0*1e3
        self.z_centregrid = np.copy(self.z_centre[:])
        
        self.rho_c         = self.grid.rho_c[:,:,:,self.tsteps]
        self.rho_m         = self.grid.rho_m[:,:,:,self.tsteps]
        #self.rho_condensate     = self.grid.rho_condensate[:,:,:,self.tsteps]
        self.qvapor             = self.grid.qvapor[:,:,:,self.tsteps] #DH24.07.2017
        self.sctot              = self.grid.sctot[:,:,:,self.tsteps] #Toshi 11.12.2022
        self.latheat            = self.grid.latheat[:,:,:,self.tsteps] #DHD 26.07.2019
        self.qnice              = self.grid.qnice[:,:,:,self.tsteps] #DHD 20.09.2019
        self.qncloud            = self.grid.qncloud[:,:,:,self.tsteps] #DHD 20.09.2019
        self.qnrain             = self.grid.qnrain[:,:,:,self.tsteps] #DHD 20.09.2019
        self.noninduc           = self.grid.noninduc[:,:,:,self.tsteps] #Toshi 11.12.2022
        self.cldnuc             = self.grid.cldnuc[:,:,:,self.tsteps] #DHD 23.07.2020
        self.qice               = self.grid.qice[:,:,:,self.tsteps] #DHD 12.10.2020
        self.qcloud             = self.grid.qcloud[:,:,:,self.tsteps] #DHD 12.10.2020
        self.qrain              = self.grid.qrain[:,:,:,self.tsteps] #DHD 12.10.2020
        self.rh                 = self.grid.rh[:,:,:,self.tsteps] #DHD 23.07.2020
        self.u_c                = self.grid.u_c[:,:,:,self.tsteps]
        self.v_c                = self.grid.v_c[:,:,:,self.tsteps]
        self.w_c                = self.grid.w_c[:,:,:,self.tsteps]
        self.ptot               = self.grid.ptot[:,:,:,self.tsteps]
        self.mse                = self.grid.mse[:,:,:,self.tsteps]
        self.epotential         = self.grid.epotential[:,:,:,self.tsteps] #Toshi 08.2023
        self.qngraupel          = self.grid.qngraupel[:,:,:,self.tsteps] #Toshi 08.2023
        self.qicesnow           = self.grid.qicesnow[:,:,:,self.tsteps] #DHD 20.09.2019
        self.qghail             = self.grid.qghail[:,:,:,self.tsteps] #DHD 20.09.2019

        self.log                = []
        self.up                 = up
        if prev_thermal!=None:
            self.cell           = prev_thermal.cell
            self.shifted        = prev_thermal.shifted
            self.split          = True
            self.W_min          = prev_thermal.W_min
            self.tsteps         = prev_thermal.tsteps[t00:].astype(int)
            self.nt             = len(self.tsteps)
            self.i_left         = prev_thermal.i_left            [t00:]
            self.i_right        = prev_thermal.i_right           [t00:]
            self.k0             = prev_thermal.k0                [t00:]
            self.k1             = prev_thermal.k1                [t00:]
            self.nnx            = prev_thermal.nnx               [t00:]    
            self.nny            = prev_thermal.nny               [t00:]
            self.nnz            = prev_thermal.nnz               [t00:]
            self.subgrid        = prev_thermal.subgrid           [t00:]
            self.dz_here        = prev_thermal.dz_here           [t00:] 
            self.rho_here       = prev_thermal.rho_here          [t00:]
            #self.rho_condensate_here= prev_thermal.rho_condensate_here  [t00:]
            self.qvapor_here    = prev_thermal.qvapor_here          [t00:] #DH24.07.2017
            self.rho_m_here     = prev_thermal.rho_m_here        [t00:]
            self.u_c_here       = prev_thermal.u_c_here          [t00:]
            self.v_c_here       = prev_thermal.v_c_here          [t00:]
            self.w_c_here       = prev_thermal.w_c_here          [t00:]
            self.sctot_here     = prev_thermal.sctot_here        [t00:] #Toshi 11.12.2022 
            self.latheat_here   = prev_thermal.latheat_here      [t00:] #DHD26.07.2019
            self.qnice_here     = prev_thermal.qnice_here        [t00:] #DHD20.09.2019
            self.qncloud_here   = prev_thermal.qncloud_here      [t00:] #DHD20.09.2019
            self.qnrain_here    = prev_thermal.qnrain_here       [t00:] #DHD20.09.2019
            self.noninduc_here  = prev_thermal.noninduc_here     [t00:] #Toshi 11.12.2022
            self.cldnuc_here    = prev_thermal.cldnuc_here       [t00:] #DHD23.07.2020
            self.qice_here      = prev_thermal.qice_here         [t00:] #DHD12.10.2020 
            self.qcloud_here    = prev_thermal.qcloud_here       [t00:] #DHD12.10.2020
            self.qrain_here     = prev_thermal.qrain_here        [t00:] #DHD12.10.2020
            self.rh_here        = prev_thermal.rh_here           [t00:] #DHD23.07.2020
            self.epotential_here= prev_thermal.epotential_here   [t00:] #Toshi08.2023
            self.qngraupel_here = prev_thermal.qngraupel_here    [t00:] #Toshi08.2023
            self.qicesnow_here  = prev_thermal.qicesnow_here     [t00:] #DHD20.09.2019
            self.qghail_here    = prev_thermal.qghail_here       [t00:] #DHD20.09.2019
            self.x_subgrid      = prev_thermal.x_subgrid         [t00:]
            self.y_subgrid      = prev_thermal.y_subgrid         [t00:]
            self.z_subgrid      = prev_thermal.z_subgrid         [t00:]
    
            self.R_thermal      = prev_thermal.R_thermal        [t00:]
            self.PnzdS          = prev_thermal.PnzdS            [t00:] 
            self.zmomflux       = prev_thermal.zmomflux         [t00:]
            self.entr_dist      = prev_thermal.entr_dist        [t00:]
            self.detr_dist      = prev_thermal.detr_dist        [t00:]
            self.epsilon        = prev_thermal.epsilon          [t00:]
            self.entr_rate      = prev_thermal.entr_rate        [t00:]
            self.PnxdS          = prev_thermal.PnxdS            [t00:]
            self.PnydS          = prev_thermal.PnydS            [t00:]
            self.xmomflux       = prev_thermal.xmomflux         [t00:]
            self.ymomflux       = prev_thermal.ymomflux         [t00:]
            self.w_mean         = prev_thermal.w_mean           [t00:]
            self.v_mean         = prev_thermal.v_mean           [t00:]
            self.u_mean         = prev_thermal.u_mean           [t00:]
            self.buoy           = prev_thermal.buoy             [t00:]
            self.buoy_m         = prev_thermal.buoy_m           [t00:]
            self.mass           = prev_thermal.mass             [t00:]
            self.volume         = prev_thermal.volume           [t00:]
            self.volume_err     = prev_thermal.volume_err       [t00:]
            #self.mass_cond      = prev_thermal.mass_cond        [t00:]
            self.massflux       = prev_thermal.massflux         [t00:]
            self.data           = prev_thermal.data             [t00:]
            self.integrals      = prev_thermal.integrals        [t00:]
            self.R_ord_data_ind = prev_thermal.R_ord_data_ind   [t00:]
            self.angles         = prev_thermal.angles           [t00:]
            self.entr_distr     = prev_thermal.entr_distr       [t00:]
            self.mse_thermal    = prev_thermal.mse_thermal      [t00:]
            self.sctot_thermal  = prev_thermal.sctot_thermal  [t00:]
            self.maxsctot_thermal   = prev_thermal.maxsctot_thermal  [t00:]
            self.latheat_thermal    = prev_thermal.latheat_thermal  [t00:]
            self.maxlatheat_thermal = prev_thermal.maxlatheat_thermal[t00:]
            self.qnice_thermal      = prev_thermal.qnice_thermal     [t00:]
            self.qncloud_thermal    = prev_thermal.qncloud_thermal   [t00:]
            self.qnrain_thermal     = prev_thermal.qnrain_thermal    [t00:]
            self.noninduc_thermal   = prev_thermal.noninduc_thermal  [t00:]
            self.cldnuc_thermal     = prev_thermal.cldnuc_thermal    [t00:]
            self.rh_thermal         = prev_thermal.rh_thermal        [t00:]
            self.qice_thermal       = prev_thermal.qice_thermal      [t00:]
            self.qcloud_thermal     = prev_thermal.qcloud_thermal    [t00:]
            self.qvapor_thermal     = prev_thermal.qvapor_thermal    [t00:]
            self.qrain_thermal      = prev_thermal.qrain_thermal     [t00:]
            self.net_entr_term      = prev_thermal.net_entr_term     [t00:]
            self.epotential_thermal = prev_thermal.epotential_thermal[t00:]
            self.maxepotential_thermal = prev_thermal.maxepotential_thermal[t00:]
            self.qngraupel_thermal  = prev_thermal.qngraupel_thermal [t00:]
            self.qicesnow_thermal   = prev_thermal.qicesnow_thermal [t00:]
            self.qghail_thermal     = prev_thermal.qghail_thermal   [t00:]
            self.up                 = prev_thermal.up
        if self.up:
            self.OBJ ='thermal'
        else:
            self.OBJ ='Downdraft'
        self._load()


    def _load(self):
        skip=False
        if os.path.isfile('oldlog.out'): # if a previous run of tracking code exited before finishing, make sure you rename the old log.out file to oldlog.out and this will avoid re-computing.
            logfile=file('oldlog.out','r')
            i=0
            lines = logfile.readlines()
            for line in lines:
                if line.strip()=='Computing '+self.OBJ+' based on w-peaks from ' + hhmm(self.time[0]) + ' until ' + hhmm(self.time[-1]) + ' (x0=%d m, y0=%d m, z0=%d m)'%(self.x_centre[0],self.y_centre[0],self.z_centre[0]):
                    print (self.OBJ,'based on w-peaks from ' + hhmm(self.time[0]) + ' until ' + hhmm(self.time[-1]) +' (x0=%d m, y0=%d m, z0=%d m)'%(self.x_centre[0],self.y_centre[0],self.z_centre[0]) +' has already been computed. Will skip it!')
                    skip=True
                    j=i+1
                    while j<(len(lines)-1) and lines[j]!='\n':
                        j+=1
                    old_log=lines[i:j]
                i+=1
            logfile.close()
        if not skip:
            self.log.append( '\nComputing '+self.OBJ+' based on w-peaks from ' + hhmm(self.time[0]) + ' until ' + hhmm(self.time[-1]) + ' (x0=%d m, y0=%d m, z0=%d m)'%(self.x_centre[0],self.y_centre[0],self.z_centre[0]) )
            close2edges = self._load_step1()
            if not(close2edges):
                self._check_for_discont_r( recompute_vel=True, threshold=self.disc_r )
                self._check_min_W()
                if self.nt >= self.min_t and self._W_condition():
                    self.log.append( self.OBJ+' is ok from '+ hhmm(self.time[0]) + ' until ' + hhmm(self.time[-1]))
                    self._check_for_multiple_thermals()
                    self._check_for_discont_r( threshold=self.disc_r )
                    self._check_consistency( large_slow=True )
                    if self.nt >= self.min_t:
                        self._get_mean_W_env()
                        self._check_min_W()
                        if self.nt >= self.min_t and self._W_condition():
                            self._get_buoyancy()
                            self._compute_expected_trajectory()
                            self._check_mom_budget_fit()
                            self._check_always_rising()
                            if self.nt >= self.min_t and self._W_condition():
                                self._get_mse_environment()
                                self._compute_mse_mixing()
                                self._create_folder_and_fname()
                                self._write_data_to_file()
                                #make_plots( fname=self.fname, pdf=False )
                                self.log.append('Computed '+self.OBJ+' : ' + self.fname+'\n')
                if self.nt<self.min_t:
                    self.log.append( self.OBJ +' does not live long enough (%01d tsteps)\n'%(self.nt) )
                elif ((self.up and (self._rel_vel() < self.W_min)) or ((not(self.up)) and (self._rel_vel() > self.W_min))):
                    self.log.append( self.OBJ +' is too slow (W=%.2f)'%(self._rel_vel()) )
                    if hasattr(self, 'mean_W_env'):
                        self.log.append( '(W_env=%.2f)\n'%(np.mean(self.mean_W_env)) )
            else:
                self.log.append('\n')
        else:
            self.log.append('\n')
            for line in old_log:
                self.log.append(line.strip())
            #self.log.append('\n')

    def _load_step1( self ):
        start=time.time()
        close2edges = self._create_subgrid()
        if not(close2edges):
            self._get_centre_values()
            self._select_subgrid_data()
            self._catalogue_data()
            if not self.split:
                self._get_r_from_w()
                self._check_consistency(var=self.R_thermal)
                if self.nt >= self.min_t:    
                    self._get_vol_integrals_r()
                    self._compute_surface_integrals()
                    self._check_consistency()
        return close2edges

    def _shift_thermal_down( self, ecc=1.05 ):
        self.z_centre = self.z_centre - np.sqrt(ecc*ecc-1.)*self.R_thermal
        self.iz_centre = np.interp(self.z_centre, self.hgt_c, np.arange(self.nz))
        self.z_centregrid = np.copy(self.z_centre)
        close2edges=self._create_subgrid()
        self._get_centre_values()
        self._select_subgrid_data()
        self._catalogue_data()
        self._get_r_from_w()

    def _get_mse_environment( self, R_range = 0.5 ):
        """
        Here we compute the moist (and generalized) static energy of the environment of the thermal (around it, a distance R_range*R_thermal from it)
        """
        mse_out = np.zeros(len(self.R_thermal))
        for it in range(len(self.R_thermal)):
            dV          = np.outer(np.ones(len(self.x_grid)*len(self.y_grid))*self.dx*self.dy, self.dh).reshape([len(self.x_grid),len(self.y_grid),len(self.hgt_c)])
            mass_gbox   = self.rho_c[:,:,:,it]*dV
            # get mse in the environment of the thermal at t=it
            indxleft    = np.where((self.x_grid>self.x_centre[it]-(1+R_range)*self.R_thermal[it])*(self.x_grid<=self.x_centre[it]-self.R_thermal[it]))[0]
            indxright   = np.where((self.x_grid<self.x_centre[it]+(1+R_range)*self.R_thermal[it])*(self.x_grid>=self.x_centre[it]+self.R_thermal[it]))[0]
            indx1       = np.concatenate((indxleft,indxright))
            indyleft    = np.where((self.y_grid>self.y_centre[it]-(1+R_range)*self.R_thermal[it])*(self.y_grid<=self.y_centre[it]-self.R_thermal[it]))[0]
            indyright   = np.where((self.y_grid<self.y_centre[it]+(1+R_range)*self.R_thermal[it])*(self.y_grid>=self.y_centre[it]+self.R_thermal[it]))[0]
            indy1       = np.concatenate((indyleft,indyright))
            indz1       = np.where((self.hgt_c<=self.z_centre[it]+self.R_thermal[it])*(self.hgt_c>=self.z_centre[it]-self.R_thermal[it]))[0]
            mse_slice1  = self.mse[indx1,:,:,it][:,indy1,:][:,:,indz1].flatten()
            mass1       = mass_gbox[indx1,:,:][:,indy1,:][:,:,indz1].flatten()
            indx2       = np.where((self.x_grid>self.x_centre[it]-(1+R_range)*self.R_thermal[it])*(self.x_grid<self.x_centre[it]+(1+R_range)*self.R_thermal[it]))[0]
            indy2       = np.where((self.y_grid>self.y_centre[it]-(1+R_range)*self.R_thermal[it])*(self.y_grid<self.y_centre[it]+(1+R_range)*self.R_thermal[it]))[0]
            indz2       = np.where((self.hgt_c>(self.z_centre[it]+self.R_thermal[it]))*(self.hgt_c<=self.z_centre[it]+(1+R_range)*self.R_thermal[it]))[0]
            indz3       = np.where((self.hgt_c<(self.z_centre[it]-self.R_thermal[it]))*(self.hgt_c>=self.z_centre[it]-(1+R_range)*self.R_thermal[it]))[0]
            mse_slice2  = self.mse[indx2,:,:,it][:,indy2,:][:,:,indz2].flatten()
            mass2       = mass_gbox[indx2,:,:][:,indy2,:][:,:,indz2].flatten()
            mse_slice3  = self.mse[indx2,:,:,it][:,indy2,:][:,:,indz3].flatten()
            mass3       = mass_gbox[indx2,:,:][:,indy2,:][:,:,indz3].flatten()
            mass_env    = np.concatenate((np.concatenate((mass1,mass2)),mass3))
            mse_out[it] = np.sum(np.concatenate((np.concatenate((mse_slice1,mse_slice2)),mse_slice3))*mass_env)/np.sum(mass_env)
        self.mse_env = mse_out


    def _compute_mse_mixing(self, R_range = 0.5):
        """
        This function computes the mixing due to changes in the moist static energy of the thermal from one timestep to the next. However, it assumes homogeneous distribution inside (and outside) of the thermal, so it might not be useful at all!
        """
        self.mixing_mse = np.zeros(len(self.R_thermal)-1)
        self.mixing_mse[:] = (self.mse_thermal[1:]-self.mse_thermal[:-1])/(0.5*(self.w_mean[:-1]+self.w_mean[1:])*self.dt*(self.mse_env[:-1]-self.mse_thermal[:-1]))


    def _create_subgrid( self, fwd_tr=False, bkwd_tr=False ):
        """
        create the subgrid at each time step (or at the last one, if tracing points forward or backward), using the center of the thermal and the maximum radius
        """
        too_close_to_edges=False
        if fwd_tr and bkwd_tr:
            print( 'Error: cannot create subgrid for fwd and bkwd tracing at the same time!' )
        else:
            self.i_left = ((self.x_centregrid[:]-self.max_radius)/self.dx).astype(int) - 1
            self.i_right = ((self.x_centregrid[:]+self.max_radius)/self.dx).astype(int) + 1
            self.j_left = ((self.y_centregrid[:]-self.max_radius)/self.dy).astype(int) - 1
            self.j_right = ((self.y_centregrid[:]+self.max_radius)/self.dy).astype(int) + 1
            t0 = 0
            tf = self.nt
            self.k0 = np.zeros(self.nt)
            self.k1 = np.zeros(self.nt)
            if np.any(self.i_left<0) or np.any(self.j_left<0) or np.any(self.i_right>=self.nx) or np.any(self.j_right>=self.ny):
                # this thermal is too close to the domain edges. It will be skipped!
                too_close_to_edges=True
                self.log.append( self.OBJ+' is too close to one of the domain edges. It will be skipped.' )
            for it in range(t0,tf):
                self.i_left[it] = np.amax([0,self.i_left[it]])
                self.i_right[it] = np.amin([self.nx-1,self.i_right[it]])
                self.j_left[it] = np.amax([0,self.j_left[it]])
                self.j_right[it] = np.amin([self.ny-1,self.j_right[it]])
                k00 = np.where(self.hgt_c < self.z_centregrid[it] - self.max_radius)
                if k00[0].size != 0:
                      self.k0[it] = np.amax([np.amax(k00) - 1, 1])
                else:
                    self.k0[it] = 1
                if len(np.where(self.hgt_c[:] > self.z_centregrid[it] + self.max_radius)[0]) > 0:
                    self.k1[it] = np.amin([np.amin(np.where(self.hgt_c[:] > self.z_centregrid[it] + self.max_radius)) + 1, self.nz-2])
                else:
                    self.k1[it] = self.nz - 2
                    if not(too_close_to_edges):
                        self.log.append( 'Warning: domain is not high enough! (subgrid will be smaller than expected)' )
                    #print 'Warning: domain is not high enough! (subgrid will be smaller than expected)' 
            self.k0  = self.k0.astype(int)
            self.k1  = self.k1.astype(int)
            #make sure k0 and k1 are the same for all timesteps:
            self.k1[self.k1<np.max(self.k1)]=np.max(self.k1)
            self.k0[self.k0>np.min(self.k0)]=np.min(self.k0)
            self.nnx = (self.i_right - self.i_left + 1).astype(int)              # number of gridpoints of original grid where the circle is contained (1 additional point on each side)
            self.nny = (self.j_right - self.j_left + 1).astype(int)
            self.nnz = (self.k1 - self.k0 + 1).astype(int)
            self.subgrid = []
            self.x_subgrid = []
            self.y_subgrid = []
            self.z_subgrid = []
            for it in range(t0,tf):
                self.subgrid.append( index_grid(self.nnx[it],self.nny[it],self.nnz[it],x0=self.i_left[it],y0=self.j_left[it],z0=self.k0[it]) )
                self.x_subgrid.append( self.x_grid[self.i_left[it]:self.i_right[it]+1] )
                self.y_subgrid.append( self.y_grid[self.j_left[it]:self.j_right[it]+1] )
                self.z_subgrid.append( self.hgt_c[self.k0[it]:self.k1[it]+1] )
        return too_close_to_edges
                
    def _get_centre_values( self, fwd_tr=False, bkwd_tr=False ):
        w_centre = []
        v_centre = []
        u_centre = []
        t0 = 0
        tf = self.nt
        for it in range(t0,tf):
            # do not interpolate if the smoothed centre coincides with a grid point:
            if (np.abs(self.ix_centre[it]-np.rint(self.ix_centre[it]))<=1e-10) and (np.abs(self.iy_centre[it]-np.rint(self.iy_centre[it]))<=1e-10) and (np.abs(self.iz_centre[it]-np.rint(self.iz_centre[it]))<=1e-10):
                wc = self.w_c[int(np.rint(self.ix_centre[it])),int(np.rint(self.iy_centre[it])),int(np.rint(self.iz_centre[it])),it]
                uc = self.u_c[int(np.rint(self.ix_centre[it])),int(np.rint(self.iy_centre[it])),int(np.rint(self.iz_centre[it])),it]
                vc = self.v_c[int(np.rint(self.ix_centre[it])),int(np.rint(self.iy_centre[it])),int(np.rint(self.iz_centre[it])),it]
            else:
                # extract a small subgrid 4x4x4 around the centres:
                subgrid, ixlow, ixhigh, iylow, iyhigh, izlow, izhigh = create_small_subgrid_for_interpolation( self.ix_centre[it], self.iy_centre[it], self.iz_centre[it], self.nx, self.ny, self.nz, dim=4 )
                
                wc = pol.griddata( subgrid, self.w_c[ixlow:ixhigh, iylow:iyhigh, izlow:izhigh,it].flatten(), (self.ix_centre[it], self.iy_centre[it], self.iz_centre[it]), method='linear' )
                uc = pol.griddata( subgrid, self.u_c[ixlow:ixhigh, iylow:iyhigh, izlow:izhigh,it].flatten(), (self.ix_centre[it], self.iy_centre[it], self.iz_centre[it]), method='linear' )
                vc = pol.griddata( subgrid, self.v_c[ixlow:ixhigh, iylow:iyhigh, izlow:izhigh,it].flatten(), (self.ix_centre[it], self.iy_centre[it], self.iz_centre[it]), method='linear' )
            w_centre.append( wc )
            u_centre.append( uc )
            v_centre.append( vc )
        self.w_centre = np.asarray(w_centre)
        self.u_centre = np.asarray(u_centre)
        self.v_centre = np.asarray(v_centre)


    def _select_subgrid_data( self ):
        nt = self.nt
        self.rho_m_here     = []
        self.qvapor_here    = [] #DH24.07.2017
        self.w_c_here       = []
        self.u_c_here       = []
        self.v_c_here       = []
        self.rho_here       = []
        self.dz_here        = []
        self.mse_here       = []
        self.sctot_here     = [] #Toshi 11.12.2022
        self.latheat_here   = [] #DHD26.07.2019
        self.qnice_here     = [] #DHD20.09.2019
        self.qncloud_here   = [] #DHD20.09.2019
        self.qnrain_here    = [] #DHD20.09.2019
        self.noninduc_here  = [] #Toshi 11.12.2022
        self.cldnuc_here    = [] #DHD23.07.2020
        self.rh_here        = [] #DHD23.07.2020
        self.qice_here      = [] #DHD20.09.2019
        self.qcloud_here    = [] #DHD20.09.2019
        self.qrain_here     = [] #DHD20.09.2019
        self.epotential_here= [] #Toshi 08.2023
        self.qngraupel_here = [] #Toshi 08.2023
        self.qicesnow_here  = [] #DHD
        self.qghail_here    = [] #DHD
        for it in range(nt):
            self.rho_m_here.append(          self.rho_m[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.rho_here.append(            self.rho_c[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.qvapor_here.append( self.qvapor[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() ) #DH24.07.2017
            self.w_c_here.append(            self.w_c[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.u_c_here.append(            self.u_c[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.v_c_here.append(            self.v_c[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.dz_here.append(             self.dh[self.k0[it]:self.k1[it]+1] )
            self.mse_here.append(            self.mse   [self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.latheat_here.append(        self.latheat[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.qnice_here.append(          self.qnice [self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.qncloud_here.append(        self.qncloud[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.qnrain_here.append(         self.qnrain[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.cldnuc_here.append(         self.cldnuc[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.rh_here.append(             self.rh    [self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.qice_here.append(           self.qice  [self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.qcloud_here.append(         self.qcloud[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.qrain_here.append(          self.qrain [self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.sctot_here.append(          self.sctot[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.noninduc_here.append(       self.noninduc[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.epotential_here.append(        self.epotential[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.qngraupel_here.append(         self.qngraupel[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.qicesnow_here.append(        self.qicesnow[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )
            self.qghail_here.append(         self.qghail[self.i_left[it]:self.i_right[it]+1, self.j_left[it]:self.j_right[it]+1, self.k0[it]:self.k1[it]+1,it].flatten() )


    def _catalogue_data( self ):
        """
        Creates a database that makes it possible to compute the volume integrals later for any radius. It sorts every point of the subgrid 
        according to its distance to the centre of the grid (or to the specified centre xc,zc).
        """
        dx=self.dx
        dy=self.dy
        nt=self.nt
        size= self.nnx[0]*self.nny[0]*self.nnz[0] # should be the same for all timestesps!
        self.data = np.zeros([self.nt,size,25])
        R_ord_data_ind = np.ones([self.nt,size])*np.nan
        xc = self.x_centregrid[:] + self.x0*1e3
        yc = self.y_centregrid[:] + self.y0*1e3
        zc = np.copy(self.z_centregrid[:])
        for it in range(self.nt):
            l_ind = np.unravel_index(np.arange(int(size)),(self.nnx[it],self.nny[it],self.nnz[it]))
            self.data[it,:,0]  = _dist_to_centre(self.x_subgrid[it][l_ind[0]], self.y_subgrid[it][l_ind[1]], self.z_subgrid[it][l_ind[2]], xc[it], yc[it], zc[it])
            self.data[it,:,1]  = self.rho_here[it][:]  # this is density of moist air plus density of condensate
            self.data[it,:,2]  = self.w_c_here[it][:]
            self.data[it,:,3]  = self.u_c_here[it][:]
            self.data[it,:,4]  = self.v_c_here[it][:]
            self.data[it,:,5]  = self.dz_here[it][l_ind[2]]
            #data[it,:,6] and data[it,:,7] are left as zero intentionally! (will be filled with buoyancy later)
            self.data[it,:,8]  = self.mse_here[it][:]
            self.data[it,:,9]  = self.latheat_here[it][:] #DHD26.07.2019
            self.data[it,:,10] = self.qnice_here[it][:] #DHD20.09.2019
            self.data[it,:,11] = self.qncloud_here[it][:]#DHD20.09.2019
            self.data[it,:,12] = self.qnrain_here[it][:] #DHD20.09.2019
            self.data[it,:,13] = self.cldnuc_here[it][:] #DHD23.07.2020
            self.data[it,:,14] = self.rh_here[it][:] #DHD23.07.2020
            self.data[it,:,15] = self.qice_here[it][:] #DHD20.09.2019
            self.data[it,:,16] = self.qcloud_here[it][:]#DHD20.09.2019
            self.data[it,:,17] = self.qrain_here[it][:] #DHD20.09.2019
            self.data[it,:,18] = self.qvapor_here[it][:] #DH24.07.2017
            self.data[it,:,19] = self.sctot_here[it][:] #Toshi 11.12.2022
            self.data[it,:,20] = self.noninduc_here[it][:] #Toshi 11.12.2022
            self.data[it,:,21] = self.epotential_here[it][:] #Toshi08.2023
            self.data[it,:,22] = self.qngraupel_here[it][:]  #Toshi08.2023
            self.data[it,:,23] = self.qicesnow_here[it][:]#DHD20.09.2019
            self.data[it,:,24] = self.qghail_here[it][:] #DHD20.09.2019
            R_ord_data_ind[it,:] = np.argsort( self.data[it,:,0],0 )
        self.R_ord_data_ind = R_ord_data_ind.astype(int)
        self._compute_vol_integrals_allradii()


    def _compute_vol_integrals_allradii( self ):
        dx = self.dx
        dy = self.dy
        nt = self.nt
        R_ord_data_ind = self.R_ord_data_ind
        self.integrals = np.zeros([nt,self.nnx[0]*self.nny[0]*self.nnz[0], 28])
        for it in range(nt):
            vol_here        = dx*dy*self.data[it,R_ord_data_ind[it,:],5]
            mass_here       = self.data[it,R_ord_data_ind[it,:],1]*vol_here[:]
            mass_vapor_here = self.data[it,R_ord_data_ind[it,:],19]*vol_here[:] #DH24.07.2017
            massflux_here   = mass_here[:]*self.data[it,R_ord_data_ind[it,:],2]
            xmassflux_here  = mass_here[:]*self.data[it,R_ord_data_ind[it,:],3]
            ymassflux_here  = mass_here[:]*self.data[it,R_ord_data_ind[it,:],4]
            mse_rho_here    = mass_here[:]*self.data[it,R_ord_data_ind[it,:],8]
            latheat_here    = self.data[it,R_ord_data_ind[it,:],9]
            qnice_here      = self.data[it,R_ord_data_ind[it,:],10]
            qncloud_here    = self.data[it,R_ord_data_ind[it,:],11]
            qnrain_here     = self.data[it,R_ord_data_ind[it,:],12]
            cldnuc_here     = self.data[it,R_ord_data_ind[it,:],13]
            rh_here         = self.data[it,R_ord_data_ind[it,:],14]
            qice_here       = self.data[it,R_ord_data_ind[it,:],15]
            qcloud_here     = self.data[it,R_ord_data_ind[it,:],16]
            qrain_here      = self.data[it,R_ord_data_ind[it,:],17]
            qvapor_here     = self.data[it,R_ord_data_ind[it,:],18]
            sctot_here      = self.data[it,R_ord_data_ind[it,:],19] # DHD 03.03.2023
            noninduc_here   = self.data[it,R_ord_data_ind[it,:],20] # DHD 03.03.2023
            epotential_here = self.data[it,R_ord_data_ind[it,:],21]
            qngraupel_here  = self.data[it,R_ord_data_ind[it,:],22]
            qicesnow_here   = self.data[it,R_ord_data_ind[it,:],23]
            qghail_here     = self.data[it,R_ord_data_ind[it,:],24]

            volume = np.cumsum(vol_here)
            mass = np.cumsum(mass_here)
            #mass_condensate = np.cumsum(mass_c_here)
            #mass_vapor = np.cumsum(mass_vapor_here) #DH24.07.2017
            massflux = np.cumsum(massflux_here)
            xmassflux = np.cumsum(xmassflux_here)
            ymassflux = np.cumsum(ymassflux_here)
            mse_total = np.cumsum(mse_rho_here)
            latheat_avg = np.cumsum(latheat_here)/np.arange(1,len(latheat_here)+1) # this gives the "cumulative average" with equal weights
            latheat_max = np.maximum.accumulate(latheat_here) 
            qnice_avg   = np.cumsum(qnice_here)/np.arange(1,len(qnice_here)+1)
            qncloud_avg = np.cumsum(qncloud_here)/np.arange(1,len(qncloud_here)+1)
            qnrain_avg  = np.cumsum(qnrain_here)/np.arange(1,len(qnrain_here)+1)
            cldnuc_avg  = np.cumsum(cldnuc_here)/np.arange(1,len(cldnuc_here)+1)
            rh_avg      = np.cumsum(rh_here)/np.arange(1,len(rh_here)+1)
            qice_avg    = np.cumsum(qice_here)/np.arange(1,len(qice_here)+1)
            qcloud_avg  = np.cumsum(qcloud_here)/np.arange(1,len(qcloud_here)+1)
            qrain_avg   = np.cumsum(qrain_here)/np.arange(1,len(qrain_here)+1)
            qvapor_avg  = np.cumsum(qvapor_here)/np.arange(1,len(qvapor_here)+1)
            sctot_avg   = np.cumsum(sctot_here)/np.arange(1,len(sctot_here)+1) # DHD 03.03.2023
            sctot_max   = np.maximum.accumulate(sctot_here) # DHD 03.03.2023
            noninduc_avg= np.cumsum(noninduc_here)/np.arange(1,len(noninduc_here)+1) # DHD 03.03.2023
            epotential_avg = np.cumsum(epotential_here)/np.arange(1,len(epotential_here)+1)
            epotential_max = np.maximum.accumulate(epotential_here) 
            qngraupel_avg  = np.cumsum(qngraupel_here)/np.arange(1,len(qngraupel_here)+1)
            qicesnow_avg = np.cumsum(qicesnow_here)/np.arange(1,len(qicesnow_here)+1)
            qghail_avg  = np.cumsum(qghail_here)/np.arange(1,len(qghail_here)+1)

            self.integrals[it,:,0]  = self.data[it,R_ord_data_ind[it,:],0]
            self.integrals[it,:,1]  = massflux/mass
            self.integrals[it,:,2]  = xmassflux/mass
            #integrals[it][:,3] = -g*B_disk/mass    # buoyancy will be computed at the end!
            self.integrals[it,:,4]  = mass
            self.integrals[it,:,5]  = volume
            #integrals[it,:,6] = -g*B_moist/mass   # buoyancy will be computed at the end!
            self.integrals[it,:,7]  = ymassflux/mass
            self.integrals[it,:,8]  = mse_total/mass
            #self.integrals[it,:,6] = mass_vapor #DH24.07.2017
            self.integrals[it,:,9]  = latheat_avg #DHD 03.08.2020
            self.integrals[it,:,10] = latheat_max #DHD 03.08.2020
            self.integrals[it,:,11] = qnice_avg #DHD 12.10.2020
            self.integrals[it,:,12] = qncloud_avg #DHD 12.10.2020
            self.integrals[it,:,13] = qnrain_avg #DHD 12.10.2020
            self.integrals[it,:,14] = cldnuc_avg #DHD 12.10.2020
            self.integrals[it,:,15] = rh_avg #DHD 12.10.2020
            self.integrals[it,:,16] = qice_avg #DHD 12.10.2020
            self.integrals[it,:,17] = qcloud_avg #DHD 12.10.2020
            self.integrals[it,:,18] = qrain_avg #DHD 12.10.2020
            self.integrals[it,:,19] = qvapor_avg # DHD 22.10.2020
            self.integrals[it,:,20] = sctot_avg #Toshi 11.12.2022
            self.integrals[it,:,21] = sctot_max #Toshi 11.12.2022
            self.integrals[it,:,22] = noninduc_avg #Toshi 11.12.2022
            self.integrals[it,:,23] = epotential_avg 
            self.integrals[it,:,24] = epotential_max 
            self.integrals[it,:,25] = qngraupel_avg 
            self.integrals[it,:,26] = qicesnow_avg #DHD 
            self.integrals[it,:,27] = qghail_avg #DHD 


    def _get_r_from_w( self ):
        w = self.w_thermal
        nt = self.nt
        w_r = np.zeros([nt,self.nnx[0]*self.nny[0]*self.nnz[0],2])
        ind_w_ord = np.zeros([nt,self.nnx[0]*self.nny[0]*self.nnz[0]])
        for it in range(nt):
            ind_w_ord[it,:] = np.argsort( self.integrals[it,:,1],axis=0 )
        ind_w_ord = ind_w_ord.astype(int)
        self.R_thermal = np.ones(nt)*np.nan
        for it in range(nt):
            w_r[it,:,0] = self.integrals[it,ind_w_ord[it,:],1]
            w_r[it,:,1] = self.integrals[it,ind_w_ord[it,:],0]
            #w_smooth = moving_average(w_r[it][:,0])
            #r_smooth = moving_average(w_r[it][:,1])
            newr=np.interp( w[it], w_r[it,:,0], w_r[it,:,1], left=np.nan, right=np.nan )
            #newr=np.interp( w[it], w_smooth, r_smooth, left=0., right=np.nan )
            if np.isnan(newr):
                self.log.append( 'Warning: too large (or invalid) radius at '+hhmm(self.time[it])+'.' )
                #print 'Warning: too large (or invalid) radius at '+hhmm(self.time[it])+'.' 
            elif newr < self.min_R:
                self.log.append( 'Warning: radius is too small at '+hhmm(self.time[it])+'.' )
                #print 'Warning: radius is too small at '+hhmm(self.time[it])+'.' 
                newr = np.nan
            self.R_thermal[it] = newr
        self.R_thermal[np.where(self.R_thermal==0.)] = np.nan


    def _check_consistency( self, var='PnzdS', wmax_positive=False, large_slow=False):
        """
        remove time steps which have yielded NaN values in a certain variable (default is PnzdS), or remove time steps
        where the updraft velocity at the centre has been found to be negative (w_centre), if wmax_positive is True.
        """
        if large_slow and self.up:
            """
            remove time steps where radius is larger than mean+2*std AND w_thermal is less than 0.5 m/s
            """
            invalid=np.where((self.R_thermal>np.mean(self.R_thermal)+2.*np.std(self.R_thermal))*(self.w_thermal<0.5))[0]
            if invalid.shape[0]>=1:
                for i in range(invalid.shape[0]):
                    self.log.append( 'large and slow thermal at ' + hhmm(self.time[invalid[i]]) )
                    #print 'large and slow thermal at ' + hhmm(self.time[invalid[i]]) 
        elif wmax_positive and self.up:
            invalid=np.where(self.w_centre<=0.)[0]
            if invalid.shape[0]>=1:
                self.log.append( 'found point(s) with negative w_max:' )
                #print 'found point(s) with negative w_max:' 
                for it in range(invalid.shape[0]):
                    self.log.append( hhmm(self.time[invalid[it]]) )
        else:
            if type(var)==str:
                if var=='PnzdS':
                    var=self.PnzdS
            invalid=np.where(np.isnan(var))[0]
        if invalid.shape[0]==1:
            if invalid[0] < self.nt/2.:
                t0_valid = invalid[0]+1
                tf_valid = self.nt
            else:
                t0_valid = 0
                tf_valid = invalid[0]-1
        if invalid.shape[0]==2:
            if invalid[0] < self.nt/2. and invalid[1] < self.nt/2.:    
                t0_valid = invalid[1]+1
                tf_valid = self.nt-1
            if invalid[0] < self.nt/2. and invalid[1] >= self.nt/2.:
                t0_valid = invalid[0]+1
                tf_valid = invalid[1]-1
            if invalid[0] >= self.nt/2. and invalid[1] >= self.nt/2.:
                t0_valid = 0
                tf_valid = invalid[0]-1
        if invalid.shape[0]>2:
            invalid2 = np.zeros(len(invalid)+2)
            invalid2[0]=-1
            invalid2[-1]=self.nt
            invalid2[1:-1] = invalid
            dist = []
            for i in range(invalid2.shape[0]-1):
                dist.append(invalid2[i+1]-invalid2[i])
            dist = np.asarray(dist)
            longest=np.where(dist==np.amax(dist))[0]
            t0_valid = int(invalid2[longest[0]] + 1)
            tf_valid = int(invalid2[longest[0]+1] - 1)
        if invalid.shape[0]>0:
            self.log.append( self.OBJ+' needs to be cropped between '+ hhmm(self.time[t0_valid])+ ' and '+ hhmm(self.time[np.amin([tf_valid,self.nt-1])]) )
            #print 'thermal needs to be cropped between '+ hhmm(self.time[t0_valid])+ ' and '+ hhmm(self.time[np.amin([tf_valid,self.nt-1])]) 
            self._crop_t_thermal( int(t0_valid),int(tf_valid)+1 )
    

    def _crop_t_thermal( self, t0, tf, reload=False ):
       
        if hasattr(self,'net_entr_term'):
            self.net_entr_term = self.net_entr_term[t0:tf-1] # this array is on half time steps, so always has one time step less.
        
        self.tsteps     = self.tsteps[t0:tf]
        self.time     = (self.tsteps*self.dt)/60. + self.grid.hr0*60 + self.grid.min0    + self.grid.sec0/60.# time in minutes where the thermal is traced
        self.nt     = len(self.tsteps)
        
        self.xmax         = self.xmax[t0:tf]
        self.ymax         = self.ymax[t0:tf]
        self.hmax         = self.hmax[t0:tf]
        self.x_centre     = self.x_centre[t0:tf]
        self.y_centre     = self.y_centre[t0:tf]
        self.z_centre     = self.z_centre[t0:tf]
        self.ix_centre    = self.ix_centre[t0:tf]
        self.iy_centre    = self.iy_centre[t0:tf] 
        self.iz_centre    = self.iz_centre[t0:tf]
        self.x_centregrid = self.x_centre[:] - self.grid.x0*1e3
        self.y_centregrid = self.y_centre[:] - self.grid.y0*1e3
        self.z_centregrid = np.copy(self.z_centre[:])
        self.u_thermal    = self.u_thermal[t0:tf]
        self.v_thermal    = self.v_thermal[t0:tf]
        self.w_thermal    = self.w_thermal[t0:tf]
        
        self.rho_c        = self.grid.rho_c[:,:,:,self.tsteps]
        self.rho_m        = self.grid.rho_m[:,:,:,self.tsteps]
        #self.rho_condensate = self.grid.rho_condensate[:,:,:,self.tsteps]
        self.u_c          = self.grid.u_c[:,:,:,self.tsteps]
        self.v_c          = self.grid.v_c[:,:,:,self.tsteps]
        self.w_c          = self.grid.w_c[:,:,:,self.tsteps]
        self.ptot         = self.grid.ptot[:,:,:,self.tsteps]
        self.mse          = self.grid.mse[:,:,:,self.tsteps]
        
        # crop the necessary data from the Grid object as well:
        self.grid.tsteps        = self.tsteps
        self.grid.xmax          = self.xmax
        self.grid.ymax          = self.ymax
        self.grid.hmax          = self.hmax        
        self.grid.x             = self.xmax
        self.grid.y             = self.ymax
        self.grid.z             = self.hmax
        self.grid.t             = self.tsteps
        self.grid.x_centre      = self.x_centre
        self.grid.y_centre      = self.y_centre  
        self.grid.z_centre      = self.z_centre  
        self.grid.ix_centre     = self.ix_centre 
        self.grid.iy_centre     = self.iy_centre 
        self.grid.iz_centre     = self.iz_centre 
        self.grid.u_thermal     = self.u_thermal 
        self.grid.v_thermal     = self.v_thermal
        self.grid.w_thermal     = self.w_thermal 
        if hasattr(self,'mean_W_env'):
            self.mean_W_env     = self.mean_W_env[t0:tf]
        if hasattr(self, 'buoy_map'):
            self.buoy_map       = self.buoy_map[t0:tf]
    
        if reload:
            # re-run the initial steps of _load() without the invalid time steps:
            self._load_step1()
        else:
            self.u_centre       = self.u_centre        [t0:tf]
            self.v_centre       = self.v_centre        [t0:tf]
            self.w_centre       = self.w_centre        [t0:tf]
            self.i_left         = self.i_left          [t0:tf]
            self.i_right        = self.i_right         [t0:tf]
            self.j_left         = self.j_left          [t0:tf]
            self.j_right        = self.j_right         [t0:tf]
            self.k0             = self.k0              [t0:tf]
            self.k1             = self.k1              [t0:tf]
            self.nnx            = self.nnx             [t0:tf]    
            self.nny            = self.nny             [t0:tf]
            self.nnz            = self.nnz             [t0:tf]
            self.subgrid        = self.subgrid         [t0:tf]
            self.x_subgrid      = self.x_subgrid       [t0:tf]
            self.y_subgrid      = self.y_subgrid       [t0:tf]
            self.z_subgrid      = self.z_subgrid       [t0:tf]
            self.rho_here       = self.rho_here        [t0:tf]
            self.rho_m_here     = self.rho_m_here      [t0:tf]
            self.u_c_here       = self.u_c_here        [t0:tf]
            self.v_c_here       = self.v_c_here        [t0:tf]
            self.w_c_here       = self.w_c_here        [t0:tf]
            self.dz_here        = self.dz_here         [t0:tf] 
            self.sctot_here     = self.sctot_here      [t0:tf] #Toshi 11.12.2022
            self.latheat_here   = self.latheat_here    [t0:tf] #DHD26.07.2019
            self.qnice_here     = self.qnice_here      [t0:tf] #DHD20.09.2019
            self.qncloud_here   = self.qncloud_here    [t0:tf] #DHD20.09.2019
            self.qnrain_here    = self.qnrain_here     [t0:tf] #DHD20.09.2019
            self.noninduc_here   = self.noninduc_here  [t0:tf] #Toshi 11.12.2022
            self.cldnuc_here     = self.cldnuc_here    [t0:tf] #DHD23.07.2020
            self.rh_here         = self.rh_here        [t0:tf] #DHD23.07.2020
            self.qice_here       = self.qice_here      [t0:tf] #DHD20.09.2019
            self.qcloud_here     = self.qcloud_here    [t0:tf] #DHD20.09.2019
            self.qrain_here      = self.qrain_here     [t0:tf] #DHD20.09.2019
            self.epotential_here = self.epotential_here[t0:tf]
            self.qngraupel_here  = self.qngraupel_here [t0:tf]
            self.qicesnow_here   = self.qicesnow_here  [t0:tf] #DHD
            self.qghail_here     = self.qghail_here    [t0:tf] #DHD
    
            self.R_thermal         = self.R_thermal            [t0:tf]

        if (not reload and hasattr(self, 'PnzdS')) or self.split:
            self.PnzdS                  = self.PnzdS            [t0:tf] 
            self.zmomflux               = self.zmomflux         [t0:tf]
            self.entr_dist              = self.entr_dist        [t0:tf]
            self.detr_dist              = self.detr_dist        [t0:tf]
            self.epsilon                = self.epsilon          [t0:tf]
            self.entr_rate              = self.entr_rate        [t0:tf]
            self.PnxdS                  = self.PnxdS            [t0:tf]
            self.PnydS                  = self.PnydS            [t0:tf]
            self.xmomflux               = self.xmomflux         [t0:tf]
            self.ymomflux               = self.ymomflux         [t0:tf]
            self.w_mean                 = self.w_mean           [t0:tf]
            self.u_mean                 = self.u_mean           [t0:tf]
            self.v_mean                 = self.v_mean           [t0:tf]
            self.buoy                   = self.buoy             [t0:tf]
            self.buoy_m                 = self.buoy_m           [t0:tf]
            self.mass                   = self.mass             [t0:tf]
            self.volume                 = self.volume           [t0:tf]
            self.volume_err             = self.volume_err       [t0:tf]
            #self.mass_cond              = self.mass_cond        [t0:tf]
            #self.mass_wvapor           = self.mass_wvapor      [t0:tf] #DH24.07.2017
            self.massflux               = self.massflux         [t0:tf]
            self.mse_thermal            = self.mse_thermal      [t0:tf]
            self.sctot_thermal          = self.sctot_thermal    [t0:tf] #Toshi 11.12.2022
            self.maxsctot_thermal       = self.maxsctot_thermal[t0:tf] #Toshi 11.12.2022
            self.latheat_thermal        = self.latheat_thermal  [t0:tf] #DHD 03.08.2020
            self.maxlatheat_thermal     = self.maxlatheat_thermal  [t0:tf] #DHD 03.08.2020
            self.qnice_thermal          = self.qnice_thermal    [t0:tf] #DHD 13.10.2020 
            self.qncloud_thermal        = self.qncloud_thermal  [t0:tf] #DHD 13.10.2020 
            self.qnrain_thermal         = self.qnrain_thermal   [t0:tf] #DHD 13.10.2020 
            self.noninduc_thermal       = self.noninduc_thermal[t0:tf] #Toshi 11.12.2022
            self.cldnuc_thermal         = self.cldnuc_thermal   [t0:tf] #DHD 13.10.2020 
            self.rh_thermal             = self.rh_thermal       [t0:tf] #DHD 13.10.2020 
            self.qice_thermal           = self.qice_thermal     [t0:tf] #DHD 13.10.2020 
            self.qcloud_thermal         = self.qcloud_thermal   [t0:tf] #DHD 13.10.2020 
            self.qrain_thermal          = self.qrain_thermal    [t0:tf] #DHD 13.10.2020 
            self.qvapor_thermal         = self.qvapor_thermal   [t0:tf] #DHD 13.10.2020 
            self.epotential_thermal     = self.epotential_thermal[t0:tf]
            self.maxepotential_thermal  = self.maxepotential_thermal[t0:tf] 
            self.qngraupel_thermal      = self.qngraupel_thermal[t0:tf]
            self.qicesnow_thermal       = self.qicesnow_thermal  [t0:tf] #DHD 
            self.qghail_thermal         = self.qghail_thermal   [t0:tf] #DHD
            if hasattr(self, 'entr_distr'):
                self.angles     = self.angles       [t0:tf] 
                self.entr_distr = self.entr_distr   [t0:tf]
            if hasattr(self, 'pos_exp'):
                self.pos_exp    = self.pos_exp      [t0:tf]
                self.acc        = self.acc          [t0:tf]

        if (not reload) and hasattr(self, 'integrals'):
            self.data           = self.data             [t0:tf]
            self.integrals      = self.integrals        [t0:tf]
            self.R_ord_data_ind = self.R_ord_data_ind   [t0:tf]
    

    def _get_vol_integrals_r( self ):
        nt=self.nt
        integrals = self.integrals
        r = self.R_thermal
        w_mean                  = [] 
        u_mean                  = [] 
        v_mean                  = []
        buoy                    = [] 
        mass                    = [] 
        volume                  = [] 
        volume_err              = [] 
        #mass_cond               = [] 
        buoy_m                  = [] 
        mse_thermal             = []
        latheat_thermal         = []
        maxlatheat_thermal      = []
        qnice_thermal           = []
        qncloud_thermal         = []
        qnrain_thermal          = []
        cldnuc_thermal          = []
        rh_thermal              = []
        qice_thermal            = []
        qcloud_thermal          = []
        qrain_thermal           = []
        qvapor_thermal          = []
        sctot_thermal           = []
        maxsctot_thermal        = []
        noninduc_thermal        = []
        epotential_thermal      = []
        maxepotential_thermal   = []
        qngraupel_thermal       = []
        qicesnow_thermal        = []
        qghail_thermal          = []

        for it in range(nt):
            w_mean.append(               np.interp( r[it], integrals[it,:,0], integrals[it,:,1], left=integrals[it,0,1] ) )
            u_mean.append(               np.interp( r[it], integrals[it,:,0], integrals[it,:,2], left=integrals[it,0,2] ) )
            v_mean.append(               np.interp( r[it], integrals[it,:,0], integrals[it,:,7], left=integrals[it,0,7] ) )
            buoy.append( 0. ) # buoyancy will be computed at the end!
            mass.append(                 np.interp( r[it], integrals[it,:,0], integrals[it,:,4], left=integrals[it,0,4] ) ) 
            volume.append(               np.interp( r[it], integrals[it,:,0], integrals[it,:,5], left=integrals[it,0,5] ) ) 
            buoy_m.append( 0. )
            mse_thermal.append(          np.interp( r[it], integrals[it,:,0], integrals[it,:,8],  left=integrals[it,0,8] ) )
            latheat_thermal.append(      np.interp( r[it], integrals[it,:,0], integrals[it,:,9],  left=integrals[it,0,9] ) )
            maxlatheat_thermal.append(   np.interp( r[it], integrals[it,:,0], integrals[it,:,10], left=integrals[it,0,10] ) )
            qnice_thermal.append(        np.interp( r[it], integrals[it,:,0], integrals[it,:,11], left=integrals[it,0,11] ) ) #DHD12.10.2020
            qncloud_thermal.append(      np.interp( r[it], integrals[it,:,0], integrals[it,:,12], left=integrals[it,0,12] ) )#DHD12.10.2020
            qnrain_thermal.append(       np.interp( r[it], integrals[it,:,0], integrals[it,:,13], left=integrals[it,0,13] ) )#DHD12.10.2020
            cldnuc_thermal.append(       np.interp( r[it], integrals[it,:,0], integrals[it,:,14], left=integrals[it,0,14] ) )#DHD12.10.2020
            rh_thermal.append(           np.interp( r[it], integrals[it,:,0], integrals[it,:,15], left=integrals[it,0,15] ) )#DHD12.10.2020
            qice_thermal.append(         np.interp( r[it], integrals[it,:,0], integrals[it,:,16], left=integrals[it,0,16] ) )#DHD12.10.2020
            qcloud_thermal.append(       np.interp( r[it], integrals[it,:,0], integrals[it,:,17], left=integrals[it,0,17] ) )#DHD12.10.2020
            qrain_thermal.append(        np.interp( r[it], integrals[it,:,0], integrals[it,:,18], left=integrals[it,0,18] ) )#DHD12.10.2020
            qvapor_thermal.append(       np.interp( r[it], integrals[it,:,0], integrals[it,:,19], left=integrals[it,0,19] ) )#DHD22.10.2020
            sctot_thermal.append(        np.interp( r[it], integrals[it,:,0], integrals[it,:,20], left=integrals[it,0,20] ) ) #DHD 03.03.2023
            maxsctot_thermal.append(     np.interp( r[it], integrals[it,:,0], integrals[it,:,21], left=integrals[it,0,21] ) ) #DHD 03.03.2023
            noninduc_thermal.append(     np.interp( r[it], integrals[it,:,0], integrals[it,:,22], left=integrals[it,0,22] ) )#DHD12.10.2020
            epotential_thermal.append(   np.interp( r[it], integrals[it,:,0], integrals[it,:,23], left=integrals[it,0,23] ) )
            maxepotential_thermal.append(np.interp( r[it], integrals[it,:,0], integrals[it,:,24], left=integrals[it,0,24] ) )
            qngraupel_thermal.append(    np.interp( r[it], integrals[it,:,0], integrals[it,:,25], left=integrals[it,0,25] ) )
            qicesnow_thermal.append(     np.interp( r[it], integrals[it,:,0], integrals[it,:,26], left=integrals[it,0,26] ) )#DHD
            qghail_thermal.append(       np.interp( r[it], integrals[it,:,0], integrals[it,:,27], left=integrals[it,0,27] ) )#DHD
        
            theor_vol = 4.*np.pi*r[it]*r[it]*r[it]/3.
            volume_err.append( (volume[it]-theor_vol)/(theor_vol) )
        self.w_mean                 = np.asarray( w_mean         ) 
        self.u_mean                 = np.asarray( u_mean         ) 
        self.v_mean                 = np.asarray( v_mean         )
        self.buoy                   = np.asarray( buoy           ) 
        self.mass                   = np.asarray( mass           ) 
        self.volume                 = np.asarray( volume         ) 
        self.volume_err             = np.asarray( volume_err     ) 
        #self.mass_cond              = np.asarray( mass_cond     ) 
        self.buoy_m                 = np.asarray( buoy_m         )
        self.mse_thermal            = np.asarray( mse_thermal    )
        self.sctot_thermal          = np.asarray( sctot_thermal) #Toshi 11.12.2022
        self.maxsctot_thermal       = np.asarray( maxsctot_thermal) #Toshi 11.12.2022
        self.latheat_thermal        = np.asarray( latheat_thermal) #DHD 03.08.2020
        self.maxlatheat_thermal     = np.asarray( maxlatheat_thermal) #DHD 03.08.2020
        self.qnice_thermal          = np.asarray( qnice_thermal ) #DHD 12.10.2020
        self.qncloud_thermal        = np.asarray( qncloud_thermal ) #DHD 12.10.2020
        self.qnrain_thermal         = np.asarray( qnrain_thermal ) #DHD 12.10.2020
        self.noninduc_thermal       = np.asarray( noninduc_thermal ) #Toshi 11.12.2022
        self.cldnuc_thermal         = np.asarray( cldnuc_thermal ) #DHD 12.10.2020
        self.rh_thermal             = np.asarray( rh_thermal ) #DHD 12.10.2020
        self.qice_thermal           = np.asarray( qice_thermal ) #DHD 12.10.2020
        self.qcloud_thermal         = np.asarray( qcloud_thermal ) #DHD 12.10.2020
        self.qrain_thermal          = np.asarray( qrain_thermal ) #DHD 12.10.2020
        self.qvapor_thermal         = np.asarray( qvapor_thermal ) #DHD 22.10.2020
        self.epotential_thermal     = np.asarray( epotential_thermal) 
        self.maxepotential_thermal  = np.asarray( maxepotential_thermal)
        self.qngraupel_thermal      = np.asarray( qngraupel_thermal ) 
        self.qicesnow_thermal       = np.asarray( qicesnow_thermal ) #DHD
        self.qghail_thermal         = np.asarray( qghail_thermal ) #DHD
     
        self.mass = self.mass/(self.volume_err + 1.)               # adjustment according to error in volume integral
        self.massflux = self.w_mean*self.mass
   

    def _compute_surface_integrals( self, joblib_mode=None ):
        n_jobs = self.n_jobs
        dx = self.dx
        dy = self.dy
        nt = self.nt

        r = self.R_thermal
        x_centre = self.x_centre - self.grid.x0*1e3
        y_centre = self.y_centre - self.grid.y0*1e3
        z_centre = self.z_centre
        u_mean = self.u_mean
        v_mean = self.v_mean
        w_mean = self.w_mean
        mass   = self.mass
        volume = self.volume
       
        N_alphas = 4*2*(np.pi*r/dx).astype(int)        # This means the side of each surface segment is about dx/4
        if np.all(N_alphas==N_alphas[0]):              # this avoids a very weird problem that appears if N_alpha is the same for all timesteps (which is very unlikely) DHD 22082023
            N_alphas[-1]=N_alphas[-1]+2
        for it in range(len(N_alphas)):
            N_alphas[it] = np.amax([8,N_alphas[it]])   # at least 8 angles! (though that's never going to happen...)
        d_alpha = 2.*np.pi/N_alphas                    # angle for each arc in the xy plane
        d_phi   = d_alpha                              # angle for each arc in the xz plane

        alpha = []
        phi   = []
        for it in range(nt):
            alpha.append( np.arange(N_alphas[it]+1)*d_alpha[it]- d_alpha[it]*0.5 )           # the angles of the corners of the arcs on the xy plane, -d_alpha/2 to 2pi-d_alpha/2 (to include 0)
            phi.append( np.arange(N_alphas[it]/2 + 1)*d_phi[it] -np.pi/2 )  # the angles on the vertical plane (xz),  -pi/2 to pi/2
        alpha   = np.asarray(alpha, dtype=object)
        phi     = np.asarray(phi, dtype=object)
        alpha_c = []
        phi_c   = []
        angles  = []
        dS      = []
        n_x     = []
        n_y     = []
        n_z     = []
        for it in range(nt):
            alpha_c.append( (alpha[it][1:]+alpha[it][:-1])*0.5 )    # the angles of the center of the arcs on the horizontal plane
            phi_c.append( (phi[it][1:] + phi[it][:-1])*0.5 )        # the angles of the centre of the arcs on the vertical plane
            angles.append( np.zeros([int(N_alphas[it]*N_alphas[it]/2),2]) )
            dS.append( np.zeros([int(N_alphas[it]*N_alphas[it]/2)]) )
            for l in range(int(N_alphas[it])):
                for j in range(int(N_alphas[it]/2.)):
                    angles[it][j+int((N_alphas[it]/2)*l),0] = alpha_c[it][l]
                    angles[it][j+int((N_alphas[it]/2)*l),1] = phi_c[it][j]
                    dS[it][j+int((N_alphas[it]/2)*l)] = r[it]*r[it]*(np.sin(phi[it][j+1])-np.sin(phi[it][j]))*(alpha[it][l+1]-alpha[it][l]) # the surface area of a lat-lon quadrangle

            n_x.append( -np.cos(angles[it][:,1])*np.cos(angles[it][:,0]) )  # x-component of normal vector at the arc-centres (pointing inward)
            n_y.append( -np.cos(angles[it][:,1])*np.sin(angles[it][:,0]) ) 
            n_z.append( -np.sin(angles[it][:,1]) )                    # z-component of normal vector at the arc-centres (pointing inward)
        
        dS = np.asarray(dS, dtype=object)
        n_x = np.asarray(n_x, dtype=object)
        n_y = np.asarray(n_y, dtype=object)
        n_z = np.asarray(n_z, dtype=object)
        nxarr = np.arange(self.nx)
        nyarr = np.arange(self.ny)
        nzarr = np.arange(self.nz)

        press_circ  = [] 
        u_circ      = []
        v_circ      = []
        w_circ      = []
        dens_circ   = []
        nx = self.nx 
        ny = self.ny
        nz = self.nz
        x_centre = self.x_centre - self.grid.x0*1e3
        y_centre = self.y_centre - self.grid.y0*1e3
        z_centre = self.z_centre
        r = self.R_thermal
        if joblib_mode==None:       # this will choose the most appropriate way of splitting jobs between vars and time splitting, based on number of time steps
            if self.nt<=5 or (self.nt>5 and self.n_jobs<=5):
                joblib_mode='vars'
            if self.nt>5 and self.n_jobs>5:
                joblib_mode='time'
        #****************************************************
        # SPLIT INTO 5 VARIABLES (jobs) WITH JOBLIB:
        if joblib_mode=='vars':
            #print 'vars'
            for it in range(self.nt):
                x_m = x_centre[it] + r[it]*np.cos(angles[it][:,1])*np.cos(angles[it][:,0]) # coordinates of the centers of the dS arcs
                y_m = y_centre[it] + r[it]*np.cos(angles[it][:,1])*np.sin(angles[it][:,0])
                z_m = z_centre[it] + r[it]*np.sin(angles[it][:,1])
                ix_c = pol.griddata( self.x_grid-self.x0*1e3, np.arange(nx), x_m, method='linear' ) 
                iy_c = pol.griddata( self.y_grid-self.y0*1e3, np.arange(ny), y_m, method='linear' ) 
                iz_c = pol.griddata( self.hgt_c, np.arange(nz), z_m, method='linear' ) 
    
                press_circ.append( np.zeros(len(angles[it]), dtype=prec) )
                u_circ.append( np.zeros(len(angles[it]), dtype=prec) ) 
                v_circ.append( np.zeros(len(angles[it]), dtype=prec) )
                w_circ.append( np.zeros(len(angles[it]), dtype=prec) )
                dens_circ.append( np.zeros(len(angles[it]), dtype=prec) )
            
                if np.any(np.isnan( np.concatenate( (np.concatenate( (ix_c,iy_c) ),iz_c) ) )):  # this would mean the thermal lies (partly) outside the domain, and the interpolation below would crash.
                    press_circ[it][:] = np.nan
                    u_circ[it][:]     = np.nan
                    v_circ[it][:]     = np.nan
                    w_circ[it][:]     = np.nan
                    dens_circ[it][:]  = np.nan
                else:
                    if np.any(np.isnan(np.stack((ix_c,iy_c,iz_c)))):
                        print( 'Warning: nan values in ix_c, iy_c or iz_c!')
                    i0, i1, j0, j1, k0, k1 = create_small_subgrid_for_interpolation2( ix_c[:], iy_c[:], iz_c[:], nx, ny, nz )
                    jobs = [( self.ptot [:,:,:,it], i0, i1, j0, j1, k0, k1, ix_c, iy_c, iz_c ),
                            ( self.u_c  [:,:,:,it], i0, i1, j0, j1, k0, k1, ix_c, iy_c, iz_c ), 
                            ( self.v_c  [:,:,:,it], i0, i1, j0, j1, k0, k1, ix_c, iy_c, iz_c ), 
                            ( self.w_c  [:,:,:,it], i0, i1, j0, j1, k0, k1, ix_c, iy_c, iz_c ), 
                            ( self.rho_c[:,:,:,it], i0, i1, j0, j1, k0, k1, ix_c, iy_c, iz_c )]
                    if self.parallel_thermals:
                        press_circ[it][:]   = joblib_sfc_interp_split_var(*jobs[0])
                        u_circ[it][:]       = joblib_sfc_interp_split_var(*jobs[1])
                        v_circ[it][:]       = joblib_sfc_interp_split_var(*jobs[2])
                        w_circ[it][:]       = joblib_sfc_interp_split_var(*jobs[3])
                        dens_circ[it][:]    = joblib_sfc_interp_split_var(*jobs[4])
                    else:
                        ( press_circ[it][:], u_circ[it][:], v_circ[it][:], w_circ[it][:], dens_circ[it][:] ) = Parallel(n_jobs=5)(delayed(joblib_sfc_interp_split_var)(*jobs[i]) for i in range(5))
        #************************************************************
        if joblib_mode=='time':
            # SPLIT TIME STEPS WITH JOBLIB
            n_jobs, dtt = optimize_njobs( self.nt, self.n_jobs )
            it0 = 0
            subjob = []
            for ijob in range(n_jobs-1):
                it1 = it0 + dtt
                subjob.append( [it0,it1] )
                it0 = it1
            subjob.append( [it0, self.nt] )
            jobs = []
            for ijob in range(n_jobs):
                jobs.append( ( subjob[ijob], x_centre, y_centre, z_centre, r, angles, self.x_grid, self.y_grid, self.x0, self.y0, self.hgt_c, self.ptot[:,:,:,subjob[ijob][0]:subjob[ijob][1]], self.u_c[:,:,:,subjob[ijob][0]:subjob[ijob][1]], self.v_c[:,:,:,subjob[ijob][0]:subjob[ijob][1]], self.w_c[:,:,:,subjob[ijob][0]:subjob[ijob][1]], self.rho_c[:,:,:,subjob[ijob][0]:subjob[ijob][1]] ) )
            ( interp_vars ) = Parallel(n_jobs=n_jobs)(delayed(joblib_sfc_interp_split_time)(*jobs[i]) for i in range(len(jobs)))
            for ijob in range(n_jobs):
                for i in range(subjob[ijob][1]-subjob[ijob][0]):
                    press_circ.append( interp_vars[ijob][0][i] )
                    u_circ.append( interp_vars[ijob][1][i] )
                    v_circ.append( interp_vars[ijob][2][i] )
                    w_circ.append( interp_vars[ijob][3][i] )
                    dens_circ.append( interp_vars[ijob][4][i] )
        #**********************************
        # SPLIT THE SPHERE WITH JOBLIB
        if joblib_mode=='sphere':
            for it in range(self.nt):
                x_m = x_centre[it] + r[it]*np.cos(angles[it][:,1])*np.cos(angles[it][:,0]) # coordinates of the centers of the dS arcs
                y_m = y_centre[it] + r[it]*np.cos(angles[it][:,1])*np.sin(angles[it][:,0])
                z_m = z_centre[it] + r[it]*np.sin(angles[it][:,1])
                ix_c = pol.griddata( self.x_grid-self.x0*1e3, np.arange(nx), x_m, method='linear' ) 
                iy_c = pol.griddata( self.y_grid-self.y0*1e3, np.arange(ny), y_m, method='linear' ) 
                iz_c = pol.griddata( self.hgt_c, np.arange(nz), z_m, method='linear' ) 
    
                press_circ.append( np.zeros(len(angles[it]), dtype=prec) )
                u_circ.append( np.zeros(len(angles[it]), dtype=prec) ) 
                v_circ.append( np.zeros(len(angles[it]), dtype=prec) )
                w_circ.append( np.zeros(len(angles[it]), dtype=prec) )
                dens_circ.append( np.zeros(len(angles[it]), dtype=prec) )
    
                if np.any(np.isnan( np.concatenate( (np.concatenate( (ix_c,iy_c) ),iz_c) ) )):  # this would mean the thermal lies (partly) outside the domain, and the interpolation below would crash.
                    press_circ[it][:] = np.nan
                    u_circ[it][:]     = np.nan
                    v_circ[it][:]     = np.nan
                    w_circ[it][:]     = np.nan
                    dens_circ[it][:]  = np.nan
                else:
                    n_jobs0 = np.amin([len(angles[it])/220,n_jobs])        # number of jobs in order to have ~250 points for each job (seems to be the most efficient)
                    n_jobs1, ds = optimize_njobs(len(angles[it]), n_jobs0)
    
                    iS0 = 0
                    subjob = []
                    for ijob in range(n_jobs1-1):
                        iS1 = iS0+ds
                        subjob.append([iS0,iS1])
                        iS0 = iS1
                    subjob.append([iS0,len(angles[it])])
                    jobs = []
                    for ijob in range(n_jobs1):
                        jobs.append( (subjob[ijob], ix_c, iy_c, iz_c, self.ptot[:,:,:,it], self.u_c[:,:,:,it], self.v_c[:,:,:,it], self.w_c[:,:,:,it], self.rho_c[:,:,:,it], self.nx, self.ny, self.nz) )
                    ( interp_vars ) = Parallel(n_jobs=n_jobs1)(delayed(joblib_sfc_interp_split_sphere)(*jobs[i]) for i in range(len(jobs)))
                    jobs=None
                   
                    for ijob in range(n_jobs1):
                        press_circ[it][subjob[ijob][0]:subjob[ijob][1]] = interp_vars[ijob][0]
                        u_circ[it][subjob[ijob][0]:subjob[ijob][1]]     = interp_vars[ijob][1]
                        v_circ[it][subjob[ijob][0]:subjob[ijob][1]]     = interp_vars[ijob][2]
                        w_circ[it][subjob[ijob][0]:subjob[ijob][1]]     = interp_vars[ijob][3]
                        dens_circ[it][subjob[ijob][0]:subjob[ijob][1]]  = interp_vars[ijob][4]
        
        press_circ  = np.asarray(press_circ, dtype=object)
        u_circ      = np.asarray(u_circ, dtype=object)
        v_circ      = np.asarray(v_circ, dtype=object)
        w_circ      = np.asarray(w_circ, dtype=object)
        dens_circ   = np.asarray(dens_circ, dtype=object)
    
        pnxdS = press_circ*n_x*dS
        pnydS = press_circ*n_y*dS
        pnzdS = press_circ*n_z*dS

        rho_dS = dens_circ*dS
        #mflux = (u_circ*n_x + v_circ*n_y + w_circ*n_z)*rho_dS
        #print 'interpolating on the sphere takes %f seconds'%(time.time()-start)

        u_dev = []
        v_dev = []
        w_dev = []
        #area_sphere = []
        unx_vny_wnz = []
        epsilon     = []
        zmomflux    = []
        entr_dist   = []
        detr_dist   = []
        entr_rate   = []
        PnzdS         = []
        PnxdS         = []
        PnydS       = []
        xmomflux    = []
        ymomflux    = []
        for it in range(nt):
            area_sphere = np.sum(dS[it])
            u_dev.append( u_circ[it] - u_mean[it] )
            v_dev.append( v_circ[it] - v_mean[it] )
            w_dev.append( w_circ[it] - w_mean[it] )
            
            pfx             = np.sum(pnxdS[it])
            pfy             = np.sum(pnydS[it])
            pfz             = np.sum(pnzdS[it])
            #integral_mass_flux  = np.sum(mflux[it])
            integral_mass_flux = np.sum((u_dev[it]*n_x[it] + v_dev[it]*n_y[it] + w_dev[it]*n_z[it])*rho_dS[it])
            epsilon.append( integral_mass_flux/area_sphere )  # small correction (mass should be conserved in the volume of the thermal instantaneously)

            #unx_vny_wnz.append( (u_dev[it]*n_x[it] + v_dev[it]*n_y[it] + w_dev[it]*n_z[it] - epsilon[it])*rho_dS[it] )
            unx_vny_wnz.append( (u_dev[it]*n_x[it] + v_dev[it]*n_y[it] + w_dev[it]*n_z[it])*rho_dS[it] - epsilon[it]*dS[it] )
            
            zmomtr  = unx_vny_wnz[it]*w_dev[it]
            xmomtr  = unx_vny_wnz[it]*u_dev[it]
            ymomtr  = unx_vny_wnz[it]*v_dev[it]

            integral_zmomentum_transport = np.sum(zmomtr)
            integral_xmomentum_transport = np.sum(xmomtr)
            integral_ymomentum_transport = np.sum(ymomtr)
            integral_entrainment         = np.sum( unx_vny_wnz[it][np.where( np.ma.greater(unx_vny_wnz[it],0) )[0]] )
            integral_detrainment         = np.sum( unx_vny_wnz[it][np.where( np.ma.less( unx_vny_wnz[it],0) )[0]] )*(-1.) # make it positive by definition 

            if integral_entrainment > 0:
                entr_dist.append( w_mean[it]*mass[it]/integral_entrainment )         # entrainment distance in m. (inverse of fractional entrainment)
            else:
                entr_dist.append( np.nan )
            if integral_detrainment > 0:
                detr_dist.append( w_mean[it]*mass[it]/integral_detrainment )
            else:
                detr_dist.append( np.nan )
            entr_rate.append( integral_entrainment/volume[it] )                      # in kg m^{-3} s^{-1}
            PnzdS.append( (pfz-g*mass[it])/mass[it] )
            PnxdS.append( pfx/mass[it] )
            PnydS.append( pfy/mass[it] )
            zmomflux.append( integral_zmomentum_transport/mass[it] )
            xmomflux.append( integral_xmomentum_transport/mass[it] )
            ymomflux.append( integral_ymomentum_transport/mass[it] )
        self.entr_dist  = np.asarray( entr_dist)
        self.detr_dist  = np.asarray( detr_dist)               # detrainment distance in m (inverse of fractional detrainment)
        self.epsilon    = np.asarray( epsilon  )
        self.entr_rate  = np.asarray( entr_rate)
        self.PnzdS      = np.asarray( PnzdS     )
        self.PnxdS      = np.asarray( PnxdS     )
        self.PnydS      = np.asarray( PnydS     )
        self.zmomflux   = np.asarray( zmomflux )
        self.xmomflux   = np.asarray( xmomflux )
        self.ymomflux   = np.asarray( ymomflux )

        self.angles     = np.asarray( angles, dtype=object )
        entr_distr = []
        for it in range(nt):
            entr_distr.append( (unx_vny_wnz[it]/(mass[it]*w_mean[it]))/dS[it])
        self.entr_distr = np.asarray( entr_distr, dtype=object )         # fractional entrainment/detrainment distribution per arc dS. Note: if interpolated to a different angle discretization, each contribution must be then multiplied by the new dS!
        net_entr_term = [] 
        for it in range(1,nt): # this is the momentum budget term related to the change in size of the thermal in between time steps (net entrainment)
            net_entr_term.append( (self.mass[it]-self.mass[it-1])*np.mean([np.mean(w_dev[it]),np.mean(w_dev[it-1])])/(np.mean([self.mass[it],self.mass[it-1]])*self.dt) )
        self.net_entr_term = np.asarray( net_entr_term )

    def _check_for_discont_r( self, recompute_vel=False, threshold=0.6 ):
        """
        Finds any change in the thermal's radius that is larger than 'threshold' (0.4 means 40%) of the smallest radius 
        between any two timesteps. Such a large change in radius suggests that the thermal is not properly identified, so 
        we remove this part.
        """
        bad_points_low = []
        bad_points_high = []
        for it in range(1,self.nt):
            if abs(self.R_thermal[it]-self.R_thermal[it-1]) > np.amin([self.R_thermal[it-1],self.R_thermal[it]])*threshold:
                if it >= self.nt/2:
                    bad_points_high.append(it)
                if it < self.nt/2:
                    bad_points_low.append(it-1)
        if len(bad_points_low)>0:
            bad_low =np.amax(np.asarray(bad_points_low))
            self.log.append( 'Very large change in the radius at '+hhmm(self.time[bad_low])+'. I will start from '+hhmm(self.time[bad_low+1]) )
            #print 'Very large change in the radius at '+hhmm(self.time[bad_low])+'. I will start from '+hhmm(self.time[bad_low+1]) 
        else:
            bad_low = -1
        if len(bad_points_high)>0:
            bad_high = np.amin(np.asarray(bad_points_high))
            self.log.append( 'Very large change in the radius at '+hhmm(self.time[bad_high])+'. I will keep until '+hhmm(self.time[bad_high-1]) )
            #print 'Very large change in the radius at '+hhmm(self.time[bad_high])+'. I will keep until '+hhmm(self.time[bad_high-1]) 
        else:
            bad_high = self.nt
        if len(bad_points_low)>0 or len(bad_points_high)>0:
            #self.grid._counter_discont_r+=len(bad_points_low)+len(bad_points_high)
            if recompute_vel:
                if len(bad_points_low)>0:
                    self.u_thermal[bad_low+1] = (self.x_centre[bad_low+2]-self.x_centre[bad_low+1])/self.dt
                    self.v_thermal[bad_low+1] = (self.y_centre[bad_low+2]-self.y_centre[bad_low+1])/self.dt
                    self.w_thermal[bad_low+1] = (self.z_centre[bad_low+2]-self.z_centre[bad_low+1])/self.dt
                if len(bad_points_high)>0:
                    self.u_thermal[bad_high] = (self.x_centre[bad_high]-self.x_centre[bad_high-1])/self.dt
                    self.v_thermal[bad_high] = (self.y_centre[bad_high]-self.y_centre[bad_high-1])/self.dt
                    self.w_thermal[bad_high] = (self.z_centre[bad_high]-self.z_centre[bad_high-1])/self.dt
            if bad_low > -1 or bad_high < self.nt:
                self._crop_t_thermal( bad_low+1, bad_high, reload=False )


    def _check_min_W( self ): 
        #start=True

        if self.up:
            while self.nt>=self.min_t+1 and self._rel_vel()<self.W_min:
                if self.w_thermal[0]<self.w_thermal[-1]:
                    self.log.append( self.OBJ+' is too slow. Will remove first time step and recheck...' )
                    self._crop_t_thermal( 1, self.nt )
                else:
                    self.log.append( self.OBJ+' is too slow. Will remove last time step and recheck...' )
                    self._crop_t_thermal( 0, self.nt-1 )
        else:
            while self.nt>=self.min_t+1 and self._rel_vel()>self.W_min:
                if self.w_thermal[0]>self.w_thermal[-1]:
                    self.log.append( self.OBJ+' is too slow. Will remove first time step and recheck...' )
                    self._crop_t_thermal( 1, self.nt )
                else:
                    self.log.append( self.OBJ+' is too slow. Will remove last time step and recheck...' )
                    self._crop_t_thermal( 0, self.nt-1 )


    def _W_condition( self ):
        """
        this is to test whether the thermal is rising with a minimum average W, but also if it is rising with that same 
        minimum average W relative to its surroundings.
        """
        if self.up:
            return (self._rel_vel() >= self.W_min) and (np.mean(self.w_mean[:]) >= self.W_min)
        else:
            return (self._rel_vel() <= self.W_min) and (np.mean(self.w_mean[:]) <= self.W_min)


    def _check_for_multiple_thermals( self ):
        if self.dt==60:
            dist=7
        else:
            dist = 5*60./self.dt
        wmax_max = filters.maximum_filter(self.w_centre, dist)
        wmax_min = filters.minimum_filter(self.w_centre, dist)
        locmax = np.where(wmax_max==self.w_centre)[0]
        locmin = np.where(wmax_min==self.w_centre)[0]
        if len(locmax)>1 and len(locmin>1):
            if self.up:
                if locmax[0]==0:
                    locmax=locmax[1:]
                if locmax[-1]==len(self.w_centre)-1:
                    locmax=locmax[:-1]
                if len(locmax)>1:
                    # for cases in which there is a local min in the first timestep, followed by a local max in the second or third time steps (this would not be captured by the filter) 
                    if self.w_centre[0]<self.w_centre[locmax[0]] and locmin[0]!=0 and (locmax[0]==1 or locmax[0]==2):
                        locmin = np.concatenate(([0],locmin))
                    # (symmetric for last timestep):
                    if self.w_centre[-1]<self.w_centre[locmax[-1]] and locmin[-1]!=self.nt and (locmax[-1]==self.nt-2 or locmax[-1]==self.nt-3):
                        locmin = np.concatenate((locmin,[self.nt-1]))
                    thermals=[]
                    for i in range(len(locmin)-1):
                        if np.any((locmax>locmin[i])*(locmax<locmin[i+1])) and (locmin[i+1]-locmin[i]>=self.min_t):
                            thermals.append((locmin[i], locmin[i+1]))
                    self._split_thermal( thermals )
            else:
                print('split downdraft? Will not try to do that, sorry.')
            

    def _get_mean_W_env( self ):
        """
        Compute the mean vertical velocity in the environment that surrounds the thermal (excludes the thermal, includes a distance self.avg_dist_R*self.R_thermal 
        on either side, and 0.75 of that above and under it)
        """
        mean_W_env = np.zeros(self.nt)
        #avg_dist = np.rint(self.avg_dist_R*self.R_thermal/self.dx)
        entire_grid = index_grid(self.nx, self.ny, self.nz)
        subgrid = []
        n_jobs=self.n_jobs
        if n_jobs > 1:      
            # ***************************************************************
            #   parallelize by splitting into timesteps
            n_jobs, dtt = optimize_njobs( self.nt, n_jobs )
            it0 = 0
            subjob = []
            for ijob in range(n_jobs-1):
                it1 = it0 + dtt
                subjob.append( [it0,it1] )
                it0 = it1
            subjob.append( [it0, self.nt] )
            jobs = []
            for ijob in range(n_jobs):
                t0=subjob[ijob][0]
                t1=subjob[ijob][1]
                jobs.append( (self.x_grid, self.y_grid, self.hgt_c, self.x_centre[t0:t1], self.y_centre[t0:t1], self.z_centre[t0:t1], self.R_thermal[t0:t1], entire_grid, self.avg_dist_R, self.w_c[:,:,:,t0:t1] ) )
            ( result ) = Parallel(n_jobs=n_jobs)(delayed(_get_W_env_tstep)(*jobs[i]) for i in range(len(jobs)))

            for ijob in range(n_jobs):
                mean_W_env[subjob[ijob][0]:subjob[ijob][1]] = result[ijob][0]
                subgrid = subgrid + result[ijob][1]
        else:
            # ***************************************************************
            #   serial computation
            for it in range(self.nt):
                # these 3 condidtions must be met in order to be inside a box that contains the thermal. This box will be excluded from the mean_W_env:
                box_x = np.where((self.x_grid[:]>=self.x_centre[it]-self.R_thermal[it])*(self.x_grid[:]<=self.x_centre[it]+self.R_thermal[it]))[0]
                box_y = np.where((self.y_grid[:]>=self.y_centre[it]-self.R_thermal[it])*(self.y_grid[:]<=self.y_centre[it]+self.R_thermal[it]))[0]
                box_z = np.where((self.hgt_c[:]>=self.z_centre[it]-self.R_thermal[it])*(self.hgt_c[:]<=self.z_centre[it]+self.R_thermal[it]))[0]
                # 'True' elements are outside the thermal box:
                outside_thermal_box = ~(np.in1d(entire_grid[:,0],box_x)*np.in1d(entire_grid[:,1],box_y)*np.in1d(entire_grid[:,2],box_z))
                
                # points too far from the thermal are also excluded:
                x_2large = np.where((self.x_grid>(self.x_centre[it]+self.R_thermal[it]+self.avg_dist_R*self.R_thermal[it])))[0]
                x_2small = np.where((self.x_grid<(self.x_centre[it]-self.R_thermal[it]-self.avg_dist_R*self.R_thermal[it])))[0]
                # 'True' elements are at an avg_dist_R from the centre in the x direction:
                x_valid  = ~(np.in1d(entire_grid[:,0],np.concatenate((x_2small, x_2large))))
                y_2large = np.where((self.y_grid>(self.y_centre[it]+self.R_thermal[it]+self.avg_dist_R*self.R_thermal[it])))[0]
                y_2small = np.where((self.y_grid<(self.y_centre[it]-self.R_thermal[it]-self.avg_dist_R*self.R_thermal[it])))[0]
                # 'True' elements are at an avg_dist_R from the centre in the y direction:
                y_valid  = ~(np.in1d(entire_grid[:,1],np.concatenate((y_2small, y_2large))))
                z_2large = np.where((self.hgt_c > (self.z_centre[it] + self.R_thermal[it] + 0.75*self.avg_dist_R*self.R_thermal[it])))[0]
                z_2small = np.where((self.hgt_c < (self.z_centre[it] - self.R_thermal[it] - 0.75*self.avg_dist_R*self.R_thermal[it])))[0]
                # 'True' elements are outside of the subgrid in the z direction:
                z_valid  = ~(np.in1d(entire_grid[:,2],np.concatenate((z_2small, z_2large))))

                env = np.where((outside_thermal_box)*x_valid*y_valid*z_valid)[0]
                subgrid.append( entire_grid[env].astype(int) )
                slice_w_c = self.w_c[subgrid[it][:,0],subgrid[it][:,1],subgrid[it][:,2],it]
                masked_slice_w_c = np.ma.masked_array( slice_w_c, np.isnan(slice_w_c) )
                mean_W_env[it] = np.mean( masked_slice_w_c )
        self.mean_W_env = mean_W_env
        self.thermal_env_grid = subgrid


    def _get_buoyancy( self ):
        """
        Computes the buoyancy of the thermal, once its radius is known. Must be run after mean_W_env is computed, so that thermal_env_grid is already known.
        """
        self._compute_density_profile()
        
        for it in range(self.nt):
            ix, iy, iz = np.unravel_index( np.arange(self.nnx[it]*self.nny[it]*self.nnz[it]), (self.nnx[it],self.nny[it],self.nnz[it]) )
            if ( (self.nnx[it]*self.nny[it]*self.nnz[it]>len(self.rho_here[it])) or (self.nnx[it]*self.nny[it]*self.nnz[it]>len(iz)) or (self.nnx[it]*self.nny[it]*self.nnz[it]>len(self.rho_m_here[it])) or (self.nnx[it]*self.nny[it]*self.nnz[it]>len(self.data[it])) ):
                pdb.set_trace()
            for l in range(self.nnx[it]*self.nny[it]*self.nnz[it]):
                self.data[it,l,6] = (self.rho_here[it][l] - self.rhoavg[it][iz[l]])/self.rhoavg[it][iz[l]]
            self.data[it,l,7] = (self.rho_m_here[it][l] - self.rhoavg_m[it][iz[l]])/self.rhoavg_m[it][iz[l]]

        buoy     = []
        buoy_m   = []
        buoy_map = []
        buoy_i   = []
        for it in range(self.nt):
            vol_here = self.dx*self.dy*self.data[it,self.R_ord_data_ind[it,:],5]
            mass_here = self.data[it,self.R_ord_data_ind[it,:],1]*vol_here[:]
            B_disk_here = self.data[it,self.R_ord_data_ind[it,:],6]*mass_here[:]
            B_moist_here = self.data[it,self.R_ord_data_ind[it,:],7]*mass_here[:]
            B_disk = np.cumsum(B_disk_here)
            B_moist = np.cumsum(B_moist_here)

            mass = self.integrals[it,:,4]   
            self.integrals[it,:,3] = -g*B_disk/mass
            self.integrals[it,:,6] = -g*B_moist/mass
            buoy.append(   np.interp( self.R_thermal[it], self.integrals[it,:,0], self.integrals[it,:,3], left=self.integrals[it,0,3] ) )
            buoy_m.append( np.interp( self.R_thermal[it], self.integrals[it,:,0], self.integrals[it,:,6], left=self.integrals[it,0,6] ) )
            # save the bouyancy field of the thermal:
            ind = np.where(self.integrals[it,:,0]<=self.R_thermal[it])[0]
            new_grid = index_grid(self.nnx[it],self.nny[it],self.nnz[it])
            indices = new_grid[self.R_ord_data_ind[it][ind]]
            # save x-coor, z-coor, buoyancy:
            buoy_map.append( np.vstack((self.x_subgrid[it][indices[:,0].astype(int)],self.y_subgrid[it][indices[:,1].astype(int)],self.z_subgrid[it][indices[:,2].astype(int)], -g*B_disk_here[ind]/mass_here[ind])) )
            if np.isnan(buoy[it]):
                self.log.append( '***Warning*** invalid buoyancy at '+hhmm(self.time[it])+'.' )#self.R_thermal=',self.R_thermal[it], '\nself.integrals[it][:,0]=',self.integrals[it][:,0],'\nself.integrals[it][:,3]=',self.integrals[it][:,3] 
                #print '***Warning*** invalid buoyancy at '+hhmm(self.time[it])+'.'

        self.buoy_map   = buoy_map
        #self.buoy_map   = np.asarray(buoy_map)
        self.buoy       = np.asarray(buoy)
        self.buoy_m     = np.asarray(buoy_m)

    
    def _compute_density_profile( self ):
        """
        computes the average density around the thermal, as a function of height. 
        not now!!!: Uses self.avg_dist_R*Radius on each side, excluding the thermal.
        now uses the entire domain
        """
        rhoavg = []
        rhoavg_m = []
        subgrid = self.thermal_env_grid
        for it in range(self.nt):
            rhoavg.append( np.zeros(self.nnz[it], dtype=prec) )
            rhoavg_m.append( np.zeros(self.nnz[it], dtype=prec) )
                
            for k in range(self.nnz[it]):
                indices_levk = np.where(subgrid[it][:,2]==k+self.k0[it])[0]
                if indices_levk.size>0:
                    #rho_c_around = self.rho_c[:,:,k+self.k0[it]]
                    #rho_m_around = self.rho_m[:,:,k+self.k0[it]]
                    rho_c_around = self.rho_c[subgrid[it][indices_levk,0], subgrid[it][indices_levk,1], k+self.k0[it]]
                    rho_m_around = self.rho_m[subgrid[it][indices_levk,0], subgrid[it][indices_levk,1], k+self.k0[it]]
                else:
                    rho_c_around = np.array([np.nan])
                    rho_m_around = np.array([np.nan])
                rho_c_around = np.ma.masked_array( rho_c_around, np.isnan(rho_c_around) )
                rho_m_around = np.ma.masked_array( rho_m_around, np.isnan(rho_m_around) )
                if np.all(np.isnan(rho_c_around.data)):
                    rhoavg[it][k] = np.nan
                else:
                    rhoavg[it][k]   = np.ma.average( rho_c_around )
                if np.all(np.isnan(rho_m_around.data)):
                    rhoavg_m[it][k] = np.nan
                else:
                    rhoavg_m[it][k] = np.ma.average( rho_m_around )
        self.rhoavg = rhoavg
        self.rhoavg_m = rhoavg_m


    def _check_always_rising( self ):
        if np.any(self.z_centre[1:]-self.z_centre[:-1]<0) and self.up: #only for rising thermals, not downdrafts!
            it = np.where(self.z_centre[1:]-self.z_centre[:-1]<0)[0]
            if len(it)>1:
                it=it[0]
            self.log.append( 'Thermal is falling at at least one time step! will crop thermal...')
            #print 'Thermal is falling at at least one time step! will crop thermal...'
            self._crop_t_thermal(0,int(it))

    def _check_mom_budget_fit( self ):
        # if the expected trajectory according to the (vertical?) momentum budgdet is too far from the actual trajectory,
        # we remove the first or the last time step, whichever has the slowest updraft.
        #*******
        # 'too far' means that the distance between expected and actual centres at the last time step is
        # larger than twice the average radius of the thermal and larger than 20% of the vertical distance traveled.
        while (np.abs(self.pos_exp[-1]-self.z_centre[-1])>2.*np.mean(self.R_thermal)) and (np.abs(self.pos_exp[-1]-self.z_centre[-1])>0.2*(self.z_centre[-1]-self.z_centre[0])) and (self.nt >= self.min_t):
            if self.w_centre[-1]>self.w_centre[0]:
                self.log.append( 'Very large discrepancy between tracked and expected position (mom. budget). Will remove first timestep and recheck...' )
                #print 'Very large discrepancy between tracked and expected position (mom. budget). Will remove first timestep and recheck...' 
                self._crop_t_thermal(1,self.nt)
            else:
                self.log.append( 'Very large discrepancy between tracked and expected position (mom. budget). Will remove last timestep and recheck...' )
                #print 'Very large discrepancy between tracked and expected position (mom. budget). Will remove last timestep and recheck...' 
                self._crop_t_thermal(0,self.nt-1)
            self._compute_expected_trajectory()


    def _rel_vel( self ):
        rel_vel = 0.
        if self.nt > 0:
            if hasattr(self, 'mean_W_env'):
                rel_vel = np.mean(self.w_mean[:]) - np.mean(self.mean_W_env[:])
            else:
                rel_vel = np.mean(self.w_mean[:])
        return rel_vel


    def _compute_expected_trajectory( self ):
        nt = self.nt
        dt = self.dt
        self.acc = self.PnzdS + self.zmomflux # this is the total acceleration of the thermal at whole time steps without including the net entrainment effect (change in size)
        #av_acc = 0.5*(self.acc[1:] + self.acc[:-1]) # average acceleration in between time-steps
        av_acc = 0.5*(self.acc[1:] + self.acc[:-1]) + self.net_entr_term # average acceleration in between time-steps, including net entrainment effect (change in size)
        # integrate the position, starting at z_center with vertical velocity = w_parcel:
        self.pos_exp = np.zeros(nt)
        w_theory = np.zeros(nt)
        self.pos_exp[0] = self.z_centre[0]
        w_theory[0] = self.w_thermal[0]
        for it in range(1,nt):
            self.pos_exp[it] = self.pos_exp[it-1] + w_theory[it-1]*dt + 0.5*av_acc[it-1]*dt*dt
            w_theory[it] = w_theory[it-1] + av_acc[it-1]*dt
    

    def _split_thermal( self, limits_list ):
        if len(limits_list)>1:
            self.log.append( 'Splitting into %d thermals...'%(len(limits_list)) )
            #print 'Splitting into %d thermals...'%(len(limits_list))
            i=len(limits_list)-1
            while i > 0:
                thermal = self.grid.create_thermal_grid( self.max_radius, t00=limits_list[i][0], prev_thermal=self )
                self._crop_t_thermal( 0, limits_list[i][0] + 1 )
                i -= 1
            

    def _create_folder_and_fname( self ):
        if self.shifted!=0.:
            prefix = 'shifted'+'0%d'%(int(np.abs(self.shifted*100.)))+'_'
        else:
            prefix = ''
        if self.cell!='':
            if os.path.isdir( self.cell ):
                pass
            else:
                os.mkdir ( self.cell )
        if self.up:
            folder =  prefix + 'thermal_' + hhmm(self.time[0],char='_') + '_to_' + hhmm(self.time[-1],char='_') + '_x' + str(int(self.x_centre[0])) + 'y' + str(int(self.y_centre[0])) + 'z' + str(int(self.z_centre[0]))
        else:
            folder = prefix + 'downdraft_' + hhmm(self.time[0],char='_') + '_to_' + hhmm(self.time[-1],char='_') + '_x' + str(int(self.x_centre[0])) + 'y' + str(int(self.y_centre[0])) + 'z' + str(int(self.z_centre[0]))
        if self.cell!='':
            self.fname = self.cell + '/' + folder + '/' + folder
            if os.path.isdir( self.cell + '/' + folder ):
                print('folder exists...')
                #os.system( 'rm -r ' + folder )
            else:
                os.mkdir( self.cell + '/' + folder )
        else:
            self.fname = folder + '/' + folder
            if os.path.isdir( folder ):
                print('folder exists...')
                #os.system( 'rm -r ' + folder )
            else:
                os.mkdir( folder )

    def _expand_attributes_one_tstep( self, direction=1 ):
        direc = direction
        self.tsteps = add_one_tstep( self.tsteps, direc ).astype(int)
        if direc==1:
            self.tsteps[-1] = self.tsteps[-2] + 1
        if direc==-1:
            self.tsteps[0]     = self.tsteps[1] - 1
    
        self.nt                 = len(self.tsteps)
        self.time               = self.tsteps + self.grid.hr0*60 + self.grid.min0 + self.grid.sec0/60.    # time in minutes where the thermal is traced
        self.xmax               = add_one_tstep( self.xmax          , direc )
        self.ymax               = add_one_tstep( self.ymax          , direc )
        self.hmax               = add_one_tstep( self.hmax          , direc )
        self.x_centre           = add_one_tstep( self.x_centre      , direc )
        self.y_centre           = add_one_tstep( self.y_centre      , direc )
        self.z_centre           = add_one_tstep( self.z_centre      , direc ) 
        self.ix_centre          = add_one_tstep( self.ix_centre     , direc ) 
        self.iy_centre          = add_one_tstep( self.iy_centre     , direc )
        self.iz_centre          = add_one_tstep( self.iz_centre     , direc )
        self.u_thermal          = add_one_tstep( self.u_thermal     , direc )
        self.v_thermal          = add_one_tstep( self.v_thermal     , direc )
        self.w_thermal          = add_one_tstep( self.w_thermal     , direc ) 
        self.x_centregrid       = add_one_tstep( self.x_centregrid  , direc )
        self.z_centregrid       = add_one_tstep( self.z_centregrid  , direc )
        self.i_left             = add_one_tstep( self.i_left        , direc )
        self.i_right            = add_one_tstep( self.i_right       , direc )
        self.k0                 = add_one_tstep( self.k0            , direc )
        self.k1                 = add_one_tstep( self.k1            , direc )
        self.nnx                = add_one_tstep( self.nnx           , direc )
        self.nny                = add_one_tstep( self.nny           , direc )
        self.nnz                = add_one_tstep( self.nnz           , direc )
    
        self.R_thermal          = add_one_tstep( self.R_thermal     , direc )
        self.PnxdS              = add_one_tstep( self.PnxdS         , direc )
        self.PnzdS              = add_one_tstep( self.PnzdS         , direc ) 
        self.buoy               = add_one_tstep( self.buoy          , direc )
        self.buoy_m             = add_one_tstep( self.buoy_m        , direc )
        self.entr_dist          = add_one_tstep( self.entr_dist     , direc )
        self.detr_dist          = add_one_tstep( self.detr_dist     , direc )
        self.entr_rate          = add_one_tstep( self.entr_rate     , direc )
        self.epsilon            = add_one_tstep( self.epsilon       , direc )
        self.xmomflux           = add_one_tstep( self.xmomflux      , direc )
        self.ymomflux           = add_one_tstep( self.ymomflux      , direc )
        self.mass               = add_one_tstep( self.mass          , direc )
        #self.mass_cond          = add_one_tstep( self.mass_cond     , direc )
        self.massflux           = add_one_tstep( self.massflux      , direc )
        self.u_centre           = add_one_tstep( self.u_centre      , direc )
        self.u_mean             = add_one_tstep( self.u_mean        , direc )
        self.v_mean             = add_one_tstep( self.v_mean        , direc )
        self.zmomflux           = add_one_tstep( self.zmomflux      , direc )
        self.volume             = add_one_tstep( self.volume        , direc )
        self.volume_err         = add_one_tstep( self.volume_err    , direc )
        self.w_centre           = add_one_tstep( self.w_centre      , direc )
        self.w_mean             = add_one_tstep( self.w_mean        , direc )
        self.mse_thermal        = add_one_tstep( self.mse_thermal   , direc )
        self.latheat_thermal    = add_one_tstep( self.latheat_thermal, direc )
        self.maxlatheat_thermal = add_one_tstep( self.maxlatheat_thermal, direc )
        self.sctot_thermal      = add_one_tstep( self.sctot_thermal   , direc )
        self.maxsctot_thermal   = add_one_tstep( self.maxsctot_thermal, direc )
        self.noninduc_thermal   = add_one_tstep( self.noninduc_thermal, direc )
        self.epotential_thermal   = add_one_tstep( self.epotential_thermal, direc )
        self.maxepotential_thermal= add_one_tstep( self.maxepotential_thermal, direc )
        self.qngraupel_thermal    = add_one_tstep( self.qngraupel_thermal, direc )

        self.rho_c         = self.grid.rho_c[:,:,self.tsteps] # total density (including condensate)
        self.rho_m         = self.grid.rho_m[:,:,self.tsteps]
        #self.rho_condensate     = self.grid.rho_condensate[:,:,self.tsteps]
        self.u_c         = self.grid.u_c[:,:,self.tsteps]
        self.v_c         = self.grid.v_c[:,:,self.tsteps]
        self.w_c         = self.grid.w_c[:,:,self.tsteps]
        self.ptot         = self.grid.ptot[:,:,self.tsteps]
        self.vorticity         = self.grid.vorticity[:,:,self.tsteps]
      
    def make_movie( self, cropx=False, cropy=False, gif=False ):
        x = self.x_centre
        z = self.z_centre
        r = self.R_thermal
        t = self.time
        fname = self.fname
        xlims = [np.amax( [np.amin(x[:])/1e3 - 3, self.x_grid[0]/1e3] ), np.amin( [np.amax(x[:])/1e3 + 3, self.x_grid[-1]/1e3] )]
        ylims = [ 0, np.amin( [np.amax(z[:]+self.R_thermal[:])/1e3 + 2, self.hgt_c[-1]/1e3] )]
        for k in range(len(x)):
            s = '%02d'%int(k)
            self.show_wmax( t=t[k], x=x[:k+1], z=z[:k+1], R=r[k], n=k+1, fname='frame_'+s+'.jpg', xlims=xlims, ylims=ylims )
        os.system( 'cp frame_' + '%02d'%int(len(x)-1) + '.jpg frame_' + '%02d'%int(len(x)) + '.jpg' ) # repeat the last frame
        if gif:
            os.system( 'convert frame_??.jpg -delay 40 ' + fname + '.gif' )
        os.system( 'mkdir tmp_movie_folder' )
        os.system( 'convert frame_??.jpg tmp_movie_folder/frame%05d.jpg' )
        os.system( 'rm -rf frame_??.jpg' )
        os.system( 'ffmpeg -loglevel 0 -r 3 -qscale 1 -i tmp_movie_folder/frame%05d.jpg -y -an ' + fname + '.avi' )
        os.system( 'rm -R tmp_movie_folder' )
            #self.show_wmax( self.time[-1], self.x_centre, self.z_centre, self.R_thermal[-1], n=self.nt, fname=self.fname + 'vort_streamlines.pdf' )
    

    def show_wmax( self, t, x, z, R, n=1, qcloud_thr=1e-5, fname=None, xlims=None, ylims=None ): 
        from matplotlib import rc
        import matplotlib.pyplot as plt
        it = int(( t- (self.grid.hr0*60+self.grid.min0) )*60./self.dt)
        psi = self.grid.compute_streamfunction( it )
        
        nx = self.nx
        nz = self.nz
        vorticity = self.grid.vorticity
        dh = self.dh
        dh_c = self.dh_c
        p_max = np.nanmax(abs(vorticity))
        p_min = -p_max
        N = 150
        rc('font',**{'family':'FreeSans'})
        rc('text', usetex=False)
        fig=plt.figure(figsize=(16,10))
        ax=plt.gca()
        plt.contourf(self.x_grid/1e3, self.hgt_c/1e3, np.swapaxes(vorticity[:,:,it],0,1), N, levels=np.arange(N)*(2*p_max)/(N-1)-p_max)
        CB = plt.colorbar( pad=0, ticks=[-p_max, 0, p_max], format='%.2f')
        for i in range(n):
            plt.plot(x[i]/1e3,z[i]/1e3, 'k+', ms=10, mew=3)
        if xlims!=None:
            plt.xlim( xlims[0], xlims[1] )
        if ylims!=None:
            plt.ylim( ylims[0], ylims[1] )
        #if not cropx and not cropy:
        #    plt.axis('tight')
        plt.title( hhmm(t) )
        #plt.title( '%02d'%(t/60) + ':' +  '%02d'%(np.mod(t,60)) )
        plt.xlabel('X (km)')
        plt.ylabel('Height (km)')
        l,b,w,h = plt.gca().get_position().bounds
        ll,bb,ww,hh = CB.ax.get_position().bounds
        CB.ax.set_position([ll+0.1*w, b, ww, h])
        N=30
        plt.contour(self.x_grid/1e3, self.hgt_c/1e3, np.swapaxes(self.grid.qcloud[:,:,it],0,1),1,levels=[qcloud_thr],colors='r',linewidths=2)
        plt.matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        plt.contour(self.x_grid[1:-1]/1e3, self.hgt_c[1:-1]/1e3, np.swapaxes(psi,0,1),N,colors='k')
        circle = Circle((x[-1]/1e3,z[-1]/1e3), R/1e3, facecolor='none',edgecolor='k',linewidth=3 )
        ax.add_artist(circle)
        if fname==None:
            plt.show() 
        else:
            plt.savefig( fname )
    

    def square_distance_to_centre( self, t, x, z ):
        result = np.square(x-self.x_centre[t])+np.square(z-self.z_centre[t])
        if result == np.nan:
            self.log.append( 'Warning! nan encountered!' )
            #print 'Warning! nan encountered!' 
        return result
    
    
    def _write_data_to_file( self ):
        import datetime as dt
        fname = self.fname + '_data.dat'
        date=np.zeros_like(self.time)
        for it in range(len(self.time)):
            idate = dt.datetime(self.YY0, self.MM0, self.DD0,0,0,0)+dt.timedelta(minutes=self.time[it])
            date[it] = int(idate.strftime('%Y%m%d'))

        data = np.vstack( (self.time, self.w_centre, self.R_thermal, self.mass, self.xmax/1e3, self.ymax/1e3, self.hmax/1e3, self.x_centre/1e3, self.y_centre/1e3, self.z_centre/1e3, self.u_mean, self.v_mean, self.w_mean, self.PnzdS, self.zmomflux, self.buoy, self.buoy_m, self.acc, self.PnxdS, self.xmomflux, self.PnxdS + self.xmomflux, self.PnydS, self.ymomflux, self.PnydS + self.ymomflux, self.entr_dist/1e3, self.entr_rate, self.pos_exp, self.detr_dist/1e3, self.mse_thermal, self.mse_env, self.latheat_thermal, self.maxlatheat_thermal, self.qnice_thermal, self.qncloud_thermal, self.qnrain_thermal, self.cldnuc_thermal, self.rh_thermal, self.qcloud_thermal, self.qrain_thermal, self.qice_thermal, self.qvapor_thermal, self.sctot_thermal, self.maxsctot_thermal, self.noninduc_thermal, self.epotential_thermal, self.maxepotential_thermal, self.qngraupel_thermal, self.qicesnow_thermal, self.qghail_thermal, date) )#, self.mass_wvapor) )

        fname = self.fname + '_data.npy'
        np.save( fname, data )
        fname = self.fname + '_buoy_map.npz'
        np.savez( fname, *[self.buoy_map[i] for i in arange(len(self.buoy_map))] )
        fname = self.fname + '_angles.npy'
        np.save( fname, self.angles )
        fname = self.fname + '_entr_distr.npy'
        np.save( fname, self.entr_distr )
        fname = self.fname + '_mse_mixing.npy'
        np.save( fname, self.mixing_mse )
        fname = self.fname + '_net_entr_term.npy'
        np.save( fname, self.net_entr_term )


    def _release_memory( self ):
        del self.grid           
        del self.nx             
        del self.ny             
        del self.nz             
        del self.x0             
        del self.hgt            
        del self.hgt_c          
        del self.x_grid         
        del self.y_grid         
        del self.dh             
        del self.dh_c           
        del self.tsteps         
        del self.time           
         
        del self.xmax           
        del self.ymax           
        del self.hmax           
        del self.x_centre       
        del self.y_centre       
        del self.z_centre       
        del self.ix_centre      
        del self.iy_centre      
        del self.iz_centre      
        del self.u_thermal      
        del self.v_thermal      
        del self.w_thermal      
        del self.x_centregrid   
        del self.y_centregrid   
        del self.z_centregrid   
         
        del self.rho_c          
        del self.rho_m          
        #del self.rho_condensate 
        del self.u_c            
        del self.v_c            
        del self.w_c            
        del self.ptot           
        del self.mse            
        gc.collect()

        if hasattr(self, 'PnzdS'):
            del self.u_centre        
            del self.v_centre       
            del self.w_centre        
            del self.i_left         
            del self.j_left         
            del self.i_right        
            del self.j_right        
            del self.k0              
            del self.k1              
            del self.nnx             
            del self.nny            
            del self.nnz             
            del self.subgrid         
            del self.rho_here        
            #del self.rho_condensate_here 
            del self.rho_m_here      
            #del self.rhoavg_int      
            #del self.rhoavg_m_int    
            del self.u_c_here       
            del self.v_c_here       
            del self.w_c_here        
            del self.latheat_here   
            del self.sctot_here     
            del self.qnice_here     
            del self.qncloud_here   
            del self.qnrain_here    
            del self.noninduc_here  
            del self.cldnuc_here    
            del self.rh_here        
            del self.qice_here     
            del self.qcloud_here   
            del self.qrain_here    
            del self.qvapor_here    
            del self.epotential_here   
            del self.qngraupel_here   
            del self.qicesnow_here
            del self.qghail_here
            del self.x_subgrid       
            del self.y_subgrid      
            del self.z_subgrid      
            del self.R_thermal      

            del self.PnzdS          
            del self.zmomflux        
            del self.entr_dist       
            del self.detr_dist      
            del self.epsilon         
            del self.entr_rate       
            del self.PnxdS           
            del self.xmomflux        
            del self.PnydS          
            del self.ymomflux       
            del self.w_mean         
            del self.v_mean         
            del self.u_mean          
            del self.buoy            
            del self.buoy_m          
            del self.mass            
            del self.volume          
            del self.volume_err      
            #del self.mass_cond       
            del self.massflux        
            del self.data            
            del self.integrals       
            del self.R_ord_data_ind  
            del self.mse_thermal    
            del self.latheat_thermal
            del self.maxlatheat_thermal
            del self.epotential_thermal
            del self.maxepotential_thermal
            del self.qngraupel_thermal
            if hasattr(self, 'buoy_map'):
                del self.buoy_map
            if hasattr(self, 'mse_env'): 
                del self.mse_env       
            gc.collect()


def _get_W_env_tstep( x_grid, y_grid, hgt_c, x_centre, y_centre, z_centre, R_thermal, entire_grid, avg_dist_R, w_c ):
    mean_W_env = np.zeros(len(x_centre))
    subgrid=[]
    for it in range(len(x_centre)):
        # these 3 condidtions must be met in order to be inside a box that contains the thermal. This box will be excluded from the mean_W_env:
        box_x = np.where((x_grid[:]>=x_centre[it]-R_thermal[it])*(x_grid[:]<=x_centre[it]+R_thermal[it]))[0]
        box_y = np.where((y_grid[:]>=y_centre[it]-R_thermal[it])*(y_grid[:]<=y_centre[it]+R_thermal[it]))[0]
        box_z = np.where((hgt_c[:]>=z_centre[it]-R_thermal[it])*(hgt_c[:]<=z_centre[it]+R_thermal[it]))[0]
        # 'True' elements are outside the thermal box:
        outside_thermal_box = ~(np.in1d(entire_grid[:,0],box_x)*np.in1d(entire_grid[:,1],box_y)*np.in1d(entire_grid[:,2],box_z))
        
        # points too far from the thermal are also excluded:
        x_2large = np.where((x_grid>(x_centre[it]+R_thermal[it]+avg_dist_R*R_thermal[it])))[0]
        x_2small = np.where((x_grid<(x_centre[it]-R_thermal[it]-avg_dist_R*R_thermal[it])))[0]
        # 'True' elements are at an avg_dist_R from the centre in the x direction:
        x_valid  = ~(np.in1d(entire_grid[:,0],np.concatenate((x_2small, x_2large))))
        y_2large = np.where((y_grid>(y_centre[it]+R_thermal[it]+avg_dist_R*R_thermal[it])))[0]
        y_2small = np.where((y_grid<(y_centre[it]-R_thermal[it]-avg_dist_R*R_thermal[it])))[0]
        # 'True' elements are at an avg_dist_R from the centre in the y direction:
        y_valid  = ~(np.in1d(entire_grid[:,1],np.concatenate((y_2small, y_2large))))
        z_2large = np.where((hgt_c > (z_centre[it] + R_thermal[it] + 0.75*avg_dist_R*R_thermal[it])))[0]
        z_2small = np.where((hgt_c < (z_centre[it] - R_thermal[it] - 0.75*avg_dist_R*R_thermal[it])))[0]
        # 'True' elements are outside of the subgrid in the z direction:
        z_valid  = ~(np.in1d(entire_grid[:,2],np.concatenate((z_2small, z_2large))))

        env = np.where((outside_thermal_box)*x_valid*y_valid*z_valid)[0]
        subgrid.append( entire_grid[env].astype(int) )
        slice_w_c = w_c[subgrid[it][:,0],subgrid[it][:,1],subgrid[it][:,2],it]
        masked_slice_w_c = np.ma.masked_array( slice_w_c, np.isnan(slice_w_c) )
        mean_W_env[it] = np.mean( masked_slice_w_c )
    return (mean_W_env, subgrid)


def create_small_subgrid_for_interpolation( ix_c, iy_c, iz_c, nx, ny, nz, dim=4 ):
    if np.mod(dim,2)!=0:
        print( 'Warning: dim must be even number!')
        return None
    else:
        left_points = dim/2. - 1 
        ixlow = np.amax([0, ix_c.astype(int)-left_points])
        iylow = np.amax([0, iy_c.astype(int)-left_points])
        izlow = np.amax([0, iz_c.astype(int)-left_points])
        ixhigh = np.amin([ixlow + dim, nx])
        iyhigh = np.amin([iylow + dim, ny])
        izhigh = np.amin([izlow + dim, nz])
        dimx=ixhigh-ixlow
        dimy=iyhigh-iylow
        dimz=izhigh-izlow
        subgrid = index_grid(dimx,dimy,dimz,x0=ixlow,y0=iylow,z0=izlow)
        return subgrid, int(ixlow), int(ixhigh), int(iylow), int(iyhigh), int(izlow), int(izhigh)


def create_small_subgrid_for_interpolation2( ix_c, iy_c, iz_c, nx, ny, nz, dim=4 ):
    """
    same as above, but works for ix_c, iy_c etc as arrays. Output is lists!
    """
    if np.mod(dim,2)!=0:
        print( 'Warning: dim must be even number!')
        return None
    else:
        left_points = int( dim/2. - 1 )
        ixlow = ix_c.astype(int)-left_points
        iylow = iy_c.astype(int)-left_points
        izlow = iz_c.astype(int)-left_points
        ixlow[np.where( ixlow < 0 )] = 0
        iylow[np.where( iylow < 0 )] = 0
        izlow[np.where( izlow < 0 )] = 0
        
        ixhigh = ixlow + int(dim)
        iyhigh = iylow + int(dim)
        izhigh = izlow + int(dim)
        ixhigh[np.where(ixhigh > nx)] = nx
        iyhigh[np.where(iyhigh > ny)] = ny
        izhigh[np.where(izhigh > nz)] = nz

        dimx=ixhigh-ixlow
        dimy=iyhigh-iylow
        dimz=izhigh-izlow
        subgrid = []
        #for i in range(len(dimx)):
        #    subgrid.append( index_grid(dimx[i],dimy[i],dimz[i],x0=ixlow[i],y0=iylow[i],z0=izlow[i]) )
        #return subgrid, ixlow, ixhigh, iylow, iyhigh, izlow, izhigh
        return ixlow, ixhigh, iylow, iyhigh, izlow, izhigh


def joblib_sfc_interp_split_sphere( subjob, ix_c, iy_c, iz_c, ptot, u_c, v_c, w_c, rho_c, nx, ny, nz ):
    """
    interpolate on the given points on the surface of the sphere
    """
    int_method='linear'
    press_circ = np.zeros(subjob[1]-subjob[0])
    u_circ = np.zeros(subjob[1]-subjob[0])
    v_circ = np.zeros(subjob[1]-subjob[0])
    w_circ = np.zeros(subjob[1]-subjob[0])
    dens_circ = np.zeros(subjob[1]-subjob[0])
    subgrid, i0, i1, j0, j1, k0, k1 = create_small_subgrid_for_interpolation2( ix_c[subjob[0]:subjob[1]], iy_c[subjob[0]:subjob[1]], iz_c[subjob[0]:subjob[1]], nx, ny, nz )
    i00 = subjob[0]
    for iS in range(subjob[0],subjob[1]):
        press_circ[iS-i00]  = pol.griddata( subgrid[iS-i00], ptot[i0[iS-i00]:i1[iS-i00], j0[iS-i00]:j1[iS-i00], k0[iS-i00]:k1[iS-i00]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
        u_circ    [iS-i00]  = pol.griddata( subgrid[iS-i00], u_c[i0[iS-i00]:i1[iS-i00], j0[iS-i00]:j1[iS-i00], k0[iS-i00]:k1[iS-i00]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
        v_circ    [iS-i00]  = pol.griddata( subgrid[iS-i00], v_c[i0[iS-i00]:i1[iS-i00], j0[iS-i00]:j1[iS-i00], k0[iS-i00]:k1[iS-i00]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
        w_circ    [iS-i00]  = pol.griddata( subgrid[iS-i00], w_c[i0[iS-i00]:i1[iS-i00], j0[iS-i00]:j1[iS-i00], k0[iS-i00]:k1[iS-i00]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
        dens_circ [iS-i00]  = pol.griddata( subgrid[iS-i00], rho_c[i0[iS-i00]:i1[iS-i00], j0[iS-i00]:j1[iS-i00], k0[iS-i00]:k1[iS-i00]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
    return press_circ, u_circ, v_circ, w_circ, dens_circ


def joblib_sfc_interp_split_time( subjob, x_centre, y_centre, z_centre, r, angles, x_grid, y_grid, x0, y0, hgt_c, ptot, u_c, v_c, w_c, rho_c ):
    """
    interpolate the points on the surface for the timesteps given in subjob
    """
    press_circ  = [] 
    u_circ      = []
    v_circ      = []
    w_circ      = []
    dens_circ   = []
    nx = len(x_grid)
    ny = len(y_grid)
    nz = len(hgt_c)
    
    for it in range(subjob[0],subjob[1]):
        x_m = x_centre[it] + r[it]*np.cos(angles[it][:,1])*np.cos(angles[it][:,0]) # coordinates of the centers of the dS arcs
        y_m = y_centre[it] + r[it]*np.cos(angles[it][:,1])*np.sin(angles[it][:,0])
        z_m = z_centre[it] + r[it]*np.sin(angles[it][:,1])
        ix_c = pol.griddata( x_grid-x0*1e3, np.arange(nx), x_m, method='linear' ) 
        iy_c = pol.griddata( y_grid-y0*1e3, np.arange(ny), y_m, method='linear' ) 
        iz_c = pol.griddata( hgt_c, np.arange(nz), z_m, method='linear' ) 

        press_circ.append( np.zeros(len(angles[it]), dtype=prec) )
        u_circ.append( np.zeros(len(angles[it]), dtype=prec) ) 
        v_circ.append( np.zeros(len(angles[it]), dtype=prec) )
        w_circ.append( np.zeros(len(angles[it]), dtype=prec) )
        dens_circ.append( np.zeros(len(angles[it]), dtype=prec) )
    
        if np.any(np.isnan( np.concatenate( (np.concatenate( (ix_c,iy_c) ),iz_c) ) )):  # this would mean the thermal lies (partly) outside the domain, and the interpolation below would crash.
            press_circ[it-subjob[0]][:] = np.nan
            u_circ[it-subjob[0]][:]     = np.nan
            v_circ[it-subjob[0]][:]     = np.nan
            w_circ[it-subjob[0]][:]     = np.nan
            dens_circ[it-subjob[0]][:]  = np.nan
        else:
            subgrid, i0, i1, j0, j1, k0, k1 = create_small_subgrid_for_interpolation2( ix_c[:], iy_c[:], iz_c[:], nx, ny, nz )
            for iS in range(len(angles[it])):
                press_circ[it-subjob[0]][iS]  = pol.griddata( subgrid[iS], ptot [i0[iS]:i1[iS], j0[iS]:j1[iS], k0[iS]:k1[iS], it-subjob[0]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
                u_circ[it-subjob[0]][iS]      = pol.griddata( subgrid[iS], u_c  [i0[iS]:i1[iS], j0[iS]:j1[iS], k0[iS]:k1[iS], it-subjob[0]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
                v_circ[it-subjob[0]][iS]      = pol.griddata( subgrid[iS], v_c  [i0[iS]:i1[iS], j0[iS]:j1[iS], k0[iS]:k1[iS], it-subjob[0]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
                w_circ[it-subjob[0]][iS]      = pol.griddata( subgrid[iS], w_c  [i0[iS]:i1[iS], j0[iS]:j1[iS], k0[iS]:k1[iS], it-subjob[0]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
                dens_circ[it-subjob[0]][iS]   = pol.griddata( subgrid[iS], rho_c[i0[iS]:i1[iS], j0[iS]:j1[iS], k0[iS]:k1[iS], it-subjob[0]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
    return press_circ, u_circ, v_circ, w_circ, dens_circ


#def joblib_sfc_interp_split_var( subgrid, var, i0, i1, j0, j1, k0, k1, ix_c, iy_c, iz_c ):
def joblib_sfc_interp_split_var( var, i0, i1, j0, j1, k0, k1, ix_c, iy_c, iz_c ):
    int_method='linear'
    l = len(ix_c)
    out = np.zeros(l)
    for iS in range(l):
        rgi=pol.RegularGridInterpolator(points=[np.arange(i0[iS],i1[iS]),np.arange(j0[iS],j1[iS]),np.arange(k0[iS],k1[iS])],values=var[i0[iS]:i1[iS], j0[iS]:j1[iS], k0[iS]:k1[iS]])
        out[iS] =  rgi((ix_c[iS], iy_c[iS], iz_c[iS]))
        #out[iS] = pol.griddata( subgrid[iS], var[i0[iS]:i1[iS], j0[iS]:j1[iS], k0[iS]:k1[iS]].flatten(), (ix_c[iS], iy_c[iS], iz_c[iS]), int_method )
    return out


def index_grid( nx, ny, nz, x0=0, y0=0, z0=0, ndivx=1., ndivy=1., ndivz=1. ):
    """
    creates a matrix with the indices of a nx*ny*nz grid (useful for scipy griddata interpolations!)
    """
    size = int(nx*ny*nz)
    grid = np.zeros([size,3], dtype=prec)
    ind = np.unravel_index(np.arange(size),(int(nx),int(ny),int(nz)))
    grid[:,0] = ind[0]/ndivx + x0
    grid[:,1] = ind[1]/ndivy + y0
    grid[:,2] = ind[2]/ndivz + z0
    return grid


def read_data( fname ):
    fname = fname + '_data.npy'
    return np.load(fname)


def read_data_old( fname, complete=False, cols=24 ):
    if complete:
        file = open(fname + '_data_complete.dat', 'r')
    else:
        file = open(fname + '_data.dat', 'r')
    data = file.readlines()
    vars = np.zeros([cols,len(data)-1])
    for i in range(1,len(data)):
        col = 0
        j = 0
        while col < cols and j < len(data[i]):
            while data[i][j] == ' ':
                j +=1
            k=j
            while k < len(data[i]) and data[i][k] != ' ':
                k +=1
            vars[col][i-1] = float(data[i][j:k])
            j=k
            col +=1
    return vars        


def _dist_to_centre( x, y, z, xc, yc, zc):
    return np.sqrt(np.square(x-xc)+np.square(y-yc)+np.square(z-zc))


def add_one_tstep( var, direc=1 ):
    result = np.zeros(len(var)+1)
    if direc==1:
        result[:-1]=var
    if direc==-1:
        result[1:]=var
    return result


def hhmm( t, char=':' ):
    sec=(np.mod(t,60)-int(np.mod(t,60)))*60
    return '%02d'%(t/60)+char+'%02d'%(np.mod(t,60))+char+'%02d'%(sec)


def fix_fname( fname ):
    fname = list( fname )
    fname = fname[len(fname)/2 + 1:]
    i=0
    while not (fname[i] in ['_'] and fname[i+1] in ['0','1','2']) and i<len(fname):
        if fname[i] in ['_']:
            fname[i] = ' '
        i+=1
    fname[i]  = ' '
    fname[i+9]  = ' '
    fname[i+12] = ' '
    fname[i+21] = ' '
    fname[i+15] = ':'
    fname[i+18] = ':'
    fname[i+3]  = ':'
    fname[i+6]  = ':'
    fname[i+12] = ' '
    return "".join(fname)


def make_plots( fname, pdf=False ):
    from matplotlib import rc
    from matplotlib.ticker import NullFormatter
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    plt.ioff()
    fname_title = fix_fname(fname)
    #fname = case + '/' + case
    data = read_data( fname )
    time            = data[0]
    nt              = len(data[0])
    wmax            = data[1]
    R_parcel        = data[2]
    mass            = data[3]
    #mass_cond       = data[4]
    xmax            = data[4]
    ymax            = data[5]
    hmax            = data[6]
    x_centre        = data[7]
    y_centre        = data[8]
    z_centre        = data[9]
    u_mean          = data[10]
    v_mean          = data[11]
    w_mean          = data[12]
    PnzdS           = data[13]
    vmf             = data[14]
    buoy            = data[15]
    buoy_moist      = data[16]
    acc             = data[17]
    PnxdS           = data[18]
    xmf             = data[19]
    acc_x           = data[20]
    PnydS           = data[21]
    ymf             = data[22]
    acc_y           = data[23]
    entr_dist       = data[24]
    entr_rate       = data[25]
    pos             = data[26]
    detr_dist       = data[27]

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #rc('font',**{'family':'FreeSans'})
    rc('text', usetex=False)
    fig=plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
    ax1 = subplot(gs[0])
    ax1.xaxis.set_major_formatter( NullFormatter() )
    plt.plot( time, hmax, 'kx', ms=10, mew=2, label=r"$\omega_{max}$" )
    plt.errorbar( time, z_centre, yerr=R_parcel/1e3, fmt='k',ms=1.5,lw=1.5, mew=1.5, label="+/- radius" )
    plt.plot( time, pos/1e3, 'r',lw=2, label="mom. budget" )
    plt.xlim( time[0]-1, time[nt-1]+1 )
    plt.ylim( np.amin([(hmax[0]-1), (z_centre[0] - R_parcel[0]/1e3 - 0.8)]),(z_centre[nt-1] + R_parcel[nt-1]/1e3 + 0.8) )
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    rc('text', usetex=True)
    plt.ylabel('height (km)', fontsize=14)
    #plt.xlabel('time (min)', fontsize=16)
    plt.title(fname_title, fontsize = 18)
    #plt.title("Thermal's height", fontsize = 18)
    plt.grid()
    rc('text', usetex=False)
    ax2 = subplot(gs[1])
    ax2.xaxis.set_major_formatter( NullFormatter() )
    plt.plot(time, wmax, 'k-', lw=1.5, ms=10, mew=1.5)
    plt.xlim(time[0]-1, time[nt-1]+1)
    plt.ylim( np.amin(wmax)*0.9, np.amax(wmax)*1.1)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    rc('text', usetex=True)
    plt.ylabel('{\huge $\omega_{max}$} (m/s)', fontsize=14)
    #plt.xlabel('time (min)', fontsize=14)
    plt.grid()
    
    rc('text', usetex=False)
    ax3 = subplot(gs[2])
    plt.plot(time, w_mean, 'k-', lw=1.5, ms=10, mew=1.5)
    plt.xlim(time[0]-1, time[nt-1]+1)
    plt.ylim( np.amin(w_mean)*0.9, np.amax(w_mean)*1.1)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    rc('text', usetex=True)
    plt.ylabel('{\huge W} (m/s)', fontsize=14)
    plt.xlabel('time (min)', fontsize=14)
    plt.grid()
    subplots_adjust(wspace=0, hspace=0.1)
    #plt.legend(loc=4, frameon=False)
    if pdf:
        plt.savefig(fname + '_hmax_wmax_W_.pdf')
    plt.savefig(fname + '_hmax_wmax_W_.eps')
    plt.close()
    plt.clf()
    
    rc('text', usetex=False)
    fig = plt.figure(figsize=(9,3.8))
    ax = fig.add_axes( [0.1,0.15,0.85,0.75] )
    plt.plot( time, hmax, 'kx', ms=10, mew=2, label=r"$\omega_{max}$" )
    plt.errorbar( time, z_centre, yerr=R_parcel/1e3, fmt='k',ms=1.5,lw=1.5, mew=1.5, label="+/- radius" )
    plt.plot( time, pos/1e3, 'r',lw=2, label="mom. budget" )
    plt.xlim( time[0]-1, time[nt-1]+6 )
    plt.ylim( np.amin([(hmax[0]-1), (z_centre[0] - R_parcel[0]/1e3 - 0.8)]),(z_centre[nt-1] + R_parcel[nt-1]/1e3 + 0.8) )
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    rc('text', usetex=True)
    plt.ylabel('height (km)', fontsize=16)
    plt.xlabel('time (min)', fontsize=16)
    plt.title(fname_title, fontsize = 18)
    plt.grid()
    plt.legend(loc=4, frameon=False)
    if pdf:
        plt.savefig(fname + '_hmax.pdf')
    plt.savefig(fname + '_hmax.eps')
    plt.close()
    plt.clf()
    
    rc('text', usetex=False)
    fig=plt.figure(figsize=(15,5))

    dx = np.amax(2*R_parcel/1e3)*0.2
    range0 = np.amax(x_centre+R_parcel/1e3)+dx - (np.amin(x_centre-R_parcel/1e3)-dx)
    range1 = np.amax(y_centre+R_parcel/1e3)+dx - (np.amin(y_centre-R_parcel/1e3)-dx)
    commonrange = np.amax([range0,range1])

    gs = gridspec.GridSpec(1,2)
    ax1= subplot(gs[0])
    #plt.figure(figsize=(8,5))
    plt.plot( xmax, hmax, 'kx-', lw=1, ms=10, mew=1, label=r"w$_{max}$ points" )
    plt.errorbar( x_centre, z_centre, xerr=R_parcel/1e3, fmt='r',ms=1,lw=2, mew=1, label="centre +/- radius" )
    dz = np.amax(z_centre)*1.2-np.amax(z_centre)
    plt.ylim(np.amin(z_centre)-dz, np.amax(z_centre)+dz)
    plt.xlim( np.amin(x_centre-R_parcel/1e3)-dx, np.amin(x_centre-R_parcel/1e3)-dx + commonrange )
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    rc('text', usetex=True)
    plt.ylabel('height (km)', fontsize=16)
    plt.xlabel('X (km)', fontsize=16)
    plt.title(fname_title, fontsize = 18)
    plt.grid()
    plt.legend(loc=0)
    
    rc('text', usetex=False)
    ax2 = subplot(gs[1])
    plt.plot( ymax, hmax, 'kx-', lw=1, ms=10, mew=1, label=r"w$_{max}$ points" )
    plt.errorbar( y_centre, z_centre, xerr=R_parcel/1e3, fmt='r',ms=1,lw=2, mew=1, label="centre +/- radius" )
    dz = np.amax(z_centre)*1.2-np.amax(z_centre)
    plt.ylim(np.amin(z_centre)-dz, np.amax(z_centre)+dz)
    dx = np.amax(2*R_parcel/1e3)*0.2
    plt.xlim( np.amin(y_centre-R_parcel/1e3)-dx, np.amin(y_centre-R_parcel/1e3)-dx + commonrange )
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    rc('text', usetex=True)
    plt.ylabel('height (km)', fontsize=16)
    plt.xlabel('Y (km)', fontsize=16)
    plt.title(fname_title, fontsize = 18)
    plt.grid()
    plt.legend(loc=0)
    if pdf:
        plt.savefig(fname + '_wmax_pos.pdf' )
    plt.savefig(fname + '_wmax_pos.eps' )
    plt.close()
    plt.clf()
    
    rc('text', usetex=False)
    ymax=1e2*np.amax([abs(PnzdS),abs(vmf),abs(acc)])*1.15
    fig = plt.figure(figsize=(9,3.8))
    ax = fig.add_axes( [0.1,0.15,0.85,0.75] )
    plt.plot(time, PnzdS*1e2, 'k-', lw=2, label=r'$\int P_y n_y dS - mg$')
    plt.plot(time, vmf*1e2, 'k--', lw=2, label=r"$\int u'\cdot\hat{n}\rho\omega'dS$")
    #plt.plot(time, self.buoyancy, 'k:', lw = 2, label="c. Buoyancy")
    plt.plot(time, acc*1e2, 'r-', lw=2, label='sum')
    plt.xlim(time[0]-1, time[nt-1]+4)
    plt.ylim(-ymax, ymax)
    plt.yticks(fontsize = 14)
    plt.xticks(fontsize = 14)
    rc('text', usetex = True)
    plt.xlabel(r'time (min)', fontsize = 16)
    plt.ylabel(r'10$^{-2}$ m~s$^{-2}$',fontsize = 16)
    plt.title(fname_title, fontsize = 18)
    plt.grid()
    plt.legend(loc=1, frameon=False)
    if pdf:
        plt.savefig(fname + '_mom_budget.pdf')
    plt.savefig(fname + '_mom_budget.eps')
    plt.close()
    plt.clf()
    
    rc('text', usetex=False)
    ymax=1e2*np.amax([abs(PnzdS-buoy),abs(vmf),abs(acc),abs(buoy),abs(buoy_moist)])*1.15
    fig = plt.figure(figsize=(9,3.8))
    ax = fig.add_axes( [0.1,0.15,0.85,0.75] )
    plt.plot(time, buoy*1e2, 'k-', lw = 2, label="Buoyancy")
    plt.plot(time, vmf*1e2, 'k--', lw=2, label=r"F$_{mix}$")
    plt.plot(time, (PnzdS-buoy)*1e2, 'b--', lw = 2, label=r"F$_{nh}$")
    plt.plot(time, acc*1e2, 'r-', lw=2, label='total')
    plt.xlim(time[0]-1, time[nt-1]+4)
    plt.ylim(-ymax, ymax)
    plt.yticks(fontsize = 14)
    plt.xticks(fontsize = 14)
    rc('text', usetex=True)
    plt.ylabel('10$^{-2}$ m~s$^{-2}$', fontsize = 16)
    plt.xlabel('time (min)', fontsize = 16)
    #plt.title(r"Buoyancy \& non-hydrost. forces", fontsize = 18)
    plt.grid()
    plt.legend(loc=1, frameon=False)
    if pdf:
        plt.savefig(fname + '_buoyancy.pdf')
    plt.savefig(fname + '_buoyancy.eps')
    plt.close()
    plt.clf()
    
    plt.close('all')


def optimize_njobs( nt, n_jobs):
    if n_jobs == 1:
        dtt = nt
    else:
        if nt <= n_jobs:
            n_jobs = nt
            dtt = 1
        elif nt > n_jobs:
            if np.mod(nt, n_jobs)==0:
                dtt = int(nt/n_jobs)
            else:
                dtt = int(nt/n_jobs) + 1
                n_jobs = int(nt/dtt)
            if np.mod(nt, n_jobs)!=0:
                n_jobs +=1
    n_jobs = np.max([n_jobs, 1])
    return n_jobs, dtt

def moving_average(a, n=5) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


