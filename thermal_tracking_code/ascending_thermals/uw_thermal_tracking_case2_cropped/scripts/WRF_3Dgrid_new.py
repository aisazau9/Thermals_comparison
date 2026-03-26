import numpy as np
import scipy as sp
from scipy import stats, interpolate
import scipy.io as io
import netCDF4 as nc4
import scipy.interpolate as pol
import aux_functions as aux
import WRF_3Dthermal as thermal3D
import os
from joblib import Parallel, delayed
import time
import gzip
import pdb
from WRF_3Dthermal import optimize_njobs
import gc
import datetime as dt

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
Lv   = 2.5e6                        # latent heat of vaporization (J/kg)


prec = np.float32                   # precision of the arrays (np.float32 or np.float64) (fields read from nc files are f32 regardless)

class Grid(object):

    def __init__( self, dx=100, YY0=2022, MM0=8, DD0=7, hr0=6, min0=00, sec0=00, nt=12, hgt000_fixed_vals = [], hgt000_c_fixed_vals = [], x0=None, y0=None, i0=None, j0=None, nxi=None, nyi=None, nxkm=None, nykm=None, nz=90, path = '/srv/ccrc/data15/z3392395/WRF/em_seabreeze2d_x/', header_fmt='wrfout_d01_YYYY-MM-DD_', ending='', dt=60, n_jobs=4, original_grid=None, t0=None, tf=None, x00=None, xf=None, y00=None, yf=None, zf=None, profiles=False, gunzip=False, gunzip_path='./', compute_rh=False, compute_theta=False, GCE=False , hgt000_fixed = None, compute_extra_fields = False):
        self.hgt000_fixed        =  hgt000_fixed # ALEJANDRA
        self.hgt000_fixed_vals   =  hgt000_fixed_vals # ALEJANDRA
        self.hgt000_c_fixed_vals =  hgt000_c_fixed_vals # ALEJANDRA
        self.compute_extra_fields = compute_extra_fields # ALEJANDRA
        self.compute_rh     = compute_rh
        self.compute_theta  = compute_theta
        self.GCE = GCE
        self.YY0    = int(YY0)
        self.MM0    = int(MM0)
        self.DD0    = int(DD0)
        if original_grid==None:
            self.dx     = dx # in m
            self.dy     = self.dx 
            self.dt     = dt
            self.hr0    = int(hr0)
            self.min0   = int(min0)
            self.sec0   = int(sec0)
            self.nt     = nt
            # if domain is given in terms of x0, y0 and nxkm, nykm 
            # (assumes x and y grids start at 0km at index 0):
            if i0==None and j0==None and nxi==None and nyi==None:
                self.x0 = x0
                self.y0 = y0
                self.nxkm = nxkm
                self.nykm = nykm
                self.i0 = int(self.x0*1e3/self.dx)
                self.j0 = int(self.y0*1e3/self.dy)
                self.i1 = int(np.rint(self.x0*1e3/self.dx + self.nxkm*1e3/self.dx))
                self.j1 = int(np.rint(self.y0*1e3/self.dy + self.nykm*1e3/self.dy))
                self.nx = self.i1 - self.i0
                self.ny = self.j1 - self.j0
            # if domain is given in terms of indices:
            else:
                self.i0 = i0
                self.j0 = j0
                self.nx = nxi
                self.ny = nyi
                self.i1 = self.i0 + self.nx
                self.j1 = self.j0 + self.ny
                # if x[0]!=0km or y[0]!=0km, x0 and y0 must be provided:
                if x0!=None and y0!=None: 
                    self.x0 = x0
                    self.y0 = y0
                else: # this assumes x and y start at 0 at index 0
                    self.x0 = self.dx*self.i0
                    self.y0 = self.dy*self.j0
                self.nxkm   = self.nx*self.dx*1e-3
                self.nykm   = self.ny*self.dy*1e-3
            self.nz     = nz
            self.path   = path
            self.ending = ending
            self.header_fmt = header_fmt
            self.profiles = profiles    # True if grid is for computing the profiles of theta and rho (to compute N)
            self.gunzip_path = gunzip_path  # if simulation data is gzipped
            
            self._load(gunzip=gunzip)
            self._interpolate_heights( n_jobs=n_jobs )
            self._compute_thermodynamic_quantities()
        else:
            """
            this is executed when extracting a smaller subgrid from the original grid
            """
            nx = xf-x00+1
            ny = yf-y00+1
            
            self.x0             = original_grid.x_grid[x00]/1e3
            self.y0             = original_grid.y_grid[y00]/1e3
            self.nx             = nx
            self.ny             = ny
            self.dx             = original_grid.dx 
            self.dy             = original_grid.dy
            self.dx_w           = original_grid.dx_w
            self.dy_w           = original_grid.dy_w
            self.dt             = original_grid.dt
            minutes             = original_grid.hr0*60. + original_grid.min0 + original_grid.sec0/60. + t0*self.dt/60.
            self.hr0            = int(minutes/60.)
            self.min0           = int(np.mod(minutes,60))
            self.sec0           = int( np.mod(minutes,1)*60. )
            self.nt             = tf-t0+1

            self.path           = original_grid.path
            self.x_grid         = original_grid.x_grid[x00:xf+1]
            self.x_grid_upoints = original_grid.x_grid_upoints[x00:xf+2]
            self.y_grid         = original_grid.y_grid[y00:yf+1]
            self.y_grid_vpoints = original_grid.y_grid_vpoints[y00:xf+2]
            x_shift             = x00*self.dx
            y_shift             = y00*self.dy
            x00_wint            = int(x_shift/self.dx_w)
            y00_wint            = int(y_shift/self.dy_w)
            xf_wint             = np.where(original_grid.x_grid_wint==original_grid.x_grid[xf])[0]
            yf_wint             = np.where(original_grid.y_grid_wint==original_grid.y_grid[yf])[0]
            self.x_grid_wint    = original_grid.x_grid_wint[x00_wint:(xf_wint+1)[0]]
            self.y_grid_wint    = original_grid.y_grid_wint[y00_wint:(yf_wint+1)[0]]
            self.nx_w           = len(self.x_grid_wint)
            self.ny_w           = len(self.y_grid_wint)
            
            #self.w             = original_grid.w     [x00:xf+1,y00:yf+1,:zf+1,t0:tf+1]
            self.u_c            = original_grid.u_c   [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.v_c            = original_grid.v_c   [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.w_c            = original_grid.w_c   [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qvapor        = original_grid.qvapor[x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qcloud         = original_grid.qcloud[x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.ptot           = original_grid.ptot  [x00:xf+1,y00:yf+1,:zf,t0:tf+1]

            self.nz             = zf
            self.hgt            = original_grid.hgt  [:zf+1]
            self.dh             = original_grid.dh   [:zf]
            self.hgt_c          = original_grid.hgt_c[:zf]
            self.dh_c           = original_grid.dh_c [:zf-1]

            zf_wint             = np.where(original_grid.hgt_w==original_grid.hgt_c[zf])[0]
            self.hgt_w          = original_grid.hgt_w   [:zf_wint[0]+1] #CHANGE: ALEJANDRA
            self.nz_w           = len(self.hgt_w)

            self.temp 	        = original_grid.temp 	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.rh	        = original_grid.rh 	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            if self.compute_theta:
                self.theta	        = original_grid.theta 	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.rho_m 	        = original_grid.rho_m 	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            #self.rho_condensate= original_grid.rho_condensate[x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.rho_c 	        = original_grid.rho_c 	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qtotal         = original_grid.qtotal        [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.mse            = original_grid.mse           [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.sctot          = original_grid.sctot	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.latheat        = original_grid.latheat	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qnice          = original_grid.qnice	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qncloud        = original_grid.qncloud	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qnrain         = original_grid.qnrain	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qice           = original_grid.qice	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qrain          = original_grid.qrain	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.noninduc       = original_grid.noninduc      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.cldnuc         = original_grid.cldnuc	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.epotential     = original_grid.epotential	  [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qngraupel      = original_grid.qngraupel	  [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qicesnow       = original_grid.qicesnow	  [x00:xf+1,y00:yf+1,:zf,t0:tf+1]
            self.qghail         = original_grid.qghail	      [x00:xf+1,y00:yf+1,:zf,t0:tf+1]

            self.s                      = original_grid.s # (smoothing factor, defined in 3D_thermal_tracing.py)
            self.min_thermal_duration   = original_grid.min_thermal_duration
            self.shift_x                = x00_wint
            self.shift_y                = y00_wint
            self.shift_t                = t0
            
    def header(self, timestep):
        """
        This function returns the corresponding file name header of the simulation output
        for a particular time step (start counting at 0), taking into account that the 
        date may have changed.
        """
        t0 = dt.datetime(self.YY0,self.MM0,self.DD0,self.hr0,self.min0,self.sec0)
        current_t = t0 + dt.timedelta(seconds = self.dt*timestep)
        header = self.header_fmt.replace('YYYY','%04d'%(current_t.year)).replace('MM','%02d'%(current_t.month)).replace('DD','%02d'%(current_t.day))
        return header
        

    def _load(self, gunzip=False):
        dx = self.dx
        dy = self.dy
        i0 = self.i0
        j0 = self.j0
        nx = self.nx
        ny = self.ny
        i1 = self.i1
        j1 = self.j1
        nz = self.nz
        nt = self.nt
        if not self.profiles:
            self.u1      = np.zeros([nx+1,ny,nz,nt], dtype=np.float32 )     
            self.v1      = np.zeros([nx,ny+1,nz,nt], dtype=np.float32 )
            self.w1      = np.zeros([nx,ny,nz+1,nt], dtype=np.float32 )    
            if self.GCE:
                self.u1_c      = np.zeros([nx,ny,nz,nt], dtype=np.float32 )     #added for GCE
                self.v1_c      = np.zeros([nx,ny,nz,nt], dtype=np.float32 )     #added for GCE

        self.ph         = np.zeros([nx,ny,nz+1,nt], dtype=np.float32 )    
        self.phb        = np.zeros([nx,ny,nz+1,nt], dtype=np.float32 )    
        self.pb1        = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )   
        self.p1         = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )    
        if self.GCE:
            self.rh_gce = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.qicesnow1  = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.qghail1    = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )

        self.sctot1     = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.latheat1   = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.qnice1     = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        
        
        self.qncloud1   = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.qnrain1    = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.qice1      = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.qcloud1    = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.qrain1     = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.noninduc1  = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.cldnuc1    = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.Th1        = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )    
        self.qvapor1    = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.epotential1= np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        self.qngraupel1 = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        if not self.profiles:
            self.qsnow1  = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
            self.qgraup1 = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
            #if self.GCE:
            self.qhail1 = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )
        if self.compute_extra_fields: #ALEJANDRA
            self.phyd1   = np.zeros([nx,ny,nz,nt],   dtype=np.float32 ) 
            self.refl1   = np.zeros([nx,ny,nz,nt],   dtype=np.float32 )   
            self.maxws1  = np.zeros([nx,ny,nt],      dtype=np.float32 ) #2D
        hr0 = np.copy(self.hr0)
        min0 = np.copy(self.min0)
        sec0 = np.copy(self.sec0)
        dt = self.dt

        print ('\nReading netcdf files:' )
        for it in range(nt):
            if sec0 >= 60:
                sec0 = np.mod(sec0,60)
                min0 += 1
            if min0 == 60:
                min0 = 0
                hr0 += 1
            if hr0==24:
                hr0=0
            sec = '%02d'%(sec0)
            minute = '%02d'%(min0)
            hr = '%02d'%(hr0)
            fname = self.path + self.header(it) + hr + ':' + minute + ':' + sec + self.ending #'_short'
            if gunzip:
                fname_wo_path = self.gunzip_path + self.header(it) + hr + ':' + minute + ':' + sec + self.ending
                fobj=gzip.open(fname+'.gz')
                #os.system('gunzip -kc ' + fname + '.gz > ' + fname_wo_path)
                ifile = nc4.Dataset( 'dummy', mode='r',memory=fobj.read() )
                #ifile = nc4.Dataset( fname_wo_path, 'r' )
            else:
                ifile = nc4.Dataset( fname, 'r' )
            #ifile = nf.NetCDFFile( fname, mode='r' )
            #ifile = io.netcdf.netcdf_file(fname, mode='r')
            print( fname)
            # SWAP AXES TO HAVE THEM AS [X,Z,T] INSTEAD OF [T,Z,X] AND SELECT SUBDOMAIN
            if self.GCE:
                if not self.profiles:
                    self.u1_c     [:,:,:,it] = np.swapaxes(ifile.variables['u'][:nz, j0:j1, i0:i1], 0,2)*1.e-2  # (m/s)
                    self.v1_c     [:,:,:,it] = np.swapaxes(ifile.variables['v'][:nz, j0:j1, i0:i1], 0,2)*1.e-2  # (m/s)

                    var3d = np.swapaxes(ifile.variables['w'][:nz, j0:j1, i0:i1], 0,2)*1.e-2  # (m/s)
                    #derive w1 staggered grid
                    self.w1[:,:,0,it] = 0. # zero at surface
                    self.w1[:,:,nz,it] = 0. # zero at TOA
                    for lev in range(1, nz):
                        #print('lev=',lev-1,lev)
                        self.w1[:,:,lev,it] = ( var3d[:,:,lev] + var3d[:,:,lev-1] ) * 0.5
                # height field			
                uht = ifile.variables['uht'][:] #non-staggered height (m)
                wht = ifile.variables['wht'][:] #staggered height (m)
                for lev in range(0, nz):
                    self.ph [:,:,lev,it] = wht[lev] * g   
                self.ph [:,:,nz,it] = ( wht[nz-1] + (uht[nz-1]-wht[nz-1])*2 ) * g  #TOA height
                self.phb    [:,:,:,it] = 0.   #purchabation term -> zero
                # Pressure field
                p0 = ifile.variables['p0'][:]*1.e-1    #base pressure [Pa]
                for lev in range(0, nz):
                    self.pb1    [:,:,lev,it] = p0[lev]
                self.p1 [:,:,:,it] = np.swapaxes( ifile.variables['dp'][:nz, j0:j1, i0:i1]*1.e-1, 0,2)  #purturbation pressure [Pa]
                # RH from GCE
                self.rh_gce [:,:,:,it] = np.swapaxes( ifile.variables['rh'][:nz, j0:j1, i0:i1]*1.e-2, 0,2)  #GCE RH [-] (0-1)
            else:
                if not self.profiles:
                    self.u1     [:,:,:,it] = np.swapaxes(ifile.variables['U'][0, :nz, j0:j1, i0:i1+1], 0,2)      # at u-points (side edges)
                    self.v1     [:,:,:,it] = np.swapaxes(ifile.variables['V'][0, :nz, j0:j1+1, i0:i1], 0,2)      # at u-points (side edges)
                    self.w1     [:,:,:,it] = np.swapaxes(ifile.variables['W'][0, :nz+1, j0:j1, i0:i1], 0,2)      # at w-points (top and bottom edges)
                self.ph     [:,:,:,it] = np.swapaxes(ifile.variables['PH'][0, :nz+1, j0:j1, i0:i1], 0,2)      # at w-points
                self.phb    [:,:,:,it] = np.swapaxes(ifile.variables['PHB'][0, :nz+1, j0:j1, i0:i1], 0,2)      # at w-points
                self.pb1    [:,:,:,it] = np.swapaxes(ifile.variables['PB'][0, :nz, j0:j1, i0:i1], 0,2)    
                self.p1     [:,:,:,it] = np.swapaxes(ifile.variables['P'][0, :nz, j0:j1, i0:i1], 0,2)     

            try:
                self.sctot1 [:,:,:,it] = np.swapaxes(ifile.variables['SCTOT'][0, :nz, j0:j1, i0:i1], 0,2)
            except:
                self.sctot1[:,:,:,it] = np.ones_like(self.sctot1[:,:,:,it])*np.nan
            try:
                self.latheat1 [:,:,:,it] = np.swapaxes(ifile.variables['LATHEAT'][0, :nz, j0:j1, i0:i1], 0,2)
            except:
                self.latheat1[:,:,:,it] = np.ones_like(self.latheat1[:,:,:,it])*np.nan
            try:
                self.qnice1 [:,:,:,it] = np.swapaxes(ifile.variables['QNICE'][0, :nz, j0:j1, i0:i1], 0,2)
            except:
                self.qnice1[:,:,:,it] = np.ones_like(self.qnice1[:,:,:,it])*np.nan
            try:
                self.qncloud1 [:,:,:,it] = np.swapaxes(ifile.variables['QNCLOUD'][0, :nz, j0:j1, i0:i1], 0,2)
            except:
                try:
                    self.qncloud1 [:,:,:,it] = np.swapaxes(ifile.variables['QNDROP'][0, :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.qncloud1[:,:,:,it] = np.ones_like(self.qncloud1[:,:,:,it])*np.nan
            try:
                self.qnrain1 [:,:,:,it] = np.swapaxes(ifile.variables['QNRAIN'][0, :nz, j0:j1, i0:i1], 0,2)
            except:
                self.qnrain1[:,:,:,it] = np.ones_like(self.qnrain1[:,:,:,it])*np.nan
                            
            if self.compute_extra_fields: #ALEJANDRA
                try:
                    self.refl1 [:,:,:,it] = np.swapaxes(ifile.variables['REFL_10CM'][0, :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.refl1[:,:,:,it] = np.ones_like(self.refl1[:,:,:,it])*np.nan    
                try:
                    self.phyd1 [:,:,:,it] = np.swapaxes(ifile.variables['P_HYD'][0, :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.phyd1[:,:,:,it] = np.ones_like(self.phyd1[:,:,:,it])*np.nan
                try:
                    self.maxws1[:,:,it] = np.swapaxes(ifile.variables['WSPD10MAX'][0, j0:j1, i0:i1], 0,1) #x,y
                except:
                    self.maxws1[:,:,it] = np.ones_like(self.maxws1[:,:,it])*np.nan            
            
            if self.GCE:
                try:
                    #self.qicesnow1 [:,:,:,it] = np.swapaxes(ifile.variables['QNCLOUD'][ :nz, j0:j1, i0:i1], 0,2)
                    qci3d = np.swapaxes( ifile.variables['qci'][:nz, j0:j1, i0:i1], 0,2)  #cloud ice [g/g]
                    qcs3d = np.swapaxes( ifile.variables['qcs'][:nz, j0:j1, i0:i1], 0,2)  #cloud snow [g/g] 
                    self.qicesnow1 [:,:,:,it] = qci3d[:,:,:] + qcs3d[:,:,:] #ice + snow mass conc.. [g/g]

                except:
                    self.qicesnow1[:,:,:,it] = np.ones_like(self.qicesnow1[:,:,:,it])*np.nan
                try:
                    #self.qghail1 [:,:,:,it] = np.swapaxes(ifile.variables['QNRAIN'][ :nz, j0:j1, i0:i1], 0,2)
                    qcg3d = np.swapaxes( ifile.variables['qcg'][:nz, j0:j1, i0:i1], 0,2)  #graupel  [g/g]
                    qch3d = np.swapaxes( ifile.variables['qch'][:nz, j0:j1, i0:i1], 0,2)  #hail     [g/g] 
                    self.qghail1 [:,:,:,it] = qcg3d[:,:,:] + qch3d[:,:,:]  #ice + snow mass conc.. [g/g]

                except:
                    self.qghail1[:,:,:,it] = np.ones_like(self.qghail1[:,:,:,it])*np.nan
                try:
                    self.qice1 [:,:,:,it] = np.swapaxes(ifile.variables['qci'][ :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.qice1[:,:,:,it] = np.ones_like(self.qice1[:,:,:,it])*np.nan
                try:
                    self.qcloud1 [:,:,:,it] = np.swapaxes(ifile.variables['qcl'][ :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.qcloud1[:,:,:,it] = np.ones_like(self.qcloud1[:,:,:,it])*np.nan
                try:
                    self.qrain1 [:,:,:,it] = np.swapaxes(ifile.variables['qrn'][ :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.qrain1[:,:,:,it] = np.ones_like(self.qrain1[:,:,:,it])*np.nan
                try:
                    self.cldnuc1 [:,:,:,it] = np.swapaxes(ifile.variables['physc'][ :nz, j0:j1, i0:i1], 0,2) #condensation (g/g/s)
                except:
                    self.cldnuc1[:,:,:,it] = np.ones_like(self.cldnuc1[:,:,:,it])*np.nan

                       # potential temperature
                exner = ifile.variables['pi'][:] #exner function [-]
                pt = ifile.variables['ta'][:]    #domain mean potential temperature [K] 
                dpt = np.swapaxes( ifile.variables['dp'][:nz, j0:j1, i0:i1], 0,2) #purturbation pot temp [K]

                for lev in range(0, nz):
                    self.Th1    [:,:,lev,it] = pt[lev] + dpt[:,:,lev] - 300.  # WRF-like Th1
                #print('pt=',pt)
                #print('tair=',pt*exner-273.1)
                #print('Th1',self.Th1[1,1,:,it] )
                #stop
                # the rest are all at cell centres 
                qa  = ifile.variables['qa'][:]    #domain mean water vapor mixing ratio [g/g] 
                dqv = np.swapaxes( ifile.variables['dqv'][:nz, j0:j1, i0:i1], 0,2) #purturbation water vapor mixing ratio [g/g]
                for lev in range(0, nz):
                    self.qvapor1 [:,:,lev,it] = qa[lev] + dqv[:,:,lev]  #water vapor mixing ratio [g/g]
                if not self.profiles:
                    #self.qcloud1[:,:,:,it] = np.swapaxes(ifile.variables['QCLOUD'][0, :nz, j0:j1, i0:i1], 0,2)
                    #self.qrain1 [:,:,:,it] = np.swapaxes(ifile.variables['QRAIN'][0, :nz, j0:j1, i0:i1], 0,2) 
                    #self.qice1  [:,:,:,it] = np.swapaxes(ifile.variables['QICE'][0, :nz, j0:j1, i0:i1], 0,2) 
                    try:
                        self.qsnow1 [:,:,:,it] = np.swapaxes(ifile.variables['qcs'][ :nz, j0:j1, i0:i1], 0,2) 
                    except:
                        self.qsnow1[:,:,:,it] = np.zeros_like(self.qsnow1[:,:,:,it])
                    try:
                        self.qgraup1[:,:,:,it] = np.swapaxes(ifile.variables['qcg'][ :nz, j0:j1, i0:i1], 0,2)
                    except:
                        self.qgraup1[:,:,:,it] = np.zeros_like(self.qgraup1[:,:,:,it])
                    try:
                        self.qhail1[:,:,:,it] = np.swapaxes(ifile.variables['qch'][ :nz, j0:j1, i0:i1], 0,2)
                    except:
                        self.qhail1[:,:,:,it] = np.zeros_like(self.qhail1[:,:,:,it])
            else:
                self.qicesnow1[:,:,:,it] = np.ones_like(self.qicesnow1[:,:,:,it])*np.nan
                self.qghail1[:,:,:,it] = np.ones_like(self.qghail1[:,:,:,it])*np.nan
                try:
                    self.qice1 [:,:,:,it] = np.swapaxes(ifile.variables['QICE'][0, :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.qice1[:,:,:,it] = np.ones_like(self.qice1[:,:,:,it])*np.nan
                try:
                    self.qcloud1 [:,:,:,it] = np.swapaxes(ifile.variables['QCLOUD'][0, :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.qcloud1[:,:,:,it] = np.ones_like(self.qcloud1[:,:,:,it])*np.nan
                try:
                    self.qrain1 [:,:,:,it] = np.swapaxes(ifile.variables['QRAIN'][0, :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.qrain1[:,:,:,it] = np.ones_like(self.qrain1[:,:,:,it])*np.nan
                try:
                    self.cldnuc1 [:,:,:,it] = np.swapaxes(ifile.variables['CLDNUC'][0, :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.cldnuc1[:,:,:,it] = np.ones_like(self.cldnuc1[:,:,:,it])*np.nan
                try:
                    self.noninduc1 [:,:,:,it] = np.swapaxes(ifile.variables['NONINDUC'][0, :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.noninduc1[:,:,:,it] = np.ones_like(self.noninduc1[:,:,:,it])*np.nan
                try:
                    self.epotential1 [:,:,:,it] = np.swapaxes(ifile.variables['POT'][0, :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.epotential1[:,:,:,it] = np.ones_like(self.epotential1[:,:,:,it])*np.nan
                try:
                    self.qngraupel1 [:,:,:,it] = np.swapaxes(ifile.variables['QNGRAUPEL'][0, :nz, j0:j1, i0:i1], 0,2)
                except:
                    self.qngraupel1[:,:,:,it] = np.ones_like(self.qngraupel1[:,:,:,it])*np.nan
                self.Th1    [:,:,:,it] = np.swapaxes(ifile.variables['T'][0, :nz, j0:j1, i0:i1], 0,2)      # the rest are all at cell centres 
                self.qvapor1[:,:,:,it] = np.swapaxes(ifile.variables['QVAPOR'][0, :nz, j0:j1, i0:i1], 0,2)
                if not self.profiles:
                    #self.qcloud1[:,:,:,it] = np.swapaxes(ifile.variables['QCLOUD'][0, :nz, j0:j1, i0:i1], 0,2)
                    #self.qrain1 [:,:,:,it] = np.swapaxes(ifile.variables['QRAIN'][0, :nz, j0:j1, i0:i1], 0,2) 
                    #self.qice1  [:,:,:,it] = np.swapaxes(ifile.variables['QICE'][0, :nz, j0:j1, i0:i1], 0,2) 
                    try:
                        self.qsnow1 [:,:,:,it] = np.swapaxes(ifile.variables['QSNOW'][0, :nz, j0:j1, i0:i1], 0,2) 
                    except:
                        self.qsnow1[:,:,:,it] = np.zeros_like(self.qsnow1[:,:,:,it])
                    try:
                        self.qgraup1[:,:,:,it] = np.swapaxes(ifile.variables['QGRAUP'][0, :nz, j0:j1, i0:i1], 0,2)
                    except:
                        self.qgraup1[:,:,:,it] = np.zeros_like(self.qgraup1[:,:,:,it])
                    try:
                        self.qhail1[:,:,:,it] = np.swapaxes(ifile.variables['QHAIL'][0, :nz, j0:j1, i0:i1], 0,2)
                    except:
                        self.qhail1[:,:,:,it] = np.zeros_like(self.qhail1[:,:,:,it])
                    #merge variables
                    self.qicesnow1 [:,:,:,it] = self.qice1 [:,:,:,it] + self.qsnow1 [:,:,:,it]  #ice + snow mass conc.. [g/g]
                    self.qghail1   [:,:,:,it] = self.qhail1[:,:,:,it]  + self.qgraup1[:,:,:,it] #graupel + hail mass conc.. [g/g]
            sec0 += dt
            if gunzip:
                fobj.close()
                ifile.close()
                #    os.remove( fname_wo_path )
            else:
                ifile.close()
        self.ptot1 = self.pb1 + self.p1
        del self.pb1 
        del self.p1  
        if (not self.profiles) or self.compute_rh:
            #if self.GCE:
            self.qrisg1 = self.qrain1 + self.qice1 + self.qgraup1 + self.qsnow1 + self.qhail1
            #else:
            #    self.qrisg1 = self.qrain1 + self.qice1 + self.qgraup1 + self.qsnow1
            self.qtotal1 = self.qcloud1 + self.qrisg1
            #del self.qsnow1   #ALEJANDRA
            #del self.qgraup1 
            #if self.GCE:      #ALEJANDRA
            #del self.qhail1   #ALEJANDRA

        # INTERPOLATE STAGGERED VARIABLES TO GRID CENTRES
        if not self.profiles:
            if not self.GCE:
                self.u1_c = (self.u1[:-1,:,:,:] + self.u1[1:,:,:,:])*0.5     	# u at grid centres
                self.v1_c = (self.v1[:,:-1,:,:] + self.v1[:,1:,:,:])*0.5        # v at grid centres
        
        # DEFINE THE GRIDS:
        self.x_grid = np.arange(nx, dtype=prec)*dx + self.x0*1e3
        self.y_grid = np.arange(ny, dtype=prec)*dy + self.y0*1e3
        self.x_grid_upoints = np.arange(nx+1, dtype=prec)*dx + self.x0*1e3-0.5*self.dx
        self.y_grid_vpoints = np.arange(ny+1, dtype=prec)*dy + self.y0*1e3-0.5*self.dy
         
    def _interpolate_heights(self, n_jobs=4 ):
        dx = self.dx
        nx = self.nx
        ny = self.ny
        nz = self.nz
        nt = self.nt

        # COMPUTE HEIGHTS
        hgt = (self.ph + self.phb)/g                        	# height of w-points (m) (time-dependent)
        hgt_c 	= (hgt[:,:,:-1,:]+hgt[:,:,1:,:])*0.5  	        	# height at grid centres (m) (time-dependent)

        if self.hgt000_fixed == None: #ALEJANDRA
            #hgt_upoints = np.zeros([self.nx+1,self.ny,self.nz,self.nt])             # height of the u-points (m) (at the side edges of gridboxes)
            #hgt_upoints[0,:,:,:] = hgt_c[0,:,:,:]
            #hgt_upoints[self.nx,:,:,:] = hgt_c[self.nx-1,:,:,:]
            #hgt_upoints[1:-1,:,:,:] = 0.5*(hgt_c[1:,:,:,:]+hgt_c[:-1,:,:,:])
            #hgt_vpoints = np.zeros([self.nx,self.ny+1,self.nz,self.nt])             # height of the v-points (m) (at the side edges of gridboxes)
            #hgt_vpoints[:,0,:,:] = hgt_c[:,0,:,:]
            #hgt_vpoints[:,self.ny,:,:] = hgt_c[:,self.ny-1,:,:]
            #hgt_vpoints[:,1:-1,:,:] = 0.5*(hgt_c[:,1:,:,:]+hgt_c[:,:-1,:,:])
            hgt000 = hgt[0,0,:,0]
            hgt000_c = (hgt000[:-1] + hgt000[1:])*0.5
            hgt_w = hgt000_c
        else: #ALEJANDRA
            hgt000   = self.hgt000_fixed_vals
            hgt000_c = self.hgt000_c_fixed_vals
            hgt_w    = hgt000_c
        
        nz_w = self.nz
        nx_w = self.nx
        ny_w = self.ny
        dx_w = self.dx
        dy_w = self.dy
        self.x_grid_wint = self.x_grid
        self.y_grid_wint = self.y_grid
        
        # CREATE A TIME-INDEPENDENT HEIGHT COORDINATE BY INTERPOLATING ALL FIELDS TO HEIGHTS AT FIRST TIME-STEP AND X=0
                
        if not self.profiles:
            self.u_c = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
            self.v_c = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
            self.w_c = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.sctot      = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.latheat    = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.qnice      = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.qnrain     = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.qncloud    = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.qcloud     = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.qrain      = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.qice       = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.noninduc   = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.cldnuc     = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.ptot       = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.Th         = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.qvapor     = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.epotential = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.qngraupel  = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.qicesnow   = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        self.qghail     = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )
        if self.compute_extra_fields: #ALEJANDRA
            self.qhail     = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )  #ALEJANDRA
            self.qsnow     = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )  #ALEJANDRA
            self.qgraup    = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )  #ALEJANDRA
            self.refl      = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )  #ALEJANDRA
            self.phyd      = np.zeros([nx,ny,nz,nt],  dtype=np.float32 )  #ALEJANDRA
        if not self.profiles: 
            self.qrisg  = np.zeros([nx,ny,nz,nt],   dtype=np.float32 ) #        rain, ice, snow and graupel
        if self.nt == 1:
            start=time.time()
            print( 'interpolating data with %d jobs...'%(n_jobs))
            # split the horizontal (nx,ny) grid into n_jobs chunks for parallel processing
            hor_size = nx*ny
            subl = []
            l0 = 0
            dl = int(hor_size/n_jobs)
            for ijob in range(n_jobs-1):
                l1 = l0+dl
                subl.append([l0,l1])
                l0 = l1
            subl.append([l0,nx*ny])
            for it in range(nt):
                jobs = []
                l = []
                for ijob in range(n_jobs):
                    l.append( np.arange(subl[ijob][0],subl[ijob][1]) )
                    if self.profiles:                            
                        jobs.append( ( l[ijob], self.Th1[:,:,:,it], self.ptot1[:,:,:,it], self.qvapor1[:,:,:,it], hgt_c[:,:,:,it], hgt000_c, nx, ny, nz ) )
                    else:
                        if self.compute_extra_fields: #ALEJANDRA
                            jobs.append( ( l[ijob], self.w1[:,:,:,it], self.u1_c[:,:,:,it], self.v1_c[:,:,:,it], self.Th1[:,:,:,it], self.ptot1[:,:,:,it], self.qvapor1[:,:,:,it], self.qcloud1[:,:,:,it], self.qrisg1[:,:,:,it], self.latheat1[:,:,:,it], self.qnice1[:,:,:,it], self.qncloud1[:,:,:,it], self.qnrain1[:,:,:,it], self.cldnuc1[:,:,:,it], self.qrain1[:,:,:,it], self.qice1[:,:,:,it], self.sctot1[:,:,:,it], self.noninduc1[:,:,:,it], self.epotential1[:,:,:,it], self.qngraupel1[:,:,:,it], self.qicesnow1[:,:,:,it], self.qghail1[:,:,:,it], hgt[:,:,:,it], hgt_c[:,:,:,it], hgt000, hgt000_c, nx, ny, nz, True, self.qhail1[:,:,:,it], self.qsnow1[:,:,:,it], self.qgraup1[:,:,:,it], self.refl1[:,:,:,it], self.phyd1[:,:,:,it] ) )

                        else: 
                            jobs.append( ( l[ijob], self.w1[:,:,:,it], self.u1_c[:,:,:,it], self.v1_c[:,:,:,it], self.Th1[:,:,:,it], self.ptot1[:,:,:,it], self.qvapor1[:,:,:,it], self.qcloud1[:,:,:,it], self.qrisg1[:,:,:,it], self.latheat1[:,:,:,it], self.qnice1[:,:,:,it], self.qncloud1[:,:,:,it], self.qnrain1[:,:,:,it], self.cldnuc1[:,:,:,it], self.qrain1[:,:,:,it], self.qice1[:,:,:,it], self.sctot1[:,:,:,it], self.noninduc1[:,:,:,it], self.epotential1[:,:,:,it], self.qngraupel1[:,:,:,it], self.qicesnow1[:,:,:,it], self.qghail1[:,:,:,it], hgt[:,:,:,it], hgt_c[:,:,:,it], hgt000, hgt000_c, nx, ny, nz ) )
                if self.profiles:
                    ( vars0 ) = Parallel(n_jobs=n_jobs)(delayed(parallel_griddata_profile)(*jobs[i]) for i in range(len(jobs)))
                else:
                    ( vars0 ) = Parallel(n_jobs=n_jobs)(delayed(parallel_griddata)(*jobs[i]) for i in range(len(jobs)))

                for ijob in range(n_jobs):
                    ix, iy = np.unravel_index( l[ijob], (nx, ny) )
                    if not self.profiles:
                        self.u_c[ix,iy,:,it]        = vars0[ijob][0][:,:]
                        self.v_c[ix,iy,:,it]        = vars0[ijob][1][:,:]
                        self.w_c[ix,iy,:,it]        = vars0[ijob][2][:,:]
                        self.ptot[ix,iy,:,it]       = vars0[ijob][3][:,:]
                        self.latheat[ix,iy,:,it]    = vars0[ijob][4][:,:]
                        self.qnice[ix,iy,:,it]      = vars0[ijob][5][:,:]
                        self.qncloud[ix,iy,:,it]    = vars0[ijob][6][:,:]
                        self.qnrain[ix,iy,:,it]     = vars0[ijob][7][:,:]
                        self.cldnuc[ix,iy,:,it]     = vars0[ijob][8][:,:]
                        self.sctot[ix,iy,:,it]      = vars0[ijob][9][:,:]
                        self.noninduc[ix,iy,:,it]   = vars0[ijob][10][:,:]
                        self.Th[ix,iy,:,it]         = vars0[ijob][11][:,:]
                        self.qvapor[ix,iy,:,it]     = vars0[ijob][12][:,:]
                        self.qcloud[ix,iy,:,it]     = vars0[ijob][13][:,:]
                        self.qrisg[ix,iy,:,it]      = vars0[ijob][14][:,:]
                        self.qrain[ix,iy,:,it]      = vars0[ijob][15][:,:]
                        self.qice[ix,iy,:,it]       = vars0[ijob][16][:,:]
                        self.epotential[ix,iy,:,it] = vars0[ijob][17][:,:]
                        self.qngraupel[ix,iy,:,it]  = vars0[ijob][18][:,:]
                        self.qicesnow[ix,iy,:,it]   = vars0[ijob][19][:,:]
                        self.qghail[ix,iy,:,it]     = vars0[ijob][20][:,:]
                        
                        if self.compute_extra_fields: #ALEJANDRA
                            self.qhail[ix,iy,:,it]    = vars0[ijob][21][:,:]
                            self.qsnow[ix,iy,:,it]    = vars0[ijob][22][:,:]
                            self.qgraup[ix,iy,:,it] = vars0[ijob][23][:,:]
                            self.refl[ix,iy,:,it]     = vars0[ijob][24][:,:]
                            self.phyd[ix,iy,:,it]     = vars0[ijob][25][:,:]

                    else:
                        self.Th[ix,iy,:,it] = vars0[ijob][0][:,:]
                        self.ptot[ix,iy,:,it]   = vars0[ijob][1][:,:]
                        self.qvapor[ix,iy,:,it] = vars0[ijob][2][:,:]
            print ('interpolation with %d jobs took %f seconds'%(n_jobs,time.time()-start))
            del self.w1        
            del self.u1        
            del self.v1        
            del self.Th1       
            del self.ptot1     
            del self.qvapor1   
            del self.qcloud1   
            del self.qrisg1    
            del self.qtotal1   
            del self.latheat1  
            del self.qnice1    
            del self.qnrain1   
            del self.qncloud1  
            del self.cldnuc1   
            del self.qice1     
            del self.qrain1    
            del self.sctot1    
            del self.noninduc1 
            del self.epotential1
            del self.qngraupel1
            if self.compute_extra_fields: #ALEJANDRA
                del self.qhail1
                del self.qsnow1
                del self.qgraup1
                del self.refl1
                del self.phyd1

            gc.collect()

        else:
            if self.profiles:
                print( 'Warning! do not use more than 1 time step if computing profiles for N. This will probably crash.')
            # parallelize the time axis
            n_jobs, dtt = optimize_njobs( self.nt, n_jobs)
            print ('Interpolating data with %d jobs...'%(n_jobs))
            start = time.time()
            subt = []
            #tt = int(self.nt/n_jobs)
            t0 = 0
            for ijobs in range(n_jobs-1):
                t1 = t0+dtt
                subt.append([t0,t1])
                t0 = t1
            subt.append([t0,self.nt])
            jobs = []
            for ijob in range(n_jobs):
                if self.compute_extra_fields: #ALEJANDRA
                    jobs.append( (self.w1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.u1_c[:,:,:,subt[ijob][0]:subt[ijob][1]], self.v1_c[:,:,:,subt[ijob][0]:subt[ijob][1]], self.Th1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.ptot1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qvapor1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qcloud1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qrisg1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.latheat1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qnice1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qncloud1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qnrain1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.cldnuc1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qrain1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qice1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.sctot1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.noninduc1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.epotential1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qngraupel1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qicesnow1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qghail1[:,:,:,subt[ijob][0]:subt[ijob][1]], hgt[:,:,:,subt[ijob][0]:subt[ijob][1]], hgt_c[:,:,:,subt[ijob][0]:subt[ijob][1]], hgt000, hgt000_c, True, self.qhail1[:,:,:,subt[ijob][0]:subt[ijob][1]],self.qsnow1[:,:,:,subt[ijob][0]:subt[ijob][1]],self.qgraup1[:,:,:,subt[ijob][0]:subt[ijob][1]],self.refl1[:,:,:,subt[ijob][0]:subt[ijob][1]],self.phyd1[:,:,:,subt[ijob][0]:subt[ijob][1]]) )                
                else:
                    jobs.append( (self.w1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.u1_c[:,:,:,subt[ijob][0]:subt[ijob][1]], self.v1_c[:,:,:,subt[ijob][0]:subt[ijob][1]], self.Th1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.ptot1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qvapor1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qcloud1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qrisg1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.latheat1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qnice1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qncloud1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qnrain1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.cldnuc1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qrain1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qice1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.sctot1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.noninduc1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.epotential1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qngraupel1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qicesnow1[:,:,:,subt[ijob][0]:subt[ijob][1]], self.qghail1[:,:,:,subt[ijob][0]:subt[ijob][1]], hgt[:,:,:,subt[ijob][0]:subt[ijob][1]], hgt_c[:,:,:,subt[ijob][0]:subt[ijob][1]], hgt000, hgt000_c) )
            (vars0) = Parallel(n_jobs=n_jobs, verbose = 50)(delayed(parallel_time_RGI)(*jobs[i]) for i in range(len(jobs)))
            print ('ran parallel_time_RGI with %d jobs in %f minutes'%(n_jobs,(time.time()-start)/60.))
            #start=time.time()
            del self.w1        
            del self.u1        
            del self.v1        
            del self.Th1       
            del self.ptot1     
            del self.qvapor1   
            del self.qcloud1   
            del self.qrisg1    
            del self.qtotal1   
            del self.latheat1  
            del self.qnice1    
            del self.qncloud1  
            del self.qnrain1   
            del self.cldnuc1   
            del self.qice1     
            del self.qrain1    
            del self.sctot1    
            del self.noninduc1
            del self.epotential1
            del self.qngraupel1
            del self.qicesnow1
            del self.qghail1
            if self.compute_extra_fields: #ALEJANDRA
                del self.qhail1
                del self.qsnow1
                del self.qgraup1
                del self.refl1
                del self.phyd1
            gc.collect()

            # concatenate the different fields in time:
            ijob=0
            self.u_c        = vars0[ijob][0]
            self.v_c        = vars0[ijob][1]
            self.w_c        = vars0[ijob][2]
            self.ptot       = vars0[ijob][3]
            self.Th         = vars0[ijob][4]
            self.qvapor     = vars0[ijob][5]
            self.qcloud     = vars0[ijob][6]
            self.qrisg      = vars0[ijob][7]
            self.latheat    = vars0[ijob][8]
            self.qnice      = vars0[ijob][9]
            self.qncloud    = vars0[ijob][10]
            self.qnrain     = vars0[ijob][11]
            self.cldnuc     = vars0[ijob][12]
            self.qrain      = vars0[ijob][13] 
            self.qice       = vars0[ijob][14]
            self.sctot      = vars0[ijob][15]
            self.noninduc   = vars0[ijob][16]
            self.epotential = vars0[ijob][17]
            self.qngraupel  = vars0[ijob][18]
            self.qicesnow   = vars0[ijob][19]
            self.qghail     = vars0[ijob][20]
            if self.compute_extra_fields: #ALEJANDRA
                self.qhail   = vars0[ijob][21][:,:]
                self.qsnow   = vars0[ijob][22][:,:]
                self.qgraup  = vars0[ijob][23][:,:]
                self.refl    = vars0[ijob][24][:,:]
                self.phyd    = vars0[ijob][25][:,:]
            for ijob in range(1,n_jobs):
                self.u_c        = np.concatenate( (self.u_c,        vars0[ijob][0]),axis=3 )    
                self.v_c        = np.concatenate( (self.v_c,        vars0[ijob][1]),axis=3 ) 
                self.w_c        = np.concatenate( (self.w_c,        vars0[ijob][2]),axis=3 )
                self.ptot       = np.concatenate( (self.ptot,       vars0[ijob][3]),axis=3 )
                self.Th         = np.concatenate( (self.Th,         vars0[ijob][4]),axis=3 )
                self.qvapor     = np.concatenate( (self.qvapor,     vars0[ijob][5]),axis=3 )
                self.qcloud     = np.concatenate( (self.qcloud,     vars0[ijob][6]),axis=3 )
                self.qrisg      = np.concatenate( (self.qrisg,      vars0[ijob][7]),axis=3 )
                self.latheat    = np.concatenate( (self.latheat,    vars0[ijob][8]),axis=3 )
                self.qnice      = np.concatenate( (self.qnice,      vars0[ijob][9]),axis=3 )
                self.qncloud    = np.concatenate( (self.qncloud,    vars0[ijob][10]),axis=3 )
                self.qnrain     = np.concatenate( (self.qnrain,     vars0[ijob][11]),axis=3 )
                self.cldnuc     = np.concatenate( (self.cldnuc,     vars0[ijob][12]),axis=3 )
                self.qrain      = np.concatenate( (self.qrain,      vars0[ijob][13]),axis=3 )
                self.qice       = np.concatenate( (self.qice,       vars0[ijob][14]),axis=3 )
                self.sctot      = np.concatenate( (self.sctot,      vars0[ijob][15]),axis=3 )
                self.noninduc   = np.concatenate( (self.noninduc,   vars0[ijob][16]),axis=3 )
                self.epotential = np.concatenate( (self.epotential, vars0[ijob][17]),axis=3 )
                self.qngraupel  = np.concatenate( (self.qngraupel,  vars0[ijob][18]),axis=3 )
                self.qicesnow   = np.concatenate( (self.qicesnow,   vars0[ijob][19]),axis=3 )
                self.qghail     = np.concatenate( (self.qghail,     vars0[ijob][20]),axis=3 )
                if self.compute_extra_fields: #ALEJANDRA
                    self.qhail   = np.concatenate( (self.qhail,     vars0[ijob][21]),axis=3 )
                    self.qsnow   = np.concatenate( (self.qsnow,     vars0[ijob][22]),axis=3 )
                    self.qgraup  = np.concatenate( (self.qgraup, vars0[ijob][23]),axis=3 )
                    self.refl    = np.concatenate( (self.refl,     vars0[ijob][24]),axis=3 )
                    self.phyd    = np.concatenate( (self.phyd,     vars0[ijob][25]),axis=3 )
            #print ('assigned interpolated variables in %f seconds'%(time.time()-start))

        if not self.profiles:
            self.w_interp = self.w_c
            self.qtotal_interp = self.qcloud + self.qrisg

        # THE TOTAL PRESSURE (base state + perturbation):
        #self.ptot = pb + p
        #pb = None
        #p = None

        # THE HEIGHT FIELD IS NOW ONLY FUNCTION OF THE VERTICAL INDEX:
        self.hgt  = hgt000
        self.dh   = self.hgt[1:] - self.hgt[:-1]
        self.hgt_c= hgt000_c
        self.dh_c = self.hgt_c[1:] - self.hgt_c[:-1]
        self.hgt_w = hgt_w
        self.nz_w = nz_w
        self.nx_w = nx_w
        self.ny_w = ny_w
        self.dx_w = dx_w
        self.dy_w = dy_w
 
    def _compute_thermodynamic_quantities(self):
        self.temp 	= (300. + self.Th)/np.power((pref/self.ptot),(Rd/cp))                                                 	# temperature in K          
        theta 	        = 300. + self.Th                                                                                     	# potential temperature in K
        #self.Th         = None
        #del self.Th 
        theta_m 	= (theta)*(1. + (Rv/Rd)*self.qvapor)				                                # virtual potential temperature
        #self.theta_e 	= aux.theta_e( theta, self.qvapor )                                                            	# equivalent potential temperature
        if self.compute_rh:
            self.rh 	= aux.relhum( self.temp, self.ptot, self.qvapor )                                                       # relative humidity (eq in aux_functions.py)
        else:
            self.rh     = np.ones_like(self.temp)*np.nan
        rho_d 	        = (pref/(Rd*theta_m))*np.power((self.ptot/pref),(cv/cp))		                        	# density of the dry air
        self.rho_m 	= rho_d*(1 + self.qvapor)						                        	# density of the dry + moist air
        #self.rho_c 	= self.rho_d*(1 + self.qvapor + self.qcloud + self.qrain + self.qice + self.qgraup + self.qsnow)	# density of cloudy air
        if not self.profiles:
            #self.rho_condensate = rho_d*(self.qcloud + self.qrisg)#self.qrain + self.qice + self.qgraup + self.qsnow)
            if self.GCE:
                #toshii (debbug for testing negative buoyancy GCE)
                self.rho_c         = self.rho_m  #just using temperature and water vapor only (debbug)
            else:
                self.rho_c 	= self.rho_m + rho_d*(self.qcloud + self.qrisg) # total density, including all condensate
            self.qtotal     = self.qcloud + self.qrisg #self.qice + self.qgraup + self.qrain + self.qsnow
            #self.qrisg      = None
            #del self.qrisg
            mse0            = cp*self.temp + Lv*self.qvapor
            self.mse        = np.zeros_like(self.temp)
            for iz in range(len(self.hgt_c)):
                self.mse[:,:,iz,:] = mse0[:,:,iz,:] + g*self.hgt_c[iz]
        #self.qvapor = None
        if self.compute_theta:
            self.theta = theta
            self.theta_e = aux.theta_e( theta, self.qvapor ) #ALEJANDRA
        #del self.qvapor

          
    def crop_grid_z( self, z_i ):
        """
        Crop the grid in the vertical direction. z_i must be an array with the consecutive z-indices to keep.
        """
        #self.w      = self.w     [:,:,z_i,:]
        self.u_c    = self.u_c   [:,:,z_i,:]
        self.v_c    = self.v_c   [:,:,z_i,:]
        self.w_c    = self.w_c   [:,:,z_i,:]
        self.hgt_c  = self.hgt_c[z_i]
        self.dh_c   = self.hgt_c[1:] - self.hgt_c[:-1]

        self.qvapor         = self.qvapor[:,:,z_i,:]
        self.qcloud         = self.qcloud[:,:,z_i,:]
        self.temp 	        = self.temp 	     [:,:,z_i,:] 
        self.rh             = self.rh 	     [:,:,z_i,:] 
        if self.compute_theta:
            self.theta      = self.theta 	     [:,:,z_i,:] 
        self.rho_m 	        = self.rho_m 	     [:,:,z_i,:] 
        #self.rho_condensate = self.rho_condensate[:,:,z_i,:]
        self.rho_c 	        = self.rho_c 	     [:,:,z_i,:] 
        self.qtotal	        = self.qtotal 	     [:,:,z_i,:] 
        self.ptot           = self.ptot          [:,:,z_i,:] 
        self.mse            = self.mse           [:,:,z_i,:]

        self.nz = len(z_i)

    def create_thermal_grid( self, max_radius, t00=0, prev_thermal=None, W_min=1., min_thermal_duration=4, avg_dist_R=5., min_R=200., max_steps=2, disc_r=0.6, n_jobs=1, shifted=0.,parallel_thermals=False, up=True, cell='' ):
        return thermal3D.Thermal( max_radius, self, dx=self.dx, dt=self.dt, t00=t00, prev_thermal=prev_thermal, W_min=W_min, min_thermal_duration=min_thermal_duration, avg_dist_R=avg_dist_R, min_R=min_R, max_steps=max_steps, disc_r=disc_r, n_jobs=n_jobs, shifted=shifted, parallel_thermals=parallel_thermals, up=up, cell=cell )

    def release_memory( self ):
        #del self.u                
        #del self.v              
        #del self.w              
        #del self.Th             
        del self.qvapor         
        del self.qcloud         
        del self.qrisg          
        del self.u_c            
        del self.v_c            
        del self.w_c            
        del self.w_interp       
        del self.qtotal_interp  
        del self.temp
        if self.compute_theta:
            del self.theta
            del self.theta_e #ALEJANDRA
        del self.ptot           
        del self.hgt            
        del self.hgt_c          
        del self.x_grid         
        del self.y_grid         
        del self.x_grid_upoints 
        del self.y_grid_vpoints 
        del self.x_grid_wint    
        del self.y_grid_wint    
        del self.rh             
        gc.collect()

def parallel_griddata_profile( l, Th1, ptot1, qvapor1, hgt_c, hgt000_c, nx, ny, nz ):
    """
    interpolate a horizontal slice of 3d data, given by the flat indices in l
    """
    ix, iy = np.unravel_index( l, (nx, ny) )
    l0 = l[0]
    nl = len(l)
    Th      = np.zeros([nl,nz],   dtype=prec )
    ptot    = np.zeros([nl,nz],   dtype=prec )
    qvapor  = np.zeros([nl,nz],   dtype=prec )
    for il in range(nl):
        Th[il,:]        = pol.griddata( hgt_c[ix[il],iy[il],:], Th1[ix[il],iy[il],:]    , hgt000_c )
        qvapor[il,:]    = pol.griddata( hgt_c[ix[il],iy[il],:], qvapor1[ix[il],iy[il],:], hgt000_c )
        ptot[il,:]      = pol.griddata( hgt_c[ix[il],iy[il],:], ptot1[ix[il],iy[il],:]  , hgt000_c )
    return (Th, ptot, qvapor)

def parallel_griddata( l, w1, u1_c, v1_c, Th1, ptot1, qvapor1, qcloud1, qrisg1, latheat1, qnice1, qncloud1, qnrain1, cldnuc1, qrain1, qice1, sctot1, noninduc1, epotential1, qngraupel1, qicesnow1, qghail1, hgt, hgt_c, hgt000, hgt000_c, nx, ny, nz, compute_extra_fields = False, qhail1 = False, qsnow1 = False, qgraup1 = False, refl1 = False, phyd1 = False):
    """
    interpolate a horizontal slice of 3d data, given by the flat indices in l
    """
    ix, iy = np.unravel_index( l, (nx, ny) )
    l0 = l[0]
    nl = len(l)
    #w       = np.zeros([nl,nz+1], dtype=prec )
    u_c         = np.zeros([nl,nz],   dtype=prec )
    v_c         = np.zeros([nl,nz],   dtype=prec )
    w_c         = np.zeros([nl,nz],   dtype=prec )
    Th          = np.zeros([nl,nz],   dtype=prec )
    ptot        = np.zeros([nl,nz],   dtype=prec )
    qvapor      = np.zeros([nl,nz],   dtype=prec )
    qcloud      = np.zeros([nl,nz],   dtype=prec )
    qrisg       = np.zeros([nl,nz],   dtype=prec )
    latheat     = np.zeros([nl,nz],   dtype=prec )
    qnice       = np.zeros([nl,nz],   dtype=prec )
    qncloud     = np.zeros([nl,nz],   dtype=prec )
    qnrain      = np.zeros([nl,nz],   dtype=prec )
    cldnuc      = np.zeros([nl,nz],   dtype=prec )
    qrain       = np.zeros([nl,nz],   dtype=prec )
    qice        = np.zeros([nl,nz],   dtype=prec )
    sctot       = np.zeros([nl,nz],   dtype=prec )
    noninduc    = np.zeros([nl,nz],   dtype=prec )
    epotential  = np.zeros([nl,nz],   dtype=prec )
    qngraupel   = np.zeros([nl,nz],   dtype=prec )
    qicesnow    = np.zeros([nl,nz],   dtype=prec )
    qghail      = np.zeros([nl,nz],   dtype=prec )
    if compute_extra_fields: #ALEJANDRA
        qhail      = np.zeros([nl,nz],   dtype=prec )
        qsnow      = np.zeros([nl,nz],   dtype=prec )
        qgraup   = np.zeros([nl,nz],   dtype=prec )
        refl      = np.zeros([nl,nz],   dtype=prec )
        phyd      = np.zeros([nl,nz],   dtype=prec )
        
    for il in range(nl):
        #w[il,:]      = pol.griddata( hgt[ix[il],iy[il],:],   w1[ix[il],iy[il],:]      , hgt000 )
        u_c[il,:]    = pol.griddata( hgt_c[ix[il],iy[il],:], u1_c[ix[il],iy[il],:]    , hgt000_c )
        v_c[il,:]    = pol.griddata( hgt_c[ix[il],iy[il],:], v1_c[ix[il],iy[il],:]    , hgt000_c )
        w_c[il,:]    = pol.griddata( hgt[ix[il],iy[il],:]  , w1[ix[il],iy[il],:]      , hgt000_c )
        Th[il,:]     = pol.griddata( hgt_c[ix[il],iy[il],:], Th1[ix[il],iy[il],:]     , hgt000_c )
        ptot[il,:]   = pol.griddata( hgt_c[ix[il],iy[il],:], ptot1[ix[il],iy[il],:]   , hgt000_c )
        qvapor[il,:] = pol.griddata( hgt_c[ix[il],iy[il],:], qvapor1[ix[il],iy[il],:] , hgt000_c )
        qcloud[il,:] = pol.griddata( hgt_c[ix[il],iy[il],:], qcloud1[ix[il],iy[il],:] , hgt000_c )
        qrisg[il,:]  = pol.griddata( hgt_c[ix[il],iy[il],:], qrisg1[ix[il],iy[il],:]  , hgt000_c )
        latheat[il,:]= pol.griddata( hgt_c[ix[il],iy[il],:], latheat1[ix[il],iy[il],:], hgt000_c )
        qnice[il,:]  = pol.griddata( hgt_c[ix[il],iy[il],:], qnice1[ix[il],iy[il],:]  , hgt000_c )
        qncloud[il,:]= pol.griddata( hgt_c[ix[il],iy[il],:], qncloud1[ix[il],iy[il],:], hgt000_c )
        qnrain[il,:] = pol.griddata( hgt_c[ix[il],iy[il],:], qnrain1[ix[il],iy[il],:] , hgt000_c )
        cldnuc[il,:] = pol.griddata( hgt_c[ix[il],iy[il],:], cldnuc1[ix[il],iy[il],:] , hgt000_c )
        qrain[il,:]  = pol.griddata( hgt_c[ix[il],iy[il],:], qrain1[ix[il],iy[il],:] , hgt000_c )
        qice[il,:]   = pol.griddata( hgt_c[ix[il],iy[il],:], qice1[ix[il],iy[il],:] , hgt000_c )
        sctot[il,:]  = pol.griddata( hgt_c[ix[il],iy[il],:], sctot1[ix[il],iy[il],:], hgt000_c )
        noninduc[il,:] = pol.griddata( hgt_c[ix[il],iy[il],:], noninduc1[ix[il],iy[il],:] , hgt000_c )
        epotential[il,:] = pol.griddata( hgt_c[ix[il],iy[il],:], epotential1[ix[il],iy[il],:] , hgt000_c )
        qngraupel[il,:] = pol.griddata( hgt_c[ix[il],iy[il],:], qngraupel1[ix[il],iy[il],:] , hgt000_c )
        qicesnow[il,:]= pol.griddata( hgt_c[ix[il],iy[il],:], qicesnow1[ix[il],iy[il],:], hgt000_c )
        qghail[il,:]  = pol.griddata( hgt_c[ix[il],iy[il],:], qghail1[ix[il],iy[il],:], hgt000_c )
        
        if compute_extra_fields: #ALEJANDRA
            qhail[il,:]  = pol.griddata( hgt_c[ix[il],iy[il],:], qhail1[ix[il],iy[il],:], hgt000_c )
            qsnow[il,:]  = pol.griddata( hgt_c[ix[il],iy[il],:], qsnow1[ix[il],iy[il],:], hgt000_c )
            qgraup[il,:]  = pol.griddata( hgt_c[ix[il],iy[il],:], qgraup1[ix[il],iy[il],:], hgt000_c )
            refl[il,:]  = pol.griddata( hgt_c[ix[il],iy[il],:], refl1[ix[il],iy[il],:], hgt000_c )
            phyd[il,:]  = pol.griddata( hgt_c[ix[il],iy[il],:], phyd1[ix[il],iy[il],:], hgt000_c )

    if compute_extra_fields: #ALEJANDRA
        return (u_c, v_c, w_c, ptot, latheat, qnice, qncloud, qnrain, cldnuc, sctot, noninduc, Th, qvapor, qcloud, qrisg, qrain, qice, epotential, qngraupel, qicesnow, qghail, qhail, qsnow, qgraup, refl, phyd)
        
    else:
        return (u_c, v_c, w_c, ptot, latheat, qnice, qncloud, qnrain, cldnuc, sctot, noninduc, Th, qvapor, qcloud, qrisg, qrain, qice, epotential, qngraupel, qicesnow, qghail)


def parallel_time_RGI( w1, u1_c, v1_c, Th1, ptot1, qvapor1, qcloud1, qrisg1, latheat1, qnice1, qncloud1, qnrain1, cldnuc1, qrain1, qice1, sctot1, noninduc1, epotential1, qngraupel1, qicesnow1, qghail1, hgt, hgt_c, hgt000, hgt000_c, compute_extra_fields = False, qhail1 = False, qsnow1 = False, qgraup1 = False, refl1 = False, phyd1 = False):
    """
    interpolate a time-slice of 3d data (input fields must be previously sliced).
    """
    shape = u1_c.shape[:4] 
    u_c         = np.zeros( shape, dtype=prec )
    v_c         = np.zeros( shape, dtype=prec )
    w_c         = np.zeros( shape, dtype=prec )
    Th          = np.zeros( shape, dtype=prec )
    ptot        = np.zeros( shape, dtype=prec )
    qvapor      = np.zeros( shape, dtype=prec )
    qcloud      = np.zeros( shape, dtype=prec )
    qrisg       = np.zeros( shape, dtype=prec )
    latheat     = np.zeros( shape, dtype=prec )
    qnice       = np.zeros( shape, dtype=prec )
    qncloud     = np.zeros( shape, dtype=prec )
    qnrain      = np.zeros( shape, dtype=prec )
    cldnuc      = np.zeros( shape, dtype=prec )
    qrain       = np.zeros( shape, dtype=prec )
    qice        = np.zeros( shape, dtype=prec )
    sctot       = np.zeros( shape, dtype=prec )
    noninduc    = np.zeros( shape, dtype=prec )
    epotential  = np.zeros( shape, dtype=prec )
    qngraupel   = np.zeros( shape, dtype=prec )
    qicesnow    = np.zeros( shape, dtype=prec )
    qghail      = np.zeros( shape, dtype=prec )
    
    if compute_extra_fields: #ALEJANDRA
        qhail      = np.zeros( shape, dtype=prec )
        qsnow      = np.zeros( shape, dtype=prec )
        qgraup    = np.zeros( shape, dtype=prec )
        refl      = np.zeros( shape, dtype=prec )
        phyd      = np.zeros( shape, dtype=prec )
        
    for it in range(shape[-1]):
        ix, iy = np.unravel_index( np.arange(shape[0]*shape[1]), (shape[0],shape[1]) )
        for l in range(shape[0]*shape[1]):
            rgi=pol.RegularGridInterpolator(points=[hgt[ix[l],iy[l],:,it]],values=w1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            w_c[ix[l],iy[l],:,it]    = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=u1_c[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            u_c[ix[l],iy[l],:,it]    = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=v1_c[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            v_c[ix[l],iy[l],:,it]    = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=Th1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            Th[ix[l],iy[l],:,it]     = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=ptot1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            ptot[ix[l],iy[l],:,it]   = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qvapor1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qvapor[ix[l],iy[l],:,it] = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qcloud1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qcloud[ix[l],iy[l],:,it] = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qrisg1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qrisg[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=latheat1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            latheat[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qnice1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qnice[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qncloud1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qncloud[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qnrain1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qnrain[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=cldnuc1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            cldnuc[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qrain1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qrain[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qice1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qice[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=sctot1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            sctot[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=noninduc1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            noninduc[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=epotential1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            epotential[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qngraupel1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qngraupel[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qicesnow1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qicesnow[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qghail1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
            qghail[ix[l],iy[l],:,it]  = rgi(hgt000_c)
            
            if compute_extra_fields: #ALEJANDRA
                rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qhail1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
                qhail[ix[l],iy[l],:,it]  = rgi(hgt000_c)
                rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qsnow1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
                qsnow[ix[l],iy[l],:,it]  = rgi(hgt000_c)
                rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=qgraup1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
                qgraup[ix[l],iy[l],:,it]  = rgi(hgt000_c)
                rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=refl1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
                refl[ix[l],iy[l],:,it]  = rgi(hgt000_c)
                rgi=pol.RegularGridInterpolator(points=[hgt_c[ix[l],iy[l],:,it]],values=phyd1[ix[l],iy[l],:,it],fill_value=None,bounds_error=False)
                phyd[ix[l],iy[l],:,it]  = rgi(hgt000_c)
    if compute_extra_fields: #ALEJANDRA
        return (u_c, v_c, w_c, ptot, Th, qvapor, qcloud, qrisg, latheat, qnice, qncloud, qnrain, cldnuc, qrain, qice, sctot, noninduc, epotential, qngraupel, qicesnow, qghail, qhail, qsnow, qgraup, refl, phyd)        
    else:    
        return (u_c, v_c, w_c, ptot, Th, qvapor, qcloud, qrisg, latheat, qnice, qncloud, qnrain, cldnuc, qrain, qice, sctot, noninduc, epotential, qngraupel, qicesnow, qghail )

def index_grid( nx, ny, nz, x_grid, y_grid, hgt_c, x0=0, y0=0, z0=0 ):
    """
    creates a matrix with the indices of a nx*ny*nz grid (useful for scipy griddata interpolations!)
    """
    size = int(nx*ny*nz)
    grid = np.zeros([size,3], dtype=np.float64)
    ind = np.unravel_index(np.arange(size),(int(nx),int(ny),int(nz)))
    grid[:,0] = x_grid[np.asarray(ind)[0,:]+x0] #+ x0
    grid[:,1] = y_grid[np.asarray(ind)[1,:]+y0] #+ y0
    grid[:,2] = hgt_c[np.asarray(ind)[2,:]+z0]  #+ z0
    return grid


