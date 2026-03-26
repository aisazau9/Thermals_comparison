import numpy as np
import pdb
import plotting_thermals as pltfile
import matplotlib.pyplot as plt
from pylab import *
import WRF_3Danalysis as composite


Rd   = 287.                         # Gas constant for dry air, J/(kg K)
cp   = 7.*0.5*Rd                    # Specific heat of dry air at constant pressure, J/(kg K)
Lv   = 2.260e6
g    = 9.81                         # gravitational constant (m/s^2)

# this is to have a common discretization for the vertical profiles
dz   = 100.

nlevs = 15. # number of levels for vertical profiles


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

class Composite(composite.Composite):
    def make_mean_composite_tseries( self, tminplot=-4, tmaxplot=4, ymaxplot=0.022 ): #ALEJANDRA ymaxplot
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
        
        #ALEJANDRA
        
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
            pltfile.composite_plot( self.t_range-0.5, net_entr_mean*1e3, net_entr_std*1e3, ylabel='net entrainment (m$^{-1}$)', fname='net_entr', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=6 )
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
            pltfile.composite_plot( self.t_range[:-1]-0.5, net_entr_term_mean, net_entr_term_std, ylabel='net entr. term (m s$^{-2}$)', fname='net_entr_term', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot )
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
        pltfile.composite_plot( self.t_range, Ni, error=None, ylabel='Number of cases', fname='N', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=N_total + 5, ymin0=0, grid=True )
         
        pltfile.composite_plot( self.t_range, buoy_mean, buoy_std, ylabel=buoyl, fname='buoy', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None )
        pltfile.composite_plot( self.t_range, Fres_mean, Fres_std, ylabel=Fresl, fname='Fres', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None )
        pltfile.composite_plot( self.t_range, Fnh_mean, Fnh_std, ylabel=Fnhl, fname='Fnh', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None )
        pltfile.composite_plot( self.t_range, acc_mean, acc_std, ylabel=accl, fname='acc', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None )
        pltfile.composite_plot( self.t_range, Pnz_mean, Pnz_std, ylabel=Pnzl, fname='Pnz', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None )
        pltfile.composite_plot( self.t_range, D_mean, D_std, ylabel=Dl, fname='D', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=ymaxplot )
        pltfile.composite_plot( self.t_range, R_mean, R_std, ylabel=Rl, fname='R', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=.6, ymin0=0. )
        
        pltfile.composite_plot( self.t_range, buoy_mean, buoy_std, ylabel=buoyl, fname='buoy_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None, pctl=[buoy_10, buoy_90] )
        pltfile.composite_plot( self.t_range, Fres_mean, Fres_std, ylabel=Fresl, fname='Fres_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None, pctl=[Fres_10, Fres_90] )
        pltfile.composite_plot( self.t_range, Fnh_mean, Fnh_std, ylabel=Fnhl, fname='Fnh_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None, pctl=[Fnh_10, Fnh_90] )
        pltfile.composite_plot( self.t_range, acc_mean, acc_std, ylabel=accl, fname='acc_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None, pctl=[acc_10, acc_90] )
        pltfile.composite_plot( self.t_range, Pnz_mean, Pnz_std, ylabel=Pnzl, fname='Pnz_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None, pctl=[Pnz_10, Pnz_90] )
        pltfile.composite_plot( self.t_range, D_mean, D_std, ylabel=Dl, fname='D_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None, pctl=[D_10, D_90] )

        pltfile.composite_plot( self.t_range, R_mean, R_std, ylabel=Rl, fname='R_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=.7, ymin0=0., pctl=[R_10, R_90] )
        pltfile.composite_plot( self.t_range, W_mean, W_std, ylabel=Wl, fname='W_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None, ymin0=None, pctl=[W_10, W_90] )
        pltfile.composite_plot( self.t_range, wmax_mean, wmax_std, ylabel=wmaxl, fname='wmax_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=None, ymin0=None, pctl=[wmax_10, wmax_90] )

        pltfile.composite_plot( self.t_range, loge_mean, loge_std, ylabel=logel, fname='loge_pctl', folder=self.folder, xmin=tminplot, xmax=tmaxplot, ymax0=-2., ymin0=-3.2, pctl=[loge_10, loge_90], zero_y=False )

        np.save( self.folder+'/mean_composite_mom_budget.npy', np.vstack((self.t_range, acc_mean, Fres_mean, Fnh_mean, buoy_mean, acc_10, acc_90, Fres_10, Fres_90, Fnh_10, Fnh_90, buoy_10, buoy_90)).data)
        np.save( self.folder+'/weighted_mean_composite_mom_budget.npy', np.vstack((self.t_range, acc_wmean, Fres_wmean, Fnh_wmean, buoy_wmean, acc_10, acc_90, Fres_10, Fres_90, Fnh_10, Fnh_90, buoy_10, buoy_90)).data)
        np.save( self.folder+'/mean_composite_RW.npy', np.vstack((self.t_range, R_mean, W_mean, R_std, R_10, R_90, W_std, W_10, W_90)).data )
        np.save( self.folder+'/weighted_mean_composite_RW.npy', np.vstack((self.t_range, R_wmean, W_wmean, R_std, R_10, R_90, W_std, W_10, W_90)).data )
        np.save( self.folder+'/mean_net_entr_term.npy', np.vstack((self.t_range[:-1]+0.5,net_entr_term_mean, net_entr_term_std)).data )
        np.save( self.folder+'/weighted_mean_net_entr_term.npy', np.vstack((self.t_range[:-1]+0.5,net_entr_term_wmean, net_entr_term_std)).data )

        #Alejandra
        np.save( self.folder+'/mean_composite_NumberCases.npy', np.vstack((self.t_range,Ni)).data )

    
    def tracer_results( self ):
        """
        Here we plot the results for entrainment and detrainment based on tracking tracers throughout the lifetime of each thermal
        (one value per thermal, not per timestep!)
        """
        xmin = -5
        xmax = -1
        e_raw   = self.tracer_entr[:,0]
        d_raw   = self.tracer_detr[:,0]
        print( 'zero entrainment cases with tracers: %d'%(len(np.where(e_raw==0.))))
        e_raw[np.where(e_raw==0.)] = np.nan
        print( 'mean entrainment (tracers) = %.4f'%(np.nanmean(e_raw)))
        d_raw[np.where(d_raw==0.)] = np.nan
        print( 'mean detrainment (tracers) = %.4f'%(np.nanmean(d_raw)))
        e_raw   = np.ma.masked_array( e_raw, mask=np.isnan(e_raw) )
        d_raw   = np.ma.masked_array( d_raw, mask=np.isnan(d_raw) )
        
        #if self.folder[10:13]=='LBA':
        #    hmax=10750
        #else:
        #    hmax=16000
        #Z = np.arange(h0+dz*0.5, hmax, dz)  # 50m between steps for profiles
        
        loge_raw = np.log10( e_raw )
        logd_raw = np.log10( d_raw )
        h_raw = np.ma.masked_array(self.tracer_entr[:,1],mask=np.isnan(self.tracer_entr[:,1]))

        Ntotal= np.where(~np.isnan(h_raw))[0].shape[0]
        Nz = int(Ntotal/nlevs)

        Z               = np.ones(int(nlevs))*np.nan
        N               = np.ones(int(nlevs))*np.nan
        
        mean_loge_raw   = np.ones_like(Z)*np.nan
        mean_logd_raw   = np.ones_like(Z)*np.nan
        mean_e          = np.ones_like(Z)*np.nan
        mean_d          = np.ones_like(Z)*np.nan
        e_l             = np.ones_like(Z)*np.nan
        e_r             = np.ones_like(Z)*np.nan
        d_l             = np.ones_like(Z)*np.nan
        d_r             = np.ones_like(Z)*np.nan

        z0 = np.nanmin(h_raw.data) # in m
        z1 = z0
        for k in range(int(nlevs)):
            nz=0
            while nz<Nz and z1<20000.:
                z1+=1.
                i   = np.where( np.ma.greater_equal(h_raw,z0)*np.ma.less(h_raw,z1) )[0]
                nz  = len(i)
            N[k]            = nz
            Z[k]            = np.ma.mean( h_raw[i] )
            mean_loge_raw[k] = np.ma.mean( loge_raw[i] )
            mean_logd_raw[k] = np.ma.mean( logd_raw[i] )
            mean_e[k] = np.ma.mean( e_raw[i] )
            mean_d[k] = np.ma.mean( d_raw[i] )
            e_l[k] = np.nanpercentile(e_raw[i],10 )
            e_r[k] = np.nanpercentile(e_raw[i],90 )
            d_l[k] = np.nanpercentile(d_raw[i],10 )
            d_r[k] = np.nanpercentile(d_raw[i],90 )
            z0=np.copy(z1)

        pltfile.tracer_mixing( loge_raw, h_raw/1e3, xlabel = '$\log_{10}(\epsilon)$ (m$^{-1}$)', fname=self.folder +'/tracers_entr_raw.png', xmin=xmin, xmax=xmax, mean=mean_loge_raw, Z=Z/1e3, title='tracers' )
        pltfile.tracer_mixing( logd_raw, h_raw/1e3, xlabel = '$\log_{10}(\delta)$ (m$)^{-1}$', fname=self.folder +'/tracers_detr_raw.png', xmin=xmin, xmax=xmax, mean=mean_logd_raw, Z=Z/1e3, title='tracers' )

        np.save( self.folder+'/loge_tracers_mean_pctls.npy', [np.ma.mean(loge_raw), np.percentile(loge_raw,10), np.percentile(loge_raw,90)] )
        np.save( self.folder+'/logd_tracers_mean_pctls.npy', [np.ma.mean(logd_raw), np.percentile(logd_raw,10), np.percentile(logd_raw,90)] )
        self.loge_raw = loge_raw
        self.logd_raw = logd_raw
        self.tracer_e_avg = mean_e
        self.tracer_d_avg = mean_d
        self.tracer_e_l = e_l
        self.tracer_e_r = e_r
        self.tracer_d_l = d_l
        self.tracer_d_r = d_r

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
            if len(self.mse_env_or[i])>0:
                mse_out = np.nanmean(self.mse_env_or[i])
                mse_in  = np.nanmean(self.mse_thermal_or[i])
                mse_entr.append((self.mse_thermal_or[i][-1]-self.mse_thermal_or[i][0])/(dZ[i]*(mse_out-mse_in)))
            else:
                mse_entr.append(np.nan)

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
        entr_rate = pltfile.flatten_array(self.entr_rate)
        net_entr = self.net_entr_tsteps
        z_net_entr = self.z_for_net_entr_tsteps
        Fentr = pltfile.flatten_array(self.net_entr_term)
        
        
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

        np.save( self.folder+'/profile_mom_budget_'+case+'.npy', np.vstack((Z, acc_avg, Fnh_avg, Fres_avg, buoy_avg, Fentr_avg, Z2)) )
        np.save( self.folder+'/weighted_profile_mom_budget_'+case+'.npy', np.vstack((Z_w, acc_wavg, Fnh_wavg, Fres_wavg, buoy_wavg, Fentr_wavg, Z2_w)) )
        np.save( self.folder+'/profile_RW_'+case+'.npy', np.vstack((Z, R_avg/1e3, W_avg, net_entr_avg, Z2, net_entr_l, net_entr_r)) )
        np.save( self.folder+'/weighted_profile_RW_'+case+'.npy', np.vstack((Z_w, R_wavg/1e3, W_wavg, net_entr_wavg, Z2_w, net_entr_l, net_entr_r)) )

        #Alejandra
        np.save( self.folder+'/weighted_profile_OtherVars_'+case+'.npy', np.vstack((Z, N, lifetime_avg, wmax_avg, massflux_avg)) )


        
        #pltfile.height_profile( [self.tracer_e_avg,mse_entr_avg,e_avg],[Z/1e3,Z3/1e3,Z/1e3],label=['tracers','mse','direct'],fname=self.folder+'/profile_entrainment_new.pdf')

        #pltfile.height_profile( [tracer_entr_wavg,mse_entr_wavg,e_wavg],[Z3/1e3,Z3/1e3,Z/1e3],label=['tracers','mse','direct'],fname=None)
        
        # ALEJANDRA: Dic with limits for mom budget profile
        x_lims_case = {"case1_d02":(-0.02,0.04),
                      "case1_d03":(-0.02,0.02),
                      "case2_new_d02":(-0.02,0.04),
                      "case2_new_d03":(-0.02,0.02),
                      "case3_new_d02":(-0.02,0.04),
                      "case3_new_d03":(-0.025,0.025)}
        xmin_case, x_max_case = x_lims_case[self.exp_name]
        pltfile.height_profile( [acc_avg,Fnh_avg,Fres_avg,buoy_avg], [Z/1e3,Z/1e3,Z/1e3,Z/1e3], label=['dW/dt','Fnh','Fmix','buoy'], xticks=np.arange(-0.02,0.021,0.01), fname=self.folder+'/profile_mom_budget_'+case+'.pdf', title=self.exp_name, xmin=xmin_case, xmax=x_max_case, xlabel='m$\,$s$^{-2}$' )
        pltfile.height_profile( [lifetime_avg], Z/1e3, zero=False, fname=self.folder+'/profile_lifetime_'+case+'.pdf', title=self.exp_name, xlabel='lifetime (min)' )
        pltfile.height_profile( [D_avg], Z/1e3, zero=False, fname=self.folder+'/profile_D_'+case+'.pdf', title=self.exp_name, range_l=[D_l], range_r=[D_r], xlabel='D (km)' )
        #pltfile.height_profile( [n], Z/1e3, zero=False, fname=self.folder+'/profile_n_'+case+'.png',title=self.exp_name, xmin=None, xmax=None, xlabel='Number of thermals' )
        pltfile.height_profile( [W_avg], Z/1e3, zero=False, fname=self.folder+'/profile_W_'+case+'_talk.pdf', title=None, xlabel='W (m/s)', xmin=0, xmax=8, thin=True, ylabel=False )
        pltfile.height_profile( [W_avg], Z/1e3, zero=False, fname=self.folder+'/profile_W_'+case+'.pdf', title='W '+case, xlabel='W (m/s)', xmin=0, xmax=12, range_l=[W_l], range_r=[W_r] )
        pltfile.height_profile( [wmax_avg], Z/1e3, zero=False, fname=self.folder+'/profile_wmax_'+case+'.pdf', title='wmax '+case, xlabel='wmax (m/s)' )
        pltfile.height_profile( [R_avg*1e-3], Z/1e3, zero=False, fname=self.folder+'/profile_R_'+case+'.png', title=self.exp_name, xlabel='R (km)', xmin=0, xmax=1.2, range_l=[R_l*1e-3], range_r=[R_r*1e-3] )
        pltfile.height_profile( [loge_avg], Z/1e3, zero=False, fname=self.folder+'/profile_fract_entr_'+case+'.pdf', title=None, xlabel='$\log_{10}(\epsilon)$ $(m^{-1})$', xmin=-4, xmax=-2, range_l=[loge_l], range_r=[loge_r] )
        pltfile.tracer_mixing( loge, z_centre/1e3, xlabel='$\log_{10}(\epsilon)$ $(m^{-1})$', fname=self.folder+'/profile_fract_entr_'+case+'.pdf', xmin=-5, xmax=-1, mean=loge_avg, Z=Z/1e3, title='Direct method' )
        pltfile.tracer_mixing( loge, z_centre/1e3, xlabel='$\log_{10}(\epsilon)$ $(m^{-1})$', fname=self.folder+'/profile_fract_entr_'+case+'.png', xmin=-5, xmax=-1, mean=loge_avg, Z=Z/1e3, title='Direct method' )
        pltfile.height_profile( [massflux_avg], Z/1e3, zero=False, fname=self.folder+'/profile_massflux_'+case+'.pdf', title='Average mass flux '+self.exp_name, xlabel='10$^{5}$ kg m/s', xmin=None, xmax=None )
        pltfile.height_profile( [mass_avg*1e-3], Z/1e3, zero=False, fname=self.folder+'/profile_mass_'+case+'.pdf', title='Mass '+self.exp_name, xlabel='10$^{3}$ kg', range_l=[mass_l*1e-3], range_r=[mass_r*1e-3], xmin=None, xmax=None )
        valid=np.where(~np.isnan(mse_in_l))[0]
        #pltfile.height_profile( [tot_mflux*1e-5,n*mass_avg*W_avg*1e-10], [Z/1e3,Z/1e3], label=['M. flux', 'n*$\overline{m}$*$\overline{W}$'], zero=False, fname=self.folder+'/profile_tot_mflux_'+case+'.png', title='Total mass flux '+self.exp_name, xlabel='x 10$^{10}$ kg m/s', xmin=None, xmax=None )
        pltfile.height_profile( [mse_in_avg[valid]*1e-5, mse_out_avg[valid]*1e-5], [Z[valid]/1e3,Z[valid]/1e3], label=['MSE$_{thermal}$', 'MSE$_{env}$'], zero=False, fname=self.folder+'/profile_mse_'+case+'.png', title=self.exp_name, xlabel='x 10$^{5}$ J/kg', xmin=3.35, xmax=3.55, range_l=[mse_in_l[valid]*1e-5, None], range_r=[mse_in_r[valid]*1e-5, None], filled=True )
        
        pltfile.height_profile( [entr_rate_avg], Z/1e3, zero=False, fname=self.folder+'/profile_entr_rate'+case+'.pdf', title=None, xlabel='E (kg s$^{-1}$ m$^{-2}$)', xmin=None, xmax=None, range_l=[entr_rate_l], range_r=[entr_rate_r] )

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
        pltfile.height_profile( [mixing_mse_nan*1e3], Z0/1e3, label=None, zero=False, fname=self.folder+'/profile_mixing_mse_'+case+'.png', title=self.exp_name, xlabel='$\epsilon_{mse}$ (x10$^{-3}$ m$^{-1}$)', xmin=0, xmax=2.5 )
        pltfile.height_profile( [log_mixing_mse], Z0/1e3, fname=self.folder+'/profile_mixing_mse_'+case+'.png', title='MSE', xlabel='$\log_{10}(\epsilon)$ (m$^{-1}$)', xmin=-5, xmax=-1 )
        

        self.tracer_e_avg = np.ones_like(e_avg)*np.nan # old tracer estimate. Set to nan to avoid confusion!
        self.tracer_e_l = np.ones_like(e_avg)*np.nan 
        self.tracer_e_r = np.ones_like(e_avg)*np.nan 

        pltfile.corr_plot( R_avg*1e-3, e_avg*1e3, ylabel='$\epsilon$ (x 10$^{-3}$ m$^{-1}$)', xlabel=Rl, fname='profile_R_e', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=0.1, xmax=0.6, ymin=0.2, ymax=3.2, markersize=10, label_regr=True )
        
        pltfile.corr_plot( 1./(R_avg*1e-3), e_avg*1e3, ylabel='$\epsilon$ (x 10$^{-3}$ m$^{-1}$)', xlabel='R$^{-1}$ (x 10$^{-3}$ m$^{-1}$)', fname='profile_Rinv_e', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=0.6, xmax=5, ymin=0.6, ymax=5, markersize=10, label_regr=True, title=self.exp_name+' profile (direct)' )
        
        try:
            pltfile.corr_plot( np.log10(R_avg), np.log10(e_avg), ylabel=logel, xlabel='$\log_{10}$(R) (m)', fname='profile_logR_loge', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=2.1, xmax=3., ymin=-3.5, ymax=-2.2, markersize=10 )
        except:
            print('could not plot correlation plot of log10(R_avg) and log10(e_avg)')

        pltfile.corr_plot( W_avg, e_avg*1e3, ylabel='$\epsilon$ (x 10$^{-3}$ m$^{-1}$)', xlabel=Wl, fname='profile_W_e', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=1.5, xmax=6.5, ymin=0.2, ymax=3.2, markersize=10 )
        try:
            pltfile.corr_plot( np.log10(W_avg), np.log10(e_avg), ylabel=logel, xlabel='$\log_{10}$(W) (m s$^{-1}$)', fname='profile_logW_loge', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=None, xmax=None, ymin=-3.5, ymax=-2.2, markersize=10 )
        except:
            print('could not plot correlation plot of log10(W_avg) and log10(e_avg)')
       
        B = buoy_avg[np.where(buoy_avg>0)]
        W = W_avg[np.where(buoy_avg>0)]
        e = e_avg[np.where(buoy_avg>0)]
        w = wmax_avg[np.where(buoy_avg>0)]
        #e_tr = self.tracer_e_avg[np.where(buoy_avg>0)]
        pltfile.corr_plot( B*1e3/(W*W), e*1e3, ylabel='$\epsilon$ (x 10$^{-3}$ m$^{-1}$)', xlabel='B W$^{-2}$ (x 10$^{-3}$ m$^{-1}$)', fname='profile_BW2_e', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=None, xmax=None, ymin=0.2, ymax=3.2, markersize=10 )
        pltfile.corr_plot( B*1e3/(w*w), e*1e3, ylabel='$\epsilon$ (x 10$^{-3}$ m$^{-1}$)', xlabel='B w$_{max}$$^{-2}$ (x 10$^{-3}$ m$^{-1}$)', fname='profile_Bwmax2_e', folder=self.folder, xsym=False, ysym=False, flatten=False, bothreg=False, linewidth=1, xmin=None, xmax=None, ymin=0.2, ymax=3.2, markersize=10 )

        np.save( self.folder+'/mse_mixing_mean.npy', np.ma.mean(np.ma.masked_array(mixing_mse, mask=np.isnan(mixing_mse))) )


    def plot_histograms( self, vars=None, disc_r=80, Rmax=None, Wmax=None, lifetimemax=None, z0max=None, delta_zmax=None, Fnhmax=None, buoymax=None, Fmixmax=None, accmax=None, emax=None ):
        try:
            weights = np.ones_like( self.iz_up[:,3] )*100./len(self.iz_up[:,3])
            bins = np.arange(-self.R_range-self.delta_R,self.R_range+1.5*self.delta_R,self.delta_R)
            pltfile.histogram_plot( self.iz_up[:,3],     second_data=self.iz_low[:,3],   bins=bins, folder=self.folder, ylabel='Rz', fname='iz_straight', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights, weights_2=weights, xlabel='percent (%)', orientation='horizontal', title=self.exp_name )
            np.savez( self.folder+'/iz_straight_histogram.npz', self.iz_up[:,3], self.iz_low[:,3], bins, weights )
            
            pltfile.histogram_plot( self.iy_left[:,3],   second_data=self.iy_right[:,3], bins=bins, folder=self.folder, xlabel='Ry', fname='iy_straight', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights,weights_2=weights, ylabel='percent (%)', title=self.exp_name)
            pltfile.histogram_plot( self.ix_left[:,3],   second_data=self.ix_right[:,3], bins=bins, folder=self.folder, xlabel='Rx', fname='ix_straight', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights,weights_2=weights, ylabel='percent (%)', title=self.exp_name)
            np.savez( self.folder+'/ix_straight_histogram.npz', self.ix_left[:,3], self.ix_right[:,3], bins, weights )

            pltfile.histogram_plot( self.iz_up[:,2],     second_data=self.iz_low[:,2],   bins=bins, folder=self.folder, ylabel='Rz', fname='iz', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights,weights_2=weights, xlabel='percent (%)', orientation='horizontal' , title=self.exp_name)

            pltfile.histogram_plot( self.ix_left[:,0],   second_data=self.ix_right[:,0], bins=bins, folder=self.folder, xlabel='Rx', fname='ix', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights,weights_2=weights, ylabel='percent (%)' , title=self.exp_name)
            pltfile.histogram_plot( self.iy_left[:,1],   second_data=self.iy_right[:,1], bins=bins, folder=self.folder, xlabel='Ry', fname='iy', xmin=-self.R_range-self.delta_R,xmax=self.R_range+self.delta_R, second_color='gray', mean=True, weights=weights,weights_2=weights, ylabel='percent (%)', title=self.exp_name )

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
            pltfile.histogram_plot( x=delta_R, bins=bins, fname='deltaR', xlabel='$\Delta$ R (\%)', folder=self.folder, xmin=-R_range, xmax=R_range, mean=True, xticks=ticks, title=self.exp_name )

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
                Fentr = pltfile.flatten_array(self.net_entr_term)
                fname = ''
                flatten=False
                wghts = W*mass/(np.sum(W*mass))
                wghts_Fentr = []
                for i in range(self.W_c.shape[0]):
                    ind = np.where(~np.isnan(self.W_c[i]))[0]
                    wghts_Fentr.append(0.5*((self.W_c[i]*self.mass_c[i])[ind][1:]+(self.W_c[i]*self.mass_c[i])[ind][:-1]))
                wghts_Fentr = pltfile.flatten_array(wghts_Fentr)
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

            #radius = pltfile.flatten_array(R)
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
            pltfile.histogram_plot( x=R,  bins=np.logspace(0,4,100), fname=fname+'R_log', xlabel=Rl, folder=self.folder, flatten=flatten, xmin=100, xmax=3000, mean=False, log=True, weights=weights, y_units=y_units, cumulative=False, ymax=Rmax, title='mean radius$\,=\,$%d m'%(int(np.ma.mean(R))), histtype='step' )
            #********************************************************* 'main characteristics of the thermals' *********************************************************
            pltfile.histogram_plot( x=R*1e-3,  N=100, fname=fname+'R', xlabel=Rl, folder=self.folder, flatten=flatten, xmin=0, xmax=3, mean=False, weights=None, y_units=y_units, cumulative=True, ymax=None, title='mean radius$\,=\,$%d m'%(int(np.ma.mean(R))) )
            print( 'mean radius = %.3f +/- %.3f km'%(np.ma.mean(R*1e-3), np.ma.std(R*1e-3)))
            print( 'mean weighted radius = %.3f km'%(np.ma.average(R*1e-3,weights=wghts)))
            pltfile.histogram_plot( x=W, bins=np.arange(0.05,14.,0.1), fname=fname+'W', xlabel=Wl, folder=self.folder, flatten=flatten, mean=False, weights=weights, xmin=0, xmax=14, y_units=y_units, cumulative=True, ymax=Wmax, deltay=0.2, title='mean ascent rate$\,=\,$%.1f m$\,$s$^{-1}$'%(np.ma.mean(W)))
            print ('mean ascent rate (W) = %.2f +/- %.2f m/s'%(np.ma.mean(W), np.ma.std(W)))
            print ('mean weighted ascent rate (W) = %.2f m/s'%(np.ma.average(W,weights=wghts)))
            pltfile.histogram_plot( x=length, bins=bins, xlabel='Thermal lifetime (min)', fname='lifetime', folder=self.folder, xmin=0, xmax=15, mean=False, weights=weights2, y_units=y_units, ymax=lifetimemax, cumulative=True, title='average lifetime = %.1f min'%(np.ma.mean(length)) )
            print ('mean lifetime = %.2f +/- %.2f min'%(np.ma.mean(length), np.ma.std(length)))
            print ('mean weighted lifetime = %.2f min'%(np.ma.average(length,weights=weights_lifetime)))
            pltfile.histogram_plot( x=self.z0*1e-3, bins=np.arange(0,15.01,0.3), fname=fname+'z0', xlabel='$Z_0 (km)$', folder=self.folder,flatten=False, xmin=0, xmax=15, mean=False, weights=weights2, y_units=y_units, ymax=z0max, cumulative=True, title='average $Z_0$ = %.2f km'%(np.ma.mean(self.z0*1e-3)) )
            pltfile.histogram_plot( x=self.delta_z*1e-3, bins=np.arange(0,3.61,0.04), fname=fname+'delta_z', xlabel='$\Delta Z (km)$', folder=self.folder,flatten=False, xmin=0, xmax=3.5, mean=False, weights=weights2, y_units=y_units, ymax=delta_zmax, title='average $\Delta Z$ = %d m'%(np.ma.mean(self.delta_z)), cumulative=True )
            print ('mean distance traveled = %.3f +/- %.3f km'%(np.ma.mean(self.delta_z*1e-3),np.ma.std(self.delta_z*1e-3)))
            print ('mean weighted distance traveled = %.3f km'%(np.ma.average(self.delta_z*1e-3,weights=weights_lifetime)))
            pltfile.histogram_plot( x=self.z0*1e-3, bins=np.arange(0,15.01,0.3), fname=fname+'z0_rotated', ylabel='$Z_0 (km)$', folder=self.folder,flatten=False, xmin=0, xmax=14.1, mean=False, weights=weights2, ymax=z0max, cumulative=False, orientation='horizontal', xlabel='counts '+ y_units, title=None)
            print ('mean starting level = %.2f +/- %.2f km'%(np.ma.mean(self.z0*1e-3),np.ma.std(self.z0*1e-3)))
            print ('mean weighted starting level = %.2f km'%(np.ma.average(self.z0*1e-3,weights=weights_lifetime)))

            #**********************************************************************************************************************************************************

            pltfile.histogram_plot( x=self.deltazR, N=100, fname=fname+'deltazR', xlabel='$\Delta Z/R$', folder=self.folder,flatten=False, xmin=0, xmax=6, mean=False, weights=weights2, y_units=y_units, title=self.exp_name)
            #limit = np.around( np.amax(np.concatenate((np.abs(Fres),np.abs(Fnh),np.abs(buoy),np.abs(acc)))), decimals=3 )
            limit = 0.2
            dl = 2.*limit/200.
            bins = np.arange(-limit, limit+0.5*dl, dl)
            #********************************************************* 'momentum budget' ******************************************************************************
            pltfile.histogram_plot( x=Fnh , bins=bins, fname=fname+'Fnh',  folder=self.folder, flatten=flatten, xmin=-0.07, xmax=0.07, zero=True, weights=weights, y_units=y_units, ymax=Fnhmax,  deltax=0.05, cumulative=True, title='Fnh' )
            print ('mean Fnh = %.3f +/- %.3f'%(np.ma.mean(Fnh),np.ma.std(Fnh, ddof=1)))
            print ('mean weighted Fnh = %.3f'%(np.ma.average(Fnh,weights=wghts)))
            pltfile.histogram_plot( x=buoy, bins=bins, fname=fname+'buoy', folder=self.folder, flatten=flatten, xmin=-0.07, xmax=0.07, zero=True, weights=weights, y_units=y_units, ymax=buoymax, deltax=0.05, cumulative=True, title='Buoyancy' )
            print ('mean buoy = %.3f +/- %.3f'%(np.ma.mean(buoy),np.ma.std(buoy, ddof=1)))
            print ('mean weighted buoy = %.3f'%(np.ma.average(buoy,weights=wghts)))
            pltfile.histogram_plot( x=Fentr, bins=bins, fname=fname+'Fentr', xlabel='m s$^{-2}$', folder=self.folder, flatten=flatten, xmin=-0.07, xmax=0.07, zero=True, weights=weights_Fentr, y_units=y_units, ymax=Fmixmax, deltax=0.05, cumulative=True, title='Fentr' )
            print ('mean Fentr = %.3f +/- %.3f'%(np.ma.mean(Fentr),np.ma.std(Fentr, ddof=1)))
            print ('mean weighted Fentr = %.4f'%(np.ma.average(Fentr,weights=wghts_Fentr)))
            pltfile.histogram_plot( x=Fmix, bins=bins, fname=fname+'Fmix', xlabel='m s$^{-2}$', folder=self.folder, flatten=flatten, xmin=-0.07, xmax=0.07, zero=True, weights=weights, y_units=y_units, ymax=Fmixmax, deltax=0.05, cumulative=True, title='Fmix' )
            print( 'mean Fmix = %.3f +/- %.3f'%(np.ma.mean(Fmix),np.ma.std(Fmix, ddof=1)))
            print( 'mean weighted Fmix = %.3f'%(np.ma.average(Fmix,weights=wghts)))
            pltfile.histogram_plot( x=acc , bins=bins, fname=fname+'acc',  xlabel='m s$^{-2}$', folder=self.folder, flatten=flatten, xmin=-0.07, xmax=0.07, zero=True, weights=weights, y_units=y_units, ymax=accmax,  deltax=0.05, cumulative=True, title='acc' )
            print( 'mean acc = %.3f +/- %.3f'%(np.ma.mean(acc),np.ma.std(acc, ddof=1)))
            print( 'mean weighted acc = %.3f'%(np.ma.average(acc,weights=wghts)))
            #**********************************************************************************************************************************************************

            np.save( self.folder + '/' + prefix + '_histograms_3.npy', np.vstack((weights, buoy, Fnh, Fmix, acc, wghts)) )
            np.save( self.folder + '/' + prefix + '_histograms_Fentr.npy', np.vstack((weights_Fentr, Fentr, wghts_Fentr)) )
            self.Fentr=Fentr
            if vars==None:
                #np.save( self.folder + '/' + prefix + '_histogram_net_entr_gross.npy', np.vstack((weights, net_gross_entr)) )
                bins=np.arange(-8,8.1,0.5)
                pltfile.histogram_plot( x=net_gross_entr*1e3 , bins=bins, fname=fname+'netgross_entr',  xlabel='x10$^{-3}$m$^{-1}$', folder=self.folder, flatten=flatten, xmin=-8, xmax=8, zero=True, weights=None, y_units='counts', ymax=None,  deltax=2, cumulative=True, title='net gross $\epsilon$', mean=True )
                pltfile.histogram_plot( x=net_entr_tsteps*1e3, bins=bins, fname=fname+'net_entr',  xlabel='x10$^{-3}$m$^{-1}$', folder=self.folder, flatten=flatten, xmin=-8, xmax=8, zero=True, weights=None, y_units='counts', ymax=None,  deltax=2, cumulative=False, title='net $\epsilon$', mean=True )
                np.save( self.folder + '/' + prefix + '_histogram_net_entr_tsteps.npy', net_entr_tsteps*1e3 )
                print( 'average of net entrainment (per time step): %f'%(np.mean(net_entr_tsteps)))
                print( 'weighted average of net entrainment (per time step): %f'%(np.average(net_entr_tsteps,weights=wghts_Fentr)))
                print( 'average of net entrainment (per thermal): %f'%(np.mean(net_gross_entr)))

            min_mixing = -0.01
            max_mixing = 0.01
            pltfile.histogram_plot( x=mix     , N=100, fname=fname+'mixing',    xlabel='mse mixing (m$^{-1}$)', folder=self.folder, flatten=flatten, xmin=min_mixing, xmax=max_mixing, zero=True, mean=True, weights=weights, y_units=y_units, title=self.exp_name )
            if vars!=None:
                n, bins = np.histogram( self.D, bins=100 )
                pltfile.histogram_plot( x=D       , bins=bins, fname=fname+'D',          xlabel=Dl       , folder=self.folder, flatten=flatten, mean=True, title=self.exp_name )
            else:
                loge = entr # np.log10(e)
                pltfile.histogram_plot( x=D       , N=100, fname=fname+'D',          xlabel=Dl       , folder=self.folder, flatten=flatten, mean=True, title=self.exp_name )
                bins = np.arange(0,7, 0.05)
                pltfile.histogram_plot( x=e*1e3, bins=bins, fname=fname+'e',  xlabel='$\epsilon$ (x 10$^{-3}$ m$^{-1})$', folder=self.folder, flatten=False, mean=False, xmin=0, xmax=7, weights=weights, y_units=y_units, title=self.exp_name+' (direct method)', cumulative=True, ymax=emax)
                print( 'mean epsilon (direct method) = %.4f'%(np.nanmean(e)))
                nonnan=np.where(~np.isnan(e))[0]
                print( 'weighted mean epsilon (direct method) = %.4f'%(np.average(e[nonnan],weights=wghts[nonnan])))
                #tracer_entr = np.ma.masked_array(self.tracer_entr[:,0], mask=np.isnan(self.tracer_entr[:,0]))
                #pltfile.histogram_plot( x=tracer_entr*1e3, bins=bins, fname=fname+'e_tracers',  xlabel='$\epsilon$ (x 10$^{-3}$ m$^{-1})$', folder=self.folder, flatten=False, mean=False, xmin=0, xmax=7, weights=weights2, y_units=y_units, title=self.exp_name+' (tracers)', cumulative=True )
                bins = np.arange(-4.0, -1.5, (2.5/100))
                pltfile.histogram_plot( x=loge, bins=bins, fname=fname+'fract_entr',  xlabel='$\log_{10}(\epsilon) (m^{-1})$', folder=self.folder, flatten=False, mean=True, xmin=-4., xmax=-2., weights=weights, y_units=y_units, title=self.exp_name)#, ymax=1.1 )#, ymax=120 )
                #pltfile.histogram_plot( x=pltfile.flatten_array(self.entr_rate), fname=fname+'entr_rate',  xlabel='E (kg s$^{-1}$ m$^{-2})$', folder=self.folder, flatten=False, mean=False, xmin=None, xmax=None, weights=weights, y_units=y_units, title=self.exp_name, cumulative=True, ymax=None)
            pltfile.histogram_plot( x=mass    , N=100, fname=fname+'mass',       xlabel=massl    , folder=self.folder, flatten=flatten, mean=True, title=self.exp_name )

            pltfile.histogram_plot( x=z_centre*1e-3, N=80, fname=fname+'z_centre',   xlabel=z_centrel, folder=self.folder, flatten=flatten, mean=False, title=self.exp_name, xmin=0, xmax=15 )
            pltfile.histogram_plot( x=Pnz     , N=100, fname=fname+'Pnz',        xlabel=Pnzl     , folder=self.folder, flatten=flatten, zero=True, mean=True, title=self.exp_name )
            
            if vars!=None:
                n, bins = np.histogram(self.wmax, bins=100)
                pltfile.histogram_plot( x=wmax, bins=bins, fname=fname+'wmax',       xlabel=wmaxl    , folder=self.folder, flatten=flatten, mean=False, xmax=25, xmin=0, weights=weights, y_units=y_units, title=self.exp_name, ymax=.6 )
            else:
                pltfile.histogram_plot( x=wmax    , N=100, fname=fname+'wmax',       xlabel=wmaxl    , folder=self.folder, flatten=flatten, mean=False, xmax=25, xmin=0, weights=weights, y_units=y_units, title=self.exp_name, ymax=.6 )
            pltfile.histogram_plot( x=time    , N=100, fname=fname+'time',       xlabel='time (min)', folder=self.folder, flatten=flatten, mean=True, title=self.exp_name )
            #bins = np.arange(-15.5,15.6,1)
            #pltfile.histogram_plot( x=it, bins=bins, fname=fname+'stages',       xlabel='stage (tsteps)', folder=self.folder, flatten=flatten, mean=True )
        except:
            print('Something went wrong with the histograms.')


