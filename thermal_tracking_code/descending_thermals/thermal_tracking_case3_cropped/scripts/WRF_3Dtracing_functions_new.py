import numpy as np
import scipy as sp
from scipy import stats
import time
import pdb
import scipy.interpolate as pol
import os
import scipy.ndimage.filters as filters
from joblib import Parallel, delayed
import pickle as pkl
from io import StringIO
import WRF_3Dgrid_new as grid3D
from WRF_3Dthermal import index_grid, hhmm, optimize_njobs

class Grid(grid3D.Grid):

    def find_thermals( self, qcloud_thr=1e-8, w_thr=1., cluster_dist=500., min_thermal_duration=4, Rmax=900., W_min=1., avg_dist_R=5., min_R=200., max_steps=2, disc_r=0.6, s=30000, n_jobs=4, max_jobs=4, shift_amounts=[0.], cell='', cell_mask=None ):
        """
        find updrafts related to thermal-centres.

        shift_amounts is used if thermal centers are assumed to be shifted vertically from the wmax centers. In that case, it can be assigned a list of values (fractions of radius). Otherwise, default is no shift, so shift_amounts=[0]
        """
        self.cell_mask = cell_mask
        self.shift_amounts = shift_amounts
        self.min_thermal_duration = min_thermal_duration
        x, y, z, t = self._extract_wmax( min_dist=cluster_dist, qcloud_thr=qcloud_thr, w_thr=w_thr )
        self.clusters = self._create_clusters2( x, y, z, t )
        x, y, z, t = None, None, None, None
        self.w_interp       = None
        self.qtotal_interp  = None
        self.s = s                      # smoothing factor for the splines fitted to the centre trajectories
        logfile = open('log'+cell+'.out','w')
        self.total_number_clusters=len(self.clusters)
        print ('\nFound %d possible thermals!'%(self.total_number_clusters))
        logfile.write('Found %d possible thermals!\n'%(len(self.clusters)))
        
        start=time.time()
        log, log_shifted = process_thermals_JOBLIB( logfile, self.clusters, self, Rmax, W_min, avg_dist_R, min_R, max_steps, disc_r, n_jobs, self.shift_amounts, cell=cell )
        print ('processed all thermals in %f minutes.'%((time.time()-start)/60.))
        if log_shifted!=None:
            for j in log_shifted:
                logfile.write(j+'\n')
        logfile.close()

    def find_downdrafts( self, w_thr=-0.8, cluster_dist=500., min_thermal_duration=4, Rmax=900., W_min=-1., avg_dist_R=5., min_R=200., max_steps=2, disc_r=0.6, s=30000, n_jobs=4, max_jobs=4, shift_amounts=[0.], cell='', cell_mask=None ):
        """
        find updrafts related to thermal-centres.
        Parameters:
        ----------
        self	: grid object
        """
        self.cell_mask = cell_mask
        self.shift_amounts = shift_amounts
        self.min_thermal_duration = min_thermal_duration
        print ('Going to start extracting wmin points') #ALEJANDRA
        start=time.time()                #ALEJANDRA
        x, y, z, t = self._extract_wmin( min_dist=cluster_dist, qcloud_thr=0, w_thr=w_thr )
        print ('got all w min points in %f minutes.'%((time.time()-start)/60.)) #ALEJANDRA
        start=time.time()                #ALEJANDRA
        self.clusters = self._create_clusters2( x, y, z, t, up=False)
        x, y, z, t = None, None, None, None
        self.w_interp       = None
        self.qtotal_interp  = None
        self.s = s                      # smoothing factor for the splines fitted to the centre trajectories
        logfile = open('log'+cell+'_downdr.out','w')
        self.total_number_clusters=len(self.clusters)
        print ('\nFound %d possible downdrafts!'%(self.total_number_clusters))
        print ('Downdrafts found in %f minutes.'%((time.time()-start)/60.)) #ALEJANDRA
        logfile.write('Found %d possible downdrafts!\n'%(len(self.clusters)))
        start=time.time()
        log, log_shifted = process_thermals_JOBLIB( logfile, self.clusters, self, Rmax, W_min, avg_dist_R, min_R, max_steps, disc_r, n_jobs, self.shift_amounts, up=False, cell=cell )
        print ('processed all descending thermals in %f minutes.'%((time.time()-start)/60.))
        if log_shifted!=None:
            for j in log_shifted:
                logfile.write(j+'\n')
        logfile.close()


    def extract_subgrid( self, clusters, dist=3000. ):
        """
        extract a sub-grid object where all the thermals in clusters are contained, adding
        a distance dist (in m) on either side and at the top.
        """
        t0, tf, x0, xf, y0, yf, zf = get_t0tf( clusters )
        x0 = np.amax([0, x0 - int(dist/self.dx)])
        xf = np.amin([self.nx-1, xf + int(dist/self.dx)])
        y0 = np.amax([0, y0 - int(dist/self.dy)])
        yf = np.amin([self.ny-1, yf + int(dist/self.dy)])
        zf = np.amin([self.nz-1, zf + int(dist/self.dh[zf])])
        return Grid( original_grid=self, t0=t0, tf=tf, x00=x0, xf=xf, y00=y0, yf=yf, zf=zf, compute_rh=self.compute_rh, YY0=self.YY0, MM0=self.MM0, DD0=self.DD0 )
 

    def smooth_trajectory( self ):
        self.x_centre, self.y_centre, self.z_centre, self.u_thermal, self.v_thermal, self.w_thermal = smooth_xyz( self.x, self.y, self.z, dt=self.dt, s=self.s )         # smoothed centre positions in m
        self.ix_centre = np.interp(self.x_centre, self.x_grid, np.arange(self.nx)) # indices (as floats) of centre positions
        self.iy_centre = np.interp(self.y_centre, self.y_grid, np.arange(self.ny))
        self.iz_centre = np.interp(self.z_centre, self.hgt_c, np.arange(self.nz))


    def _extract_wmax( self, min_dist, qcloud_thr, w_thr ):
        """
        Extracts the peaks of vertical velocity above a certain threshold that are at least a min_dist distance appart,
        have a minimum qcloud content and are at least min_dist+2dx away from the edges of the horizontal domain.
        """
        for it in range(self.nt):
            wmax_filtered = filters.maximum_filter( self.w_interp[:,:,:,it], int(2*min_dist/self.dx_w)+1 )
            #wmax_filtered = filters.maximum_filter( self.w_interp[:,:,:,it], int(2*min_dist/self.dx_w) )
            maxima = wmax_filtered == self.w_interp[:,:,:,it]
            # (only consider wmax points inside mask)
            filtered_indices = np.where(maxima*self.cell_mask[it,:,:,:])
            above_w_thr = np.where( np.ma.greater(self.w_interp[filtered_indices[0],filtered_indices[1],filtered_indices[2],it], w_thr) )
            above_q_thr = np.where( np.ma.greater(self.qtotal_interp[filtered_indices[0][above_w_thr],filtered_indices[1][above_w_thr],filtered_indices[2][above_w_thr],it], qcloud_thr) )
            far_from_edges = np.where( (filtered_indices[0][above_w_thr][above_q_thr] > (min_dist/self.dx) + 2)*(filtered_indices[0][above_w_thr][above_q_thr] < (self.nx - ((min_dist/self.dx) + 2)))*(filtered_indices[1][above_w_thr][above_q_thr] > (min_dist/self.dx) + 2)*(filtered_indices[1][above_w_thr][above_q_thr] < (self.ny - ((min_dist/self.dy) + 2))) )
            sort_ind = np.flipud(np.argsort(self.w_interp[filtered_indices[0][above_w_thr][above_q_thr][far_from_edges],filtered_indices[1][above_w_thr][above_q_thr][far_from_edges],filtered_indices[2][above_w_thr][above_q_thr][far_from_edges],it])) 
            # this sorts the indices according to wmax, so it will start looking for clusters starting with the strongest updraft
            if 'x' in vars():
                x = np.concatenate( (x, filtered_indices[0][above_w_thr][above_q_thr][far_from_edges][sort_ind]) )
                y = np.concatenate( (y, filtered_indices[1][above_w_thr][above_q_thr][far_from_edges][sort_ind]) )
                z = np.concatenate( (z, filtered_indices[2][above_w_thr][above_q_thr][far_from_edges][sort_ind]) )
                t = np.concatenate( (t, np.ones(len(filtered_indices[0][above_w_thr][above_q_thr][far_from_edges][sort_ind]))*it) )
            else:
                x = filtered_indices[0][above_w_thr][above_q_thr][far_from_edges][sort_ind]
                y = filtered_indices[1][above_w_thr][above_q_thr][far_from_edges][sort_ind]
                z = filtered_indices[2][above_w_thr][above_q_thr][far_from_edges][sort_ind]
                t = np.ones(len(filtered_indices[0][above_w_thr][above_q_thr][far_from_edges][sort_ind]))*it
        return x, y, z, t

    def _extract_wmin( self, min_dist, qcloud_thr, w_thr ):
        """
        Extracts the peaks of negative vertical velocity below a certain threshold that are at least a min_dist distance appart,
        have a minimum qcloud content and are at least min_dist+2dx away from the edges of the horizontal domain.
        """
        for it in range(self.nt):
            print ("Getting wmin points in ", it)
            #wmax_filtered = filters.minimum_filter( self.w_interp[:,:,:,it], int(2*min_dist/self.dx_w)+1 )
            wmax_filtered = filters.minimum_filter( self.w_interp[:,:,:,it], int(2*min_dist/self.dx_w) )
            minima = wmax_filtered == self.w_interp[:,:,:,it]
            filtered_indices = np.where(minima)
            above_w_thr = np.where( np.ma.less(self.w_interp[filtered_indices[0],filtered_indices[1],filtered_indices[2],it], w_thr) )
            #above_q_thr = np.where( np.ma.greater(self.qtotal_interp[filtered_indices[0][above_w_thr],filtered_indices[1][above_w_thr],filtered_indices[2][above_w_thr],it], qcloud_thr) )
            far_from_edges = np.where( (filtered_indices[0][above_w_thr] > (min_dist/self.dx) + 2)*(filtered_indices[0][above_w_thr] < (self.nx - ((min_dist/self.dx) + 2)))*(filtered_indices[1][above_w_thr] > (min_dist/self.dx) + 2)*(filtered_indices[1][above_w_thr] < (self.ny - ((min_dist/self.dy) + 2))) )
            sort_ind = np.flipud(np.argsort(self.w_interp[filtered_indices[0][above_w_thr][far_from_edges],filtered_indices[1][above_w_thr][far_from_edges],filtered_indices[2][above_w_thr][far_from_edges],it])) 
            # this sorts the indices according to wmax, so it will start looking for clusters starting with the strongest updraft
            if 'x' in vars():
                x = np.concatenate( (x, filtered_indices[0][above_w_thr][far_from_edges][sort_ind]) )
                y = np.concatenate( (y, filtered_indices[1][above_w_thr][far_from_edges][sort_ind]) )
                z = np.concatenate( (z, filtered_indices[2][above_w_thr][far_from_edges][sort_ind]) )
                t = np.concatenate( (t, np.ones(len(filtered_indices[0][above_w_thr][far_from_edges][sort_ind]))*it) )
            else:
                x = filtered_indices[0][above_w_thr][far_from_edges][sort_ind]
                y = filtered_indices[1][above_w_thr][far_from_edges][sort_ind]
                z = filtered_indices[2][above_w_thr][far_from_edges][sort_ind]
                t = np.ones(len(filtered_indices[0][above_w_thr][far_from_edges][sort_ind]))*it
        return x, y, z, t


    def _create_clusters2( self, x, y, z, t, up=True):
        t=t.astype(int)
        data = []
        data_next = np.ones(len(x))*-1
        data_prev = np.ones(len(x))*-1
        for i in range(len(x)):
            data.append((x[i],y[i],z[i],t[i]))
            if up:
                next_t_pts = np.where((t==t[i]+1)*(z>=z[i]))[0]
                prev_t_pts = np.where((t==t[i]-1)*(z<=z[i]))[0]
            else:
                next_t_pts = np.where((t==t[i]+1)*(z<=z[i]))[0]
                prev_t_pts = np.where((t==t[i]-1)*(z>=z[i]))[0]
            dist_next = []
            dist_prev = []
            for ind in next_t_pts:
                dist_next.append( distance3D( x0=self.x_grid_wint[x[i]], x1=self.x_grid_wint[x[ind]], y0=self.y_grid_wint[y[i]], y1=self.y_grid_wint[y[ind]], z0=self.hgt_w[z[i]], z1=self.hgt_w[z[ind]] ) )
            for ind in prev_t_pts:
                dist_prev.append( distance3D( x0=self.x_grid_wint[x[i]], x1=self.x_grid_wint[x[ind]], y0=self.y_grid_wint[y[i]], y1=self.y_grid_wint[y[ind]], z0=self.hgt_w[z[i]], z1=self.hgt_w[z[ind]] ) )
            # we assume that the next centre will be at the most a distance w_max*dt*1.5 away from the current centre:
            max_dist = np.absolute((self.w_interp[x[i],y[i],z[i],t[i]]*self.dt)*1.5)
            # choose the closest one (if there is more than 1 to choose from):
            if len(np.where(dist_next<max_dist)[0]) >= 1:
                data_next[i] = next_t_pts[np.where(dist_next==np.amin(dist_next))[0]][0]
            if len(np.where(dist_prev<max_dist)[0]) >= 1:
                data_prev[i] = prev_t_pts[np.where(dist_prev==np.amin(dist_prev))[0]][0]
        cluster_number = np.ones_like( data_next )*-1
        used_points = np.zeros_like(data_next, dtype='bool')
        counter = 1
        used_points[np.where((data_next==-1)*(data_prev==-1))] = True
        while np.any(~used_points):
            first_choice=np.where((~used_points)*(data_prev==-1))[0]
            if len(first_choice)>0:
                point = first_choice[0]
            else:
                point = np.where((~used_points))[0][0]
            cluster_number[point]=counter
            used_points[point]=True
            while data_next[point]!=-1 and np.any(~used_points):
                point = int(data_next[point])
                cluster_number[point]=int(counter)
                used_points[point]=True
                if data_next[point] == -1:
                    possible=np.where(data_prev==point)[0]
                    if len(possible)>0:
                        data_next[point] = possible[0]
            counter+=1
        n_clusters = int(np.amax(cluster_number))
        clusters = []
        for i in range(n_clusters):
            centres = np.where(cluster_number==i+1)[0]
            if len(centres)>=self.min_thermal_duration:
                clusters.append( np.asarray(data)[centres] )
        # sort the clusters in time, to make the parallelization more memory efficient:
        starting_point = np.zeros(len(clusters))
        for i in range(len(clusters)):
            starting_point[i] = clusters[i][0][3]
        ordered_clusters_index = np.argsort( starting_point )
        sorted_clusters = []
        for i in range(len(clusters)):
            sorted_clusters.append(clusters[ordered_clusters_index[i]])
        return sorted_clusters


    def _rule_out_bad_points( self, up=True):
        """
        remove time steps from a thermal at the start and/or end, which are ruled out due to their vertical velocity.
        returns True if the resulting thermal is at least min_thermal-duration time-steps long, False otherwise.
        """
        log = []
        if up:
            rule_out2 = np.where( self.w_thermal <= 0.1 )[0]	# rule out any descending points, or very slow thermals!
        else:
            rule_out2 = np.where( self.w_thermal>=-0.1 )[0] # rule out any ascending points, or very slow downdrafts!
        rule_out = rule_out2
   
        first_j = 0
        last_j=len(self.x)
        half = int(last_j/2.)
    
        j0, j1 = get_rule_out_slice( rule_out, half )
        if j1 == -1:
            j1 = last_j
        first_j = np.amax([first_j, j0])
        last_j = np.amin([last_j,j1])
    
        if last_j-first_j >= self.min_thermal_duration:
            if first_j > 0 or last_j < len(self.x):
                self.x		= self.x[first_j:last_j]
                self.y          = self.y[first_j:last_j]
                self.z		= self.z[first_j:last_j]
                self.t		= self.t[first_j:last_j]
                #self.smooth_trajectory()                #re-smooth the centres and update the velocities using the remaining points
                # since we don't smooth now, we must also crop these:
                self.x_centre   = self.x_centre[first_j:last_j]
                self.y_centre   = self.y_centre[first_j:last_j]
                self.z_centre   = self.z_centre[first_j:last_j]
                self.u_thermal  = self.u_thermal[first_j:last_j]
                self.v_thermal  = self.v_thermal[first_j:last_j]
                self.w_thermal  = self.w_thermal[first_j:last_j]

                self.tsteps 	= self.tsteps 	[first_j:last_j]
            self.xmax 	    = self.x
            self.ymax       = self.y
            self.hmax 	    = self.z
            valid_thermal   = True
        else:
            log.append( hhmm(self.hr0*60+self.min0+self.sec0/60.+self.t[first_j]) + ' - Thermal at x0=%.2f, z0=%.2f'%(self.x_centre[0], self.z_centre[0]) + ' is too short (nt=%01d)'%(last_j-first_j) )
            #print hhmm(self.hr0*60+self.min0+self.sec0/60.+self.t[first_j]) + ' - Thermal at x0=%.2f, z0=%.2f'%(self.x_centre[0], self.z_centre[0]) + ' is too short (nt=%01d)'%(last_j-first_j)
            valid_thermal = False
        return valid_thermal, log

    def make_movie( self, x, z, r, t, fname, cropx=False, cropy=False, gif=False ):
        for k in range(len(x)):
            s = '%02d'%int(k)
            self.show_wmax( t[k], x[:k+1], z[:k+1], r[k], n=k+1, fname='frame_'+s+'.jpg', cropx=cropx, cropy=cropy )
        os.system( 'cp frame_' + '%02d'%int(len(x)-1) + '.jpg frame_' + '%02d'%int(len(x)) + '.jpg' ) # repeat the last frame
        if gif:
            os.system( 'convert frame_??.jpg -delay 40 ' + fname + '.gif' )
        if os.path.isdir( 'tmp_movie_folder' ):
            os.system( 'rm -r tmp_movie_folder' )
        os.mkdir( 'tmp_movie_folder' )
        os.system( 'convert frame_??.jpg tmp_movie_folder/frame%05d.jpg' )
        os.system( 'rm -rf frame_??.jpg' )
        os.system( 'ffmpeg -loglevel 0 -r 3 -qscale 1 -i tmp_movie_folder/frame%05d.jpg -y -an ' + fname + '.avi' )
        os.system( 'rm -R tmp_movie_folder' )
    
    
    def show_wmax( self, t, x, y, R, n=1, qcloud_thr=1e-5, fname=None, cropx=False, cropy=False ): 
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib import rc
        #it = t-int(self.hr0*60+self.min0)
        it = t
        psi = self.compute_streamfunction( it )
        
        nx = self.nx
        nz = self.nz
        vorticity = self.vorticity
        dh = self.dh
        dh_c = self.dh_c
        p_max = np.nanmax(abs(vorticity))
        p_min = -p_max
        w_max = np.nanmax(abs(self.w_c[:,:,it]))
        w_min = -w_max
        N = 150
        rc('text', usetex=False)
        fig=plt.figure(figsize=(16,10))
        ax=plt.gca()
        #plt.contourf(self.x_grid/1e3, self.hgt_c/1e3, np.swapaxes(vorticity[:,:,it],0,1), N, levels=np.arange(N)*(2*p_max)/(N-1)-p_max)
        plt.contourf(self.x_grid/1e3, self.hgt_c/1e3, np.swapaxes(self.w_c[:,:,it],0,1), N, levels=np.arange(N)*(2*w_max)/(N-1)-w_max)
        #CB = plt.colorbar( pad=0, ticks=[-p_max, 0, p_max], format='%.2f')
        CB = plt.colorbar( pad=0, ticks=[-w_max, 0, w_max], format='%.2f')
        for i in range(n):
            plt.plot(x[i]/1e3,y[i]/1e3, 'k+', ms=10, mew=3)
        if cropx:
            plt.xlim( np.amax( [np.amin(x[:])/1e3 - 6, self.x_grid[0]/1e3] ), np.amin( [np.amax(x[:])/1e3 + 6, self.x_grid[-1]/1e3] ) )
        if cropy:
            plt.ylim( 0, np.amin([np.amax(y[:])/1e3 + 6, self.hgt_c[-1]/1e3]) )
        if not cropx and not cropy:
            plt.axis('tight')
        plt.title( '%02d'%(t/60) + ':' +  '%02d'%(np.mod(t,60)) )
        plt.xlabel('X (km)')
        plt.ylabel('Height (km)')
        l,b,w,h = plt.gca().get_position().bounds
        ll,bb,ww,hh = CB.ax.get_position().bounds
        CB.ax.set_position([ll+0.1*w, b, ww, h])
        N=30
        plt.contour(self.x_grid/1e3, self.hgt_c/1e3, np.swapaxes(self.qtotal[:,:,it],0,1),1,levels=[qcloud_thr],colors='r',linewidths=2)
        plt.matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
        plt.contour(self.x_grid[1:-1]/1e3, self.hgt_c[1:-1]/1e3, np.swapaxes(psi,0,1),N,colors='k')
        circle = Circle((x[-1]/1e3,y[-1]/1e3), R/1e3, facecolor='none',edgecolor='k',linewidth=3 )
        ax.add_artist(circle)
        if fname==None:
            plt.show() 
        else:
            plt.savefig( fname )
    
    
    def expand_attributes_one_tstep( self, direc=1 ):
        """
        Adds one timestep (filled with zero) to the relevant attributes either at the end or at the beginning,
        according to dir (1 means forward tracing, so does it at the end, -1 means backward, means at the beginning).
        """	
        self.xmax 	= add_one_tstep( self.xmax 	 , direc )
        self.hmax 	= add_one_tstep( self.hmax 	 , direc )
        self.ix_centre 	= add_one_tstep( self.ix_centre  , direc ) 
        self.iz_centre 	= add_one_tstep( self.iz_centre  , direc )
        self.x_centre  	= add_one_tstep( self.x_centre   , direc )
        self.z_centre  	= add_one_tstep( self.z_centre   , direc )
        self.tsteps	= add_one_tstep( self.tsteps     , direc ) 
        self.u_thermal  = add_one_tstep( self.u_thermal  , direc )
        self.w_thermal  = add_one_tstep( self.w_thermal  , direc )

		
def add_one_tstep( var, dir=1 ):
    result = np.zeros(len(var)+1)
    if dir==1:
        result[:-1]=var
    if dir==-1:
        result[1:]=var
    return result
	

def remove_cluster( tmp_cluster, x, y, t ):
    """
    remove the peak-w data in tmp_cluster from x, y, and t
    """
    ind = np.zeros(len(tmp_cluster))
    for i in range(len(tmp_cluster)):
        ind[i]=tmp_cluster[i][0]
    x0 = np.zeros((len(x)-len(ind)))
    y0 = np.zeros((len(x)-len(ind)))
    t0 = np.zeros((len(x)-len(ind)))
    j = 0
    for i in range(len(x)):
        if len(np.where(ind==i)[0])==0:
            x0[j]=x[i]
            y0[j]=y[i]
            t0[j]=t[i]
            j+=1
    return x0,y0,t0


def smooth_xyz( x, y, z, dt=60., s=30000, k=None ):
    """
    smooth the trajectory of the thermal's centre by interpolating with a 1d spline of 3d order 
    on each coordinate, with smoothing factor s. Get the velocities at once from the first 
    derivative of the spline.
    """
    l = len(x)
    ll = np.arange(l)
    if k==None:
        k=3
        if l<=4:
            k=2
        if l<3:
            k=1
    xspline = pol.UnivariateSpline( ll, x, k=k, s=s )
    yspline = pol.UnivariateSpline( ll, y, k=k, s=s )
    zspline = pol.UnivariateSpline( ll, z, k=k, s=s )
    x_smooth = xspline(ll)
    y_smooth = yspline(ll)
    z_smooth = zspline(ll)
    u = np.zeros(l)
    v = np.zeros(l)
    w = np.zeros(l)
    for i in range(l):
        u[i] = xspline.derivatives(i)[1]/dt
        v[i] = yspline.derivatives(i)[1]/dt
        w[i] = zspline.derivatives(i)[1]/dt
    return x_smooth, y_smooth, z_smooth, u, v, w


def plot_smoothing( x, y, x_smooth, y_smooth, title='' ):
    """
    function for plotting the smoothing results in order to tune the smoothing factor s
    """
    import matplotlib.pyplot as plt
    l = np.arange(len(x))
    plt.figure(figsize=(5,15))
    plt.subplot(311)
    plt.title(title)
    plt.scatter(l,x)
    plt.plot(l,x_smooth)
    plt.subplot(312)
    plt.scatter(l,y)
    plt.plot(l,y_smooth)
    plt.subplot(313)
    plt.scatter(x,y)
    plt.plot(x_smooth,y_smooth)


def running_mean( x_in, l, weights ):
    """NOT USED NOW!!!
    """
    if np.mod(l,2)==0:
        l=l-1
        print ('using l=',l,' (running mean length)')
    l2=int((l-1)/2) #number of points at either side
    
    x_new = np.zeros_like( x_in )
    x_new[0]  = x_in[0]
    x_new[-1] = x_in[-1]
    if l == 5:
        x_new[1]= x_in[0]*(weights[0]+weights[1]*0.5) + x_in[1]*(weights[1]*0.5+weights[2]+weights[3]*0.5) + x_in[2]*(weights[3]*0.5 + weights[4])
        x_new[-2]= x_in[-3]*(weights[0]+weights[1]*0.5) + x_in[-2]*(weights[1]*0.5+weights[2]+weights[3]*0.5) + x_in[-1]*(weights[3]*0.5 + weights[4])
    #x_new[:l2]=x_in[:l2]
    #x_new[-l2:]=x_in[-l2:]
    for i in range(l2,len(x_in)-l2):
        for j in range(-l2,l2+1):
            x_new[i] += weights[j+l2]*x_in[i+j]
    return x_new


#def compute_vel( x, y, dt ):
#    """NOT USED NOW!!!
#    """
#    W = np.zeros_like( x)
#    U = np.zeros_like( x )
#    W[0] = (y[1]-y[0])/dt
#    W[1:-1] = (y[2:]-y[0:-2])/(2*dt)
#    #W[0] = W[1]
#    #W[-1] = W[-2]
#    W[-1] = (y[-1]-y[-2])/dt
#    U[0] = (x[1]-x[0])/dt
#    U[1:-1] = (x[2:]-x[0:-2])/(2*dt)
#    U[-1] = (x[-1]-x[-2])/dt
#    return U, W


def get_rule_out_slice (rule_out, half):
    """
    This function returns the indices needed to slice data in order to remove "bad points" found in _rule_out_bad_points()
    """
    if rule_out.shape[0] > 0:
        low = np.where(rule_out <= half )[0]
        high = np.where(rule_out > half)[0]
        if low.shape[0]>0:
            j0 = np.amax(rule_out[low]) + 1
        else:
            j0 = 0
        if high.shape[0]>0:
            jf = np.amin(rule_out[high])
        else:
            jf = -1
    else:
        j0 = 0
        jf = -1
    return j0, jf


def rotate(ang, or_grid):
    """
    rotate a grid (NOT USED NOW!, was used for searching vorticity dipoles)
    """
    rot = []
    for j in range(len(or_grid)):
        rot.append([or_grid[j][0]*np.cos(ang)-or_grid[j][1]*np.sin(ang),or_grid[j][0]*np.sin(ang)+or_grid[j][1]*np.cos(ang)])
    return np.asarray(rot)


def avg(data,indices):
    """
    returns the average of a slice of data (over a set of given indices) (NOT USED NOW!!!)
    """
    result = 0.
    count = 0.
    for x in indices:
        if not np.isnan(data[tuple(x)]):
            result += data[tuple(x)]
            count += 1.
    if count > 0:
        result = result/count
    else:
        result = np.nan
    return result


def distance3D( x0, y0, z0, x1, y1, z1 ):
    return np.sqrt(np.square(x1-x0) + np.square(y1-y0) + np.square(z1-z0))


def process_thermals_JOBLIB( logfile, list_clusters, grid_file, Rmax, W_min, avg_dist_R, min_R, max_steps, disc_r, n_jobs_max=1, shift_amounts=[0.], up=True, cell='' ):
    """
    Tracks all the thermals whose centres are given in 'list_clusters' in the grid 'grid'.
    """
    import copy
    if type(grid_file)==str:
        file = open(grid_file, 'rb')
        data = file.read()
        stream = StringIO.StringIO(data)
        grid = pkl.load(stream)
    else:
        grid = grid_file
        grid.shift_x = 0
        grid.shift_y = 0
        grid.shift_t = 0

    n_jobs=n_jobs_max
     
    jobs=[]
    for i in range(len(list_clusters)):
        jobs.append(([list_clusters[i]], grid.extract_subgrid([list_clusters[i]], dist=1.8*Rmax), Rmax,W_min, avg_dist_R, min_R, max_steps, disc_r, shift_amounts, up, 1, cell ))
    if up:
        print( 'processing %d thermals with %d jobs...'%(len(list_clusters),np.min([n_jobs,len(jobs)])))
    else:
        print( 'processing %d downdrafts with %d jobs...'%(len(list_clusters),np.min([n_jobs,len(jobs)])))

    #(output) = Parallel(n_jobs=np.min([n_jobs,len(jobs)]),max_nbytes=None, verbose = 50)(delayed(process_clusters)(*jobs[k]) for k in range(len(jobs)))
    (output) = Parallel(n_jobs=np.min([n_jobs,len(jobs)]),max_nbytes=None, verbose = 50)(delayed(process_clusters)(*jobs[k]) for k in range(len(jobs)))#ALEJANDRA
    log=[]
    for i in range(np.min([n_jobs,len(jobs)])):
        log= log+output[i][0]
    if ~np.any(0. in shift_amounts): # if computing with shifted centres
        log_shifted=[]
        for i in range(np.min([n_jobs,len(jobs)])):
            log_shifted=log_shifted+output[i][1]
    else:
        log_shifted=None
    for line in log:
        logfile.write(line+'\n')

    if type(grid_file)==str:
        os.system('rm -r '+grid_file)
    return log, log_shifted

def process_clusters( list_clusters, grid, Rmax, W_min, avg_dist_R, min_R, max_steps, disc_r, shift_amounts=[0.], up=True, n_jobs=1, cell='' ):
    log=[]
    thermals=len(list_clusters)
    if ~np.any(0. in shift_amounts): # if computing with shifted centres
        #if shift_amount!=0:
        log_shifted=[]
    else:
        log_shifted=None
    while len(list_clusters)>0:
        #print '\nprocessing %d / %d...'%(case_number,grid.total_number_clusters)
        grid.x = grid.x_grid_wint[np.asarray(list_clusters[0])[:,0].astype(int)-grid.shift_x]
        grid.y = grid.y_grid_wint[np.asarray(list_clusters[0])[:,1].astype(int)-grid.shift_y]
        grid.z = grid.hgt_w[np.asarray(list_clusters[0])[:,2].astype(int)]
        grid.t = np.asarray(list_clusters[0])[:,3]-grid.shift_t
        grid.smooth_trajectory()
        grid.tsteps = grid.t.astype(int)	                                        # timesteps in which this thermal (cluster) exists
        valid_thermal, log00 = grid._rule_out_bad_points( up=up )                       # rule out 'bad' points and check if the thermal is valid (e.g., long enough)
        xx = np.copy(grid.x)
        yy = np.copy(grid.y)
        zz = np.copy(grid.z)
        tt = np.copy(grid.t)
        if valid_thermal:
            thermal = grid.create_thermal_grid( max_radius=Rmax, W_min=W_min, min_thermal_duration=grid.min_thermal_duration, avg_dist_R=avg_dist_R, min_R=min_R, max_steps=max_steps, disc_r=disc_r, n_jobs=n_jobs, parallel_thermals=True, up=up, cell=cell )

            for shift_amount in shift_amounts:  # (shift_amounts should be a list of values!):
                if shift_amount != 0. and len(thermal.tsteps)>0:
                    grid.x = np.copy(xx)
                    grid.y = np.copy(yy)
                    grid.z = np.copy(zz)
                    grid.t = np.copy(tt)
                    grid.tsteps = grid.t.astype(int)

                    grid.z[:np.where(grid.tsteps==thermal.tsteps[0])[0]] += shift_amount*thermal.R_thermal[0]
                    for it in range(len(thermal.tsteps)):
                        grid.z[np.where(grid.tsteps==thermal.tsteps[it])[0]] += shift_amount*thermal.R_thermal[it]
                    if thermal.tsteps[-1] < grid.tsteps[-1]:
                        grid.z[np.where(grid.tsteps==thermal.tsteps[-1])[0]+1:] += shift_amount*thermal.R_thermal[-1]
                    
                    grid.smooth_trajectory()
                    valid_thermal2, log00_shifted = grid._rule_out_bad_points( )
                    if valid_thermal2:
                        thermal_shifted = grid.create_thermal_grid( max_radius=Rmax, W_min=W_min, min_thermal_duration=grid.min_thermal_duration, avg_dist_R=avg_dist_R, min_R=min_R, max_steps=max_steps, disc_r=disc_r, n_jobs=n_jobs, shifted=shift_amount,parallel_thermals=True, up=up, cell=cell)
                        thermal_shifted._release_memory()
                        log_shifted = log_shifted + thermal_shifted.log
                    else:
                        log_shifted = log_shifted + log00_shifted
            thermal._release_memory()
            log = log + thermal.log
        else:
            log = log + log00
        thermal = None
        del list_clusters[0]
    return log, log_shifted

def distribute_clusters_temporal( grid, list_clusters, Rmax, nt_max=10 ):
    t0=[]
    nt=[]
    for case in range(len(list_clusters)):
        t0.append((np.asarray(list_clusters[case])[:,3]-grid.shift_t)[0])
        nt.append((np.asarray(list_clusters[case])[:,3]-grid.shift_t)[-1]-t0[-1]+1)
    t0=np.asarray(t0)
    nt=np.asarray(nt)
    jobclusters=[]
    longcases=np.where(nt>=nt_max)[0]
    for i in longcases: # isolate the longer thermals for individual jobs to avoid creating a very large subgrid
        jobclusters.append(np.array([i]))
    t00=0
    max_length=0
    while t00<grid.nt:
        commont0=np.where((t0==t00)*(nt<nt_max))[0]
        if len(commont0)>0:
            jobclusters.append(commont0)
            if len(jobclusters[-1])>max_length:
                max_length=len(jobclusters[-1])
        t00+=1
    return jobclusters

def distribute_clusters_spatial( grid, list_clusters, Rmax ):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    x_jobs=6
    y_jobs=6
    n_jobs=x_jobs*y_jobs
    xmin=[] 
    xmax=[]
    ymin=[]
    ymax=[]
    centers_x=[]
    centers_y=[]
    plt.figure()
    for case in range(len(list_clusters)):
        xmin.append(np.min(grid.x_grid_wint[np.asarray(list_clusters[case])[:,0].astype(int)-grid.shift_x])-Rmax) 
        xmax.append(np.max(grid.x_grid_wint[np.asarray(list_clusters[case])[:,0].astype(int)-grid.shift_x])+Rmax)
        ymin.append(np.min(grid.y_grid_wint[np.asarray(list_clusters[case])[:,1].astype(int)-grid.shift_x])-Rmax)
        ymax.append(np.max(grid.y_grid_wint[np.asarray(list_clusters[case])[:,1].astype(int)-grid.shift_x])+Rmax)
        centers_x.append(np.mean(grid.x_grid_wint[np.asarray(list_clusters[case])[:,0].astype(int)-grid.shift_x]))
        centers_y.append(np.mean(grid.y_grid_wint[np.asarray(list_clusters[case])[:,1].astype(int)-grid.shift_x]))
        plt.scatter(centers_x[-1]*1e-3,centers_y[-1]*1e-3,marker='.',s=20)
        #plt.plot(np.array([xmin[-1],xmin[-1],xmax[-1],xmax[-1],xmin[-1]])*1e-3,np.array([ymin[-1],ymax[-1],ymax[-1],ymin[-1],ymin[-1]])*1e-3)
    plt.xlim(grid.x0,grid.x0+grid.nxkm)
    plt.ylim(grid.y0,grid.y0+grid.nykm)
    x0=[]
    xf=[]
    y0=[]
    yf=[]
    jobclusters=[]
    x_ind=np.argsort(np.asarray(centers_x))
    nclustersx=len(x_ind)/x_jobs
    for i in range(x_jobs):
        if i<x_jobs-1:
            y_ind=np.argsort(np.asarray(centers_y)[x_ind[i*nclustersx:(i+1)*nclustersx]])
        else:
            y_ind=np.argsort(np.asarray(centers_y)[x_ind[i*nclustersx:]])
        nclustersy=len(y_ind)/y_jobs
        for j in range(y_jobs):
            x=[]
            y=[]
            if j<y_jobs-1:
                if i<x_jobs-1:
                    jobclusters.append(x_ind[i*nclustersx:(i+1)*nclustersx][y_ind[j*nclustersy:(j+1)*nclustersy]])
                else:
                    jobclusters.append(x_ind[i*nclustersx:][y_ind[j*nclustersy:(j+1)*nclustersy]])
            else:
                if i<x_jobs-1:
                    jobclusters.append(x_ind[i*nclustersx:(i+1)*nclustersx][y_ind[j*nclustersy:]])
                else:
                    jobclusters.append(x_ind[i*nclustersx:][y_ind[j*nclustersy:]])
            for k in range(len(jobclusters[-1])):
               for l in range(len(np.asarray(list_clusters)[jobclusters[-1][k]][:,0])):
                    x.append(grid.x_grid_wint[np.asarray(list_clusters)[jobclusters[-1][k]][:,0][l]])
                    y.append(grid.y_grid_wint[np.asarray(list_clusters)[jobclusters[-1][k]][:,1][l]])
            x0.append(np.max([(np.min(np.asarray(x))-Rmax)*1e-3,grid.x0]))
            xf.append(np.min([(np.max(np.asarray(x))+Rmax)*1e-3,grid.x0+grid.nxkm]))
            y0.append(np.max([(np.min(np.asarray(y))-Rmax)*1e-3,grid.y0]))
            yf.append(np.min([(np.max(np.asarray(y))+Rmax)*1e-3,grid.y0+grid.nykm]))
            plt.plot(np.asarray([x0[-1],x0[-1],xf[-1],xf[-1],x0[-1]]), np.asarray([y0[-1],yf[-1],yf[-1],y0[-1],y0[-1]]),lw=3)
    #plt.show()
    plt.savefig('subgrids.png')
    return jobclusters


def get_t0tf( clusters ):
    """
    search the smallest boundaries of the grid where the thermal centres in 'clusters' fit.
    Note that the indices of x, y and z are given in terms of the interpolated w field, if w has
    been interpolated!
    """
    t0 = clusters[0][0][-1]
    tf = 0
    x0 = clusters[0][0][0]
    y0 = clusters[0][0][1]
    xf = 0
    yf = 0
    zf = 0
    for i in range(len(clusters)):
        t0 = int(np.amin([t0, np.amin(np.asarray(clusters[i])[:,-1])]))
        tf = int(np.amax([tf, np.amax(np.asarray(clusters[i])[:,-1])]))
        x0 = int(np.amin([x0, np.amin(np.asarray(clusters[i])[:,0])])) 
        xf = int(np.amax([xf, np.amax(np.asarray(clusters[i])[:,0])]))
        y0 = int(np.amin([y0, np.amin(np.asarray(clusters[i])[:,1])])) 
        yf = int(np.amax([yf, np.amax(np.asarray(clusters[i])[:,1])]))
        zf = int(np.amax([zf, np.amax(np.asarray(clusters[i])[:,2])]))
    return t0, tf, x0, xf, y0, yf, zf


