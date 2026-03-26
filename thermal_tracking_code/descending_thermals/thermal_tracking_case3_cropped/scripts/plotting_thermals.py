import numpy as np
import scipy as sp
import pdb
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
import pylab as pl


def tracer_mixing( mixing, z, xlabel, fname=None, xmin=None, xmax=None, mean=None, Z=None, title=None, zmin=0, zmax=14.1 ):
    #rc('font',**{'family':'FreeSans'})
    fig=plt.figure(figsize=(5.5,7))
    rc('text', usetex=False)
    ax=fig.add_axes([0.14,0.14,0.82,0.76])
    plt.plot( mixing, z, 'ko', markersize=1 )
    if xlabel!=None:
        plt.xlabel( xlabel, fontsize=20 )
    plt.ylabel( 'Height (km)', fontsize=20 )
    plt.ylim(zmin,zmax)
    if np.any(mean!=None):
        plt.plot(mean, Z, 'k', lw=3)
    if xmax!=None and xmin!=None:
        plt.xlim(xmin,xmax)
    if title!=None:
        plt.title(title, fontsize=24)
    plt.tick_params( axis='both', which='major', labelsize=16 )
    if fname!=None:
        plt.savefig(fname, bbox_inches = "tight") #ALEJANDRA: bbox)
    else:
        plt.show()

    plt.close()
    plt.clf()

def height_profile( x, z, label=None, fname=None, xmin=None, xmax=None, zero=True, zmin=0, zmax=5.5, title=None, xlabel=None, xticks=None, range_l=None, range_r=None, clr='grey', filled=False, thin=False, ylabel=True, colors=['k','b','g','r'], linestyles=['-','-','-','-'] ):
    #rc('font',**{'family':'FreeSans'})
    if thin:
        fig=plt.figure(figsize=(3.5,7))
        ax = fig.add_axes([0.1,0.14,0.81,0.76])
    else:
        fig=plt.figure(figsize=(5.5,7))
        ax=fig.add_axes([0.14,0.14,0.81,0.76])
    rc('text', usetex=False)
    #ax = fig.add_subplot(111)
    if type(x)!=list:
        x=[x]
    if type(z)!=list:
        z=[z]
    for i in range(len(x)):
        #plt.errorbar( x[i], z, xerr=error[i], label=label[i] )
        if label!=None:
            plt.plot( x[i], z[i], label=label[i], color=colors[i], ls=linestyles[i], lw=2. )
        else:
            plt.plot( x[i], z[i], color=colors[i], ls=linestyles[i], lw=2. )
        if range_l!=None and not filled:
            plt.plot(range_l[i], z[i], ls='--', color=colors[i], alpha=0.5)
        if range_r!=None and not filled:
            plt.plot(range_r[i], z[i], ls='--', color=colors[i], alpha=0.5)
        if (np.any(range_l!=None) and np.any(range_r!=None)) and filled:
            if(np.any(range_l[i]!=None) and np.any(range_r[i]!=None)):
                plt.fill_betweenx( z[i], range_l[i], range_r[i], alpha=0.3, facecolor=clr )
    if zero:
        plt.axvline(0, color='k', ls='-', lw=2.)
    if xmin!=None and xmax!=None:
        plt.xlim(xmin,xmax)
    plt.ylim(zmin,zmax)
    #rc('text', usetex=True)
    if label!=None:
       plt.legend(fontsize=20,frameon=False)
    if xlabel!=None:
        plt.xlabel(xlabel, fontsize=26)
    if title!=None:
        plt.title(title, fontsize=26)
    if ylabel:
        plt.ylabel('height (km)', fontsize=26)
    plt.tick_params( axis='both', which='major', labelsize=18 )
    if np.any(xticks!=None):
        ax.set_xticks(xticks)
    if fname==None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches = "tight") #ALEJANDRA: bbox
    plt.close()
    plt.clf()


def histogram_plot( x, bins=None, N=100, fname=None, xlabel=None, ylabel='counts', folder='', xmin=None, xmax=None, flatten=False, density=False, zero=False, mean=False, xticks=None, second_data=None, ymax=None, ylines=None, weights=None, weights_2=None, y_units=None, deltay=None, title=None, second_color='red', orientation='vertical', cumulative=False, deltax=None, log=False, histtype='bar' ):
    if flatten:
        x=flatten_array(x)
    if orientation=='vertical':
        fig=plt.figure(figsize=(6,4))
        ax=fig.add_axes( [0.14,0.16,0.75,0.75] )
    elif orientation=='horizontal':
        fig = plt.figure(figsize=(4,6))
        ax=fig.add_axes( [0.17,0.12,0.76,0.80] )
    #rc('font',**{'family':'FreeSans'})
    rc( 'text', usetex=False )
    if ymax!=None and type(ymax) is list:
        deltay=ymax[1]
        ymax=ymax[0]
    if xmax!=None and xmin!=None:
        range=( xmin,xmax )
    else:
        range=None
    if np.any(bins==None):
        n, bins1, patches = ax.hist( x, N, facecolor='gray', alpha=0.75, density=density, zorder=0, weights=weights, range=range, orientation=orientation, histtype=histtype )
    else:
        n, bins1, patches = ax.hist( x, bins, facecolor='gray', alpha=0.75, density=density, zorder=0, weights=weights, range=range, orientation=orientation, histtype=histtype )
    if np.any(second_data!=None):
        n, bins1, patches = ax.hist( second_data, bins=bins, facecolor=second_color, alpha=0.75, density=density, zorder=1, weights=weights_2, range=range, orientation=orientation, histtype=histtype )
    if log:
        pl.gca().set_xscale('log')
    if zero:
        if orientation=='vertical':
            plt.axvline(0, color='k', ls='--', lw=1, zorder=2)
        else:
            plt.axhline(0, color='k', ls='--', lw=1, zorder=2)
    if mean:
        x = np.ma.masked_array(x, np.isnan(x))
        if orientation=='vertical':
            plt.axvline(np.ma.mean(x), color='k', ls='--', lw=3, zorder=3)
            if np.any(second_data!=None):
                x2 = np.ma.masked_array(second_data,np.isnan(second_data))
                plt.axvline(np.ma.mean(x2), color='k', ls='--', lw=3, zorder=4)
        elif orientation=='horizontal':
            plt.axhline(np.ma.mean(x), color='k', ls='--', lw=3, zorder=3)
            if np.any(second_data!=None):
                x2 = np.ma.masked_array(second_data,np.isnan(second_data))
                plt.axhline(np.ma.mean(x2), color='k', ls='--', lw=3, zorder=4)

    if np.any(ylines != None):
        for yval in ylines:
            plt.axvline( yval, color='r', ls='-', lw=1, zorder=5 )
    if ymax!=None:
        if deltay!=None:
            plt.ylim(0,ymax)
            ax.yaxis.set_ticks(np.arange(0,ymax+0.00001,deltay))
        else:
            plt.ylim(0,ymax)
    plt.tick_params( axis='both', which='major', labelsize=16 )
    if np.any(xticks!=None):
        plt.xticks(xticks)
    #plt.grid()
    if title!=None:
        plt.title(title, fontsize=20)
    if np.any(y_units!=None):
        ylabel = ylabel + ' ' + y_units
    plt.rc('text', usetex=True)
    plt.ylabel(ylabel, fontsize=18)
    if xlabel!=None:
        plt.xlabel(xlabel, fontsize=18)
    plt.rc( 'text', usetex=False )
    if cumulative==True:
        ax2 = ax.twinx()
        if xmin!=None:
            xminx = np.min([np.min(x), xmin])
        else:
            xminx = np.min(x)
        if xmax!=None:
            xmaxx = np.max([np.max(x),xmax])
        else:
            xmaxx = np.max(x)
        bins= np.arange(xminx,xmaxx+0.001,(xmaxx-xminx)/10000.)
        n, bins, patches = ax2.hist( x, bins, density=True, histtype='step', cumulative=True, color='k', linewidth=2, range=(np.nanmin(x),np.nanmax(x)) )
        ax2.set_ylabel('accum. probability', fontsize=18, labelpad=-50)
        plt.ylim(0,1.)
        plt.tick_params( axis='both', which='major', labelsize=16, direction='out' )
    if xmin!=None and xmax!=None:
        if orientation=='vertical':
            plt.xlim(xmin,xmax)
        elif orientation=='horizontal':
            plt.ylim(xmin,xmax)
    if orientation=='vertical':
        xmin, xmax = ax.get_xlim()
    elif orientation=='horizontal':
        xmin, xmax = ax.get_ylim()
    if deltax!=None:
        if xmin==0:
            ax.xaxis.set_ticks(np.arange(xmin,xmax+deltax*0.5,deltax))
        else:
            ax.xaxis.set_ticks(np.concatenate((np.arange(0,xmin-deltax*0.5, -deltax)[::-1][:-1],np.arange(0,xmax+deltax*0.5, deltax))))
    if fname==None:
        plt.show()
    else:
        #plt.savefig(folder+'/histogram_'+fname+'.eps')
        #plt.savefig(folder+'/histogram_'+fname+'.pdf')
        plt.savefig(folder+'/histogram_'+fname+'.png', bbox_inches = "tight") #ALEJANDRA: bbox
    plt.close()
    plt.clf()
    return n, bins, patches


def tseries_plot( it, data, ylabel='data', fname='data', symmetric=True, ymax0=None, folder='', xmin=None, xmax=None, z=None, wthr=2., t=None, w=None, W=None, Wthr=2., xtick_dist=2 ):
    majorLocator = MultipleLocator(xtick_dist)
    #rc('font',**{'family':'FreeSans'})
    rc('text', usetex=False)
    fig = plt.figure(figsize=(9,4.5))
    ax = fig.add_axes([0.12,0.15,0.85,0.75] )
    ymax = 0.
    for i in range(len(data)):
        if z==None and t==None and w==None and W==None:
            plt.plot( it[i], data[i], lw=1.5 )
        else:
            if z != None:
                if z[i][0] < wthr:
                    clr = 'b'
                else:
                    clr = 'r'
            if t != None:
                if t[i][0] < 1000:
                    clr = 'b'
                else:
                    clr = 'r'
            if w != None:
                if np.amax(w[i][:])<6:
                    clr = 'b'
                else:
                    clr = 'r'
            if W != None:
                if np.mean(W[i][:]) < Wthr:
                    clr = 'b'
                else:
                    clr = 'r'
            plt.plot( it[i], data[i], color=clr, lw=1.5 )
        ymax=np.nanmax([ymax,np.nanmax([np.abs(data[i])])])
    plt.axhline(0, color='k', ls='--', lw=2)
    plt.axvline(0, color='k', ls='--', lw=2)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    rc('text', usetex=True)
    plt.ylabel(ylabel, fontsize=22)
    plt.xlabel('time (min)', fontsize=22)
    if symmetric:
        if ymax0 != None:
            plt.ylim(-ymax0, ymax0)
        else:
            plt.ylim(-ymax*1.15, ymax*1.15)
    if xmin!= None and xmax!=None:
        plt.xlim(xmin,xmax)
    ax.xaxis.set_major_locator( majorLocator )
    if fname!=None:
        plt.savefig( folder+'/composite_'+fname+'.eps' , bbox_inches = "tight") #ALEJANDRA: bbox)
        plt.savefig( folder+'/composite_'+fname+'.pdf' , bbox_inches = "tight") #ALEJANDRA: bbox)
    else:
        plt.show()
    plt.close()
    plt.clf()

def flatten_array( data ):
    n = 0
    for i in range(len(data)):
        n = n + len(data[i])
    out = np.zeros([n])
    k=0
    for i in range(len(data)):
        for j in range(len(data[i])):
            out[k]=data[i][j]
            k = k+1
    return out

def corr_plot( datax, datay, xlabel='datax', ylabel='datay', fname=None, folder='.', verbose=False, xmin=None, xmax=None, ymin=None, ymax=None, flatten=True, xy=None, labelpos1=None, labelpos2=(0.7,0.04), xsym=False, ysym=False, bothreg=True, mark='.', linewidth=2,plot_regr=True, title=None, markersize=1, grid=False, label_regr=False):
    try:
        if flatten:
            datax_flat = flatten_array(datax)
            datay_flat = flatten_array(datay)
        else:
            datax_flat = datax
            datay_flat = datay
        if hasattr(datax_flat, 'mask'):
            datax_flat = datax_flat.data
        if hasattr(datay_flat, 'mask'):
            datay_flat = datay_flat.data
        x=[]
        y=[]
        for i in range(len(datax_flat)):
            if (~np.isnan(datax_flat[i])) and (not np.isnan(datay_flat[i])):
                x.append(datax_flat[i])
                y.append(datay_flat[i])
        x=np.asarray(x)
        y=np.asarray(y)
        try:
            m0,b0,R0,p0,stderr0 = sp.stats.linregress(x,y)
            m1,b1,R1,p1,stderr1 = sp.stats.linregress(y,x)
            if verbose:
                print (xlabel, ylabel, 'slope='+str(m0), 'intercept='+str(b0))
                print (ylabel, xlabel, 'slope='+str(m1), 'intercept='+str(b1))
        except:
            m0,b0,R0,p0,stderr0 = np.nan, np.nan, np.nan, np.nan, np.nan
            m1,b1,R1,p1,stderr1 = np.nan, np.nan, np.nan, np.nan, np.nan
        if verbose:
            print (xlabel, ylabel, 'slope='+str(m0), 'intercept='+str(b0))
            print (ylabel, xlabel, 'slope='+str(m1), 'intercept='+str(b1))
        xarr = np.arange(int(np.nanmin(datax_flat))-1, int(np.nanmax(datax_flat))+2)
        yarr = np.arange(int(np.nanmin(datay_flat))-1, int(np.nanmax(datay_flat))+2)
        
        #rc('font',**{'family':'FreeSans'})
        rc('text', usetex=False)
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_axes([0.16,0.12,0.8,0.8] )
        plt.scatter( datax_flat, datay_flat, c='k', marker='.', s=markersize )
        #for i in range(len(datax)):
        #    plt.plot( datax[i], datay[i], mark, markersize=3, markeredgecolor=None )
        if plot_regr:
            plt.plot(xarr, m0*xarr+b0, '-k', lw=linewidth )
            if bothreg:
                plt.plot(m1*yarr+b1, yarr, '-k', lw=linewidth )
                #plt.annotate('m$_{2}$=%.3f'%(1./m1), xy=labelpos2, xycoords='axes fraction', fontsize=22)
            #if title==None:
            #    plt.title('R=%.2f'%(R0), fontsize=22)
        if title!=None:
            plt.title(title, fontsize=20)
        if label_regr:
            if labelpos1==None:
                if m0>0:
                    labelpos1=(0.5,0.1)
                else:
                    labelpos1=(0.1,0.1)
            plt.annotate('m=%.2f,  r=%.2f'%(m0,R0), xy=labelpos1, xycoords='axes fraction', fontsize=22)
        # ****************
        #plt.plot(xarr, 0.15*xarr, '-r', lw=2 )
        #plt.plot(xarr, 1.2*xarr, '-r', lw=2 )
        #****************
        if grid:
            plt.grid()
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        if xmin==None:
            if xsym:
                xm = np.nanmax(np.abs(datax_flat))*1.3
                plt.xlim(-xm, xm)
            else:
                xmn = np.nanmin(datax_flat)
                xmx = np.nanmax(datax_flat)
                try:
                    plt.xlim( xmn-0.15*(xmx-xmn), xmx+0.15*(xmx-xmn) )
                except:
                    pass
        else:
            plt.xlim(xmin, xmax)
        if ymin==None:
            if ysym:
                ym = np.nanmax( np.abs(datay_flat) )*1.3
                plt.ylim( -ym,ym )
            else:
                ymn = np.nanmin(datay_flat)
                ymx = np.nanmax(datay_flat)
                plt.ylim(ymn-0.15*(ymx-ymn), ymx+0.15*(ymx-ymn))
        else:
            plt.ylim(ymin, ymax)
        if xy != None:
            plt.plot(xy[0],xy[1], 'o', markerfacecolor='none', markeredgecolor='red', ms=20, markeredgewidth=3)
        #rc('text', usetex=True)
        plt.ylabel(ylabel, fontsize=22)
        plt.xlabel(xlabel, fontsize=22)
        if fname!=None:
            #plt.savefig( folder+'/corr_'+fname+'.eps' )
            plt.savefig( folder+'/corr_'+fname+'.png' , bbox_inches = "tight") #ALEJANDRA: bbox)
        else:
            plt.show()
        plt.close()
        plt.clf()
    except:
        print('could not make correlation plot',title)

def composite_plot( t, data, error, ylabel='ydata', clr='r', fname='data', folder='', ymax0=None, ymin0=None, xmin=None, xmax=None, xtick_dist=2, pctl=None, zero_x=True, zero_y=True, grid=False, title=None ):
    majorLocator = MultipleLocator(xtick_dist)
    #rc('font',**{'family':'FreeSans'})
    rc('text', usetex=False)
    fig = plt.figure(figsize=(6,3.1))
    ax = fig.add_axes([0.165,0.18,0.79,0.72] )
    #ax = fig.add_subplot(111)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    #rc('text', usetex=True)
    plt.ylabel( ylabel, fontsize=20 )
    plt.plot( t, data, lw=2, color='k' )
    if zero_y:
        plt.axhline(0, color='grey', ls=':', lw=2 )
    if zero_x:
        plt.axvline(0, color='grey', ls=':', lw=2 )
    if not np.any(error==None):
        if pctl==None:
            plt.plot( t, data+error, ls='--', lw=2, color='k' )
            plt.plot( t, data-error, ls='--', lw=2, color='k' )
            #plt.fill_between( t, data-error, data+error, alpha=0.3, facecolor=clr )
        else:
            plt.plot( t, pctl[0], ls='--', lw=2, color='k' )
            plt.plot( t, pctl[1], ls='--', lw=2, color='k' ) 
            #plt.fill_between( t, pctl[0], pctl[1], alpha=0.3, facecolor=clr )
    plt.xlabel('time (min)', fontsize=16)
    if ymax0!=None:
        if ymin0==None:
            plt.ylim(-ymax0, ymax0)
        else:
            plt.ylim(ymin0,ymax0)
    #else:
        #plt.ylim(-1.15*np.amax(np.abs(data)+error), 1.15*np.amax(np.abs(data)+error))
    if grid:
        plt.grid()
    if title!=None:
        plt.title(title, fontsize=22)
    if xmin!= None and xmax!=None:
        plt.xlim(xmin, xmax)
    #plt.tick_params( axis='both', which='major', labelsize=16 )
    ax.xaxis.set_major_locator( majorLocator )
    plt.savefig( folder+'/mean_composite_'+fname+'.png' , bbox_inches = "tight") #ALEJANDRA: bbox)
    #plt.savefig( folder+'/mean_composite_'+fname+'.pdf' )
    plt.close()
    plt.clf()


def normalize( array ):
    return (array - np.mean(array))/np.std(array)

