#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:23:51 2024

@author: ymai0110
"""

import sys
#sys.path.insert(0,'/project/blobby3d/Blobby3D/pyblobby3d/src/pyblobby3d/')
sys.path.insert(0,'/Users/ymai0110/Documents/Blobby3D_develop/Blobby3D/pyblobby3d/src/pyblobby3d/')


import numpy as np
import matplotlib.pyplot as plt



from pyblobby3d.post_blobby3d import PostBlobby3D
from pyblobby3d.moments import SpectralModel
 
import matplotlib as mpl
from astropy.io import fits

class cmap:
    flux = 'Oranges'
    v = 'RdYlBu_r'
    vdisp = 'YlOrBr'
    residuals = 'RdYlBu_r'


def plot_cubecompare(datapath,emi_line,emi_sn_path,savepath,mode='random'):
    '''compare the convolved data cube with observed data cube
    emi_line: Ha or Oii
    mode: random or highsn; 
          random, randomly select 100 spaxels.
          highsn, select spaxels with highest SN
          
    
    
    '''
    
    np.random.seed(0)
    
    emi_sn_fits = fits.open(emi_sn_path)
    emi_sn_cut = emi_sn_fits[0].data
    
    
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    
    sample = 0
    
    if emi_line == 'Ha':
        sm = SpectralModel(
                lines=[[6562.81], [6583.1, 6548.1, 0.3333]],
                lsf_fwhm=0.846,)
        
    elif emi_line == 'Oii':
        sm = SpectralModel(
                lines=[[3727.092], [3729.875]],
                lsf_fwhm=0.846,)
        
        
    wave = post_b3d.metadata.get_axis_array('r')
    
    con_cubes_sample = post_b3d.con_cubes[sample]
    data_cubes_obs = post_b3d.data
    
    # randomly select 100 spaxels to plot
    # selec spaxels sn greater than 5
    
    spaxel_sn5_num = np.sum(emi_sn_cut>=5)
    column_n = 10
    if spaxel_sn5_num<=100:
        row_n = int(spaxel_sn5_num/column_n)
    else:
        row_n = 10
        
    fig, axs = plt.subplots(row_n, column_n,figsize=(column_n*2,row_n*2),
                            gridspec_kw={'wspace':0.2,'hspace':0.45})
    axs = axs.ravel()
    
    flag = 0
    
    if spaxel_sn5_num<=100:
    
        for i in range(data_cubes_obs.shape[0]):
            for j in range(data_cubes_obs.shape[1]):
                if emi_sn_cut[i,j] >= 5:
                    axs[flag].plot(wave,data_cubes_obs[i,j],c='b',alpha=0.5,label='data')
                    axs[flag].plot(wave,con_cubes_sample[i,j],c='k',alpha=0.5,label='convolved')
                    axs[flag].set_title('({},{})'.format(i,j))
                    if flag%column_n == 0:
                        axs[flag].set_ylabel('flux')
                    if flag >= ((row_n-1)*column_n):
                        axs[flag].set_xlabel('wavelength [$\\AA$]')
                    flag += 1
                if flag >= row_n*column_n:
                    break
            if flag>= row_n*column_n:
                break
    
    else:
        
        x = np.arange(data_cubes_obs.shape[0])
        y = np.arange(data_cubes_obs.shape[1])
        xx, yy = np.meshgrid(x,y,indexing='ij')
        xx_sn5 = xx[emi_sn_cut>=5]
        yy_sn5 = yy[emi_sn_cut>=5]
        
        
        if mode == 'random':
            index = np.random.choice(range(len(xx_sn5)),100,replace=False)
        elif mode == 'highsn':
            emi_sn_cut_sn5 = emi_sn_cut[emi_sn_cut>=5]
            
            
            idx_sort = emi_sn_cut_sn5.argsort()
            idx_sort_desc = idx_sort[::-1]
            index = idx_sort_desc[:100]
            
            
            
        xx_100 = xx_sn5[index]
        yy_100 = yy_sn5[index]
        
        for flag in range(100):
            i = xx_100[flag]
            j = yy_100[flag]
            axs[flag].plot(wave,data_cubes_obs[i,j],c='b',alpha=0.5,label='data')
            axs[flag].plot(wave,con_cubes_sample[i,j],c='k',alpha=0.5,label='convolved')
            axs[flag].set_title('({},{})'.format(i,j))
            if flag%column_n == 0:
                axs[flag].set_ylabel('flux')
            if flag >= ((row_n-1)*column_n):
                axs[flag].set_xlabel('wavelength [$\\AA$]')
        
            
    axs[0].legend()
    plt.savefig(savepath,bbox_inches='tight',dpi=300)
    plt.show()
    
    
def plot_ha_oii_cubecompare(datapath_ha,emi_sn_path_ha,datapath_oii,
                            emi_sn_path_oii,mode,ha_savepath,oii_savepath):
    
    plot_cubecompare(datapath_ha,'Ha',emi_sn_path_ha,ha_savepath,mode)
    plot_cubecompare(datapath_oii,'Oii',emi_sn_path_oii,oii_savepath,mode)
    

def oii_doublet_ratio(datapath):
    
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    
    sample = 0
    sm = SpectralModel(
            lines=[[3727.092], [3729.875]],
            lsf_fwhm=0.846,)
    precon_flux_oii_3727 = post_b3d.maps[sample, 0]
    precon_flux_oii_3729 = post_b3d.maps[sample,1]
    
    
    plt.scatter(precon_flux_oii_3727+precon_flux_oii_3729,precon_flux_oii_3729/precon_flux_oii_3727)
    plt.xlabel('oii 3727+3729')
    plt.ylabel('oii 3729/3727')
    plt.show()

def oii_doublet_ratio_compare(datapath1,datapath2,datapath3,datapath4,savepath):
    '''compare three version of oii doublet ratio
    datapath1: no constrain
    datapath2: constrain velocity profile, pa
    datapath3: constrain velocity profile and vdisp, pa
    datapath4: constrain pa only
    
    '''
    fig, ax = plt.subplots()
    
    sample = 0
    datapath = datapath1
    post_b3d_1 = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    precon_flux_oii_3727 = post_b3d_1.maps[sample, 0]
    precon_flux_oii_3729 = post_b3d_1.maps[sample,1]
    ax.scatter((precon_flux_oii_3727+precon_flux_oii_3729).ravel(),
                (precon_flux_oii_3729/precon_flux_oii_3727).ravel(),
                label='no constrain',c='b',alpha=0.3,s=8)
    
    
    datapath = datapath2
    post_b3d_2 = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    precon_flux_oii_3727 = post_b3d_2.maps[sample, 0]
    precon_flux_oii_3729 = post_b3d_2.maps[sample,1]
    ax.scatter((precon_flux_oii_3727+precon_flux_oii_3729).ravel(),
                (precon_flux_oii_3729/precon_flux_oii_3727).ravel(),
                label='constrain v pro, pa',c='r',alpha=0.3,s=8)
    
    datapath = datapath3
    post_b3d_3 = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    precon_flux_oii_3727 = post_b3d_3.maps[sample, 0]
    precon_flux_oii_3729 = post_b3d_3.maps[sample,1]
    ax.scatter((precon_flux_oii_3727+precon_flux_oii_3729).ravel(),
                (precon_flux_oii_3729/precon_flux_oii_3727).ravel(),
                label='constrain vpro vdisp pa',c='green',alpha=0.3,s=8)
    
    
    datapath = datapath4
    post_b3d_4 = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    precon_flux_oii_3727 = post_b3d_4.maps[sample, 0]
    precon_flux_oii_3729 = post_b3d_4.maps[sample,1]
    ax.scatter((precon_flux_oii_3727+precon_flux_oii_3729).ravel(),
                (precon_flux_oii_3729/precon_flux_oii_3727).ravel(),
                label='constrain pa',c='orange',alpha=0.3,s=8)
    
    
    total_flux = (precon_flux_oii_3727+precon_flux_oii_3729).ravel()
    non_zero_flux = total_flux[total_flux>0]
    
    percent = 95
    
    flux_min = np.nanmin(non_zero_flux)
    flux_max = np.nanmax((precon_flux_oii_3727+precon_flux_oii_3729).ravel())
    
    ax.set_xlabel('oii 3727+3729')
    ax.set_ylabel('oii 3729/3727')
    ax.legend()
    # ratio limit from theory 0.22-1.5, page 123, Astrophysics of Gaseous 
    # Nebulae and Active Galactic Nuclei, 2nd edition
    ax.axhline(y=0.22,xmin=flux_min,xmax=flux_max,linestyle='--')
    ax.axhline(y=1.5,xmin=flux_min,xmax=flux_max,linestyle='--')
    
    ax.set_yscale('log')
    ax.set_xscale('log') # careful of zero value in log scale..
    plt.savefig(savepath,bbox_inches='tight',dpi=300)
    plt.show()

def oii_flux_compare(datapath1,datapath2,datapath3,datapath4,oiierrpath,savepath):
    '''compare three version of oii flux
    datapath1: no constrain
    datapath2: constrain velocity profile pa
    datapath3: constrain velocity profile and vdisp pa
    datapath4: constrain pa only
    
    (3,4) plot
    oii_flux_no_constrain, oii_flux_vpro, oii_flux_vp_vd, oii_flux_pa--> (log-scale)
    v1-v2, v1-v3, v2-v3, v1-v4  --> lin-scale (can't be log as could be negative)
    (version1-version2)/fluxerr, (v1-v3)/fluxerr, (v2-v3)/fluxerr, (v1-v4)/err --> (lin-scale)
    
    todo:
        each line has the same colorbar limit
    
    
    
    '''
    oii_err = fits.open(oiierrpath)
    oii_err = oii_err[0].data
    sample = 0
    
    
    oii_1 = get_oii_sum_flux(datapath1,sample)
    oii_2 = get_oii_sum_flux(datapath2,sample)
    oii_3 = get_oii_sum_flux(datapath3,sample)
    oii_4 = get_oii_sum_flux(datapath4,sample)
    
    fig, axs = plt.subplots(3, 4,figsize=(13,10),
                            gridspec_kw={'wspace':0.4,'hspace':0.45})
    percent = 90
    
    oii_1_non_zero = oii_1[oii_1>0]
    
    clim_flux = [np.percentile(oii_1_non_zero,100-percent),
                 np.percentile(oii_1_non_zero,percent)]
    if clim_flux[0] == 0:
        clim_flux[0] = 1  # get rid of 0 in logscale...
    plot_map(axs[0,0],oii_1,clim=clim_flux,logscale=True,
             title='no constrain (v1)')
    plot_map(axs[0,1],oii_2,clim=clim_flux,logscale=True,
             title='constrain vpro pa (v2)')
    plot_map(axs[0,2],oii_3,clim=clim_flux,logscale=True,
             title='constrain vp vd pa (v3)')
    plot_map(axs[0,3],oii_4,clim=clim_flux,logscale=True,plot_cbar=True,
             title='constrain pa (v4)')
    
    
    
    diff_max_1 = np.nanpercentile(np.abs(oii_1-oii_2),percent)
    diff_max_2 = np.nanpercentile(np.abs(oii_1-oii_3),percent)
    diff_max_3 = np.nanpercentile(np.abs(oii_2-oii_3),percent)
    diff_max_4 = np.nanpercentile(np.abs(oii_1-oii_4),percent)
    diff_max = np.nanmax([diff_max_1,diff_max_2,diff_max_3,diff_max_4])
    
    clim_diff = [-diff_max,diff_max]
    plot_map(axs[1,0],oii_1-oii_2,clim=clim_diff,cmap=cmap.residuals,
             title='v1-v2')
    plot_map(axs[1,1],oii_1-oii_3,clim=clim_diff,cmap=cmap.residuals,
             title='v1-v3')
    plot_map(axs[1,2],oii_2-oii_3,clim=clim_diff,cmap=cmap.residuals,
             title='v2-v3')
    plot_map(axs[1,3],oii_1-oii_4,clim=clim_diff,cmap=cmap.residuals,
             title='v1-v4',plot_cbar=True,cbar_label='$\Delta$flux')
    
    # fractional diff 
    fdiff_max_1 = np.nanpercentile(np.abs(oii_1-oii_2)/oii_err,percent)
    fdiff_max_2 = np.nanpercentile(np.abs(oii_1-oii_3)/oii_err,percent)
    fdiff_max_3 = np.nanpercentile(np.abs(oii_2-oii_3)/oii_err,percent)
    fdiff_max_4 = np.nanpercentile(np.abs(oii_1-oii_4)/oii_err,percent)
    fdiff_max = np.nanmax([fdiff_max_1,fdiff_max_2,fdiff_max_3,fdiff_max_4])
    
    clim_fdiff = [-fdiff_max,fdiff_max]
    
    plot_map(axs[2,0],(oii_1-oii_2)/oii_err,clim=clim_fdiff,cmap=cmap.residuals,
             title='(v1-v2)/err')
    plot_map(axs[2,1],(oii_1-oii_3)/oii_err,clim=clim_fdiff,cmap=cmap.residuals,
             title='(v1-v3)/err')
    plot_map(axs[2,2],(oii_2-oii_3)/oii_err,clim=clim_fdiff,cmap=cmap.residuals,
             title='(v2-v3)/err')
    plot_map(axs[2,3],(oii_1-oii_4)/oii_err,clim=clim_fdiff,cmap=cmap.residuals,
             title='(v1-v4)/err',plot_cbar=True,
             cbar_label='$\Delta$flux/$\sigma_{flux}$')
    
    plt.savefig(savepath,bbox_inches='tight',dpi=300)
    plt.show()
    
    
    
def get_oii_sum_flux(datapath,sample=0):
    
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    precon_flux_oii_3727 = post_b3d.maps[sample, 0]
    precon_flux_oii_3729 = post_b3d.maps[sample,1]
    
    oii_sum = precon_flux_oii_3727 + precon_flux_oii_3729
    
    return oii_sum

    
    
    
    
    
    
    
    
def plot_map(ax,map_2d,clim=None,logscale=False,cmap='Oranges',
             plot_cbar=False,title=None,cbar_label='flux'):
    naxis = map_2d.shape

    if clim is not None:
        if clim[0] is None:
            clim[0] = np.nanmin(map_2d)
        if clim[1] is None:
            clim[1] = np.nanmax(map_2d)
    else:
        clim = [np.nanmin(map_2d), np.nanmax(map_2d)]

    if logscale:
        norm = mpl.colors.LogNorm(vmin=clim[0], vmax=clim[1])
    
        im0 = ax.imshow(
            map_2d,
            origin='lower',
            interpolation='nearest',
            norm=norm,
            cmap=cmap
            )
    
    else:
        norm = None

        im0 = ax.imshow(
            map_2d,
            origin='lower',
            interpolation='nearest',
            norm=norm,
            vmin=clim[0], vmax=clim[1],
            cmap=cmap
            )
    if plot_cbar:
        cb = plt.colorbar(im0, ax=ax,fraction=0.046, pad=0.04)
        cb.set_label(cbar_label,fontsize=20)
        cb.ax.tick_params(labelsize=15)
        
    ax.set_title(title,fontsize=10)
        