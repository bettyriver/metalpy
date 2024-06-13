#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:23:58 2024

@author: ymai0110
"""

import sys
sys.path.insert(0,'/project/blobby3d/Blobby3D/pyblobby3d/src/pyblobby3d/')
#from pyblobby3d.b3dcomp import setup_comparison_maps,map_limits,colorbar
#from pyblobby3d.b3dplot import plot_map,plot_colorbar,cmap
from b3dcomp import setup_comparison_maps,map_limits,colorbar
from b3dplot import plot_map,plot_colorbar,cmap

import numpy as np
import matplotlib.pyplot as plt

import dnest4 as dn4
#from pyblobby3d import PostBlobby3D
#from pyblobby3d import SpectralModel
from post_blobby3d import PostBlobby3D
from moments import SpectralModel
 
import matplotlib as mpl
from astropy.io import fits

def make_compmap_oiierr(datapath,figpath=None,flux_scale_factor=1,fwhm=None,plot_nii=False,
                 set_title=None,vdispHist=False,vdispSFR=False,vdispSFRpath=None,
                 vdispcolumn=None,sfrcolumn=None,oii_err_path=None,set_ticks=False):
    '''use ha err from GIST emission line output, also update subtract median of velocity field.'''
    
    
    
    oii_err_gist = fits.open(oii_err_path)
    oii_err_gist = oii_err_gist[0].data
    
    
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=2)
    
    # choose a sample
    sample = 0
    sm = SpectralModel(
            lines=[[3727.092], [3729.875]],
            lsf_fwhm=0.846,)
    wave = post_b3d.metadata.get_axis_array('r')
    fit_con, fit_var_con = sm.fit_cube(wave, post_b3d.con_cubes[sample], post_b3d.var)
    
    fit_data, fit_var_data = sm.fit_cube(wave, post_b3d.data, post_b3d.var)
    
    
    
    # preconvolved
    # precon flux, velocity map, velocity dispersion map
    precon_flux_oii_3727 = post_b3d.maps[sample, 0]*flux_scale_factor
    ##precon_flux = post_b3d.precon_cubes[sample].sum(axis=2)
    
    precon_flux_oii_3729 = post_b3d.maps[sample,1]*flux_scale_factor
    
    precon_oii_sum = precon_flux_oii_3727 + precon_flux_oii_3729
    
    precon_vel = post_b3d.maps[sample, 2] - np.nanmedian(post_b3d.maps[sample,2])
    precon_vdisp = post_b3d.maps[sample, 3]
    
    
    
    # convolved
    
    
    con_flux_oii_3727 = fit_con[0]*flux_scale_factor
    ##con_flux = post_b3d.con_cubes[sample].sum(axis=2)
    con_flux_oii_3729 = fit_con[1]*flux_scale_factor
    con_oii_sum = con_flux_oii_3727 + con_flux_oii_3729
    con_vel = fit_con[2] - np.nanmedian(post_b3d.maps[sample,2])
    con_vdisp = fit_con[3]
    
    ## flux_err
    #con_flux_ha_err = np.sqrt(fit_var_con[0]*(flux_scale_factor**2))
    #con_flux_nii_err = np.sqrt(fit_var_con[1]*(flux_scale_factor**2))
    
    # data 
    
    data_flux_oii_3727 = fit_data[0]*flux_scale_factor
    ##data_flux = post_b3d.data.sum(axis=2)
    data_flux_oii_3729 = fit_data[1]*flux_scale_factor
    data_oii_sum = data_flux_oii_3727 + data_flux_oii_3729
    data_vel = fit_data[2] - np.nanmedian(post_b3d.maps[sample,2])
    data_vdisp = fit_data[3]
    
    
    
    # mask all
    mask = np.isnan(con_vel)
    precon_flux_oii_3727[mask] = np.nan
    precon_flux_oii_3729[mask] = np.nan
    precon_oii_sum[mask] = np.nan
    precon_vel[mask] = np.nan
    precon_vdisp[mask] = np.nan
    con_flux_oii_3727[mask] = np.nan
    con_flux_oii_3729[mask] = np.nan
    con_oii_sum[mask] = np.nan
    con_vel[mask] = np.nan
    con_vdisp[mask] = np.nan
    data_flux_oii_3727[mask] = np.nan
    data_flux_oii_3729[mask] = np.nan
    data_oii_sum[mask] = np.nan
    data_vel[mask] = np.nan
    data_vdisp[mask] = np.nan
    
    # residual
    res_flux_oii_3727 = (data_flux_oii_3727 - con_flux_oii_3727)/oii_err_gist
    res_vel = data_vel - con_vel
    res_vdisp = data_vdisp - con_vdisp
    res_flux_oii_3729 = (data_flux_oii_3729 - con_flux_oii_3729)/oii_err_gist
    res_oii_sum = (data_oii_sum - con_oii_sum)/oii_err_gist
    
    extra_row = 0
    if vdispHist==True or vdispSFR==True:
        extra_row = 1
        if vdispSFR==True:
            import pandas as pd
            vdisp_sfr_csv = pd.read_csv(vdispSFRpath)
            vdisp_all = vdisp_sfr_csv[vdispcolumn].to_numpy()
            sfr_all = vdisp_sfr_csv[sfrcolumn].to_numpy()
            id_all = vdisp_sfr_csv['MAGPIID'].to_numpy()
            magpiid_str = datapath[-11:-1]
            magpiid_int = int(magpiid_str)
            
            
            index_0 = np.where(magpiid_int==id_all)[0]
            if len(index_0)==0:
                data_avail = False
                
            else:
                data_avail = True
                index = index_0[0]
                vdisp_this = vdisp_all[index]
                sfr_this = sfr_all[index] 
    
    if plot_nii == True:
        fig,ax = setup_comparison_maps(comp_shape=(5+extra_row,3), map_shape=precon_vel.shape, figsize=(18,18))
    else :
        fig,ax = setup_comparison_maps(comp_shape=(3+extra_row,3), map_shape=precon_vel.shape, figsize=(15,8.5))
    
    if set_title is not None:
        fig.suptitle(set_title,fontsize=16)
    
    flux_max = np.nanmax(con_flux_oii_3727)+10
    flux_min = np.nanmin(con_flux_oii_3727)
    vel_max = np.nanmax(con_vel)
    vel_min = np.nanmin(con_vel)
    if np.abs(vel_min)>vel_max:
        vel_max = np.abs(vel_min)
    else:
        vel_min = -vel_max
    
    vdisp_max = np.nanmax(con_vdisp)+20
    vdisp_min = np.nanmin(con_vdisp)
    oii_sum_max = np.nanmax(con_oii_sum)
    oii_sum_min = np.nanmin(con_oii_sum)
    
    res_flux_max = np.nanmax(res_flux_oii_3727)
    res_flux_min = np.nanmin(res_flux_oii_3727)
    if np.abs(res_flux_min)>res_flux_max:
        res_flux_max = np.abs(res_flux_min)
    else:
        res_flux_min = -res_flux_max
    
    res_vel_max = 15
    res_vel_min = -15
    
    res_vdisp_max = 20
    res_vdisp_min = -20
    
    pct = 90
    
    flux_oii_3727_lim = map_limits(data=con_flux_oii_3727,pct=pct)
    flux_oii_3729_lim = map_limits(data=con_flux_oii_3729,pct=pct)
    oii_sum_lim = map_limits(data=con_oii_sum,pct=pct)
    vel_lim = map_limits(data=con_vel,pct=pct,absolute=True)
    vdisp_lim = map_limits(data=data_vdisp,pct=pct)
    
    
    
    res_flux_oii_3727_lim = map_limits(data=res_flux_oii_3727,pct=pct,absolute=True)
    res_flux_oii_3729_lim = map_limits(data=res_flux_oii_3729,pct=pct,absolute=True)
    res_oii_sum_lim = map_limits(data=res_oii_sum,pct=pct,absolute=True)
    res_vel_lim = map_limits(data=res_vel,pct=pct,absolute=True)
    res_vdisp_lim = map_limits(data=res_vdisp,pct=pct,absolute=True)
    
    
    
    ii = 0
    
    if plot_nii == True:
        ii = 2
    
    fontsize_set = 18
    
    plot_map(ax[0][0],precon_flux_oii_3727,clim=flux_oii_3727_lim,logscale=True,
             title='Pre-convolved',cmap=cmap.flux,
             title_fontsize=fontsize_set,label_fontsize=fontsize_set)
    plot_map(ax[0][1],con_flux_oii_3727,clim=flux_oii_3727_lim,logscale=True,
             title='convolved',cmap=cmap.flux,title_fontsize=fontsize_set,label_fontsize=fontsize_set)
    plot_map(ax[0][2],data_flux_oii_3727,clim=flux_oii_3727_lim,logscale=True,
             cbar_label='log(Flux(OII3727))',title='data',
             cmap=cmap.flux,title_fontsize=fontsize_set,label_fontsize=fontsize_set)
    
    if plot_nii == True:
        plot_map(ax[1][0],precon_flux_oii_3729,clim=flux_oii_3729_lim,logscale=True,ylabel='Flux(OII3729)',cmap=cmap.flux,label_fontsize=fontsize_set)
        plot_map(ax[1][1],con_flux_oii_3729,clim=flux_oii_3729_lim,logscale=True,cmap=cmap.flux,label_fontsize=fontsize_set)
        plot_map(ax[1][2],data_flux_oii_3729,clim=flux_oii_3729_lim,logscale=True,cbar_label='log(Flux(OII3729))',cmap=cmap.flux,label_fontsize=fontsize_set)
        plot_map(ax[1][4],res_flux_oii_3729,cmap=cmap.residuals,label_fontsize=fontsize_set)
        
        plot_map(ax[2][0],precon_oii_sum,clim=oii_sum_lim,cmap=cmap.flux,ylabel='Flux(OIIsum)',label_fontsize=fontsize_set)
        plot_map(ax[2][1],con_oii_sum,clim=oii_sum_lim,cmap=cmap.flux,label_fontsize=fontsize_set)
        plot_map(ax[2][2],data_oii_sum,clim=oii_sum_lim,cmap=cmap.flux,label_fontsize=fontsize_set)
        plot_map(ax[2][4],res_oii_sum,cmap=cmap.residuals,label_fontsize=fontsize_set)
    
    plot_map(ax[ii+1][0],precon_vel,clim=vel_lim,cmap=cmap.v,label_fontsize=fontsize_set)
    plot_map(ax[ii+1][1],con_vel,clim=vel_lim,cmap=cmap.v,label_fontsize=fontsize_set)
    plot_map(ax[ii+1][2],data_vel,clim=vel_lim,cmap=cmap.v,cbar_label='v(km/s)',label_fontsize=fontsize_set)
    plot_map(ax[ii+2][0],precon_vdisp,clim=vdisp_lim,cmap=cmap.vdisp,label_fontsize=fontsize_set)
    plot_map(ax[ii+2][1],con_vdisp,clim=vdisp_lim,cmap=cmap.vdisp,label_fontsize=fontsize_set)
    plot_map(ax[ii+2][2],data_vdisp,clim=vdisp_lim,cmap=cmap.vdisp,cbar_label='$\sigma_{\mathrm{gas}}$(km/s)',label_fontsize=fontsize_set)
    
    plot_map(ax[0][4], res_flux_oii_3727,clim=res_flux_oii_3727_lim,
             cbar_label='$\Delta$Flux/$\sigma_{F(OII)}$',
             cmap=cmap.residuals,title='residual',title_fontsize=fontsize_set,label_fontsize=fontsize_set)
    if plot_nii == True:
        plot_map(ax[1][4], res_flux_oii_3729,clim=res_flux_oii_3729_lim,cmap=cmap.residuals,label_fontsize=fontsize_set)
        plot_map(ax[2][4], res_oii_sum,clim=res_oii_sum_lim,cmap=cmap.residuals,label_fontsize=fontsize_set)
    
    plot_map(ax[ii+1][4], res_vel,clim=res_vel_lim,cbar_label='$\Delta$v(km/s)',cmap=cmap.residuals,label_fontsize=fontsize_set)
    plot_map(ax[ii+2][4], res_vdisp,clim=res_vdisp_lim,cbar_label='$\Delta \sigma_{\mathrm{gas}}$(km/s)',cmap=cmap.residuals,label_fontsize=fontsize_set)
    
    mpb0 = mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=flux_oii_3727_lim[0],vmax=flux_oii_3727_lim[1]),cmap=cmap.flux)
    mpb1 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vel_lim[0],vmax=vel_lim[1]),cmap=cmap.v)
    mpb2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vdisp_lim[0],vmax=vdisp_lim[1]),cmap=cmap.vdisp)
    
    colorbar(mpb0,cax=ax[0][3],clim=flux_oii_3727_lim,label='Flux(10$^{-20}$ erg/s/cm$^{-2}$)',label_fontsize=fontsize_set)
    colorbar(mpb1,cax=ax[ii+1][3],clim=vel_lim,label='$v$(km/s)',label_fontsize=fontsize_set)
    colorbar(mpb2,cax=ax[ii+2][3],clim=vdisp_lim,label='$\sigma_{\mathrm{gas}}$(km/s)',label_fontsize=fontsize_set)
    
    if plot_nii == True:
        mpb3 = mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=flux_oii_3729_lim[0],vmax=flux_oii_3729_lim[1]),cmap=cmap.flux)
        mpb4 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=oii_sum_lim[0],vmax=oii_sum_lim[1]),cmap=cmap.flux)
        colorbar(mpb3,cax=ax[1][3],clim=flux_oii_3729_lim,label='Flux(OII3729)',label_fontsize=fontsize_set)
        colorbar(mpb4,cax=ax[2][3],clim=oii_sum_lim)
    
    # residual colorbar
    
    res_mpb0 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_flux_oii_3727_lim[0],vmax=res_flux_oii_3727_lim[1]),cmap=cmap.residuals)
    res_mpb1 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_vel_lim[0],vmax=res_vel_lim[1]),cmap=cmap.residuals)
    res_mpb2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_vdisp_lim[0],vmax=res_vdisp_lim[1]),cmap=cmap.residuals)
    res_mpb3 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_flux_oii_3729_lim[0],vmax=res_flux_oii_3729_lim[1]),cmap=cmap.residuals)
    res_mpb4 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_oii_sum_lim[0],vmax=res_oii_sum_lim[1]),cmap=cmap.residuals)
    
    colorbar(res_mpb0,cax=ax[0][5],clim=res_flux_oii_3727_lim,label='$\Delta F(oii3727)/\sigma_{F(oiisum)}$',label_fontsize=fontsize_set)
    colorbar(res_mpb1,cax=ax[ii+1][5],clim=res_vel_lim,label='$\Delta v$(km/s)',label_fontsize=fontsize_set)
    colorbar(res_mpb2,cax=ax[ii+2][5],clim=res_vdisp_lim,label='$\Delta \sigma_{\mathrm{gas}}$(km/s)',label_fontsize=fontsize_set)
    if plot_nii == True:
        colorbar(res_mpb3,cax=ax[1][5],clim=res_flux_oii_3729_lim,label='$\Delta F(oii3729)/\sigma_{F(oiisum)}$',label_fontsize=fontsize_set)
        colorbar(res_mpb4,cax=ax[2][5],clim=res_oii_sum_lim,label='$\Delta F(oiisum)/\sigma_{F(oiisum)}$')
    
    if plot_nii==True:
        extra_row_index = 5
    else:
        extra_row_index = 3
    
    if vdispHist==True:
        ax[extra_row_index][0].hist(np.exp(post_b3d.global_param['VDISP0']),bins=20, color='c', edgecolor='k', alpha=0.65)
        ax[extra_row_index][0].set_xlabel('vdisp [km/s]')
        ax[extra_row_index][0].set_ylabel('N')
        ax[extra_row_index][0].axvline(np.exp(post_b3d.global_param['VDISP0']).mean(), color='k', linestyle='dashed', linewidth=1)
        
    if vdispSFR==True:
        ax[extra_row_index][1].scatter(sfr_all,vdisp_all)
        ax[extra_row_index][1].set_xscale('log')
        ax[extra_row_index][1].set_xlabel('SFR [M* yr$^{-1}$]')
        ax[extra_row_index][1].set_ylabel('vdisp [km/s]')
        if data_avail==True:
            ax[extra_row_index][1].scatter(sfr_this,vdisp_this,c='r',marker='*') 
    
    if set_ticks==True:
        # xticks
        #x_shape = precon_flux_ha.shape[1]
        naxis = precon_flux_oii_3727.shape
        x_dist = naxis[1]*0.2/2
        y_dist = naxis[0]*0.2/2
        ## todo
        set_xticks(ax[ii+2][0],xticks='lower',x_dist=x_dist,naxis=naxis)
        set_xticks(ax[ii+2][1],xticks='all',x_dist=x_dist,naxis=naxis)
        set_xticks(ax[ii+2][2],xticks='upper',x_dist=x_dist,naxis=naxis)
        set_xticks(ax[ii+2][4],xticks='all',x_dist=x_dist,naxis=naxis)
        
        set_yticks(ax[0][0],yticks='upper',y_dist=y_dist,naxis=naxis)
        set_yticks(ax[1][0],yticks='all',y_dist=y_dist,naxis=naxis)
        set_yticks(ax[2][0],yticks='lower',y_dist=y_dist,naxis=naxis)
    
    
    
    if fwhm is not None:
    
        circle = plt.Circle(
                (1.1*fwhm, 1.1*fwhm),
                radius=fwhm,
                fill=False, edgecolor='r', linewidth=1.0)
        ax[0][2].add_artist(circle)
    
    if figpath == None:
        figpath = datapath
    
    if set_title == None:
        plt.savefig(figpath+"compmap.pdf",dpi=300,bbox_inches='tight')
    else:
        plt.savefig(figpath+"compmap_"+set_title+".pdf",dpi=300,bbox_inches='tight')



def set_xticks(ax,xticks,x_dist,naxis,tick_fontsize='large'):
    ## todo
    if xticks == 'all':
        
        ax.set_xticks([-0.5, (naxis[1]-1.0)/2.0, naxis[1]-0.5])
        xtick_values = [-round(x_dist, 1), 0.0, round(x_dist, 1)]
        ax.set_xticklabels(xtick_values, fontsize=tick_fontsize)
    elif xticks == 'upper':
        
        ax.set_xticks([(naxis[1]-1.0)/2.0, naxis[1]-0.5])
        xtick_values = [0.0, round(x_dist, 1)]
        ax.set_xticklabels(xtick_values, fontsize=tick_fontsize)
    elif xticks == 'lower':
    
        ax.set_xticks([-0.5, (naxis[1]-1.0)/2.0])
        xtick_values = [-round(x_dist, 1), 0.0]
        ax.set_xticklabels(xtick_values, fontsize=tick_fontsize)

def set_yticks(ax,yticks,y_dist,naxis,tick_fontsize='large'):
    if yticks == 'all':
        
        ax.set_yticks([-0.5, (naxis[0]-1.0)/2.0, naxis[0]-0.5])
        ytick_values = [-round(y_dist, 1), 0.0, round(y_dist, 1)]
        ax.set_yticklabels(ytick_values, fontsize=tick_fontsize)
    elif yticks == 'upper':
        
        ax.set_yticks([(naxis[0]-1.0)/2.0, naxis[0]-0.5])
        ytick_values = [0.0, round(y_dist, 1)]
        ax.set_yticklabels(ytick_values, fontsize=tick_fontsize)
    elif yticks == 'lower':
        
        ax.set_yticks([-0.5, (naxis[0]-1.0)/2.0])
        ytick_values = [-round(y_dist, 1), 0.0]
        ax.set_yticklabels(ytick_values, fontsize=tick_fontsize)
        

def make_compmap_hberr(datapath,figpath=None,flux_scale_factor=1,fwhm=None,plot_nii=False,
                 set_title=None,vdispHist=False,vdispSFR=False,vdispSFRpath=None,
                 vdispcolumn=None,sfrcolumn=None,hb_err_path=None,set_ticks=False):
    '''use hb err from GIST emission line output, also update subtract median of velocity field.'''
    
    
    
    hb_err_gist = fits.open(hb_err_path)
    hb_err_gist = hb_err_gist[0].data
    
    
    post_b3d = PostBlobby3D(
            samples_path=datapath+'posterior_sample.txt',
            data_path=datapath+'data.txt',
            var_path=datapath+'var.txt',
            metadata_path=datapath+'metadata.txt',
            nlines=1)
    
    # choose a sample
    sample = 0
    sm = SpectralModel(
            lines=[[4862.683]],
            lsf_fwhm=0.846,)
    wave = post_b3d.metadata.get_axis_array('r')
    fit_con, fit_var_con = sm.fit_cube(wave, post_b3d.con_cubes[sample], post_b3d.var)
    
    fit_data, fit_var_data = sm.fit_cube(wave, post_b3d.data, post_b3d.var)
    
    
    
    # preconvolved
    # precon flux, velocity map, velocity dispersion map
    precon_flux_hb = post_b3d.maps[sample, 0]*flux_scale_factor
    
    precon_vel = post_b3d.maps[sample, 1] - np.nanmedian(post_b3d.maps[sample,1])
    precon_vdisp = post_b3d.maps[sample, 2]
    
    
    
    # convolved
    
    
    con_flux_hb = fit_con[0]*flux_scale_factor
    
    con_vel = fit_con[1] - np.nanmedian(post_b3d.maps[sample,1])
    con_vdisp = fit_con[2]
    
    ## flux_err
    #con_flux_ha_err = np.sqrt(fit_var_con[0]*(flux_scale_factor**2))
    #con_flux_nii_err = np.sqrt(fit_var_con[1]*(flux_scale_factor**2))
    
    # data 
    
    data_flux_hb = fit_data[0]*flux_scale_factor
    
    data_vel = fit_data[1] - np.nanmedian(post_b3d.maps[sample,1])
    data_vdisp = fit_data[2]
    
    
    
    # mask all
    mask = np.isnan(con_vel)
    precon_flux_hb[mask] = np.nan
    precon_vel[mask] = np.nan
    precon_vdisp[mask] = np.nan
    
    con_flux_hb[mask] = np.nan
    con_vel[mask] = np.nan
    con_vdisp[mask] = np.nan
    
    data_flux_hb[mask] = np.nan
    data_vel[mask] = np.nan
    data_vdisp[mask] = np.nan
    
    # residual
    res_flux_hb = (data_flux_hb - con_flux_hb)/hb_err_gist
    res_vel = data_vel - con_vel
    res_vdisp = data_vdisp - con_vdisp
    
    
    extra_row = 0
    if vdispHist==True or vdispSFR==True:
        extra_row = 1
        if vdispSFR==True:
            import pandas as pd
            vdisp_sfr_csv = pd.read_csv(vdispSFRpath)
            vdisp_all = vdisp_sfr_csv[vdispcolumn].to_numpy()
            sfr_all = vdisp_sfr_csv[sfrcolumn].to_numpy()
            id_all = vdisp_sfr_csv['MAGPIID'].to_numpy()
            magpiid_str = datapath[-11:-1]
            magpiid_int = int(magpiid_str)
            
            
            index_0 = np.where(magpiid_int==id_all)[0]
            if len(index_0)==0:
                data_avail = False
                
            else:
                data_avail = True
                index = index_0[0]
                vdisp_this = vdisp_all[index]
                sfr_this = sfr_all[index] 
    
    if plot_nii == True:
        fig,ax = setup_comparison_maps(comp_shape=(5+extra_row,3), map_shape=precon_vel.shape, figsize=(18,18))
    else :
        fig,ax = setup_comparison_maps(comp_shape=(3+extra_row,3), map_shape=precon_vel.shape, figsize=(15,8.5))
    
    if set_title is not None:
        fig.suptitle(set_title,fontsize=16)
    
    flux_max = np.nanmax(con_flux_hb)+10
    flux_min = np.nanmin(con_flux_hb)
    vel_max = np.nanmax(con_vel)
    vel_min = np.nanmin(con_vel)
    if np.abs(vel_min)>vel_max:
        vel_max = np.abs(vel_min)
    else:
        vel_min = -vel_max
    
    vdisp_max = np.nanmax(con_vdisp)+20
    vdisp_min = np.nanmin(con_vdisp)
    #oii_sum_max = np.nanmax(con_oii_sum)
    #oii_sum_min = np.nanmin(con_oii_sum)
    
    res_flux_max = np.nanmax(res_flux_hb)
    res_flux_min = np.nanmin(res_flux_hb)
    if np.abs(res_flux_min)>res_flux_max:
        res_flux_max = np.abs(res_flux_min)
    else:
        res_flux_min = -res_flux_max
    
    res_vel_max = 15
    res_vel_min = -15
    
    res_vdisp_max = 20
    res_vdisp_min = -20
    
    pct = 90
    
    flux_hb_lim = map_limits(data=con_flux_hb,pct=pct)
    #flux_oii_3729_lim = map_limits(data=con_flux_oii_3729,pct=pct)
    #oii_sum_lim = map_limits(data=con_oii_sum,pct=pct)
    vel_lim = map_limits(data=con_vel,pct=pct,absolute=True)
    vdisp_lim = map_limits(data=data_vdisp,pct=pct)
    
    
    
    res_flux_hb_lim = map_limits(data=res_flux_hb,pct=pct,absolute=True)
    #res_flux_oii_3729_lim = map_limits(data=res_flux_oii_3729,pct=pct,absolute=True)
    #res_oii_sum_lim = map_limits(data=res_oii_sum,pct=pct,absolute=True)
    res_vel_lim = map_limits(data=res_vel,pct=pct,absolute=True)
    res_vdisp_lim = map_limits(data=res_vdisp,pct=pct,absolute=True)
    
    
    
    ii = 0
    
    if plot_nii == True:
        ii = 2
    
    fontsize_set = 18
    
    plot_map(ax[0][0],precon_flux_hb,clim=flux_hb_lim,logscale=True,
             title='Pre-convolved',cmap=cmap.flux,
             title_fontsize=fontsize_set,label_fontsize=fontsize_set)
    plot_map(ax[0][1],con_flux_hb,clim=flux_hb_lim,logscale=True,
             title='convolved',cmap=cmap.flux,title_fontsize=fontsize_set,label_fontsize=fontsize_set)
    plot_map(ax[0][2],data_flux_hb,clim=flux_hb_lim,logscale=True,
             cbar_label='log(Flux(Hbeta))',title='data',
             cmap=cmap.flux,title_fontsize=fontsize_set,label_fontsize=fontsize_set)
    
    
    plot_map(ax[ii+1][0],precon_vel,clim=vel_lim,cmap=cmap.v,label_fontsize=fontsize_set)
    plot_map(ax[ii+1][1],con_vel,clim=vel_lim,cmap=cmap.v,label_fontsize=fontsize_set)
    plot_map(ax[ii+1][2],data_vel,clim=vel_lim,cmap=cmap.v,cbar_label='v(km/s)',label_fontsize=fontsize_set)
    plot_map(ax[ii+2][0],precon_vdisp,clim=vdisp_lim,cmap=cmap.vdisp,label_fontsize=fontsize_set)
    plot_map(ax[ii+2][1],con_vdisp,clim=vdisp_lim,cmap=cmap.vdisp,label_fontsize=fontsize_set)
    plot_map(ax[ii+2][2],data_vdisp,clim=vdisp_lim,cmap=cmap.vdisp,cbar_label='$\sigma_{\mathrm{gas}}$(km/s)',label_fontsize=fontsize_set)
    
    plot_map(ax[0][4], res_flux_hb,clim=res_flux_hb_lim,
             cbar_label='$\Delta$Flux/$\sigma_{F(hb)}$',
             cmap=cmap.residuals,title='residual',title_fontsize=fontsize_set,label_fontsize=fontsize_set)
    
    plot_map(ax[ii+1][4], res_vel,clim=res_vel_lim,cbar_label='$\Delta$v(km/s)',cmap=cmap.residuals,label_fontsize=fontsize_set)
    plot_map(ax[ii+2][4], res_vdisp,clim=res_vdisp_lim,cbar_label='$\Delta \sigma_{\mathrm{gas}}$(km/s)',cmap=cmap.residuals,label_fontsize=fontsize_set)
    
    mpb0 = mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=flux_hb_lim[0],vmax=flux_hb_lim[1]),cmap=cmap.flux)
    mpb1 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vel_lim[0],vmax=vel_lim[1]),cmap=cmap.v)
    mpb2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vdisp_lim[0],vmax=vdisp_lim[1]),cmap=cmap.vdisp)
    
    colorbar(mpb0,cax=ax[0][3],clim=flux_hb_lim,label='Flux(10$^{-20}$ erg/s/cm$^{-2}$)',label_fontsize=fontsize_set)
    colorbar(mpb1,cax=ax[ii+1][3],clim=vel_lim,label='$v$(km/s)',label_fontsize=fontsize_set)
    colorbar(mpb2,cax=ax[ii+2][3],clim=vdisp_lim,label='$\sigma_{\mathrm{gas}}$(km/s)',label_fontsize=fontsize_set)
    
    
    # residual colorbar
    
    res_mpb0 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_flux_hb_lim[0],vmax=res_flux_hb_lim[1]),cmap=cmap.residuals)
    res_mpb1 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_vel_lim[0],vmax=res_vel_lim[1]),cmap=cmap.residuals)
    res_mpb2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=res_vdisp_lim[0],vmax=res_vdisp_lim[1]),cmap=cmap.residuals)
    
    colorbar(res_mpb0,cax=ax[0][5],clim=res_flux_hb_lim,label='$\Delta F(hb)/\sigma_{F(hb)}$',label_fontsize=fontsize_set)
    colorbar(res_mpb1,cax=ax[ii+1][5],clim=res_vel_lim,label='$\Delta v$(km/s)',label_fontsize=fontsize_set)
    colorbar(res_mpb2,cax=ax[ii+2][5],clim=res_vdisp_lim,label='$\Delta \sigma_{\mathrm{gas}}$(km/s)',label_fontsize=fontsize_set)
    
    if plot_nii==True:
        extra_row_index = 5
    else:
        extra_row_index = 3
    
    if vdispHist==True:
        ax[extra_row_index][0].hist(np.exp(post_b3d.global_param['VDISP0']),bins=20, color='c', edgecolor='k', alpha=0.65)
        ax[extra_row_index][0].set_xlabel('vdisp [km/s]')
        ax[extra_row_index][0].set_ylabel('N')
        ax[extra_row_index][0].axvline(np.exp(post_b3d.global_param['VDISP0']).mean(), color='k', linestyle='dashed', linewidth=1)
        
    if vdispSFR==True:
        ax[extra_row_index][1].scatter(sfr_all,vdisp_all)
        ax[extra_row_index][1].set_xscale('log')
        ax[extra_row_index][1].set_xlabel('SFR [M* yr$^{-1}$]')
        ax[extra_row_index][1].set_ylabel('vdisp [km/s]')
        if data_avail==True:
            ax[extra_row_index][1].scatter(sfr_this,vdisp_this,c='r',marker='*') 
    
    if set_ticks==True:
        # xticks
        #x_shape = precon_flux_ha.shape[1]
        naxis = precon_flux_hb.shape
        x_dist = naxis[1]*0.2/2
        y_dist = naxis[0]*0.2/2
        ## todo
        set_xticks(ax[ii+2][0],xticks='lower',x_dist=x_dist,naxis=naxis)
        set_xticks(ax[ii+2][1],xticks='all',x_dist=x_dist,naxis=naxis)
        set_xticks(ax[ii+2][2],xticks='upper',x_dist=x_dist,naxis=naxis)
        set_xticks(ax[ii+2][4],xticks='all',x_dist=x_dist,naxis=naxis)
        
        set_yticks(ax[0][0],yticks='upper',y_dist=y_dist,naxis=naxis)
        set_yticks(ax[1][0],yticks='all',y_dist=y_dist,naxis=naxis)
        set_yticks(ax[2][0],yticks='lower',y_dist=y_dist,naxis=naxis)
    
    
    
    if fwhm is not None:
    
        circle = plt.Circle(
                (1.1*fwhm, 1.1*fwhm),
                radius=fwhm,
                fill=False, edgecolor='r', linewidth=1.0)
        ax[0][2].add_artist(circle)
    
    if figpath == None:
        figpath = datapath
    
    if set_title == None:
        plt.savefig(figpath+"compmap.pdf",dpi=300,bbox_inches='tight')
    else:
        plt.savefig(figpath+"compmap_hb_"+set_title+".pdf",dpi=300,bbox_inches='tight')

