#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:56:24 2024

for metalicity gradient plot

part of code from Di Wang

@author: ymai0110
"""

import sys
#sys.path.insert(0,'/project/blobby3d/Blobby3D/pyblobby3d/src/pyblobby3d/')
#sys.path.insert(0,'/Users/ymai0110/Documents/Blobby3D_develop/Blobby3D/pyblobby3d/src/')

import numpy as np
import matplotlib.pyplot as plt

from pyblobby3d.post_blobby3d import PostBlobby3D
from pyblobby3d.moments import SpectralModel
from pyblobby3d.b3dcomp import map_limits
 
import matplotlib as mpl
from astropy.io import fits
import pandas as pd

from astropy.cosmology import LambdaCDM
lcdm = LambdaCDM(70,0.3,0.7)
import astropy.units as u
import astropy.cosmology.units as cu
from dust_extinction.parameter_averages import F19
from scipy.optimize import curve_fit
from metal_uncertainty import intrinsic_flux_with_err
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter

def plot_oii_gradient(post_b3d,global_param,x_cen_pix,y_cen_pix,ellip,ax,
                      label=None,plot_data=False):
    '''
    post_b3d: class PostBlobby3D
    global_param: output from blobby3d
    x_cen_pix: from galaxy_cen_cut.csv file
    ellip=ellipticity of the ellipse=1-b/a
    
    '''
    
    wave = post_b3d.metadata.get_axis_array('r')
    
    
    
    pa = global_param['PA'][0] # for PA in b3d output, use angletype='NTE'
    
    
    sample = 0
    
    if plot_data:
        
        sm = SpectralModel(
                lines=[[3727.092], [3729.875]],
                lsf_fwhm=1.5,) # this value doesn't really matter, it's only 
                               # used for the initial guess of the fitting
        fit_data, fit_var_data = sm.fit_cube(wave, post_b3d.data, post_b3d.var)
        data_flux_oii3727 = fit_data[0]
        ##data_flux = post_b3d.data.sum(axis=2)
        data_flux_oii3729 = fit_data[1]
        oii_sum_map = data_flux_oii3727 + data_flux_oii3729
        
    else:
    
        precon_flux_oii_3727 = post_b3d.maps[sample, 0]
        precon_flux_oii_3729 = post_b3d.maps[sample,1]
    
        oii_sum_map = precon_flux_oii_3727 + precon_flux_oii_3729
    
    dist_arr = ellip_distarr(size=oii_sum_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    
    if plot_data:
        ax.scatter(dist_arr, oii_sum_map,label='data',alpha=0.4)
    else:
        ax.scatter(dist_arr, oii_sum_map,label=label,alpha=0.4)
    
    
    
def plot_oii_gradient_compare(post_b3d_path_list,global_param_path_list,
                              x_cen_pix_list, y_cen_pix_list, ellip,
                              version_list,savepath=None,plot_data=False):
    '''
    

    Parameters
    ----------
    post_b3d_path_list : TYPE
        DESCRIPTION.
    global_param_path_list : TYPE
        DESCRIPTION.
    ellip : TYPE
        DESCRIPTION.
    version_list : TYPE
        list of str, name of different versions of constrain.
    savepath : TYPE, optional
        DESCRIPTION. The default is None.
    plot_data : if True, plot the oii gradient of fitting of the data

    Returns
    -------
    None.

    '''
    
    fig, ax = plt.subplots()
    
    for datapath, global_param_path_i,label_i,x_cen_pix,y_cen_pix in zip(post_b3d_path_list,
                                                    global_param_path_list,
                                                    version_list,x_cen_pix_list,
                                                    y_cen_pix_list):
        post_b3d = PostBlobby3D(
                samples_path=datapath+'posterior_sample.txt',
                data_path=datapath+'data.txt',
                var_path=datapath+'var.txt',
                metadata_path=datapath+'metadata.txt',
                nlines=2)
        global_param = pd.read_csv(global_param_path_i)
        plot_oii_gradient(post_b3d=post_b3d,global_param=global_param,
                          x_cen_pix=x_cen_pix,y_cen_pix=y_cen_pix,
                          ellip=ellip,ax=ax,label=label_i)
    
    
    if plot_data:
        plot_oii_gradient(post_b3d=post_b3d,global_param=global_param,
                          x_cen_pix=x_cen_pix,y_cen_pix=y_cen_pix,
                          ellip=ellip,ax=ax,plot_data=True)
    
    ax.set_xlabel('radius')
    ax.set_ylabel('log OII sum')
    ax.legend()
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath,bbox_inches='tight',dpi=300)
    plt.show()
    
    
def plot_metal_gradient_compare(post_b3d_oii_path_list,post_b3d_nii_path,
                                global_param_path_list,
                              x_cen_pix, y_cen_pix, ellip,
                              version_list,savepath=None,plot_data=False,
                              plot_median=False):
    '''
    

    Parameters
    ----------
    post_b3d_oii_path_list : TYPE
        DESCRIPTION.
    post_b3d_nii_path : TYPE
        Halpha and nii fitting path.
        
    global_param_path_list : TYPE
        DESCRIPTION.
    ellip : TYPE
        DESCRIPTION.
    version_list : TYPE
        list of str, name of different versions of constrain.
    savepath : TYPE, optional
        DESCRIPTION. The default is None.
    plot_data : if True, plot the oii gradient of fitting of the data

    Returns
    -------
    None.

    '''
    
    fig, ax = plt.subplots()
    i = 0
    color_list = ['k','b','r','g']
    
    for datapath_oii, global_param_path_i,label_i, in zip(post_b3d_oii_path_list,
                                                    global_param_path_list,
                                                    version_list):
        post_b3d_oii = PostBlobby3D(
                samples_path=datapath_oii+'posterior_sample.txt',
                data_path=datapath_oii+'data.txt',
                var_path=datapath_oii+'var.txt',
                metadata_path=datapath_oii+'metadata.txt',
                nlines=2)
        datapath_nii = post_b3d_nii_path
        post_b3d_nii = PostBlobby3D(
                samples_path=datapath_nii+'posterior_sample.txt',
                data_path=datapath_nii+'data.txt',
                var_path=datapath_nii+'var.txt',
                metadata_path=datapath_nii+'metadata.txt',
                nlines=2)
        
        
        global_param = pd.read_csv(global_param_path_i)
        
        if plot_median==False:
            plot_metal_gradient(post_b3d_oii=post_b3d_oii,
                                post_b3d_nii=post_b3d_nii,
                                global_param=global_param,
                              x_cen_pix=x_cen_pix,y_cen_pix=y_cen_pix,
                              ellip=ellip,ax=ax,label=label_i)
        else:
            plot_metal_gradient_median(post_b3d_oii=post_b3d_oii,
                                post_b3d_nii=post_b3d_nii,
                                global_param=global_param,
                              x_cen_pix=x_cen_pix,y_cen_pix=y_cen_pix,
                              ellip=ellip,ax=ax,label=label_i,r_shift=i*0.1,
                              markercolor=color_list[i])
            i=i+1
    
    if plot_data:
        if plot_median==False:
            plot_metal_gradient(post_b3d_oii=post_b3d_oii,
                                post_b3d_nii=post_b3d_nii,
                                global_param=global_param,
                              x_cen_pix=x_cen_pix,y_cen_pix=y_cen_pix,
                              ellip=ellip,ax=ax,plot_data=True)
        else:
            plot_metal_gradient_median(post_b3d_oii=post_b3d_oii,
                                post_b3d_nii=post_b3d_nii,
                                global_param=global_param,
                              x_cen_pix=x_cen_pix,y_cen_pix=y_cen_pix,
                              ellip=ellip,ax=ax,plot_data=True,r_shift=0.4,
                              markercolor='orange')
        
    
    ax.set_xlabel('radius')
    ax.set_ylabel('log (NII6584/OII_sum)')
    ax.legend()
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath,bbox_inches='tight',dpi=300)
    plt.show()
    
    
def plot_metal_gradient_median(post_b3d_oii,post_b3d_nii,global_param,
                                       x_cen_pix,y_cen_pix,ellip,ax,markercolor,
                                       label=None,plot_data=False,r_shift=0):
    '''calculate the metal at 1 pix radius
    shift r to avoid overlap between different versions
    '''
    
    oii_flux_map = get_flux_map(post_b3d=post_b3d_oii, emi_line='Oii',
                                get_data=plot_data)
    nii_flux_map = get_flux_map(post_b3d=post_b3d_nii, emi_line='Nii',
                                get_data=plot_data)
    
    metal_map = np.log10(nii_flux_map/oii_flux_map)
    
    pa = global_param['PA'][0] # for PA in b3d output, use angletype='NTE'
    
    dist_arr = ellip_distarr(size=oii_flux_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    if plot_data:
        # mask some very low metal sapxels...
        metal_map[metal_map<-10]=np.nan
        r_max = int(np.nanmax(dist_arr))
        
        r_array = np.arange(r_max)
        metal_median_array = []
        metal_err_array = []
        for r_i in r_array:
            index = (dist_arr >= r_i) & (dist_arr < r_i+1)
            metal_median_array.append(np.nanmedian(metal_map[index]))
            metal_err_array.append(np.nanstd(metal_map[index]))
        
        r_array = r_array + r_shift
        ax.scatter(r_array,metal_median_array,c='orange',label='data')
        ax.errorbar(r_array,metal_median_array,yerr=metal_err_array,fmt='none',
                    ecolor='orange')
        
    else:
        
        r_max = int(np.nanmax(dist_arr))
        
        r_array = np.arange(r_max)
        metal_median_array = []
        metal_err_array = []
        for r_i in r_array:
            index = (dist_arr >= r_i) & (dist_arr < r_i+1)
            metal_median_array.append(np.nanmedian(metal_map[index]))
            metal_err_array.append(np.nanstd(metal_map[index]))
        
        r_array = r_array + r_shift
        ax.scatter(r_array,metal_median_array,c=markercolor,label=label)
        ax.errorbar(r_array,metal_median_array,yerr=metal_err_array,fmt='none',
                    ecolor=markercolor)
        
    
    
def plot_metal_gradient(post_b3d_oii,post_b3d_nii,global_param,x_cen_pix,
                        y_cen_pix,ellip,ax,label=None,plot_data=False):
    '''calculate the metallicity using NII6584/OII_sum, (Kewley+2019)
    
    plot log(Nii6584/oii_sum) vs radius
    
    
    '''
    
    oii_flux_map = get_flux_map(post_b3d=post_b3d_oii, emi_line='Oii',
                                get_data=plot_data)
    nii_flux_map = get_flux_map(post_b3d=post_b3d_nii, emi_line='Nii',
                                get_data=plot_data)
    
    metal_map = np.log10(nii_flux_map/oii_flux_map)
    
    pa = global_param['PA'][0] # for PA in b3d output, use angletype='NTE'
    
    dist_arr = ellip_distarr(size=oii_flux_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    if plot_data:
        # mask some very low metal sapxels...
        metal_map[metal_map<-10]=np.nan
        ax.scatter(dist_arr, metal_map,label='data',alpha=0.4)
    else:
        ax.scatter(dist_arr, metal_map,label=label,alpha=0.4)
    
    
def plot_metal_gradient_dustcorr(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,
                                 pa,x_cen_pix,y_cen_pix,ellip,
                                 foreground_E_B_V,
                                 z,re_kpc=None,savepath=None,
                                 r_option='kpc',plot_title=None,
                                 limit_metal_range=False):
    
    '''
    plot the metallicity of each spaxels as function of kpc or Re
    do dust corretion using Balmer decrement; follow Mun et al. 2024
    ...add more details here ...
    
    
    ha_map: 2d-array of ha flux
    ha_sn: 2d-array of sn of ha
    oii_map: 2d-array of oii sum map (oii3727+oii3729)
    pa: PA from b3d output, use angletype='NTE'
    x_cen_pix: center pixel of x
    ellip: ellipticity from profound
    foreground_E_B_V: foreground extinction value, 
    get from https://irsa.ipac.caltech.edu/applications/DUST/
    
    
    z: redshift
    re_kpc: half-light radius of white light image of galaxy
    r_option:'kpc' or 're'
    
    '''
    
    # dust correction
    
    # oii hb ha nii
    # [3727.092,3729.875] 4862.683 6564.61 [6549.85, 6585.28]
    
    # oii_sum hb ha nii6585
    # 3728.4835 4862.683 6564.61 6585.28
    oii_wave = 3728.4835
    hb_wave = 4862.683
    ha_wave = 6564.61
    nii_wave = 6585.28
    wavelengths_restframe = np.array([oii_wave,hb_wave,ha_wave,nii_wave])*u.AA
    wavelengths_redshift = wavelengths_restframe * (1+z)
    
    ha_correct_MW = dust_correction(flux=ha_map, 
                                               E_B_V=foreground_E_B_V, 
                                               wavelength=ha_wave* (1+z))
    
    hb_correct_MW = dust_correction(flux=hb_map, 
                                               E_B_V=foreground_E_B_V, 
                                               wavelength=hb_wave* (1+z))
    
    nii_correct_MW = dust_correction(flux=nii_map, 
                                               E_B_V=foreground_E_B_V,
                                               wavelength=nii_wave* (1+z))
    
    oii_correct_MW = dust_correction(flux=oii_map, 
                                               E_B_V=foreground_E_B_V, 
                                               wavelength=oii_wave* (1+z))
    
    
    E_44_55 = 2.5/1.16 * np.log10(ha_correct_MW/hb_correct_MW/2.86) # eq 10 in Fitzpatrick+19
    E_B_V_intrin = E_44_55 * 0.976 # based on table 4 in Fitzpatrick+19, assuming E(44-55)=0.5, RV=3.1
    
    nii_intrinsic = dust_correction_2d(flux_map=nii_correct_MW, 
                                    E_B_V_map=E_B_V_intrin, 
                                    wavelength=nii_wave)
    oii_intrinsic = dust_correction_2d(flux_map=oii_correct_MW, 
                                       E_B_V_map=E_B_V_intrin, 
                                       wavelength=oii_wave)
    
    
    logR_map = np.log10(nii_intrinsic/oii_intrinsic)
    metal_map = metal_k19(logR_map)
    
    dist_arr = ellip_distarr(size=metal_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map[mask_region] = np.nan
    
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    if limit_metal_range==True:
        metal_map[metal_map<=5] = np.nan 
        metal_map[metal_map>=10] = np.nan 
    
    if r_option=='kpc':
    
        plt.scatter(radius_kpc, metal_map)
        plt.vlines(x=re_kpc,ymin=np.nanmin(metal_map),ymax=np.nanmax(metal_map),
                   linestyle='--')
        plt.xlabel('radius [kpc]')
    elif r_option=='re':
        plt.scatter(radius_re, metal_map)
        
        plt.xlabel('radius/Re')
    plt.title(plot_title)
    
    plt.ylabel('log(O/H) + 12')
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    plt.show()

class cmap:
    flux = 'Oranges'
    v = 'RdYlBu_r'
    vdisp = 'YlOrBr'
    residuals = 'RdYlBu_r'
    
def get_e_b_v(ha_map,hb_map,foreground_E_B_V,z):
    # dust correction
    
    # oii hb ha nii
    # [3727.092,3729.875] 4862.683 6564.61 [6549.85, 6585.28]
    
    # oii_sum hb ha nii6585
    # 3728.4835 4862.683 6564.61 6585.28
    oii_wave = 3728.4835
    hb_wave = 4862.683
    ha_wave = 6564.61
    nii_wave = 6585.28
    wavelengths_restframe = np.array([oii_wave,hb_wave,ha_wave,nii_wave])*u.AA
    wavelengths_redshift = wavelengths_restframe * (1+z)
    
    ha_correct_MW = dust_correction(flux=ha_map, 
                                               E_B_V=foreground_E_B_V, 
                                               wavelength=ha_wave* (1+z))
    
    hb_correct_MW = dust_correction(flux=hb_map, 
                                               E_B_V=foreground_E_B_V, 
                                               wavelength=hb_wave* (1+z))
    E_44_55 = 2.5/1.16 * np.log10(ha_correct_MW/hb_correct_MW/2.86) # eq 10 in Fitzpatrick+19
    E_B_V_intrin = E_44_55 * 0.976 # based on table 4 in Fitzpatrick+19, assuming E(44-55)=0.5, RV=3.1
    
    return E_B_V_intrin
    
    

def get_metal_map_n2o2(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,foreground_E_B_V,
                                 z,radius_kpc=None):
    '''input flux map, flux SN, foreground E(B-V), redshift
    output metal map with dust correction and mask low SN region'''
    
    
    
    # dust correction
    
    # oii hb ha nii
    # [3727.092,3729.875] 4862.683 6564.61 [6549.85, 6585.28]
    
    # oii_sum hb ha nii6585
    # 3728.4835 4862.683 6564.61 6585.28
    oii_wave = 3728.4835
    hb_wave = 4862.683
    ha_wave = 6564.61
    nii_wave = 6585.28
    wavelengths_restframe = np.array([oii_wave,hb_wave,ha_wave,nii_wave])*u.AA
    wavelengths_redshift = wavelengths_restframe * (1+z)
    
    ha_correct_MW = dust_correction(flux=ha_map, 
                                               E_B_V=foreground_E_B_V, 
                                               wavelength=ha_wave* (1+z))
    
    hb_correct_MW = dust_correction(flux=hb_map, 
                                               E_B_V=foreground_E_B_V, 
                                               wavelength=hb_wave* (1+z))
    
    nii_correct_MW = dust_correction(flux=nii_map, 
                                               E_B_V=foreground_E_B_V,
                                               wavelength=nii_wave* (1+z))
    
    oii_correct_MW = dust_correction(flux=oii_map, 
                                               E_B_V=foreground_E_B_V, 
                                               wavelength=oii_wave* (1+z))
    
    
    E_44_55 = 2.5/1.16 * np.log10(ha_correct_MW/hb_correct_MW/2.86) # eq 10 in Fitzpatrick+19
    E_B_V_intrin = E_44_55 * 0.976 # based on table 4 in Fitzpatrick+19, assuming E(44-55)=0.5, RV=3.1
    
    nii_intrinsic = dust_correction_2d(flux_map=nii_correct_MW, 
                                    E_B_V_map=E_B_V_intrin, 
                                    wavelength=nii_wave)
    oii_intrinsic = dust_correction_2d(flux_map=oii_correct_MW, 
                                       E_B_V_map=E_B_V_intrin, 
                                       wavelength=oii_wave)
    
    if radius_kpc is not None:
        print('ha/hb intrin:')
        print((ha_correct_MW/hb_correct_MW)[radius_kpc<3])
        print('oii obs:')
        print(oii_map[radius_kpc<3])
        print('nii obs:')
        print(nii_map[radius_kpc<3])
        print('oii intrinsic:')
        print(oii_intrinsic[radius_kpc<3])
        print('nii intrinsic:')
        print(nii_intrinsic[radius_kpc<3])
    
    
    logR_map = np.log10(nii_intrinsic/oii_intrinsic)
    metal_map = metal_k19(logR_map)
    
    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map[mask_region] = np.nan
    
    return metal_map


def get_metal_map_n2ha(ha_map,nii_map,ha_sn,nii_sn,nii_sn_cut=True):
    
    logR_map = np.log10(nii_map/ha_map)
    metal_map = metal_k19(logR=logR_map,R='N2Ha')
    
    if nii_sn_cut:
        query = (ha_sn>=3) & (nii_sn>=3)
    else:
        query = ha_sn>=3
    
    mask_region = ~query
    
    metal_map[mask_region] = np.nan
    
    return metal_map
    
    
    
def diagnostic_map(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,
                                 ha_err,hb_err,oii_err,nii_err,
                                 pa,x_cen_pix,y_cen_pix,ellip,
                                 foreground_E_B_V,
                                 z,re_kpc=None,savepath=None,
                                 r_option='kpc',plot_title=None,
                                 limit_metal_range=False):
    '''
    
    ha_map: b3d pre-convolved ha flux
    ha_sn: ha sn from gist
    ha_err: ha err from gist
    
    
    
    
    make diagnostic map of metallicity, including NII/OII map, Halpha/Hbeta map
    metallicity map. with radius contour (0.5Re, 1Re..), galaxy center
    
    metal (without dust correction) vs radius plot with errorbar
    metal (with dust correction) vs radius plot with errorbar
    
    balmer decrement from GIST output
    metal map using GIST balmer decrement
    metal gradient using GIST balmer decrement
    
    metal map using nii/ha
    metal graident using nii/ha
    
    bin data in each 1kpc (std or 16,84) and fit the gradient, text of fitting
    1. nii/oii without dust corretion
    2. nii/oii with dust correction
    3. nii/ha
    
    
    how to bin: 1. mean 2. halpha weight average (two version of bin on the same plot)
    
    
    
    '''
    plt.rcParams.update({'font.size': 25})
    
    dist_arr = ellip_distarr(size=ha_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    
    
    fig, ax = plt.subplots(4,4,figsize=(34,34), 
                           gridspec_kw={'wspace':0.6,'hspace':0.35})
    ax = ax.ravel()
    
    # NII/OII map
    
    clim = map_limits(np.log10(nii_map/oii_map),pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    
    im0 = ax[0].imshow(np.log10(nii_map/oii_map),
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
    cb0.set_label(label='log10(NII/OII)',fontsize=20)
    ax[0].set_title('nii/oii')
    
    ax[0].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[0].contour(radius_re, levels=[1], colors='g', linewidths=2, linestyles='dashed')
    ax[0].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted')
    ax[0].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    
    # Halpha/Hbeta map
    
    norm = mpl.colors.Normalize(vmin=0, vmax=6)
    
    im1 = ax[1].imshow(ha_map/hb_map,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb1 = plt.colorbar(im1,ax=ax[1],fraction=0.047)
    cb1.set_label(label='Halpha/Hbeta',fontsize=20)
    ax[1].set_title('Balmer decrement')
    
    ax[1].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[1].contour(radius_re, levels=[1], colors='g', linewidths=2, linestyles='dashed')
    ax[1].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted')
    ax[1].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    
    # metal map - with dust correction
    
    metal_map_n2o2_b3ddust = get_metal_map_n2o2(ha_map=ha_map, hb_map=hb_map, oii_map=oii_map, 
                              nii_map=nii_map, ha_sn=ha_sn, hb_sn=hb_sn,
                              oii_sn=oii_sn, nii_sn=nii_sn, 
                              foreground_E_B_V=foreground_E_B_V, z=z)
    
    
    # kewley+2019 the valid range of NII/OII metallicity is [7.63,9.23]
    norm = mpl.colors.Normalize(vmin=7.0, vmax=9.3)
    
    im2 = ax[2].imshow(metal_map_n2o2_b3ddust,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb2 = plt.colorbar(im2,ax=ax[2],fraction=0.047)
    cb2.set_label(label='log(O/H)+12',fontsize=20)
    ax[2].set_title('metal map (n2o2)')
    
    ax[2].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[2].contour(radius_re, levels=[1], colors='g', linewidths=2, linestyles='dashed')
    ax[2].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted')
    ax[2].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    
    
    dist_arr = ellip_distarr(size=metal_map_n2o2_b3ddust.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    # metal as function of radius, without dust correction
    
    logR_map = np.log10(nii_map/oii_map)
    metal_map_n2o2_nodustc = metal_k19(logR_map)
    
    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map_n2o2_nodustc[mask_region] = np.nan
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    # get metal error
    #nii_err_map = nii_map/nii_sn
    #oii_err_map = oii_map/oii_sn
    
    lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_map, b=oii_map, 
                                            a_err=nii_err, 
                                            b_err=oii_err)
    
    metal_i, metal_err = calculate_z_and_error(x=lognii_oii, 
                                               x_err=lognii_oii_err, 
                                               y=-3.17)
    
    
    ax[3].scatter(radius_kpc, metal_map_n2o2_nodustc)
    ax[3].errorbar(x=radius_kpc.ravel(),y=metal_map_n2o2_nodustc.ravel(),yerr=metal_err.ravel(),fmt='none')
    ax[3].axvline(x=re_kpc,ymin=0,ymax=1,
               linestyle='--',c='g')
    ax[3].axhline(y=9.23,xmin=0,xmax=1,linestyle='--',c='r')
    ax[3].axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='r')
    ax[3].set_xlabel('radius [kpc]')
    ax[3].set_ylim(7.0,9.3)
    ax[3].set_title('metal (n2o2) w/o dust c')
    
    # metal as function of radius, with dust correction
    
    #metal_map = get_metal_map_n2o2(ha_map=ha_map, hb_map=hb_map, oii_map=oii_map, 
    #                          nii_map=nii_map, ha_sn=ha_sn, hb_sn=hb_sn,
    #                          oii_sn=oii_sn, nii_sn=nii_sn, 
    #                          foreground_E_B_V=foreground_E_B_V, z=z)
    ax[4].scatter(radius_kpc, metal_map_n2o2_b3ddust)
    ax[4].axvline(x=re_kpc,ymin=0,ymax=1,
            linestyle='--',c='g')
    ax[4].axhline(y=9.23,xmin=0,xmax=1,linestyle='--',c='r')
    ax[4].axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='r')
    ax[4].set_xlabel('radius [kpc]')
    ax[4].set_ylim(7.0,9.3)
    ax[4].set_title('metal(n2o2)-b3d balmer')
    
    
    # balmer decrement of GIST
    ha_gist = ha_sn * ha_err
    hb_gist = hb_sn * hb_err
    nii_gist = nii_sn * nii_err
    oii_gist = oii_sn * oii_err
    
    norm = mpl.colors.Normalize(vmin=0, vmax=6)
    
    im5 = ax[5].imshow(ha_gist/hb_gist,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb5 = plt.colorbar(im5,ax=ax[5],fraction=0.047)
    cb5.set_label(label='Halpha/Hbeta',fontsize=20)
    ax[5].set_title('Balmer decrement - gist')
    
    ax[5].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[5].contour(radius_re, levels=[1], colors='g', linewidths=2, linestyles='dashed')
    ax[5].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted')
    ax[5].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    
    # metal map with dust correction - gist balmer decrement
    
    # nii oii from b3d, ha hb from gist
    metal_map_n2o2_gistdust = get_metal_map_n2o2(ha_map=ha_gist, hb_map=hb_gist, oii_map=oii_map, 
                              nii_map=nii_map, ha_sn=ha_sn, hb_sn=hb_sn,
                              oii_sn=oii_sn, nii_sn=nii_sn, 
                              foreground_E_B_V=foreground_E_B_V, z=z)
    
    
    # kewley+2019 the valid range of NII/OII metallicity is [7.63,9.23]
    norm = mpl.colors.Normalize(vmin=7.0, vmax=9.3)
    
    im6 = ax[6].imshow(metal_map_n2o2_gistdust,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb6 = plt.colorbar(im6,ax=ax[6],fraction=0.047)
    cb6.set_label(label='log(O/H)+12',fontsize=20)
    ax[6].set_title('metal map(n2o2)-gist balmer')
    
    ax[6].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[6].contour(radius_re, levels=[1], colors='g', linewidths=2, linestyles='dashed')
    ax[6].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted')
    ax[6].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    
    # metal gradient - gist balmer decrement
    
    ax[7].scatter(radius_kpc, metal_map_n2o2_gistdust)
    ax[7].axvline(x=re_kpc,ymin=0,ymax=1,
            linestyle='--',c='g')
    ax[7].axhline(y=9.23,xmin=0,xmax=1,linestyle='--',c='r')
    ax[7].axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='r')
    ax[7].set_xlabel('radius [kpc]')
    ax[7].set_ylim(7.0,9.3)
    ax[7].set_title('metal(n2o2)-gist balmer')
    
    
    # metal map - nii/ha
    
    
    metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn,
                                   nii_sn=nii_sn)
    
    
    
    # kewley+2019 the valid range of NII/Ha metallicity is [7.63,8.53]
    norm = mpl.colors.Normalize(vmin=7.0, vmax=9.3)
    
    im8 = ax[8].imshow(metal_map_n2ha,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb8 = plt.colorbar(im8,ax=ax[8],fraction=0.047)
    cb8.set_label(label='log(O/H)+12',fontsize=20)
    ax[8].set_title('metal map (n2ha)')
    
    ax[8].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[8].contour(radius_re, levels=[1], colors='g', linewidths=2, linestyles='dashed')
    ax[8].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted')
    ax[8].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    # metal gradient - n2ha
    
    #nii_err_map = nii_map/nii_sn
    #ha_err_map = ha_map/ha_sn
    
    lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_map, b=ha_map, 
                                            a_err=nii_err, 
                                            b_err=ha_err)
    
    metal_i, metal_err = calculate_z_and_error(x=lognii_ha, 
                                               x_err=lognii_ha_err, 
                                               y=-3.17,R='N2Ha')
    
    ax[9].scatter(radius_kpc, metal_map_n2ha)
    ax[9].errorbar(x=radius_kpc.ravel(),y=metal_map_n2ha.ravel(),yerr=metal_err.ravel(),fmt='none')
    ax[9].axvline(x=re_kpc,ymin=0,ymax=1,
            linestyle='--',c='g')
    #ax[9].axhline(y=8.53,xmin=0,xmax=1,linestyle='--',c='r')
    #ax[9].axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='r')
    
    ax[9].axhline(y=8.86,xmin=0,xmax=1,linestyle='--',c='r')
    ax[9].axhline(y=7.17,xmin=0,xmax=1,linestyle='--',c='r')
    ax[9].set_xlabel('radius [kpc]')
    ax[9].set_ylim(7.0,9.3)
    ax[9].set_title('metal (n2ha)')
    
    
    # bin metal 
    #1. nii/oii without dust corretion
    axplot_bin_metal(ax=ax[10],metal_map=metal_map_n2o2_nodustc,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 w/o dust c',R='N2O2')
    
    #2. nii/oii with dust correction-b3d
    axplot_bin_metal(ax=ax[11],metal_map=metal_map_n2o2_b3ddust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 b3d balmer',R='N2O2')
    
    #3. nii/oii with dust correction-gist
    axplot_bin_metal(ax=ax[12],metal_map=metal_map_n2o2_gistdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 gist balmer',R='N2O2')
    
    #4. nii/ha
    axplot_bin_metal(ax=ax[13],metal_map=metal_map_n2ha,
                     ha_map=ha_map,radius_map=radius_kpc,title='n2ha',
                     re_kpc=re_kpc,R='N2Ha')
    
    
    fig.delaxes(ax[14])
    fig.delaxes(ax[15])
    
    fig.suptitle(plot_title)
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    
    plt.show()

class emi_wave():
    # just for reference ...
    oii_wave = 3728.4835
    hb_wave = 4862.683
    ha_wave = 6564.61
    nii_wave = 6585.28

def diagnostic_map_gist_dustc(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,
                                 ha_err,hb_err,oii_err,nii_err,
                                 ha_gist_err,hb_gist_err,oii_gist_err,nii_gist_err,
                                 pa,x_cen_pix,y_cen_pix,ellip,
                                 foreground_E_B_V,
                                 z,re_kpc=None,savepath=None,
                                 ha_gist=None,hb_gist=None,oii_gist=None,
                                 nii_gist=None,
                                 r_option='kpc',plot_title=None,
                                 limit_metal_range=False,
                                 savecsv=False,csvpath=None,
                                 nii_sn_cut=3,model_flux_sn=False,
                                 radius_fill_perc=0.5,
                                 bin_size=1,r_min_set_force=None,
                                 dust_model='F19'):
    
    
    '''
    calculate metal gradient for spaxel average method
    
    ----
    update 4 April 2025:
        1. make sure the range of radius for fit metallicity are the same for
        different diagnostics
        2. nii_sn_cut: make it to be float. if not nii_sn_cut, set it to None
        (the previous version is boolen)
    update 18 April 2025:
        1. bin_size: the size of the average metallicity
                    bin in the unit of kpc. This is to adjust the bin size
                    to any value, e.g. 0.5 PSF-FWHM, 0.25 Re. I just need to 
                    calculate the kpc value for those bin size.
        2. pa from MAGPI data are in the unit of degree, but ellip_distarr
            take pa in the unit of radian, so I add degree->rad conversion.
    update 2 May 2025:
        1. r_min_set_force: int, force the code to fit the first N datapoints
    update 21 July 2025:
        dust_model option, F19, SMC, LMC. default is F19
    '''
    
    ha_sn_model = ha_map/ha_err
    hb_sn_model = hb_map/hb_err
    nii_sn_model = nii_map/nii_err
    oii_sn_model = oii_map/oii_err
    
    
    plt.rcParams.update({'font.size': 25})
    
    dist_arr = ellip_distarr(size=ha_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    
    
    fig, ax = plt.subplots(2,4,figsize=(34,17), 
                           gridspec_kw={'wspace':0.6,'hspace':0.35})
    ax = ax.ravel()
    
    
    ##############################
    # NII/OII map
    
    clim = map_limits(np.log10(nii_map/oii_map),pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    
    im0 = ax[0].imshow(np.log10(nii_map/oii_map),
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
    cb0.set_label(label='log10(NII/OII)',fontsize=20)
    ax[0].set_title('nii/oii')
    
    ax[0].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[0].contour(radius_re, levels=[1], colors='g', linewidths=2, linestyles='dashed')
    ax[0].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted')
    ax[0].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    ##############################
    # balmer decrement of GIST
    #ha_gist = ha_sn * ha_err
    #hb_gist = hb_sn * hb_err
    #nii_gist = nii_sn * nii_err
    #oii_gist = oii_sn * oii_err
    
    norm = mpl.colors.Normalize(vmin=1, vmax=5)
    
    im1 = ax[1].imshow(ha_gist/hb_gist,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb1 = plt.colorbar(im1,ax=ax[1],fraction=0.047)
    cb1.set_label(label='Halpha/Hbeta',fontsize=20)
    ax[1].set_title('Balmer decrement - gist')
    
    ax[1].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[1].contour(radius_re, levels=[1], colors='g', linewidths=2, linestyles='dashed')
    ax[1].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted')
    ax[1].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    
    ##############################
    # metal n2o2 as func of radius, without dust correction
    
    logR_map = np.log10(nii_map/oii_map)
    metal_map_n2o2_nodustc = metal_k19(logR_map)
    
    if model_flux_sn:
        if nii_sn_cut is not None:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn_model>=3) & (nii_sn_model>=nii_sn_cut)
        else:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn_model>=3)# gist ha for dust corr, so gist ha sn
    else:
        if nii_sn_cut is not None:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=nii_sn_cut)
        else:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map_n2o2_nodustc[mask_region] = np.nan
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    # get metal error
    #nii_err_map = nii_map/nii_sn # wrong !!! can't divide by nii_sn, could be 0
    #oii_err_map = oii_map/oii_sn
    
    #print('no dust nii:')
    #print(nii_map[radius_kpc<3])
    #print('no dust nii err:')
    #print(nii_err[radius_kpc<3])
    #print('no dust oii:')
    #print(oii_map[radius_kpc<3])
    #print('no dust oii err:')
    #print(oii_err[radius_kpc<3])
    
    
    lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_map, b=oii_map, 
                                            a_err=nii_err, 
                                            b_err=oii_err)
    
    metal_i, metal_err_nodust = calculate_z_and_error(x=lognii_oii, 
                                               x_err=lognii_oii_err, 
                                               y=-3.17)
    
    #print('no dust logniii_oii:')
    #print(lognii_oii[radius_kpc<3])
    #print('no dust logniii_oii_err:')
    #print(lognii_oii_err[radius_kpc<3])
    
    
    #print('no dust metal:')
    #print(metal_i[radius_kpc<3])
    #print('no dust metal err:')
    #print(metal_err_nodust[radius_kpc<3])    
    
    ax[2].scatter(radius_kpc, metal_map_n2o2_nodustc)
    ax[2].errorbar(x=radius_kpc.ravel(),y=metal_map_n2o2_nodustc.ravel(),yerr=metal_err_nodust.ravel(),fmt='none')
    ax[2].axvline(x=re_kpc,ymin=0,ymax=1,
               linestyle='--',c='g')
    ax[2].axhline(y=9.23,xmin=0,xmax=1,linestyle='--',c='r')
    ax[2].axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='r')
    ax[2].set_xlabel('radius [kpc]')
    ax[2].set_ylim(7.0,9.3)
    ax[2].set_xlim(left=0)
    ax[2].set_title('metal (n2o2) w/o dust c')
    
    
    ##############################
    # metal n2o2 as func of radius, with gist dust correction and err calculation
    
    # NII correct
    nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_map,
                                                     flux_obs_err=nii_err,
                                                     wave=emi_wave.nii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z,
                                dust_model=dust_model)
    
    # OII correct 
    
    oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_map,
                                                     flux_obs_err=oii_err,
                                                     wave=emi_wave.oii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z,
                                dust_model=dust_model)
    
    logR_map = np.log10(nii_corr/oii_corr)
    metal_map_n2o2_gistdust = metal_k19(logR_map)
    
    #if nii_sn_cut:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3)
    #else:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map_n2o2_gistdust[mask_region] = np.nan
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    
    # nii_corr/oii_corr with error
    lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                            a_err=nii_err_corr, 
                                            b_err=oii_err_corr)
    
    metal_i, metal_err_withdust = calculate_z_and_error(x=lognii_oii, 
                                               x_err=lognii_oii_err, 
                                               y=-3.17)
    #print('gist balmer decre:')
    #print((ha_gist/hb_gist)[radius_kpc<3])
    #print('gist dust nii:')
    #print(nii_corr[radius_kpc<3])
    #print('gist dust nii err:')
    #print(nii_err_corr[radius_kpc<3])
    #print('gist dust oii:')
    #print(oii_corr[radius_kpc<3])
    #print('gist dust oii err:')
    #print(oii_err_corr[radius_kpc<3])
    
    #print('gist dust logniii_oii:')
    #print(lognii_oii[radius_kpc<3])
    #print('gist dust logniii_oii_err:')
    #print(lognii_oii_err[radius_kpc<3])
    
    #print('gist dust metal:')
    #print(metal_i[radius_kpc<3])
    #print('gist dust metal err:')
    #print(metal_err_withdust[radius_kpc<3]) 
    
    ax[3].scatter(radius_kpc, metal_map_n2o2_gistdust)
    ax[3].errorbar(x=radius_kpc.ravel(),y=metal_map_n2o2_gistdust.ravel(),yerr=metal_err_withdust.ravel(),fmt='none')
    ax[3].axvline(x=re_kpc,ymin=0,ymax=1,
               linestyle='--',c='g')
    ax[3].axhline(y=9.23,xmin=0,xmax=1,linestyle='--',c='r')
    ax[3].axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='r')
    ax[3].set_xlabel('radius [kpc]')
    ax[3].set_ylim(7.0,9.3)
    ax[3].set_xlim(left=0)
    ax[3].set_title('metal (n2o2) gist dust c')
    
    ##############################
    # metal gradient - n2ha
    
    if model_flux_sn:
        if nii_sn_cut is not None:
            query = (ha_sn_model>=3) & (nii_sn_model>=nii_sn_cut)
        else:
            query = (ha_sn_model>=3) # gist ha for dust corr, so gist ha sn
    else:
        if nii_sn_cut is not None:
            query = (ha_sn>=3)  & (nii_sn>=nii_sn_cut)
        else:
            query = (ha_sn>=3)
    
    
    metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn_model,
                                   nii_sn=nii_sn_model)
    
    
    mask_region = ~query
    
    metal_map_n2ha[mask_region] = np.nan
    
    #else:
    #    metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn,
    #                                   nii_sn=nii_sn)
    
    #nii_err_map = nii_map/nii_sn
    #ha_err_map = ha_map/ha_sn
    
    lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_map, b=ha_map, 
                                            a_err=nii_err, 
                                            b_err=ha_err)
    
    metal_i, metal_err_n2ha = calculate_z_and_error(x=lognii_ha, 
                                               x_err=lognii_ha_err, 
                                               y=-3.17,R='N2Ha')
    
    ax[4].scatter(radius_kpc, metal_map_n2ha)
    ax[4].errorbar(x=radius_kpc.ravel(),y=metal_map_n2ha.ravel(),yerr=metal_err_n2ha.ravel(),fmt='none')
    ax[4].axvline(x=re_kpc,ymin=0,ymax=1,
            linestyle='--',c='g')
    #ax[4].axhline(y=8.53,xmin=0,xmax=1,linestyle='--',c='r')
    #ax[4].axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='r')
    ax[4].axhline(y=7.17,xmin=0,xmax=1,linestyle='--',c='r')
    ax[4].axhline(y=8.86,xmin=0,xmax=1,linestyle='--',c='r')
    
    ax[4].set_xlabel('radius [kpc]')
    ax[4].set_ylim(7.0,9.3)
    ax[4].set_xlim(left=0)
    ax[4].set_title('metal (n2ha)')
    
    #### get the length for different diagnostics and find the min value
    
    
    n2o2_wodustc_len = axplot_bin_metal_with_err(ax=ax[5],metal_map=metal_map_n2o2_nodustc,
                              metal_err_map=metal_err_nodust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 w/o dust c',R='N2O2',get_arraylenth=True,
                     radius_fill_perc=radius_fill_perc,bin_size=bin_size)
    
    n2o2_dustc_len= axplot_bin_metal_with_err(ax=ax[6],metal_map=metal_map_n2o2_gistdust,
                              metal_err_map=metal_err_withdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 with gist dust c',R='N2O2',get_arraylenth=True,
                     radius_fill_perc=radius_fill_perc,bin_size=bin_size)
    
    n2ha_len = axplot_bin_metal_with_err(ax=ax[7],metal_map=metal_map_n2ha,
                              metal_err_map=metal_err_n2ha,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2ha',R='N2Ha',get_arraylenth=True,
                     radius_fill_perc=radius_fill_perc,bin_size=bin_size)
    
    r_min_set = np.min([n2o2_wodustc_len,n2o2_dustc_len,n2ha_len])
    
    if r_min_set_force is not None:
        r_min_set = r_min_set_force
    
    
    print(n2o2_wodustc_len,n2o2_dustc_len,n2ha_len)
    
    ##############################
    # bin metal w/o dust correction
    
    all_para = axplot_bin_metal_with_err(ax=ax[5],metal_map=metal_map_n2o2_nodustc,
                              metal_err_map=metal_err_nodust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 w/o dust c',R='N2O2',overplotax=ax[2],
                     radius_fill_perc=radius_fill_perc,r_min_set=r_min_set,
                     bin_size=bin_size)
    
    popt_mean, pcov_mean,popt_haweight, pcov_haweight,popt_inv, pcov_inv = all_para
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    a_haweight, b_haweight = popt_haweight
    a_haweight_err, b_haweight_err = np.sqrt(np.diag(pcov_haweight))
    a_inv, b_inv = popt_inv
    a_inv_err, b_inv_err = np.sqrt(np.diag(pcov_inv))
    
    
    df = pd.DataFrame({'MAGPIID':[int(plot_title)]})
    df['n2o2_wodustc_a_mean'] = a_mean
    df['n2o2_wodustc_a_mean_err'] = a_mean_err
    df['n2o2_wodustc_b_mean'] = b_mean
    df['n2o2_wodustc_b_mean_err'] = b_mean_err
    
    df['n2o2_wodustc_a_haweight'] = a_haweight
    df['n2o2_wodustc_a_haweighterr'] = a_haweight_err
    df['n2o2_wodustc_b_haweight'] = b_haweight
    df['n2o2_wodustc_b_haweight_err'] = b_haweight_err
    
    df['n2o2_wodustc_a_inv'] = a_inv
    df['n2o2_wodustc_a_inv_err'] = a_inv_err
    df['n2o2_wodustc_b_inv'] = b_inv
    df['n2o2_wodustc_b_inv_err'] = b_inv_err
    
    
    n2o2_wodustc_len = axplot_bin_metal_with_err(ax=ax[5],metal_map=metal_map_n2o2_nodustc,
                              metal_err_map=metal_err_nodust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 w/o dust c',R='N2O2',get_arraylenth=True,
                     radius_fill_perc=radius_fill_perc,
                     bin_size=bin_size)
    df['n2o2_wodustc_len'] = r_min_set
    
    
    ##############################
    # bin metal with dust corr
    
    all_para = axplot_bin_metal_with_err(ax=ax[6],metal_map=metal_map_n2o2_gistdust,
                              metal_err_map=metal_err_withdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 with gist dust c',R='N2O2',overplotax=ax[3],
                     radius_fill_perc=radius_fill_perc,r_min_set=r_min_set,
                     bin_size=bin_size)
    
    
    
    
    popt_mean, pcov_mean,popt_haweight, pcov_haweight,popt_inv, pcov_inv = all_para
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    a_haweight, b_haweight = popt_haweight
    a_haweight_err, b_haweight_err = np.sqrt(np.diag(pcov_haweight))
    a_inv, b_inv = popt_inv
    a_inv_err, b_inv_err = np.sqrt(np.diag(pcov_inv))
    
    
    df['n2o2_dustc_a_mean'] = a_mean
    df['n2o2_dustc_a_mean_err'] = a_mean_err
    df['n2o2_dustc_b_mean'] = b_mean
    df['n2o2_dustc_b_mean_err'] = b_mean_err
    
    df['n2o2_dustc_a_haweight'] = a_haweight
    df['n2o2_dustc_a_haweighterr'] = a_haweight_err
    df['n2o2_dustc_b_haweight'] = b_haweight
    df['n2o2_dustc_b_haweight_err'] = b_haweight_err
    
    df['n2o2_dustc_a_inv'] = a_inv
    df['n2o2_dustc_a_inv_err'] = a_inv_err
    df['n2o2_dustc_b_inv'] = b_inv
    df['n2o2_dustc_b_inv_err'] = b_inv_err
    
    n2o2_dustc_len= axplot_bin_metal_with_err(ax=ax[6],metal_map=metal_map_n2o2_gistdust,
                              metal_err_map=metal_err_withdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 with gist dust c',R='N2O2',get_arraylenth=True,
                     radius_fill_perc=radius_fill_perc,
                     bin_size=bin_size)
    df['n2o2_dustc_len'] = r_min_set
    ##############################
    # bin n2ha metal
    
    all_para = axplot_bin_metal_with_err(ax=ax[7],metal_map=metal_map_n2ha,
                              metal_err_map=metal_err_n2ha,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2ha',R='N2Ha',overplotax=ax[4],
                     radius_fill_perc=radius_fill_perc,r_min_set=r_min_set,
                     bin_size=bin_size)
    
    popt_mean, pcov_mean,popt_haweight, pcov_haweight,popt_inv, pcov_inv = all_para
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    a_haweight, b_haweight = popt_haweight
    a_haweight_err, b_haweight_err = np.sqrt(np.diag(pcov_haweight))
    a_inv, b_inv = popt_inv
    a_inv_err, b_inv_err = np.sqrt(np.diag(pcov_inv))
    
    df['n2ha_a_mean'] = a_mean
    df['n2ha_a_mean_err'] = a_mean_err
    df['n2ha_b_mean'] = b_mean
    df['n2ha_b_mean_err'] = b_mean_err
    
    df['n2ha_a_haweight'] = a_haweight
    df['n2ha_a_haweighterr'] = a_haweight_err
    df['n2ha_b_haweight'] = b_haweight
    df['n2ha_b_haweight_err'] = b_haweight_err
    
    df['n2ha_a_inv'] = a_inv
    df['n2ha_a_inv_err'] = a_inv_err
    df['n2ha_b_inv'] = b_inv
    df['n2ha_b_inv_err'] = b_inv_err
    
    n2ha_len = axplot_bin_metal_with_err(ax=ax[7],metal_map=metal_map_n2ha,
                              metal_err_map=metal_err_n2ha,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2ha',R='N2Ha',get_arraylenth=True,
                     radius_fill_perc=radius_fill_perc,
                     bin_size=bin_size)
    df['n2ha_len'] = r_min_set
    
    fig.suptitle(plot_title)
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    
    plt.show()
    
    if savecsv:
        df.to_csv(csvpath+plot_title+'.csv')
    
    return 0


def axplot_bin_metal(ax,metal_map,ha_map,radius_map,title,re_kpc,R='N2O2',
                     legend=False):
    '''
    note that this version have an error of the radius value, the radius should
    be the centre of each bin, but this start from the left end of the bin.
    I don't fix it because I use the new version of this function,
    see function axplot_bin_metal_with_err
    '''
    
    metal_mean, metal_haweight,metal_mean_stdonmean,radius_array = get_bin_metal(
        metal_map=metal_map, ha_map=ha_map, radius_map=radius_map)
    
    # get the max radius where data available
    #radius_max_kpc = np.nanmax(radius_map[~np.isnan(metal_map)])
    #radius_array = np.arange(int(radius_max_kpc*0.9))
    
    popt_mean, pcov_mean = curve_fit(linear_func, radius_array, metal_mean)
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    popt_haweight, pcov_haweight = curve_fit(linear_func, radius_array, metal_haweight)
    a_haweight, b_haweight = popt_haweight
    a_haweight_err, b_haweight_err = np.sqrt(np.diag(pcov_haweight))
    
    
    ax.errorbar(radius_array, metal_mean,
                    yerr=metal_mean_stdonmean,fmt='none',c='b')
    ax.scatter(radius_array,metal_mean,
                   c='b',label='mean')
    ax.scatter(radius_array,metal_haweight,label='ha weight',
                   c='r')
    
    ax.plot(radius_array, linear_func(radius_array, *popt_mean), 
                color='b', linestyle='--')
    ax.plot(radius_array, linear_func(radius_array, *popt_haweight), 
                color='r', linestyle='--')
    ax.text(0.5, 0.1, 'y=({:.2f}$\pm${:.2f})x + ({:.2f}$\pm${:.2f})'.format(a_mean,a_mean_err,b_mean,b_mean_err), 
            horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,c='b',fontsize=17)
    
    ax.text(0.5, 0.2, 'y=({:.2f}$\pm${:.2f})x + ({:.2f}$\pm${:.2f})'.format(a_haweight,a_haweight_err,b_haweight,b_haweight_err), 
            horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes,c='r',fontsize=17)
    if legend:
        ax.legend()
    ax.set_xlabel('radius [kpc]')
    ax.set_ylabel('log10(O/H)+12')
    ax.set_title(title)
    ax.set_ylim(7.0,9.3)
    ax.axvline(x=re_kpc,ymin=0,ymax=1,
            linestyle='--',c='g')
    if R=='N2O2':
        ax.axhline(y=9.23,xmin=0,xmax=1,linestyle='--',c='orange')
        ax.axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='orange')
    elif R=='N2Ha':
        #ax.axhline(y=8.53,xmin=0,xmax=1,linestyle='--',c='orange')
        #ax.axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='orange')
        ax.axhline(y=7.17,xmin=0,xmax=1,linestyle='--',c='orange')
        ax.axhline(y=8.86,xmin=0,xmax=1,linestyle='--',c='orange')



def linear_func(x, a, b):
    return a * x + b


def axplot_bin_metal_with_err(ax,metal_map,metal_err_map,ha_map,radius_map,
                              title,re_kpc,R='N2O2',
                     legend=False,get_arraylenth=False,overplotax=None,
                     radius_fill_perc=0.5,plot_indivspax=False,fit_param=True,
                     r_min_set=None,bin_size=1):
    '''
    
    overplotax: the ax of plot which I want to plot the metal gradient fitting
                on top
    
    '''
    
    if plot_indivspax:
        ax.scatter(radius_map.ravel(), metal_map.ravel(),c='grey',zorder=1)
        ax.errorbar(x=radius_map.ravel(),y=metal_map.ravel(),
                    yerr=metal_err_map.ravel(),fmt='none',c='grey',zorder=1)
    
    metal_mean, metal_mean_err, radius_array = get_bin_metal_with_err(
        metal_map=metal_map,metal_err_map=metal_err_map,ha_map=ha_map,
        radius_map=radius_map, radius_fill_perc=radius_fill_perc,weight_mode='no',
        bin_size=bin_size)
    #print('unweighted: mean_array, err_aray')
    #print(metal_mean)
    #print(metal_mean_err)
    
    metal_inv_var, metal_inv_var_err, radius_array = get_bin_metal_with_err(
        metal_map=metal_map,metal_err_map=metal_err_map,ha_map=ha_map,
        radius_map=radius_map, radius_fill_perc=radius_fill_perc,
        weight_mode='inv_var',bin_size=bin_size)
    #print('inverse variance weighted')
    #print(metal_inv_var)
    #print(metal_inv_var_err)
    
    metal_ha_ave, metal_ha_ave_err, radius_array = get_bin_metal_with_err(
        metal_map=metal_map,metal_err_map=metal_err_map,ha_map=ha_map,
        radius_map=radius_map, radius_fill_perc=radius_fill_perc,
        weight_mode='ha',bin_size=bin_size)
    #print('ha weighted')
    #print(metal_ha_ave)
    #rint(metal_ha_ave_err)
    
    if get_arraylenth:
        
        # Find the index of the last non-NaN value
        last_valid_index = np.where(~np.isnan(metal_mean))[0][-1]
        
        # length is index + 1
        # as r_min_set use get_arraylenth, so to 
        # include the last not nan value, need to +1
        return last_valid_index + 1 
    
    radius_array_to_fit = radius_array[:r_min_set]
    metal_mean_to_fit = metal_mean[:r_min_set]
    metal_mean_err_to_fit = metal_mean_err[:r_min_set]
    query_1 = np.isnan(metal_mean_to_fit) | np.isnan(metal_mean_err_to_fit)
    
    
    
    popt_mean, pcov_mean = curve_fit(linear_func, radius_array_to_fit[~query_1], 
                                     metal_mean_to_fit[~query_1],
                                     sigma=metal_mean_err_to_fit[~query_1])
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    metal_ha_ave_to_fit = metal_ha_ave[:r_min_set]
    metal_ha_ave_err_to_fit = metal_ha_ave_err[:r_min_set]
    query_2 = np.isnan(metal_ha_ave_to_fit) | np.isnan(metal_ha_ave_err_to_fit)
    
    
    popt_haweight, pcov_haweight = curve_fit(linear_func, radius_array_to_fit[~query_2], 
                                             metal_ha_ave_to_fit[~query_2],
                                             sigma=metal_ha_ave_err_to_fit[~query_2])
    a_haweight, b_haweight = popt_haweight
    a_haweight_err, b_haweight_err = np.sqrt(np.diag(pcov_haweight))
    
    metal_inv_var_to_fit = metal_inv_var[:r_min_set]
    metal_inv_var_err_to_fit = metal_inv_var_err[:r_min_set]
    query_3 = np.isnan(metal_inv_var_to_fit) | np.isnan(metal_inv_var_err_to_fit)
    
    popt_inv, pcov_inv = curve_fit(linear_func, radius_array_to_fit[~query_3],
                                   metal_inv_var_to_fit[~query_3],
                                   sigma=metal_inv_var_err_to_fit[~query_3])
    
    a_inv, b_inv = popt_inv
    a_inv_err, b_inv_err = np.sqrt(np.diag(pcov_inv))
    
    best_fit_color = 'chocolate'
    
    
    ax.errorbar(radius_array[:r_min_set], metal_mean[:r_min_set],
                    yerr=metal_mean_err[:r_min_set],fmt='none',c='b',zorder=5)
    ax.scatter(radius_array[:r_min_set],metal_mean[:r_min_set],
                   c='b',zorder=5)
    
    ax.errorbar(radius_array[:r_min_set]+0.05,metal_ha_ave[:r_min_set],yerr=metal_ha_ave_err[:r_min_set],fmt='none',
                c='r',zorder=5)
    ax.scatter(radius_array[:r_min_set]+0.05,metal_ha_ave[:r_min_set],
                   c='r',zorder=5)
    
    ax.errorbar(radius_array[:r_min_set]+0.1,metal_inv_var[:r_min_set],yerr=metal_inv_var_err[:r_min_set],fmt='none',
                c=best_fit_color ,zorder=5)
    ax.scatter(radius_array[:r_min_set]+0.1,metal_inv_var[:r_min_set],
                   c=best_fit_color ,zorder=5)
    
    # the radius_array is like [0.5,1.5,2.5]
    # but the plot range should be [0,1,2,3]
    plot_radius_array = radius_array[:r_min_set] - 0.5*bin_size
    plot_radius_array = np.concatenate((plot_radius_array,
                                        [plot_radius_array[-1]+1*bin_size]))
    
    
    ax.plot(plot_radius_array, linear_func(plot_radius_array, *popt_mean), 
                color='b', linestyle='--',zorder=6,label='unweighted')
    ax.plot(plot_radius_array, linear_func(plot_radius_array, *popt_haweight), 
                color='r', linestyle='-.',zorder=6,label='H$\\alpha$ weighted')
    ax.plot(plot_radius_array, linear_func(plot_radius_array, *popt_inv), 
                color=best_fit_color , linestyle='-',zorder=6,label='inverse variance weighted')
    
    if overplotax is not None:
        overplotax.plot(plot_radius_array, linear_func(plot_radius_array, *popt_mean), 
                    color='b', linestyle='--')
        overplotax.plot(plot_radius_array, linear_func(plot_radius_array, *popt_haweight), 
                    color='r', linestyle='-.')
        overplotax.plot(plot_radius_array, linear_func(plot_radius_array, *popt_inv), 
                    color=best_fit_color , linestyle='-')
        
    if fit_param:
        
        #ax.text(0.5, 0.1, 'y=({:.2f}$\pm${:.2f})x + ({:.2f}$\pm${:.2f})'.format(a_mean,a_mean_err,b_mean,b_mean_err), 
        #        horizontalalignment='center',
        # verticalalignment='center', transform=ax.transAxes,c='b',fontsize=17)
        
        #ax.text(0.5, 0.2, 'y=({:.2f}$\pm${:.2f})x + ({:.2f}$\pm${:.2f})'.format(a_haweight,a_haweight_err,b_haweight,b_haweight_err), 
        #        horizontalalignment='center',
        # verticalalignment='center', transform=ax.transAxes,c='r',fontsize=17)
        
        ax.text(0.5, 0.35, 'y=({:.3f}$\pm${:.3f})x + ({:.3f}$\pm${:.3f})'.format(a_inv,a_inv_err,b_inv,b_inv_err), 
                horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes,c=best_fit_color ,fontsize=14,
             bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'))
    
    
    
    if legend:
        ax.legend(prop={'size': 14})
    ax.set_xlabel('radius [kpc]')
    ax.set_ylabel('12+log(O/H)')
    ax.set_title(title)
    ax.set_ylim(7.0,9.3)
    ax.set_xlim(left=0)
    ax.axvline(x=re_kpc,ymin=0,ymax=1,
            linestyle='--',c='darkseagreen')
    if R=='N2O2':
        ax.axhline(y=9.23,xmin=0,xmax=1,linestyle='--',c='silver')
        ax.axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='silver')
    elif R=='N2Ha':
        #ax.axhline(y=8.53,xmin=0,xmax=1,linestyle='--',c='orange')
        #ax.axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='orange')
        ax.axhline(y=7.17,xmin=0,xmax=1,linestyle='--',c='silver')
        ax.axhline(y=8.86,xmin=0,xmax=1,linestyle='--',c='silver')
    
    return popt_mean, pcov_mean,popt_haweight, pcov_haweight,popt_inv, pcov_inv


def get_bin_metal_with_err(metal_map,metal_err_map,ha_map,radius_map,
                           radius_fill_perc=0.5,weight_mode='no',
                           bin_size=1):
    '''
    bin metal 
    
    radius_fill_perc: radius with > ?? percent of valid data within that bin.
    
    three versions of average and error:
        
        1. average. [no]
        2. inverse variance weighted average [inv_var]
        3. halpha flux weighted average [ha]
    
    4 April 2025 update:
        1. when radius_fill_perc is None, include up to radius have at least 
           one spaxel in that bin
    
    
    '''
    
    # get the max radius where data available
    radius_max_kpc = np.nanmax(radius_map[~np.isnan(metal_map)])
    r_set = int(radius_max_kpc/bin_size)+1
    if r_set < 2:
        r_set = 2
    radius_array = np.arange(r_set)
    
    
    metal_mean = []
    metal_ha_ave = []
    metal_inv_var = []
    
    metal_mean_err = []
    metal_ha_ave_err = []
    metal_inv_var_err = []
    
    radius_array_new = []
    for r in radius_array:
        query = (radius_map>=r*bin_size)&(radius_map<(r+1)*bin_size)
        query_all = (radius_map>=r*bin_size)&(radius_map<(r+1)*bin_size)&(~np.isnan(metal_map))&(~np.isnan(metal_err_map))
        if radius_fill_perc is None:
            if len(metal_map[query_all]) <1:
                metal_mean.append(np.nan)
                metal_ha_ave.append(np.nan)
                metal_inv_var.append(np.nan)
                
                metal_mean_err.append(np.nan)
                metal_ha_ave_err.append(np.nan)
                metal_inv_var_err.append(np.nan)
                radius_array_new.append(r)
                
                continue
        else:
            if len(metal_map[query_all])/len(metal_map[query])<radius_fill_perc:
                metal_mean.append(np.nan)
                metal_ha_ave.append(np.nan)
                metal_inv_var.append(np.nan)
                
                metal_mean_err.append(np.nan)
                metal_ha_ave_err.append(np.nan)
                metal_inv_var_err.append(np.nan)
                radius_array_new.append(r)
                
                continue
        
        
        
        # average 1
        metal_query = metal_map[query_all]
        ha_query = ha_map[query_all]
        metal_err_query = metal_err_map[query_all]
        avg_err = np.sqrt(np.nansum(metal_err_query**2)) / len(metal_err_query)
        
        
        
        
        
        # average 2
        weights = 1 / metal_err_query**2
        ave_inv_var = np.sum(metal_query * weights) / np.sum(weights)
        ave_inv_var_err = np.sqrt(1 / np.sum(weights))
        
        # don't save if error too large or metallicity below the valid range
        if ave_inv_var<7.63 or ave_inv_var_err>0.5:
            metal_mean.append(np.nan)
            metal_ha_ave.append(np.nan)
            metal_inv_var.append(np.nan)
            
            metal_mean_err.append(np.nan)
            metal_ha_ave_err.append(np.nan)
            metal_inv_var_err.append(np.nan)
            radius_array_new.append(r)
            
            continue
        
        
        
        # average 3
        ave_ha_weig = np.sum(ha_query * metal_query) / np.sum(ha_query)
        ave_ha_weig_err = np.sqrt(np.sum((ha_query**2) * (metal_err_query**2)) / (np.sum(ha_query)**2))
        
        
        metal_mean.append(np.nanmean(metal_query))
        metal_mean_err.append(avg_err)
        
        metal_inv_var.append(ave_inv_var)
        metal_inv_var_err.append(ave_inv_var_err)
        
        
        metal_ha_ave.append(ave_ha_weig)
        metal_ha_ave_err.append(ave_ha_weig_err)
        
        radius_array_new.append(r)
    
    metal_mean = np.array(metal_mean)
    metal_ha_ave = np.array(metal_ha_ave)
    metal_inv_var = np.array(metal_inv_var)
    
    metal_mean_err = np.array(metal_mean_err)
    metal_ha_ave_err = np.array(metal_ha_ave_err)
    metal_inv_var_err = np.array(metal_inv_var_err)
    
    # the radius should be the center of bin, which is 0.5, 1.5, 2.5 ...
    radius_array_new = (np.array(radius_array_new) + 0.5) * bin_size
    
    if weight_mode=='no':
        return metal_mean, metal_mean_err, radius_array_new
    if weight_mode=='inv_var':
        return metal_inv_var, metal_inv_var_err, radius_array_new
    if weight_mode=='ha':
        return metal_ha_ave, metal_ha_ave_err, radius_array_new
    
    
    
    
    
    
    
    


def get_bin_metal(metal_map,ha_map,radius_map,radius_perc=0.9):
    '''
    
    old version - the way calculate error is not right....
    
    radius_map: radius map in kpc
    radius_perc: up to ?? radius of max radius with valid data
    return: 
        metal_mean: mean of metal as function of radius
        metal_haweight: ha flux weighted metal
        metal_mean_std_on_mean: as the name
        radius_array_new: a radius array that match the length of metal_mean array
    
    '''
    # get the max radius where data available
    radius_max_kpc = np.nanmax(radius_map[~np.isnan(metal_map)])
    r_set = int(radius_max_kpc*radius_perc)
    if r_set < 2:
        r_set = 2
    radius_array = np.arange(r_set)
    
    metal_mean = []
    metal_haweight = []
    metal_mean_std_on_mean = []
    radius_array_new = []
    for r in radius_array:
        query = (radius_map>=r)&(radius_map<r+1)
        if len(metal_map[query&(~np.isnan(metal_map))])==0:
            break
        radius_array_new.append(r)
        metal_mean.append(np.nanmean(metal_map[query]))
        N = len(metal_map[query&(~np.isnan(metal_map))])
        metal_mean_std_on_mean.append(np.nanstd(metal_map[query])/np.sqrt(N))
        ma = np.ma.MaskedArray(metal_map, 
                               mask=np.isnan(metal_map))
        
        metal_haweight.append(np.ma.average(ma[query], weights=ha_map[query]))
        
    return np.array(metal_mean), np.array(metal_haweight), np.array(metal_mean_std_on_mean),np.array(radius_array_new)
    
    
    
def log10_ratio_with_error(a, b, a_err, b_err):
    # Calculate the value of log10(a/b)
    log10_ratio = np.log10(a / b)
    
    # Calculate the error using error propagation
    ln10 = np.log(10)
    log10_ratio_err = np.sqrt((a_err / (a * ln10))**2 + (b_err / (b * ln10))**2)
    
    return log10_ratio, log10_ratio_err
    

    
def metal_k19(logR,logU=-3.17,R='N2O2'):
    '''logR to log(O/H)+12 from Kewley+2019'''
    # x here is log R, could be log(NII/OII) or log(NII/Ha)
    # z here is log(O/H) + 12
    # y here is logU
    y = logU
    x = logR
    
    if R=='N2O2':
        z = 9.4772 + 1.1797 * x + 0.5085* y + 0.6879*x*y + 0.2807*x**2 \
            + 0.1612*y**2 + 0.1187*x*y**2 + 0.1200*y*x**2 + 0.2293*x**3 \
                + 0.0164*y**3
    
    elif R=='N2Ha_k19':
        z = 10.526 + 1.9958 * x - 0.6741* y + 0.2892*x*y + 0.5712*x**2 \
            - 0.6597*y**2 + 0.0101*x*y**2 + 0.0800*y*x**2 + 0.0782*x**3 \
                - 0.0982*y**3
    elif R=='N2Ha':
        # Pettini 2004
        # use this version as it has larger valid range
        
        z = 9.37 + 2.03 * x + 1.26 * x**2 + 0.32* x**3
    
    
    return z


    
def calculate_z_and_error(x, x_err, y=-3.17,R='N2O2'):
    
    # x here is log(NII/OII)
    # z here is log(O/H) + 12
    # y here is logU
    
    if R=='N2O2':
    
        # Calculate z
        z = (9.4772 + 1.1797 * x + 0.5085 * y + 0.6879 * x * y +
             0.2807 * x**2 + 0.1612 * y**2 + 0.1187 * x * y**2 +
             0.1200 * y * x**2 + 0.2293 * x**3 + 0.0164 * y**3)
        
        # Calculate the partial derivative of z with respect to x
        dz_dx = (1.1797 + 0.6879 * y + 2 * 0.2807 * x + 0.1187 * y**2 +
                 2 * 0.1200 * y * x + 3 * 0.2293 * x**2)
        
        # Calculate the error in z
        z_err = np.abs(dz_dx) * x_err
        
        # for metallicity outside the valid range, set nan
        #query = (z>9.23) | (z<7.63)
        
        #z[query] = np.nan
        #z_err[query] = np.nan
        
        
        
        
    elif R=='N2Ha_k19':
        # Kewley 2019, samll valid range.
        
        
        # Calculate z
        z = (10.526 + 1.9958 * x - 0.6741 * y + 0.2892 * x * y + 0.5712 * x**2 -
             0.6597 * y**2 + 0.0101 * x * y**2 + 0.0800 * y * x**2 +
             0.0782 * x**3 - 0.0982 * y**3)
        
        # Calculate the partial derivative of z with respect to x
        dz_dx = (1.9958 + 0.2892 * y + 2 * 0.5712 * x + 0.0101 * y**2 +
                 2 * 0.0800 * y * x + 3 * 0.0782 * x**2)
        
        # Calculate the error in z
        z_err = np.abs(dz_dx) * x_err
        
        # for metallicity outside the valid range, set nan
        #query = (z>8.53) | (z<7.63)
        
        #z[query] = np.nan
        #z_err[query] = np.nan
    
    elif R=='N2Ha':
        # Pettini 2004
        # use this version as it has larger valid range
        
        z = 9.37 + 2.03 * x + 1.26 * x**2 + 0.32* x**3
        
        dz_dx = 2.03 + 1.26*2*x + 0.32*3*x**2
        z_err = np.abs(dz_dx) * x_err
    
    
    return z, z_err


def dust_correction_2d(flux_map, E_B_V_map, wavelength):
    flux_map_correct = np.full_like(flux_map,np.nan)
    
    
    ny, nx = flux_map.shape
    for i in range(ny):
        for j in range(nx):
            flux_map_correct[i,j] = dust_correction(flux=flux_map[i,j], 
                                                    E_B_V=E_B_V_map[i,j],
                                                    wavelength=wavelength)
    
    return flux_map_correct
    
    

def dust_correction(flux,E_B_V,wavelength):
    
    flux = flux * u.erg/(u.s * u.cm**2 * u.AA)
    wavelength = wavelength * u.AA
    
    ext = F19(Rv=3.1)
    flux_correct = flux / ext.extinguish(wavelength, 
                                            Ebv=E_B_V)
    flux_correct = flux_correct.value
    return flux_correct




def SFR_gradient_sumflux(ha_map,ha_err,ha_sn,hb_sn,ha_gist_err,hb_gist_err,
                         pa,x_cen_pix,y_cen_pix,ellip,
                         foreground_E_B_V,z,gradient_radius,
                         re_kpc=None,savepath=None,r_option='kpc',
                         savecsv=False,csvpath=None,magpiid=None):
    
    '''
    ha_map: ha flux map from b3d
    ha_err: ha flux err from b3d
    ha_sn: ha sn from gist
    ha_gist_err: ha flux err from gist
    gradient_radius: int, the radius in kpc that we measure metal gradient
    magpiid: str of magpiid
    -----
    
    no, it's wrong to use sumflux method to calculate SFR gradient...
    
    can't even use SFR per pixel to get gradient, as the inclination influence
    the SFR per pixel, it's not SFR per unit area...
    
    need to modify this function.... 
    
    
    '''
    
    # balmer decrement of GIST
    ha_gist = ha_sn * ha_gist_err
    hb_gist = hb_sn * hb_gist_err
    
    dist_arr = ellip_distarr(size=ha_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    ##### SFR with dust correction
    r_set = gradient_radius
    radius_array_new = []
    SFR_array = []
    SFR_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        
        radius_array_new.append(r)
        ha_sum = np.nansum(ha_map[query_all])
        ha_gist_sum = np.nansum(ha_gist[query_all]) # use gist to calculate balmer
        hb_gist_sum = np.nansum(hb_gist[query_all]) # use gist to calculate balmer
        
        ha_err_sum = np.sqrt(np.nansum((ha_err[query_all])**2))
        ha_gist_err_sum = np.sqrt(np.nansum((ha_gist_err[query_all])**2)) # use gist err, as use gist flux  
        hb_gist_err_sum = np.sqrt(np.nansum((hb_gist_err[query_all])**2))
        
        
        ###
        ha_corr, ha_err_corr = intrinsic_flux_with_err(flux_obs=ha_sum, 
                                                       flux_obs_err=ha_err_sum, 
                                                       wave=emi_wave.ha_wave, 
                                                       ha=ha_gist_sum, 
                                                       hb=hb_gist_sum, 
                                                       ha_err=ha_gist_err_sum, 
                                                       hb_err=hb_gist_err_sum, 
                                                       foreground_E_B_V=foreground_E_B_V, 
                                                       z=z)
        
        
        ha_lumi = flux_to_luminosity(ha_corr, z)
        ha_lumi_err = flux_to_luminosity(ha_err_corr, z)
        
        SFR = luminosity_to_SFR(ha_lumi)
        SFR_err = luminosity_to_SFR(ha_lumi_err)
        
        
        
        
        
        
        SFR_array.append(SFR)
        SFR_err_array.append(SFR_err)
    
    #### gradient SFR with dust corr
    fit_radius_array = radius_array + 0.5
    popt_mean, pcov_mean = curve_fit(linear_func, fit_radius_array, SFR_array,
                                     sigma=SFR_err_array)
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    df = pd.DataFrame({'MAGPIID':[int(magpiid)]})
    
    df['SFR_gradient'] = a_mean
    df['SFR_gradient_err'] = a_mean_err
    df['SFR_centre'] = b_mean
    df['SFR_centre_err'] = b_mean_err
    
    if savecsv:
        df.to_csv(csvpath+magpiid+'.csv')
    
    
    return 0


def flux_to_luminosity(flux_map,z):
    z = z * cu.redshift
    from astropy.cosmology import LambdaCDM
    lcdm = LambdaCDM(70,0.3,0.7)
    # the unit of flux is 10^-20 erg/s/cm^2
    d = z.to(u.cm, cu.redshift_distance(lcdm, kind="luminosity"))
    #d = lcdm.luminosity_distance(z)
    lumi_map = flux_map * 1e-20 * 4 * np.pi * (d.value)**2
    
    return lumi_map


def luminosity_to_SFR(lumi):
    # from Mun et al. 2024
    SFR = lumi/(1.26 * 1e41 * 1.53)
    
    return SFR
    
def diagnostic_map_sumflux(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,
                                 ha_err,hb_err,oii_err,nii_err,
                                 ha_gist_err,hb_gist_err,oii_gist_err,nii_gist_err,
                                 pa,x_cen_pix,y_cen_pix,ellip,
                                 foreground_E_B_V,
                                 z,re_kpc=None,savepath=None,
                                 r_option='kpc',plot_title=None,
                                 limit_metal_range=False,
                                 savecsv=False,csvpath=None,
                                 nii_sn_cut=None,model_flux_sn=False,
                                 radius_fill_perc=0.5,r_min_set=None):
    
    
    '''
    ha_map: flux map to do metal calculation
    ha_sn: gist SN
    ha_err: flux err used to calculate metal err
    ha_gist_err: flux err from gist output
    
    nii_sn_cut: float, the S/N cut for NII line. If no require, the default is None
    
    r_min_set: default is None. If not, use int, to indicate the radius range to fit
                metallicity gradient. This is for the gist seeing gradient, as 
                if we use gist data, normally it goes further than b3d version
                which may cause problem, therefore i will take the range from 
                b3d output and use it to constrain the range for gist seeing 
                version.
    
    31 March 2025 update:
        1. require sn nii > 3 for sum flux spaxels selection criteria
        2. add hb, oii sn > 3 for sum flux spaxels selection criteria
        3. remove more than 50% spaxel have sn> 3 requirement
    '''
    
    plt.rcParams.update({'font.size': 25})
    
    ha_sn_model = ha_map/ha_err
    hb_sn_model = hb_map/hb_err
    nii_sn_model = nii_map/nii_err
    oii_sn_model = oii_map/oii_err
    
    # balmer decrement of GIST
    ha_gist = ha_sn * ha_gist_err
    hb_gist = hb_sn * hb_gist_err
    nii_gist = nii_sn * nii_gist_err
    oii_gist = oii_sn * oii_gist_err
    
    dist_arr = ellip_distarr(size=ha_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    # get the max radius where ha sn >= 3
    radius_max_kpc = np.nanmax(radius_kpc[ha_sn>=3])
    r_set = int(radius_max_kpc)+1
    if r_set < 2:
        r_set = 2
    radius_array = np.arange(r_set)
    
    ##### n2o2 w/o dust correction
    
    df = pd.DataFrame({'MAGPIID':[int(plot_title)]})
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        if model_flux_sn:
            if nii_sn_cut is not None: # as we use ha gist to do dust corr, so use gist sn for ha and hb
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn_model>=nii_sn_cut)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
        else:
            if nii_sn_cut is not None:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn>=nii_sn_cut)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
        # use radius_fill_perc=0.5 to ensure up to that range have reasonable
        # blob measurement
        # the range for gradient measurement will be decide later (err<0.5dex)
        if radius_fill_perc is None:
            if len(nii_map[query_all]) < 1:
                break
        else:
            if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
                break
        radius_array_new.append(r)
        
        #ha_sum = np.nansum(ha_gist[query_all]) # use gist to calculate balmer
        #hb_sum = np.nansum(hb_gist[query_all]) # use gist to calculate balmer
        oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        #ha_err_sum = np.sqrt(np.nansum((ha_err[query_all])**2))
        #hb_err_sum = np.sqrt(np.nansum((hb_err[query_all])**2))
        oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
    
        logR_map = np.log10(nii_sum/oii_sum)
        metal_map_n2o2_nodustc = metal_k19(logR_map)
        
        lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_sum, b=oii_sum, 
                                                a_err=nii_err_sum, 
                                                b_err=oii_err_sum)
        
        metal_i, metal_err_nodust = calculate_z_and_error(x=lognii_oii, 
                                                   x_err=lognii_oii_err, 
                                                   y=-3.17)
        
        metal_array.append(metal_map_n2o2_nodustc)
        metal_err_array.append(metal_err_nodust)
    
    metal_mean_n2o2wodustc = np.array(metal_array)
    metal_mean_err_n2o2wodustc = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    #print('n2o2 wo dust')
    #print(metal_mean.shape)
    #print(metal_mean_err.shape)
    #print(radius_array.shape)
    
    # center of the bin
    radius_array = radius_array + 0.5
    
    
    
    
    
    ##### n2o2 with dust correction
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        
        #query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        if model_flux_sn:
            if nii_sn_cut is not None: # as we use ha gist to do dust corr, so use gist sn for ha and hb
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn_model>=nii_sn_cut)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
        else:
            if nii_sn_cut is not None:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn>=nii_sn_cut)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
        
        
        if radius_fill_perc is None:
            if len(nii_map[query_all]) < 1:
                break
        else:
            if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
                break
        
        
        
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_gist[query_all]) # use gist to calculate balmer
        hb_sum = np.nansum(hb_gist[query_all]) # use gist to calculate balmer
        oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_gist_err[query_all])**2)) # use gist err, as use gist flux  
        hb_err_sum = np.sqrt(np.nansum((hb_gist_err[query_all])**2))
        oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        ###
        
        nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_sum,
                                                         flux_obs_err=nii_err_sum,
                                                         wave=emi_wave.nii_wave,
                                                         ha=ha_sum,hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        # OII correct 
        
        oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_sum,
                                                         flux_obs_err=oii_err_sum,
                                                         wave=emi_wave.oii_wave,
                                                         ha=ha_sum,
                                                         hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        logR_map = np.log10(nii_corr/oii_corr)
        metal_map_n2o2_gistdust = metal_k19(logR_map)
        
        # nii_corr/oii_corr with error
        lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                                a_err=nii_err_corr, 
                                                b_err=oii_err_corr)
        
        metal_i, metal_err_withdust = calculate_z_and_error(x=lognii_oii, 
                                                   x_err=lognii_oii_err, 
                                                   y=-3.17)
        
        metal_array.append(metal_map_n2o2_gistdust[0])
        metal_err_array.append(metal_err_withdust[0])
    
    
    
    
    
    metal_mean_n2o2_dustc = np.array(metal_array)
    metal_mean_err_n2o2_dustc = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    
    
    #center of the bin
    radius_array = radius_array + 0.5
    
    #print('n2o2 with dust')
    ###print(metal_mean)
    #print(metal_mean_err)
    #print(radius_array)
    #print(metal_mean.shape)
    #print(metal_mean_err.shape)
    #print(radius_array.shape)
    
    
    
    ##### n2ha 
    
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        #query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        if model_flux_sn:
            if nii_sn_cut is not None: 
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn_model>=nii_sn_cut)&(ha_sn_model>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn_model>=3)
        else:
            if nii_sn_cut is not None:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn>=nii_sn_cut)&(ha_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        
        
        
        if radius_fill_perc is None:
            if len(nii_map[query_all]) < 1:
                break
        else:
            if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
                break
        
        
        
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_map[query_all]) # use b3d to calculate metal!!
        #hb_sum = np.nansum(hb_gist[query_all]) 
        #oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_err[query_all])**2))
        #hb_err_sum = np.sqrt(np.nansum((hb_err[query_all])**2))
        #oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_sum, b=ha_sum, 
                                                a_err=nii_err_sum, 
                                                b_err=ha_err_sum)
        
        metal_map_n2ha, metal_err_n2ha = calculate_z_and_error(x=lognii_ha, 
                                                   x_err=lognii_ha_err, 
                                                   y=-3.17,R='N2Ha')
    
        
        
        metal_array.append(metal_map_n2ha)
        metal_err_array.append(metal_err_n2ha)
    
    metal_mean_n2ha = np.array(metal_array)
    metal_mean_err_n2ha = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    
    
    
    # center of the bin
    radius_array = radius_array + 0.5
    
    #print('n2ha')
    #print(metal_mean)
    #print(metal_mean_err)
    #print(radius_array)
    #print(metal_mean.shape)
    #print(metal_mean_err.shape)
    #print(radius_array.shape)
    
    
    #### decide the radius to fit
    
    r1 = find_half_dex_radius(metal_mean_err_n2o2wodustc)
    r2 = find_half_dex_radius(metal_mean_err_n2o2_dustc)
    r3 = find_half_dex_radius(metal_mean_err_n2ha)
    r_min = min(r1,r2,r3)
    if r_min_set is not None:
        if r_min_set < r_min:
            r_min = r_min_set
    if r_min<3:
        r_min=3
    
    print(r1,r2,r3)
    
    #### gradient n2o2 w/o dust corr
    
    fit_radius_array = np.array(radius_array_new)+0.5
    popt_mean, pcov_mean = curve_fit(linear_func, fit_radius_array[:r_min], metal_mean_n2o2wodustc[:r_min],
                                     sigma=metal_mean_err_n2o2wodustc[:r_min])
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    df['n2o2_wodustc_a_mean'] = a_mean
    df['n2o2_wodustc_a_mean_err'] = a_mean_err
    df['n2o2_wodustc_b_mean'] = b_mean
    df['n2o2_wodustc_b_mean_err'] = b_mean_err
    
    df['n2o2_wodustc_len'] = len(fit_radius_array[:r_min])
    
    fig, ax = plt.subplots(1,3,figsize=(20,7), 
                           gridspec_kw={'wspace':0.6,'hspace':0.35})
    ax = ax.ravel()
    
    
    axplot_sumflux_bin_metal(ax=ax[0], metal_mean=metal_mean_n2o2wodustc[:r_min], 
                             metal_mean_err=metal_mean_err_n2o2wodustc[:r_min], 
                             title='n2o2 w/o dustc', re_kpc=re_kpc, 
                             radius_array=fit_radius_array[:r_min])
    
    
    
    #### gradient n2o2 with dust corr
    
    popt_mean, pcov_mean = curve_fit(linear_func, fit_radius_array[:r_min], metal_mean_n2o2_dustc[:r_min],
                                     sigma=metal_mean_err_n2o2_dustc[:r_min])
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    
    df['n2o2_dustc_a_mean'] = a_mean
    df['n2o2_dustc_a_mean_err'] = a_mean_err
    df['n2o2_dustc_b_mean'] = b_mean
    df['n2o2_dustc_b_mean_err'] = b_mean_err
    
    df['n2o2_dustc_len'] = len(fit_radius_array[:r_min])
    
    axplot_sumflux_bin_metal(ax=ax[1], metal_mean=metal_mean_n2o2_dustc[:r_min], 
                             metal_mean_err=metal_mean_err_n2o2_dustc[:r_min], 
                             title='n2o2 with dustc', re_kpc=re_kpc, 
                             radius_array=fit_radius_array[:r_min])
    
    
    
    
    ##### gradient n2ha
    
    popt_mean, pcov_mean = curve_fit(linear_func, fit_radius_array[:r_min], metal_mean_n2ha[:r_min],
                                     sigma=metal_mean_err_n2ha[:r_min])
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    df['n2ha_a_mean'] = a_mean
    df['n2ha_a_mean_err'] = a_mean_err
    df['n2ha_b_mean'] = b_mean
    df['n2ha_b_mean_err'] = b_mean_err
    
    df['n2ha_len'] = len(fit_radius_array[:r_min])
    
    axplot_sumflux_bin_metal(ax=ax[2], metal_mean=metal_mean_n2ha[:r_min], 
                             metal_mean_err=metal_mean_err_n2ha[:r_min], 
                             title='n2ha', re_kpc=re_kpc, 
                             radius_array=fit_radius_array[:r_min],R='N2Ha')
    
    
    
    fig.suptitle(plot_title)
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    
    plt.show()
    
    if savecsv:
        df.to_csv(csvpath+plot_title+'.csv')
        
def find_half_dex_radius(err_array):
    
    # find the first index where err > 0.5
    index_array = np.where(err_array>0.5)
    if len(index_array[0])==0:
        # all <=0.5
        return len(err_array)
    else:
        return index_array[0][0]
        

def axplot_sumflux_bin_metal(ax, metal_mean, metal_mean_err,title,re_kpc,
                             radius_array, R='N2O2',legend=False,
                             set_xylabel=True,color='b',fit_param=True,
                             text_y=0.35,label=None):
    
    
    
    
    
    
    #ax.scatter(radius_array,metal_mean,c=color)
    ax.errorbar(radius_array, metal_mean,
                    yerr=metal_mean_err,c=color,linestyle='None',marker='o',
                    markersize=9)
    
    popt_mean, pcov_mean = curve_fit(linear_func, radius_array, metal_mean,
                                     sigma=metal_mean_err)
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    plot_radius_array = radius_array - 0.5
    plot_radius_array = np.concatenate((plot_radius_array,
                                        [plot_radius_array[-1]+1]))
    
    # start from r=0
    
    ax.plot(plot_radius_array, linear_func(plot_radius_array, *popt_mean), 
                color=color, linestyle='--',label=label)
    if fit_param:
        ax.text(0.5, text_y, 'y=({:.3f}$\pm${:.3f})x + ({:.3f}$\pm${:.3f})'.format(a_mean,a_mean_err,b_mean,b_mean_err), 
                horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes,c=color,fontsize=14,
             bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'))
    
    
    if legend:
        ax.legend()
    if set_xylabel:
        ax.set_xlabel('radius [kpc]')
        ax.set_ylabel('12+log(O/H)')
    ax.set_title(title)
    ax.set_ylim(7.0,9.3)
    ax.axvline(x=re_kpc,ymin=0,ymax=1,
            linestyle='--',c='darkseagreen')
    if R=='N2O2':
        ax.axhline(y=9.23,xmin=0,xmax=1,linestyle='--',c='orange')
        ax.axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='orange')
    elif R=='N2Ha':
        #ax.axhline(y=8.53,xmin=0,xmax=1,linestyle='--',c='orange')
        #ax.axhline(y=7.63,xmin=0,xmax=1,linestyle='--',c='orange')
        ax.axhline(y=8.86,xmin=0,xmax=1,linestyle='--',c='orange')
        ax.axhline(y=7.17,xmin=0,xmax=1,linestyle='--',c='orange')


    
def get_flux_map(post_b3d,emi_line,get_data=False,sample=0):
    '''
    

    Parameters
    ----------
    post_b3d : PostBlobby3d
        DESCRIPTION.
    emi_line : str
        Nii or Oii.
    get_data : boolen, optional
        if True, get the flux map from the observed cube. The default is False.

    Returns
    -------
    flux_map
        

    '''
    
    if emi_line=='Oii':
        
        if get_data==False:
            precon_flux_oii_3727 = post_b3d.maps[sample, 0]
            precon_flux_oii_3729 = post_b3d.maps[sample,1]
        
            flux_map = precon_flux_oii_3727 + precon_flux_oii_3729
        else:
            sm = SpectralModel(
                    lines=[[3727.092], [3729.875]],
                    lsf_fwhm=1.2,)
            wave = post_b3d.metadata.get_axis_array('r')
            fit_data, fit_var_data = sm.fit_cube(wave, post_b3d.data, post_b3d.var)
            data_flux_oii3727 = fit_data[0]
            ##data_flux = post_b3d.data.sum(axis=2)
            data_flux_oii3729 = fit_data[1]
            flux_map = data_flux_oii3727 + data_flux_oii3729
        
        
        
    elif emi_line=='Nii':
        
        if get_data==False:
            precon_flux_nii = post_b3d.maps[sample,1]
            flux_map = precon_flux_nii
        else:
            
            sm = SpectralModel(
                    lines=[[6562.81], [6583.1, 6548.1, 0.3333]],
                    lsf_fwhm=1.2,)
            wave = post_b3d.metadata.get_axis_array('r')
            fit_data, fit_var_data = sm.fit_cube(wave, post_b3d.data, post_b3d.var)
            data_flux_nii = fit_data[1]
            flux_map = data_flux_nii
            
    return flux_map
    

def find_closest_index(arr, target):
    # Compute the absolute differences
    differences = np.abs(arr - target)
    # Find the index of the minimum difference
    closest_index = np.argmin(differences)
    return closest_index

def radian_to_degree(radians):
    degrees = np.degrees(radians)
    return degrees



##Create an elliptical distance array for deprojected radial profiles.
#size=the size of the array. Should be the same as an image input for uf.radial_profile
#centre=centre of the ellipse
#ellip=ellipticity of the ellipse=1-b/a
#pa=position angle of the ellipse starting along the positive x axis of the image
#Angle type: 'NTE' = North-Through-East (Default). 'WTN'=West-Through-North
def ellip_distarr(size,centre,ellip,pa,scale=None, angle_type='NTE'):
    y,x=np.indices(size)
    x=x-centre[0]
    y=y-centre[1]
    r=np.sqrt(x**2 + y**2)
    theta=np.zeros(size)
    theta[np.where((x>=0) & (y>=0))]=np.arcsin((y/r)[np.where((x>=0) & (y>=0))])
    theta[np.where((x>=0) & (y<0))]=2.0*np.pi+np.arcsin((y/r)[np.where((x>=0) & (y<0))])
    theta[np.where((x<0) & (y>=0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y>=0))])
    theta[np.where((x<0) & (y<0))]=np.pi-np.arcsin((y/r)[np.where((x<0) & (y<0))])
    if angle_type=='NTE':
        theta=theta+np.pi/2.0
    scdistarr=np.nan_to_num(np.sqrt((((np.sin(theta-pa))**2)+(1-ellip)**-2*(np.cos(theta-pa))**2))*r) ##SHOULD BE (1-ellip)**-2 !!!!
    #if scale=None:
    return scdistarr


def get_x(x,y,target):
    '''
    Define linear interpolation for half-light radius analysis
    x=vector of x-values
    y=vector of y-values
    target=target y value for which to extract the x value
    '''
    ind=find_nearest_index(y,target)
    if y[ind]<target:
        x1=ind
        x2=ind+1
    else:
        x1=ind-1
        x2=ind
    m=float(y[x2]-y[x1])/float(x[x2]-x[x1])
    b=y[x1]-m*x[x1]
    
    if m==0:
        return ind
    
    return float(target-b)/float(m)



def create_dist_array(flux_array):
    lens=int(flux_array.shape[1])
    dist=np.array([[np.sqrt((x-float(lens/2))**2+(y-float(lens/2))**2) for x in range(lens)] for y in range(lens)])
    return dist


def curve_of_growth(image,centre=None,distarr=None,mask=None):
    '''
    if centre==None:
        centre=np.array(image.shape,dtype=float)/2
    if distarr==None:
        y,x=np.indices(image.shape)
        distarr=np.sqrt((x-centre[0])**2 + (y-centre[1])**2)
    '''
    if mask==None:
        mask=np.zeros(image.shape)# from ones -> zeros
    else:
        mask = mask.data
    
    rmax=int(np.max(distarr))
    r=np.linspace(0,rmax,rmax+1)
    #print(r)
    cog=np.zeros(len(r))
    for i in range(0,len(r)):
        cog[i]=np.nansum(image[np.where((mask==0) & (distarr<i))]) #from 1 --> 0
    #print(cog)
    cog[0]=0.0
    cog=cog/cog[-1]
    #print(cog,cog[-1])
    return r,cog

def find_nearest_index(array,target):
    dist=np.abs(array-target)
    #print('dist',dist)
    target_index=dist.tolist().index(np.nanmin(dist))
    return target_index

def pix_to_kpc(radius_in_pix,z,CD2=5.55555555555556e-05):
    '''
    author: yifan

    Parameters
    ----------
    radius_in_pix : float
        radius get from cont_r50_ind=get_x(r,cog,0.5)
    z : float
        redshift
    CD2 : float
        CD2 in fits

    Returns
    -------
    None.

    '''
    ang = radius_in_pix * CD2 # deg
    distance = lcdm.angular_diameter_distance(z).value # angular diameter distance
    radius_in_kpc = ang*np.pi/180*distance*1000
    return radius_in_kpc

def arcsec_to_kpc(rad_in_arcsec,z):
    from astropy.cosmology import LambdaCDM
    lcdm = LambdaCDM(70,0.3,0.7)
    distance = lcdm.angular_diameter_distance(z).value # angular diameter distance, Mpc/radian
    rad_in_kpc = rad_in_arcsec * distance * np.pi/(180*3600)*1000
    return rad_in_kpc



def check_weak_nii_on_metal(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,
                                 ha_err,hb_err,oii_err,nii_err,
                                 ha_gist_err,hb_gist_err,oii_gist_err,nii_gist_err,
                                 pa,x_cen_pix,y_cen_pix,ellip,
                                 foreground_E_B_V,
                                 z,re_kpc=None,savepath=None,
                                 r_option='kpc',plot_title=None,
                                 limit_metal_range=False,
                                 savecsv=False,csvpath=None,
                                 nii_sn_cut=True,model_flux_sn=False,
                                 sumflux_radius_fill_perc=0.5,
                                 indvspax_radius_fill_perc=0.5,fit_param=True):
    '''
    ha_map: flux map from b3d
    ha_sn: flux/err from gist
    ha_err: flux err map from b3d
    ha_gist: flux map from gist
    

    
    '''
    # balmer decrement of GIST
    ha_gist = ha_sn * ha_gist_err
    hb_gist = hb_sn * hb_gist_err
    
    ha_sn_model = ha_map/ha_err
    hb_sn_model = hb_map/hb_err
    nii_sn_model = nii_map/nii_err
    oii_sn_model = oii_map/oii_err
    
    dist_arr = ellip_distarr(size=ha_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    if model_flux_sn:
        if nii_sn_cut:
            query = (ha_sn_model>=3) & (hb_sn_model>=3) & (oii_sn_model>=3) & (nii_sn_model>=3)
        else:
            query = (ha_sn_model>=3) & (hb_sn_model>=3) & (oii_sn_model>=3)
    else:
        if nii_sn_cut:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3)
        else:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    ##############################
    # metal n2o2 as func of radius, with gist dust correction and err calculation
    
    # NII correct
    nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_map,
                                                     flux_obs_err=nii_err,
                                                     wave=emi_wave.nii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z)
    
    # OII correct 
    
    oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_map,
                                                     flux_obs_err=oii_err,
                                                     wave=emi_wave.oii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z)
    
    logR_map = np.log10(nii_corr/oii_corr)
    metal_map_n2o2_gistdust = metal_k19(logR_map)
    
    #if nii_sn_cut:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3)
    #else:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map_n2o2_gistdust[mask_region] = np.nan
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    
    # nii_corr/oii_corr with error
    lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                            a_err=nii_err_corr, 
                                            b_err=oii_err_corr)
    
    metal_i, metal_err_withdust = calculate_z_and_error(x=lognii_oii, 
                                               x_err=lognii_oii_err, 
                                               y=-3.17)
    
    ##############################
    # metal gradient - n2ha
    
    if model_flux_sn:
        metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn_model,
                                       nii_sn=nii_sn_model)
    else:
        metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn,
                                       nii_sn=nii_sn)
    
    #nii_err_map = nii_map/nii_sn
    #ha_err_map = ha_map/ha_sn
    
    lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_map, b=ha_map, 
                                            a_err=nii_err, 
                                            b_err=ha_err)
    
    metal_i, metal_err_n2ha = calculate_z_and_error(x=lognii_ha, 
                                               x_err=lognii_ha_err, 
                                               y=-3.17,R='N2Ha')
    
    ############################### sum flux ####################
    
    plt.rcParams.update({'font.size': 20})
    
    
    
    
    
    # get the max radius where ha sn >= 3
    radius_max_kpc = np.nanmax(radius_kpc[ha_sn>=3])
    r_set = int(radius_max_kpc)+1
    if r_set < 2:
        r_set = 2
    radius_array = np.arange(r_set)
    
    
    ##### n2o2 with dust correction
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        if len(nii_map[query_all])/len(nii_map[query])<sumflux_radius_fill_perc:
            break
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_gist[query_all]) # use gist to calculate balmer
        hb_sum = np.nansum(hb_gist[query_all]) # use gist to calculate balmer
        oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_gist_err[query_all])**2)) # use gist err, as use gist flux  
        hb_err_sum = np.sqrt(np.nansum((hb_gist_err[query_all])**2))
        oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        ###
        
        nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_sum,
                                                         flux_obs_err=nii_err_sum,
                                                         wave=emi_wave.nii_wave,
                                                         ha=ha_sum,hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        # OII correct 
        
        oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_sum,
                                                         flux_obs_err=oii_err_sum,
                                                         wave=emi_wave.oii_wave,
                                                         ha=ha_sum,
                                                         hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        logR_map = np.log10(nii_corr/oii_corr)
        metal_map_n2o2_gistdust_sumflux = metal_k19(logR_map)
        
        # nii_corr/oii_corr with error
        lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                                a_err=nii_err_corr, 
                                                b_err=oii_err_corr)
        
        metal_i, metal_err_withdust_sumflux = calculate_z_and_error(x=lognii_oii, 
                                                   x_err=lognii_oii_err, 
                                                   y=-3.17)
        
        metal_array.append(metal_map_n2o2_gistdust_sumflux[0])
        metal_err_array.append(metal_err_withdust_sumflux[0])
    
    
    
    
    
    metal_mean_n2o2_dustc = np.array(metal_array)
    metal_mean_err_n2o2_dustc = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    
    
    #center of the bin
    #radius_array = radius_array + 0.5
    
    #print('n2o2 with dust')
    ###print(metal_mean)
    #print(metal_mean_err)
    #print(radius_array)
    #print(metal_mean.shape)
    #print(metal_mean_err.shape)
    #print(radius_array.shape)
    
    
    
    ##### n2ha 
    
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        if len(nii_map[query_all])/len(nii_map[query])<sumflux_radius_fill_perc:
            break
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_map[query_all]) # use b3d to calculate metal!!
        #hb_sum = np.nansum(hb_gist[query_all]) 
        #oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_err[query_all])**2))
        #hb_err_sum = np.sqrt(np.nansum((hb_err[query_all])**2))
        #oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_sum, b=ha_sum, 
                                                a_err=nii_err_sum, 
                                                b_err=ha_err_sum)
        
        metal_map_n2ha_sumflux, metal_err_n2ha_sumflux = calculate_z_and_error(x=lognii_ha, 
                                                   x_err=lognii_ha_err, 
                                                   y=-3.17,R='N2Ha')
    
        
        
        metal_array.append(metal_map_n2ha_sumflux)
        metal_err_array.append(metal_err_n2ha_sumflux)
    
    metal_mean_n2ha = np.array(metal_array)
    metal_mean_err_n2ha = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    
    
    
    # center of the bin
    #radius_array = radius_array + 0.5
    
    r2 = find_half_dex_radius(metal_mean_err_n2o2_dustc)
    r3 = find_half_dex_radius(metal_mean_err_n2ha)
    r_min = min(r2,r3)
    if r_min<3:
        r_min=3
        
        
    fig, ax = plt.subplots(4,3,figsize=(23,26), 
                           gridspec_kw={'wspace':0.4,'hspace':0.35})
    ax = ax.ravel()
        
    ##############################
    # N2O2 metallicity map with dust correction
    
    clim = map_limits(metal_map_n2o2_gistdust,pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1]-0.05)
    
    im0 = ax[0].imshow(metal_map_n2o2_gistdust,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
    cb0.set_label(label='12+log(O/H)',fontsize=20)
    ax[0].set_title('metallicity map')
    ax[0].set_ylabel('N2O2')
    
    ax[0].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid',label='0.5 R$_\mathrm{e}$')
    ax[0].contour(radius_re, levels=[1], colors='darkseagreen', linewidths=2, linestyles='dashed',label='1 R$_\mathrm{e}$')
    ax[0].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted',label='1.5 R$_\mathrm{e}$')
    ax[0].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    linestylelist=['solid','dashed','dotted']
    colorlist=['r','g','b']
    label_column=['0.5 R$_\mathrm{e}$','1 R$_\mathrm{e}$','1.5 R$_\mathrm{e}$']
    columns = [ax[0].plot([], [], c=colorlist[i],linestyle=linestylelist[i])[0] for i in range(3)]

    ax[0].legend( columns,  label_column,loc='lower right', prop={'size': 12})
    
    
    #### gradient n2o2 with dust corr
    fit_radius_array = np.array(radius_array_new)+0.5
    popt_mean, pcov_mean = curve_fit(linear_func, fit_radius_array[:r_min], metal_mean_n2o2_dustc[:r_min],
                                     sigma=metal_mean_err_n2o2_dustc[:r_min])
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    
    
    axplot_sumflux_bin_metal(ax=ax[1], metal_mean=metal_mean_n2o2_dustc[:r_min], 
                             metal_mean_err=metal_mean_err_n2o2_dustc[:r_min], 
                             title='sum flux', re_kpc=re_kpc, 
                             radius_array=fit_radius_array[:r_min],fit_param=fit_param,
                             color='sienna')
    
    axplot_bin_metal_with_err(ax=ax[2],metal_map=metal_map_n2o2_gistdust,
                              metal_err_map=metal_err_withdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='spaxel average',R='N2O2',plot_indivspax=True,
                     radius_fill_perc=indvspax_radius_fill_perc,fit_param=fit_param,
                     legend=True)
    
    clim = map_limits(nii_sn*nii_gist_err,pct=90)
    if clim[0]==0:
        clim[0] = 1
    norm = mpl.colors.LogNorm(vmin=clim[0], vmax=clim[1])
    
    im3 = ax[3].imshow(nii_sn*nii_gist_err,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='pink')
    ax[3].set_title('nii gist')
    cb3 = plt.colorbar(im3,ax=ax[3],fraction=0.047)
    cb3.set_label(label='flux',fontsize=20)
    
    
    
    im5 = ax[5].imshow(nii_sn*nii_gist_err,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='pink')
    ax[5].set_title('nii gist+sn contour')
    cb5 = plt.colorbar(im5,ax=ax[5],fraction=0.047)
    cb5.set_label(label='flux',fontsize=20)
    
    # build nii sn contour map
    
    nii_sn_contour_gist = np.zeros_like(nii_sn_model)
    
    nii_sn_contour_gist[nii_sn>=3] = 3
    nii_sn_contour_gist[(nii_sn<3)&(nii_sn>=2)] = 2
    nii_sn_contour_gist[(nii_sn<2)&(nii_sn>=1)] = 1
    

    
    
    ax[5].contour(nii_sn_contour_gist, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[5].contour(nii_sn_contour_gist, levels=[1.5], colors='g', linewidths=2, linestyles='dashed')
    ax[5].contour(nii_sn_contour_gist, levels=[2.5], colors='b', linewidths=2, linestyles='dotted')
    #ax.grid(which='both')
    handles = [
    plt.Line2D([0, 1], [0, 1], color='r', linewidth=2, linestyle='solid', label='sn=1'),
    plt.Line2D([0, 1], [0, 1], color='g', linewidth=2, linestyle='dashed', label='sn=2'),
    plt.Line2D([0, 1], [0, 1], color='b', linewidth=2, linestyle='dotted', label='sn=3')
        
    ]
    ax[5].legend(handles=handles, loc='upper right', prop={'size': 12})
    
    
    
    im6 = ax[6].imshow(nii_map,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='pink')
    ax[6].set_title('nii b3d')
    cb6 = plt.colorbar(im6,ax=ax[6],fraction=0.047)
    cb6.set_label(label='flux',fontsize=20)
    
    im8 = ax[8].imshow(nii_map,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='pink')
    ax[8].set_title('nii b3d+sn contour')
    cb8 = plt.colorbar(im8,ax=ax[8],fraction=0.047)
    cb8.set_label(label='flux',fontsize=20)
    
    nii_sn_contour_b3d = np.zeros_like(nii_sn_model)
    
    nii_sn_contour_b3d[nii_sn_model>=3] = 3
    nii_sn_contour_b3d[(nii_sn_model<3)&(nii_sn_model>=2)] = 2
    nii_sn_contour_b3d[(nii_sn_model<2)&(nii_sn_model>=1)] = 1
    
    
    ax[8].contour(nii_sn_contour_b3d, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[8].contour(nii_sn_contour_b3d, levels=[1.5], colors='g', linewidths=2, linestyles='dashed')
    ax[8].contour(nii_sn_contour_b3d, levels=[2.5], colors='b', linewidths=2, linestyles='dotted')
    #ax.grid(which='both')
    handles = [
    plt.Line2D([0, 1], [0, 1], color='r', linewidth=2, linestyle='solid', label='sn=1'),
    plt.Line2D([0, 1], [0, 1], color='g', linewidth=2, linestyle='dashed', label='sn=2'),
    plt.Line2D([0, 1], [0, 1], color='b', linewidth=2, linestyle='dotted', label='sn=3')
        
    ]
    #ax[8].legend(handles=handles, loc='upper right')
    
    
    
    clim = map_limits(nii_gist_err,pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    
    im4 = ax[4].imshow(nii_gist_err,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='summer')
    ax[4].set_title('nii_err gist')
    cb4 = plt.colorbar(im4,ax=ax[4],fraction=0.047)
    cb4.set_label(label='flux',fontsize=20)
    
    im7 = ax[7].imshow(nii_err,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='summer')
    ax[7].set_title('nii_err b3d')
    cb7 = plt.colorbar(im7,ax=ax[7],fraction=0.047)
    cb7.set_label(label='flux',fontsize=20)
    
    #clim = map_limits(nii_sn,pct=90)
    clim = [0,6]
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    
    im9 = ax[9].imshow(nii_sn,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='Wistia')
    ax[9].set_title('nii sn gist')
    cb9 = plt.colorbar(im9,ax=ax[9],fraction=0.047)
    cb9.set_label(label='S/N',fontsize=20)
    
    im10 = ax[10].imshow(nii_sn_model,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='Wistia')
    ax[10].set_title('nii sn b3d')
    cb10 = plt.colorbar(im10,ax=ax[10],fraction=0.047)
    cb10.set_label(label='S/N',fontsize=20)
    
    
    
    
    ax[11].set_visible(False)
    
    
    
    fig.suptitle(plot_title)
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    
    plt.show()

def check_diff_nii_cut(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,
                                 ha_err,hb_err,oii_err,nii_err,
                                 ha_gist_err,hb_gist_err,oii_gist_err,nii_gist_err,
                                 pa,x_cen_pix,y_cen_pix,ellip,
                                 foreground_E_B_V,
                                 z,re_kpc=None,savepath=None,
                                 r_option='kpc',plot_title=None,
                                 limit_metal_range=False,
                                 savecsv=False,csvpath=None,
                                 nii_sn_cut=True,model_flux_sn=False,
                                 sumflux_radius_fill_perc=0.5,
                                 indvspax_radius_fill_perc=0.5,fit_param=True):
    '''
    check the impact of using different NII cut on the metallicity gradient 
    measurement. Two plots about N2O2 measurement. First one is the metallicity
    map. The second one is metallicity gradient measure using sumflux method.
    The second plot should include three different gradients, first is no require
    S/N[NII], the second one is S/N[NII]>=1, the third one is S/N[NII]>=3
    
    modify from check_weak_nii()
    
    with following updates    
    
    
    '''
    # balmer decrement of GIST
    ha_gist = ha_sn * ha_gist_err
    hb_gist = hb_sn * hb_gist_err
    
    ha_sn_model = ha_map/ha_err
    hb_sn_model = hb_map/hb_err
    nii_sn_model = nii_map/nii_err
    oii_sn_model = oii_map/oii_err
    
    dist_arr = ellip_distarr(size=ha_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    if model_flux_sn:
        if nii_sn_cut:
            query = (ha_sn_model>=3) & (hb_sn_model>=3) & (oii_sn_model>=3) & (nii_sn_model>=3)
        else:
            query = (ha_sn_model>=3) & (hb_sn_model>=3) & (oii_sn_model>=3)
    else:
        if nii_sn_cut:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3)
        else:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    ##############################
    # metal n2o2 as func of radius, with gist dust correction and err calculation
    
    # NII correct
    nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_map,
                                                     flux_obs_err=nii_err,
                                                     wave=emi_wave.nii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z)
    
    # OII correct 
    
    oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_map,
                                                     flux_obs_err=oii_err,
                                                     wave=emi_wave.oii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z)
    
    logR_map = np.log10(nii_corr/oii_corr)
    metal_map_n2o2_gistdust = metal_k19(logR_map)
    
    #if nii_sn_cut:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3)
    #else:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map_n2o2_gistdust[mask_region] = np.nan
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    
    # nii_corr/oii_corr with error
    lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                            a_err=nii_err_corr, 
                                            b_err=oii_err_corr)
    
    metal_i, metal_err_withdust = calculate_z_and_error(x=lognii_oii, 
                                               x_err=lognii_oii_err, 
                                               y=-3.17)
    
    ##############################
    # metal gradient - n2ha
    
    if model_flux_sn:
        metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn_model,
                                       nii_sn=nii_sn_model)
    else:
        metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn,
                                       nii_sn=nii_sn)
    
    #nii_err_map = nii_map/nii_sn
    #ha_err_map = ha_map/ha_sn
    
    lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_map, b=ha_map, 
                                            a_err=nii_err, 
                                            b_err=ha_err)
    
    metal_i, metal_err_n2ha = calculate_z_and_error(x=lognii_ha, 
                                               x_err=lognii_ha_err, 
                                               y=-3.17,R='N2Ha')
    
    ############################### sum flux ####################
    
    plt.rcParams.update({'font.size': 20})
    
    
    
    
    
    
        
    fig, ax = plt.subplots(4,3,figsize=(23,26), 
                           gridspec_kw={'wspace':0.4,'hspace':0.35})
    ax = ax.ravel()
        
    ##############################
    # N2O2 metallicity map with dust correction
    
    clim = map_limits(metal_map_n2o2_gistdust,pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1]-0.05)
    
    im0 = ax[0].imshow(metal_map_n2o2_gistdust,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
    cb0.set_label(label='12+log(O/H)',fontsize=20)
    ax[0].set_title('metallicity map')
    ax[0].set_ylabel('N2O2')
    
    ax[0].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid',label='0.5 R$_\mathrm{e}$')
    ax[0].contour(radius_re, levels=[1], colors='darkseagreen', linewidths=2, linestyles='dashed',label='1 R$_\mathrm{e}$')
    ax[0].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted',label='1.5 R$_\mathrm{e}$')
    ax[0].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    linestylelist=['solid','dashed','dotted']
    colorlist=['r','g','b']
    label_column=['0.5 R$_\mathrm{e}$','1 R$_\mathrm{e}$','1.5 R$_\mathrm{e}$']
    columns = [ax[0].plot([], [], c=colorlist[i],linestyle=linestylelist[i])[0] for i in range(3)]

    ax[0].legend( columns,  label_column,loc='lower right', prop={'size': 12})
    
    
    #### gradient n2o2 with dust corr
    df = pd.DataFrame({'MAGPIID':[int(plot_title)]})
    
    surfix = ['nocut','sn1','sn3']
    nii_cut_list = [None,1,3]
    color_list = ['sienna','olivedrab','darkmagenta']
    
    for ii in range(3):
    
        a, b, a_err, b_err, r_len = check_ax_plot_diff_sncut(ax[1],nii_map=nii_map,oii_map=oii_map,ha_map=ha_gist,
                                     hb_map=hb_gist,nii_err=nii_err,oii_err=oii_err,
                                     ha_err=ha_gist_err,
                                     hb_err=hb_gist_err, radius_kpc=radius_kpc, 
                                     foreground_E_B_V=foreground_E_B_V,z=z,re_kpc=re_kpc,
                                     nii_sn_cut=nii_cut_list[ii],fit_param=True,
                                     plot_color=color_list[ii],text_y=0.05+ii/12,
                                     label=surfix[ii])
        
        df['a_'+surfix[ii]] = a
        df['a_err_'+surfix[ii]] = a_err
        df['b_'+surfix[ii]] = b
        df['b_err_'+surfix[ii]] = b_err
        df['len_'+surfix[ii]] = r_len
    
    if savecsv:
        df.to_csv(csvpath+plot_title+'.csv')
    ax[1].legend(loc='lower right', prop={'size': 12})
    
    axplot_bin_metal_with_err(ax=ax[2],metal_map=metal_map_n2o2_gistdust,
                              metal_err_map=metal_err_withdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='spaxel average',R='N2O2',plot_indivspax=True,
                     radius_fill_perc=indvspax_radius_fill_perc,fit_param=fit_param,
                     legend=True)
    
    clim = map_limits(nii_sn*nii_gist_err,pct=90)
    if clim[0]==0:
        clim[0] = 1
    norm = mpl.colors.LogNorm(vmin=clim[0], vmax=clim[1])
    
    im3 = ax[3].imshow(nii_sn*nii_gist_err,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='pink')
    ax[3].set_title('nii gist')
    cb3 = plt.colorbar(im3,ax=ax[3],fraction=0.047)
    cb3.set_label(label='flux',fontsize=20)
    
    
    
    im5 = ax[5].imshow(nii_sn*nii_gist_err,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='pink')
    ax[5].set_title('nii gist+sn contour')
    cb5 = plt.colorbar(im5,ax=ax[5],fraction=0.047)
    cb5.set_label(label='flux',fontsize=20)
    
    # build nii sn contour map
    
    nii_sn_contour_gist = np.zeros_like(nii_sn_model)
    
    nii_sn_contour_gist[nii_sn>=3] = 3
    nii_sn_contour_gist[(nii_sn<3)&(nii_sn>=2)] = 2
    nii_sn_contour_gist[(nii_sn<2)&(nii_sn>=1)] = 1
    

    
    
    ax[5].contour(nii_sn_contour_gist, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[5].contour(nii_sn_contour_gist, levels=[1.5], colors='g', linewidths=2, linestyles='dashed')
    ax[5].contour(nii_sn_contour_gist, levels=[2.5], colors='b', linewidths=2, linestyles='dotted')
    #ax.grid(which='both')
    handles = [
    plt.Line2D([0, 1], [0, 1], color='r', linewidth=2, linestyle='solid', label='sn=1'),
    plt.Line2D([0, 1], [0, 1], color='g', linewidth=2, linestyle='dashed', label='sn=2'),
    plt.Line2D([0, 1], [0, 1], color='b', linewidth=2, linestyle='dotted', label='sn=3')
        
    ]
    ax[5].legend(handles=handles, loc='upper right', prop={'size': 12})
    
    
    
    im6 = ax[6].imshow(nii_map,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='pink')
    ax[6].set_title('nii b3d')
    cb6 = plt.colorbar(im6,ax=ax[6],fraction=0.047)
    cb6.set_label(label='flux',fontsize=20)
    
    im8 = ax[8].imshow(nii_map,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='pink')
    ax[8].set_title('nii b3d+sn contour')
    cb8 = plt.colorbar(im8,ax=ax[8],fraction=0.047)
    cb8.set_label(label='flux',fontsize=20)
    
    nii_sn_contour_b3d = np.zeros_like(nii_sn_model)
    
    nii_sn_contour_b3d[nii_sn_model>=3] = 3
    nii_sn_contour_b3d[(nii_sn_model<3)&(nii_sn_model>=2)] = 2
    nii_sn_contour_b3d[(nii_sn_model<2)&(nii_sn_model>=1)] = 1
    
    
    ax[8].contour(nii_sn_contour_b3d, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[8].contour(nii_sn_contour_b3d, levels=[1.5], colors='g', linewidths=2, linestyles='dashed')
    ax[8].contour(nii_sn_contour_b3d, levels=[2.5], colors='b', linewidths=2, linestyles='dotted')
    #ax.grid(which='both')
    handles = [
    plt.Line2D([0, 1], [0, 1], color='r', linewidth=2, linestyle='solid', label='sn=1'),
    plt.Line2D([0, 1], [0, 1], color='g', linewidth=2, linestyle='dashed', label='sn=2'),
    plt.Line2D([0, 1], [0, 1], color='b', linewidth=2, linestyle='dotted', label='sn=3')
        
    ]
    #ax[8].legend(handles=handles, loc='upper right')
    
    
    
    clim = map_limits(nii_gist_err,pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    
    im4 = ax[4].imshow(nii_gist_err,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='summer')
    ax[4].set_title('nii_err gist')
    cb4 = plt.colorbar(im4,ax=ax[4],fraction=0.047)
    cb4.set_label(label='flux',fontsize=20)
    
    im7 = ax[7].imshow(nii_err,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='summer')
    ax[7].set_title('nii_err b3d')
    cb7 = plt.colorbar(im7,ax=ax[7],fraction=0.047)
    cb7.set_label(label='flux',fontsize=20)
    
    #clim = map_limits(nii_sn,pct=90)
    clim = [0,6]
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    
    im9 = ax[9].imshow(nii_sn,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='Wistia')
    ax[9].set_title('nii sn gist')
    cb9 = plt.colorbar(im9,ax=ax[9],fraction=0.047)
    cb9.set_label(label='S/N',fontsize=20)
    
    im10 = ax[10].imshow(nii_sn_model,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap='Wistia')
    ax[10].set_title('nii sn b3d')
    cb10 = plt.colorbar(im10,ax=ax[10],fraction=0.047)
    cb10.set_label(label='S/N',fontsize=20)
    
    
    
    
    ax[11].set_visible(False)
    
    
    
    fig.suptitle(plot_title)
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    
    plt.show()


def check_ax_plot_diff_sncut(ax,nii_map,oii_map,ha_map,
                             hb_map,nii_err,oii_err,ha_err,
                             hb_err, radius_kpc, 
                             foreground_E_B_V,z,re_kpc,
                             nii_sn_cut=None,fit_param=True,
                             plot_color='sienna',text_y=0.35,label=None):
    '''
    nii and oii flux from blobby3d
    ha and hb flux from gist
    radius_kpc: radius map
    
    given flux map, and given sn cut, plot sum flux metal gradient at give ax.
    only support n2o2 here.
    
    '''
    ha_sn = ha_map/ha_err
    hb_sn = hb_map/hb_err
    nii_sn = nii_map/nii_err
    oii_sn = oii_map/oii_err
    
    
    # get the max radius where ha sn >= 3
    radius_max_kpc = np.nanmax(radius_kpc[ha_sn>=3])
    r_set = int(radius_max_kpc)+1
    if r_set < 2:
        r_set = 2
    radius_array = np.arange(r_set)
    
    
    ##### n2o2 with dust correction
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        if nii_sn_cut is None:
            query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)&(hb_sn>=3)&(oii_sn>=3)
        else:
            query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)&(hb_sn>=3)&(oii_sn>=3)&(nii_sn>=nii_sn_cut)
        if len(nii_map[query_all])<1:
            break
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_map[query_all]) # use gist to calculate balmer
        hb_sum = np.nansum(hb_map[query_all]) # use gist to calculate balmer
        oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_err[query_all])**2)) # use gist err, as use gist flux  
        hb_err_sum = np.sqrt(np.nansum((hb_err[query_all])**2))
        oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        ###
        
        nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_sum,
                                                         flux_obs_err=nii_err_sum,
                                                         wave=emi_wave.nii_wave,
                                                         ha=ha_sum,hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        # OII correct 
        
        oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_sum,
                                                         flux_obs_err=oii_err_sum,
                                                         wave=emi_wave.oii_wave,
                                                         ha=ha_sum,
                                                         hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        logR_map = np.log10(nii_corr/oii_corr)
        metal_map_n2o2_gistdust_sumflux = metal_k19(logR_map)
        
        # nii_corr/oii_corr with error
        lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                                a_err=nii_err_corr, 
                                                b_err=oii_err_corr)
        
        metal_i, metal_err_withdust_sumflux = calculate_z_and_error(x=lognii_oii, 
                                                   x_err=lognii_oii_err, 
                                                   y=-3.17)
        
        metal_array.append(metal_map_n2o2_gistdust_sumflux[0])
        metal_err_array.append(metal_err_withdust_sumflux[0])
    
    
    
    
    
    metal_mean_n2o2_dustc = np.array(metal_array)
    metal_mean_err_n2o2_dustc = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    fit_radius_array = np.array(radius_array_new)+0.5
    popt_mean, pcov_mean = curve_fit(linear_func, fit_radius_array, metal_mean_n2o2_dustc,
                                     sigma=metal_mean_err_n2o2_dustc)
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    axplot_sumflux_bin_metal(ax=ax, metal_mean=metal_mean_n2o2_dustc, 
                             metal_mean_err=metal_mean_err_n2o2_dustc, 
                             title='sum flux', re_kpc=re_kpc, 
                             radius_array=fit_radius_array,fit_param=fit_param,
                             color=plot_color,text_y=text_y,label=label)
    
    
    return a_mean, b_mean, a_mean_err, b_mean_err, len(fit_radius_array)

def paper_metal_gradient(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,
                                 ha_err,hb_err,oii_err,nii_err,
                                 ha_gist_err,hb_gist_err,oii_gist_err,nii_gist_err,
                                 pa,x_cen_pix,y_cen_pix,ellip,
                                 foreground_E_B_V,
                                 z,
                                 re_kpc=None,savepath=None,
                                 r_option='kpc',plot_title=None,
                                 limit_metal_range=False,
                                 savecsv=False,csvpath=None,
                                 nii_sn_cut=None,model_flux_sn=False,
                                 sumflux_radius_fill_perc=0.5,
                                 indvspax_radius_fill_perc=0.5,fit_param=True):
    '''
    ha_map: flux map from b3d
    ha_sn: flux/err from gist
    ha_err: flux err map from b3d
    ha_gist: flux map from gist
    
    nii_sn_cut: None or float, when float, it is the threshold of nii sn cut
    
    1 April 2025 update:
        make nii sn cut on the sum flux method
    14 July 2025 update:
        img_extent: list, the range of coordinate
                    [x_min, x_max, y_min, y_max] in arcsec

    
    '''
    # balmer decrement of GIST
    ha_gist = ha_sn * ha_gist_err
    hb_gist = hb_sn * hb_gist_err
    
    ha_sn_model = ha_map/ha_err
    hb_sn_model = hb_map/hb_err
    nii_sn_model = nii_map/nii_err
    oii_sn_model = oii_map/oii_err
    
    dist_arr = ellip_distarr(size=ha_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    if model_flux_sn:
        if nii_sn_cut is not None:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn_model>=3) & (nii_sn_model>=nii_sn_cut)
        else:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn_model>=3)
    else:
        if nii_sn_cut is not None:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3) & (nii_sn>=nii_sn_cut)
        else:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    ##############################
    # metal n2o2 as func of radius, with gist dust correction and err calculation
    
    # NII correct
    nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_map,
                                                     flux_obs_err=nii_err,
                                                     wave=emi_wave.nii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z)
    
    # OII correct 
    
    oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_map,
                                                     flux_obs_err=oii_err,
                                                     wave=emi_wave.oii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z)
    
    logR_map = np.log10(nii_corr/oii_corr)
    metal_map_n2o2_gistdust = metal_k19(logR_map)
    
    #if nii_sn_cut:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3)
    #else:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map_n2o2_gistdust[mask_region] = np.nan
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    
    # nii_corr/oii_corr with error
    lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                            a_err=nii_err_corr, 
                                            b_err=oii_err_corr)
    
    metal_i, metal_err_withdust = calculate_z_and_error(x=lognii_oii, 
                                               x_err=lognii_oii_err, 
                                               y=-3.17)
    
    ##############################
    # metal gradient - n2ha
    
    if model_flux_sn:
        metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn_model,
                                       nii_sn=nii_sn_model)
    else:
        metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn,
                                       nii_sn=nii_sn)
    
    #nii_err_map = nii_map/nii_sn
    #ha_err_map = ha_map/ha_sn
    
    lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_map, b=ha_map, 
                                            a_err=nii_err, 
                                            b_err=ha_err)
    
    metal_i, metal_err_n2ha = calculate_z_and_error(x=lognii_ha, 
                                               x_err=lognii_ha_err, 
                                               y=-3.17,R='N2Ha')
    
    ############################### sum flux ####################
    
    plt.rcParams.update({'font.size': 20})
    
    
    
    
    
    # get the max radius where ha sn >= 3
    radius_max_kpc = np.nanmax(radius_kpc[ha_sn>=3])
    r_set = int(radius_max_kpc)+1
    if r_set < 2:
        r_set = 2
    radius_array = np.arange(r_set)
    
    
    ##### n2o2 with dust correction
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        
        
        if model_flux_sn:
            if nii_sn_cut is not None: # as we use ha gist to do dust corr, so use gist sn for ha and hb
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn_model>=nii_sn_cut)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
        else:
            if nii_sn_cut is not None:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn>=nii_sn_cut)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
        
        
        #if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
        if len(nii_map[query_all]) < 1: 
            break
        
        
        #query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        #if len(nii_map[query_all])/len(nii_map[query])<sumflux_radius_fill_perc:
        #    break
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_gist[query_all]) # use gist to calculate balmer
        hb_sum = np.nansum(hb_gist[query_all]) # use gist to calculate balmer
        oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_gist_err[query_all])**2)) # use gist err, as use gist flux  
        hb_err_sum = np.sqrt(np.nansum((hb_gist_err[query_all])**2))
        oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        ###
        
        nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_sum,
                                                         flux_obs_err=nii_err_sum,
                                                         wave=emi_wave.nii_wave,
                                                         ha=ha_sum,hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        # OII correct 
        
        oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_sum,
                                                         flux_obs_err=oii_err_sum,
                                                         wave=emi_wave.oii_wave,
                                                         ha=ha_sum,
                                                         hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        logR_map = np.log10(nii_corr/oii_corr)
        metal_map_n2o2_gistdust_sumflux = metal_k19(logR_map)
        
        # nii_corr/oii_corr with error
        lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                                a_err=nii_err_corr, 
                                                b_err=oii_err_corr)
        
        metal_i, metal_err_withdust_sumflux = calculate_z_and_error(x=lognii_oii, 
                                                   x_err=lognii_oii_err, 
                                                   y=-3.17)
        
        metal_array.append(metal_map_n2o2_gistdust_sumflux[0])
        metal_err_array.append(metal_err_withdust_sumflux[0])
    
    
    
    
    
    metal_mean_n2o2_dustc = np.array(metal_array)
    metal_mean_err_n2o2_dustc = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    
    
    #center of the bin
    #radius_array = radius_array + 0.5
    
    #print('n2o2 with dust')
    ###print(metal_mean)
    #print(metal_mean_err)
    #print(radius_array)
    #print(metal_mean.shape)
    #print(metal_mean_err.shape)
    #print(radius_array.shape)
    
    
    
    ##### n2ha 
    
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        #query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        #if len(nii_map[query_all])/len(nii_map[query])<sumflux_radius_fill_perc:
        #    break
        
        if model_flux_sn:
            if nii_sn_cut is not None: 
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn_model>=nii_sn_cut)&(ha_sn_model>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn_model>=3)
        else:
            if nii_sn_cut is not None:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn>=nii_sn_cut)&(ha_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        
        
        
        #if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
        if len(nii_map[query_all]) < 1:
            break
        
        
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_map[query_all]) # use b3d to calculate metal!!
        #hb_sum = np.nansum(hb_gist[query_all]) 
        #oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_err[query_all])**2))
        #hb_err_sum = np.sqrt(np.nansum((hb_err[query_all])**2))
        #oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_sum, b=ha_sum, 
                                                a_err=nii_err_sum, 
                                                b_err=ha_err_sum)
        
        metal_map_n2ha_sumflux, metal_err_n2ha_sumflux = calculate_z_and_error(x=lognii_ha, 
                                                   x_err=lognii_ha_err, 
                                                   y=-3.17,R='N2Ha')
    
        
        
        metal_array.append(metal_map_n2ha_sumflux)
        metal_err_array.append(metal_err_n2ha_sumflux)
    
    metal_mean_n2ha = np.array(metal_array)
    metal_mean_err_n2ha = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    
    
    
    # center of the bin
    #radius_array = radius_array + 0.5
    
    r2 = find_half_dex_radius(metal_mean_err_n2o2_dustc)
    r3 = find_half_dex_radius(metal_mean_err_n2ha)
    r_min = min(r2,r3)
    if r_min<3:
        r_min=3
        
    # find the r min for spaxel average
    r_min_spax_n2o2 = axplot_bin_metal_with_err(ax=None,metal_map=metal_map_n2o2_gistdust,
                              metal_err_map=metal_err_withdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='spaxel average',R='N2O2',plot_indivspax=False,
                     radius_fill_perc=indvspax_radius_fill_perc,fit_param=fit_param,
                     legend=True,get_arraylenth=True)
    
    r_min_spax_n2ha = axplot_bin_metal_with_err(ax=None,metal_map=metal_map_n2ha,
                              metal_err_map=metal_err_n2ha,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title=None,R='N2Ha',plot_indivspax=False,
                     radius_fill_perc=indvspax_radius_fill_perc,fit_param=fit_param,
                     get_arraylenth=True)
        
    
    r_min_spax = np.min([r_min_spax_n2ha,r_min_spax_n2o2])
    
    fig, ax = plt.subplots(2,3,figsize=(23,10), 
                           gridspec_kw={'wspace':0.4,'hspace':0.35})
    ax = ax.ravel()
        
    ##############################
    # N2O2 metallicity map with dust correction
    
    clim = map_limits(metal_map_n2o2_gistdust,pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1]-0.05)
    
    im0 = ax[0].imshow(metal_map_n2o2_gistdust,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb0 = plt.colorbar(im0,ax=ax[0],fraction=0.047)
    cb0.set_label(label='12+log(O/H)',fontsize=20)
    ax[0].set_title('metallicity map')
    ax[0].set_ylabel('N2O2')
    
    ax[0].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid',label='0.5 R$_\mathrm{e}$')
    ax[0].contour(radius_re, levels=[1], colors='darkseagreen', linewidths=2, linestyles='dashed',label='1 R$_\mathrm{e}$')
    ax[0].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted',label='1.5 R$_\mathrm{e}$')
    ax[0].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    linestylelist=['solid','dashed','dotted']
    colorlist=['r','g','b']
    label_column=['0.5 R$_\mathrm{e}$','1 R$_\mathrm{e}$','1.5 R$_\mathrm{e}$']
    columns = [ax[0].plot([], [], c=colorlist[i],linestyle=linestylelist[i])[0] for i in range(3)]

    ax[0].legend( columns,  label_column,loc='lower right', prop={'size': 12})
    
    
    #### gradient n2o2 with dust corr
    fit_radius_array = np.array(radius_array_new)+0.5
    popt_mean, pcov_mean = curve_fit(linear_func, fit_radius_array[:r_min], metal_mean_n2o2_dustc[:r_min],
                                     sigma=metal_mean_err_n2o2_dustc[:r_min])
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    
    
    axplot_sumflux_bin_metal(ax=ax[1], metal_mean=metal_mean_n2o2_dustc[:r_min], 
                             metal_mean_err=metal_mean_err_n2o2_dustc[:r_min], 
                             title='sum flux', re_kpc=re_kpc, 
                             radius_array=fit_radius_array[:r_min],fit_param=fit_param,
                             color='sienna')
    
    axplot_bin_metal_with_err(ax=ax[2],metal_map=metal_map_n2o2_gistdust,
                              metal_err_map=metal_err_withdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='spaxel average',R='N2O2',plot_indivspax=True,
                     radius_fill_perc=indvspax_radius_fill_perc,fit_param=fit_param,
                     legend=True,r_min_set=r_min_spax)
    
    
    ##############################
    # N2Ha metallicity map
    
    clim = map_limits(metal_map_n2ha,pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1]-0.05)
    
    im3 = ax[3].imshow(metal_map_n2ha,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    cb3 = plt.colorbar(im3,ax=ax[3],fraction=0.047)
    cb3.set_label(label='12+log(O/H)',fontsize=20)
    #ax[3].set_title('metallicity map')
    ax[3].set_ylabel('N2H$\\alpha$')
    
    ax[3].contour(radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[3].contour(radius_re, levels=[1], colors='darkseagreen', linewidths=2, linestyles='dashed')
    ax[3].contour(radius_re, levels=[1.5], colors='b', linewidths=2, linestyles='dotted')
    ax[3].scatter(x_cen_pix,y_cen_pix,c='b',marker='x')
    
    
    
    ##### gradient n2ha
    
    popt_mean, pcov_mean = curve_fit(linear_func, fit_radius_array[:r_min], metal_mean_n2ha[:r_min],
                                     sigma=metal_mean_err_n2ha[:r_min])
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    
    axplot_sumflux_bin_metal(ax=ax[4], metal_mean=metal_mean_n2ha[:r_min], 
                             metal_mean_err=metal_mean_err_n2ha[:r_min], 
                             title=None, re_kpc=re_kpc, 
                             radius_array=fit_radius_array[:r_min],R='N2Ha',
                             fit_param=fit_param,color='sienna')
    
    axplot_bin_metal_with_err(ax=ax[5],metal_map=metal_map_n2ha,
                              metal_err_map=metal_err_n2ha,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title=None,R='N2Ha',plot_indivspax=True,
                     radius_fill_perc=indvspax_radius_fill_perc,
                     fit_param=fit_param,r_min_set=r_min_spax)
    
    fig.suptitle(plot_title)
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    
    plt.show()
    

def paper_metal_gradient_spaxelave(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,
                                 ha_err,hb_err,oii_err,nii_err,
                                 ha_gist_err,hb_gist_err,oii_gist_err,nii_gist_err,
                                 pa,x_cen_pix,y_cen_pix,ellip,
                                 foreground_E_B_V,
                                 z,img_extent,white_light_img_path,
                                 re_kpc=None,savepath=None,
                                 r_option='kpc',plot_title=None,
                                 limit_metal_range=False,
                                 savecsv=False,csvpath=None,
                                 nii_sn_cut=None,model_flux_sn=False,
                                 sumflux_radius_fill_perc=0.5,
                                 indvspax_radius_fill_perc=0.5,fit_param=True,
                                 bin_size=1,r_min_spax=None,colorbar_xpos=-0.08,
                                 set_cb_ticks=None):
    '''
    ha_map: flux map from b3d
    ha_sn: flux/err from gist
    ha_err: flux err map from b3d
    ha_gist: flux map from gist
    
    nii_sn_cut: None or float, when float, it is the threshold of nii sn cut
    
    1 April 2025 update:
        make nii sn cut on the sum flux method
    16 May 2025 update:
        this version only include metallicity map and spaxel ave plot
    22 July 2025 update:
        colorbar_xpos: set the x pos of colorbar
        set_cb_ticks: mannually set the colorbar ticks
    
    '''
    # balmer decrement of GIST
    ha_gist = ha_sn * ha_gist_err
    hb_gist = hb_sn * hb_gist_err
    
    ha_sn_model = ha_map/ha_err
    hb_sn_model = hb_map/hb_err
    nii_sn_model = nii_map/nii_err
    oii_sn_model = oii_map/oii_err
    
    dist_arr = ellip_distarr(size=ha_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    if model_flux_sn:
        if nii_sn_cut is not None:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn_model>=3) & (nii_sn_model>=nii_sn_cut)
        else:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn_model>=3)
    else:
        if nii_sn_cut is not None:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3) & (nii_sn>=nii_sn_cut)
        else:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    ##############################
    # metal n2o2 as func of radius, with gist dust correction and err calculation
    
    # NII correct
    nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_map,
                                                     flux_obs_err=nii_err,
                                                     wave=emi_wave.nii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z)
    
    # OII correct 
    
    oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_map,
                                                     flux_obs_err=oii_err,
                                                     wave=emi_wave.oii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z)
    
    logR_map = np.log10(nii_corr/oii_corr)
    metal_map_n2o2_gistdust = metal_k19(logR_map)
    
    #if nii_sn_cut:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3)
    #else:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map_n2o2_gistdust[mask_region] = np.nan
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    
    # nii_corr/oii_corr with error
    lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                            a_err=nii_err_corr, 
                                            b_err=oii_err_corr)
    
    metal_i, metal_err_withdust = calculate_z_and_error(x=lognii_oii, 
                                               x_err=lognii_oii_err, 
                                               y=-3.17)
    
    ##############################
    # metal gradient - n2ha
    
    if model_flux_sn:
        metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn_model,
                                       nii_sn=nii_sn_model)
    else:
        metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn,
                                       nii_sn=nii_sn)
    
    #nii_err_map = nii_map/nii_sn
    #ha_err_map = ha_map/ha_sn
    
    lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_map, b=ha_map, 
                                            a_err=nii_err, 
                                            b_err=ha_err)
    
    metal_i, metal_err_n2ha = calculate_z_and_error(x=lognii_ha, 
                                               x_err=lognii_ha_err, 
                                               y=-3.17,R='N2Ha')
    
    ############################### sum flux ####################
    
    plt.rcParams.update({'font.size': 20})
    
    
    
    
    
    # get the max radius where ha sn >= 3
    radius_max_kpc = np.nanmax(radius_kpc[ha_sn>=3])
    r_set = int(radius_max_kpc)+1
    if r_set < 2:
        r_set = 2
    radius_array = np.arange(r_set)
    
    
    ##### n2o2 with dust correction
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        
        
        if model_flux_sn:
            if nii_sn_cut is not None: # as we use ha gist to do dust corr, so use gist sn for ha and hb
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn_model>=nii_sn_cut)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
        else:
            if nii_sn_cut is not None:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn>=nii_sn_cut)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
        
        
        #if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
        if len(nii_map[query_all]) < 1: 
            break
        
        
        #query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        #if len(nii_map[query_all])/len(nii_map[query])<sumflux_radius_fill_perc:
        #    break
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_gist[query_all]) # use gist to calculate balmer
        hb_sum = np.nansum(hb_gist[query_all]) # use gist to calculate balmer
        oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_gist_err[query_all])**2)) # use gist err, as use gist flux  
        hb_err_sum = np.sqrt(np.nansum((hb_gist_err[query_all])**2))
        oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        ###
        
        nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_sum,
                                                         flux_obs_err=nii_err_sum,
                                                         wave=emi_wave.nii_wave,
                                                         ha=ha_sum,hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        # OII correct 
        
        oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_sum,
                                                         flux_obs_err=oii_err_sum,
                                                         wave=emi_wave.oii_wave,
                                                         ha=ha_sum,
                                                         hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        logR_map = np.log10(nii_corr/oii_corr)
        metal_map_n2o2_gistdust_sumflux = metal_k19(logR_map)
        
        # nii_corr/oii_corr with error
        lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                                a_err=nii_err_corr, 
                                                b_err=oii_err_corr)
        
        metal_i, metal_err_withdust_sumflux = calculate_z_and_error(x=lognii_oii, 
                                                   x_err=lognii_oii_err, 
                                                   y=-3.17)
        
        metal_array.append(metal_map_n2o2_gistdust_sumflux[0])
        metal_err_array.append(metal_err_withdust_sumflux[0])
    
    
    
    
    
    metal_mean_n2o2_dustc = np.array(metal_array)
    metal_mean_err_n2o2_dustc = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    
    
    #center of the bin
    #radius_array = radius_array + 0.5
    
    #print('n2o2 with dust')
    ###print(metal_mean)
    #print(metal_mean_err)
    #print(radius_array)
    #print(metal_mean.shape)
    #print(metal_mean_err.shape)
    #print(radius_array.shape)
    
    
    
    ##### n2ha 
    
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        #query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        #if len(nii_map[query_all])/len(nii_map[query])<sumflux_radius_fill_perc:
        #    break
        
        if model_flux_sn:
            if nii_sn_cut is not None: 
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn_model>=nii_sn_cut)&(ha_sn_model>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn_model>=3)
        else:
            if nii_sn_cut is not None:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn>=nii_sn_cut)&(ha_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        
        
        
        #if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
        if len(nii_map[query_all]) < 1:
            break
        
        
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_map[query_all]) # use b3d to calculate metal!!
        #hb_sum = np.nansum(hb_gist[query_all]) 
        #oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_err[query_all])**2))
        #hb_err_sum = np.sqrt(np.nansum((hb_err[query_all])**2))
        #oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_sum, b=ha_sum, 
                                                a_err=nii_err_sum, 
                                                b_err=ha_err_sum)
        
        metal_map_n2ha_sumflux, metal_err_n2ha_sumflux = calculate_z_and_error(x=lognii_ha, 
                                                   x_err=lognii_ha_err, 
                                                   y=-3.17,R='N2Ha')
    
        
        
        metal_array.append(metal_map_n2ha_sumflux)
        metal_err_array.append(metal_err_n2ha_sumflux)
    
    metal_mean_n2ha = np.array(metal_array)
    metal_mean_err_n2ha = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    
    
    
    # center of the bin
    #radius_array = radius_array + 0.5
    
    r2 = find_half_dex_radius(metal_mean_err_n2o2_dustc)
    r3 = find_half_dex_radius(metal_mean_err_n2ha)
    r_min = min(r2,r3)
    if r_min<3:
        r_min=3
        
    # find the r min for spaxel average
    r_min_spax_n2o2 = axplot_bin_metal_with_err(ax=None,metal_map=metal_map_n2o2_gistdust,
                              metal_err_map=metal_err_withdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='spaxel average',R='N2O2',plot_indivspax=False,
                     radius_fill_perc=indvspax_radius_fill_perc,fit_param=fit_param,
                     legend=True,get_arraylenth=True)
    
    r_min_spax_n2ha = axplot_bin_metal_with_err(ax=None,metal_map=metal_map_n2ha,
                              metal_err_map=metal_err_n2ha,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title=None,R='N2Ha',plot_indivspax=False,
                     radius_fill_perc=indvspax_radius_fill_perc,fit_param=fit_param,
                     get_arraylenth=True)
        
    
    #r_min_spax = np.min([r_min_spax_n2ha,r_min_spax_n2o2])
    
    fig, ax = plt.subplots(2,3,figsize=(22,12),gridspec_kw={'hspace':0.3}) 
                           #gridspec_kw={'wspace':0.5,'hspace':0.35})
    #formatter = FuncFormatter(one_decimal)
    ax = ax.ravel()
    
    # Get original positions
    pos0 = ax[0].get_position()
    pos1 = ax[1].get_position()
    pos2 = ax[2].get_position()
    
    # Row 0: move ax[1] left, and ax[2] to follow it
    ax[1].set_position([
        pos0.x1 + 0.05,
        pos1.y0,
        pos1.width,
        pos1.height
    ])
    ax[2].set_position([
        pos0.x1 + 0.05 + pos1.width + 0.05,
        pos2.y0,
        pos2.width,
        pos2.height
    ])
    
    # Row 1: same for ax[3], ax[4], ax[5]
    pos3 = ax[3].get_position()
    pos4 = ax[4].get_position()
    pos5 = ax[5].get_position()
    
    ax[4].set_position([
        pos3.x1 + 0.05,
        pos4.y0,
        pos4.width,
        pos4.height
    ])
    ax[5].set_position([
        pos3.x1 + 0.05 + pos4.width + 0.05,
        pos5.y0,
        pos5.width,
        pos5.height
    ])

    
    
    
    
    
    img = mpimg.imread(white_light_img_path)
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[3].axis('off')
    
        
    ##############################
    # N2O2 metallicity map with dust correction
    
    clim = map_limits(metal_map_n2o2_gistdust,pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    
    im0 = ax[1].imshow(metal_map_n2o2_gistdust,extent=img_extent,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    #cb0 = plt.colorbar(im0,ax=ax[1],fraction=0.047)
    #cb0.set_label(label='12+log(O/H)',fontsize=20)
    
    # Add colorbar inside the image
    cax0 = inset_axes(ax[1], width="5%", height="40%", loc='upper right', 
                     bbox_to_anchor=(colorbar_xpos, 0.0, 1, 1),
                     bbox_transform=ax[1].transAxes, borderpad=0)
    
    cb0 = fig.colorbar(im0, cax=cax0)
    #cb0.set_label(label='12+log(O/H)',fontsize=12)
    cb0.ax.tick_params(labelsize=12)
    #cb0.ax.yaxis.set_major_formatter(formatter)  # force one decimal place

    
    ax[1].set_title('N2O2')
    
    
    ax[1].set_xlabel(r'$\Delta$RA(")')
    ax[1].set_ylabel(r'$\Delta$Dec(")')
    #ax[0].set_ylabel('N2O2')
    
    
    # Prepare X, Y in real-world coordinates for contour
    ny, nx = metal_map_n2o2_gistdust.shape
    x = np.linspace(img_extent[0], img_extent[1], nx)
    y = np.linspace(img_extent[2], img_extent[3], ny)
    X, Y = np.meshgrid(x, y)
    
    ax[1].contour(X,Y, radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid',label='0.5 R$_\mathrm{e}$')
    ax[1].contour(X,Y, radius_re, levels=[1], colors='darkseagreen', linewidths=2, linestyles='dashed',label='1 R$_\mathrm{e}$')
    ax[1].contour(X,Y, radius_re, levels=[1.5], colors='steelblue', linewidths=2, linestyles='-.',label='1.5 R$_\mathrm{e}$')
    
    x_real = x_cen_pix / nx * (img_extent[1] - img_extent[0]) + 1*img_extent[0]
    y_real = y_cen_pix / ny * (img_extent[3] - img_extent[2]) + 1*img_extent[2]
    
    #ax[1].scatter(x_real,y_real,c='gold',marker='x')
    
    linestylelist=['solid','dashed','-.']
    colorlist=['r','darkseagreen','steelblue']
    label_column=['0.5 R$_\mathrm{e}$','1 R$_\mathrm{e}$','1.5 R$_\mathrm{e}$']
    columns = [ax[1].plot([], [], c=colorlist[i],linestyle=linestylelist[i])[0] for i in range(3)]

    ax[1].legend( columns,  label_column,loc='lower right', prop={'size': 14})
    
    
    #### gradient n2o2 with dust corr
    fit_radius_array = np.array(radius_array_new)+0.5
    popt_mean, pcov_mean = curve_fit(linear_func, fit_radius_array[:r_min], metal_mean_n2o2_dustc[:r_min],
                                     sigma=metal_mean_err_n2o2_dustc[:r_min])
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    
    
    #axplot_sumflux_bin_metal(ax=ax[1], metal_mean=metal_mean_n2o2_dustc[:r_min], 
    #                         metal_mean_err=metal_mean_err_n2o2_dustc[:r_min], 
    #                         title='sum flux', re_kpc=re_kpc, 
    #                         radius_array=fit_radius_array[:r_min],fit_param=fit_param,
    #                         color='sienna')
    
    axplot_bin_metal_with_err(ax=ax[2],metal_map=metal_map_n2o2_gistdust,
                              metal_err_map=metal_err_withdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title=None,R='N2O2',plot_indivspax=True,
                     radius_fill_perc=indvspax_radius_fill_perc,fit_param=fit_param,
                     legend=True,r_min_set=r_min_spax,bin_size=bin_size)
    
    
    ##############################
    # N2Ha metallicity map
    
    clim = map_limits(metal_map_n2ha,pct=90)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    
    im3 = ax[4].imshow(metal_map_n2ha,extent=img_extent,
                        origin='lower',
                        interpolation='nearest',
                        norm=norm,
                        cmap=cmap.flux)
    #cb3 = plt.colorbar(im3,ax=ax[4],fraction=0.047)
    #cb3.set_label(label='12+log(O/H)',fontsize=20)
    # Add colorbar inside the image
    cax3 = inset_axes(ax[4], width="5%", height="40%", loc='upper right', 
                     bbox_to_anchor=(colorbar_xpos, 0.0, 1, 1),
                     bbox_transform=ax[4].transAxes, borderpad=0)
    
    cb3 = fig.colorbar(im3, cax=cax3)
    #cb3.set_label(label='12+log(O/H)',fontsize=12)
    cb3.ax.tick_params(labelsize=12)
    #cb3.ax.yaxis.set_major_formatter(formatter)  # force one decimal place
    if set_cb_ticks is not None:
        cb3.set_ticks(set_cb_ticks)
    
    #ax[3].set_title('metallicity map')
    ax[4].set_xlabel(r'$\Delta$RA(")')
    ax[4].set_ylabel(r'$\Delta$Dec(")')
    ax[4].set_title('N2H$\\alpha$')
    
    ax[4].contour(X,Y, radius_re, levels=[0.5], colors='r', linewidths=2, linestyles='solid')
    ax[4].contour(X,Y, radius_re, levels=[1], colors='darkseagreen', linewidths=2, linestyles='dashed')
    ax[4].contour(X,Y, radius_re, levels=[1.5], colors='steelblue', linewidths=2, linestyles='-.')
    #ax[4].scatter(x_real,y_real,c='gold',marker='x')
    
    
    
    ##### gradient n2ha
    
    popt_mean, pcov_mean = curve_fit(linear_func, fit_radius_array[:r_min], metal_mean_n2ha[:r_min],
                                     sigma=metal_mean_err_n2ha[:r_min])
    a_mean, b_mean = popt_mean
    a_mean_err, b_mean_err = np.sqrt(np.diag(pcov_mean))
    
    
    #axplot_sumflux_bin_metal(ax=ax[4], metal_mean=metal_mean_n2ha[:r_min], 
    #                         metal_mean_err=metal_mean_err_n2ha[:r_min], 
    #                         title=None, re_kpc=re_kpc, 
    #                         radius_array=fit_radius_array[:r_min],R='N2Ha',
    #                         fit_param=fit_param,color='sienna')
    
    axplot_bin_metal_with_err(ax=ax[5],metal_map=metal_map_n2ha,
                              metal_err_map=metal_err_n2ha,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title=None,R='N2Ha',plot_indivspax=True,
                     radius_fill_perc=indvspax_radius_fill_perc,
                     fit_param=fit_param,r_min_set=r_min_spax,bin_size=bin_size)
    
    if plot_title is not None:
        ax[0].set_title(plot_title)
    #plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    
    plt.show()

def one_decimal(x, pos):
    return f"{x:.1f}"

def paperplot_data_sumflux_all(datapath,id_list,color='b',savepath=None):
    
    gala_num = len(id_list)
    column_num = int(np.sqrt(gala_num))
    row_num = int(gala_num/column_num)+1
    
    v221_path = '/Users/ymai0110/Documents/Blobby3D/v221/'
    r50_csv = pd.read_csv(v221_path+'data_from_others/vdisp_sfr_mass_re_combinererun.csv')
    profit_csv = pd.read_csv('/Users/ymai0110/Documents/Blobby3D/v221/data_from_others/MAGPI_ProfitSersicCat_v0.4.csv',header=None,usecols=[0,2])
    profit_csv.columns = ['MAGPIID', 're']
    
    plt.rcParams.update({'font.size': 25})
    fig, ax = plt.subplots(row_num,column_num,figsize=(column_num*5,row_num*5), 
                           gridspec_kw={'wspace':0.3,'hspace':0.35})
    ax = ax.ravel()
    v221_path = '/Users/ymai0110/Documents/Blobby3D/v221/'
    
    for i, idd in enumerate(id_list):
        id_str = str(idd)
    
        #index = r50_csv['MAGPIID']==idd
        #re_kpc = r50_csv['R50_conti_profound_kpc'][index]
        #re_kpc = re_kpc.to_numpy()[0]
        profound = pd.read_csv(v221_path+'profoundsources_v221/MAGPI'+id_str[:4]+'_profoundsources.csv')
        z_list = profound['Z']
        profound_id_list = profound['MAGPIID'].to_numpy()
        id_index = np.where(profound_id_list==idd)[0][0]
        z = z_list[id_index]
        re_profit_arcsec = profit_csv.loc[profit_csv['MAGPIID']==idd,'re'].values[0]
        re_kpc = arcsec_to_kpc(re_profit_arcsec,z)
        
        #parent_path = '/Users/ymai0110/Documents/Blobby3D_metal/v221/'
        #metal_csv = pd.read_csv(parent_path + 'metalgradient/allgalaplot_sumflux/'+id_str+'_sumflux.csv')
        metal_csv = pd.read_csv(datapath+id_str+'_sumflux.csv')
        
        radius_array = metal_csv['r_kpc'].to_numpy()
        metal_mean = metal_csv['n2o2_dustc_metal'].to_numpy()
        metal_mean_err = metal_csv['n2o2_dustc_metal_err'].to_numpy()
        
        axplot_sumflux_bin_metal(ax=ax[i], metal_mean=metal_mean, 
                                 metal_mean_err=metal_mean_err, 
                                 title=id_str, re_kpc=re_kpc, 
                                 radius_array=radius_array,set_xylabel=False,
                                 color=color,fit_param=False)
        if i%column_num==0:
            ax[i].set_ylabel('log(O/H)+12')
        if i>=len(id_list)-column_num:
            ax[i].set_xlabel('radius [kpc]')
    for i in range(len(id_list),column_num*row_num):
        ax[i].set_visible(False)
    
    if savepath is not None:
        plt.savefig(savepath,dpi=300,bbox_inches='tight')
    
    plt.show()
    
    
    

def data_sumflux(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,
                                 ha_err,hb_err,oii_err,nii_err,
                                 ha_gist_err,hb_gist_err,oii_gist_err,nii_gist_err,
                                 pa,x_cen_pix,y_cen_pix,ellip,
                                 foreground_E_B_V,
                                 z,re_kpc=None,savepath=None,
                                 r_option='kpc',plot_title=None,
                                 limit_metal_range=False,
                                 savecsv=False,csvpath=None,
                                 nii_sn_cut=None,model_flux_sn=True,
                                 radius_fill_perc=0.5):
    
    plt.rcParams.update({'font.size': 25})
    
    
    ha_sn_model = ha_map/ha_err
    hb_sn_model = hb_map/hb_err
    nii_sn_model = nii_map/nii_err
    oii_sn_model = oii_map/oii_err
    
    # balmer decrement of GIST
    ha_gist = ha_sn * ha_gist_err
    hb_gist = hb_sn * hb_gist_err
    nii_gist = nii_sn * nii_gist_err
    oii_gist = oii_sn * oii_gist_err
    
    dist_arr = ellip_distarr(size=ha_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    # get the max radius where ha sn >= 3
    radius_max_kpc = np.nanmax(radius_kpc[ha_sn>=3])
    r_set = int(radius_max_kpc)+1
    if r_set < 2:
        r_set = 2
    radius_array = np.arange(r_set)
    
    
    ##### n2o2 w/o dust correction
    
    
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        
        
        
        if model_flux_sn:
            if nii_sn_cut is not None: # as we use ha gist to do dust corr, so use gist sn for ha and hb
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn_model>=nii_sn_cut)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
        else:
            if nii_sn_cut is not None:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn>=nii_sn_cut)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
        # use radius_fill_perc=0.5 to ensure up to that range have reasonable
        # blob measurement
        # the range for gradient measurement will be decide later (err<0.5dex)
        #if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
        if len(nii_map[query_all]) < 1:
            break
        
        
        radius_array_new.append(r)
        
        #ha_sum = np.nansum(ha_gist[query_all]) # use gist to calculate balmer
        #hb_sum = np.nansum(hb_gist[query_all]) # use gist to calculate balmer
        oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        #ha_err_sum = np.sqrt(np.nansum((ha_err[query_all])**2))
        #hb_err_sum = np.sqrt(np.nansum((hb_err[query_all])**2))
        oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
    
        logR_map = np.log10(nii_sum/oii_sum)
        metal_map_n2o2_nodustc = metal_k19(logR_map)
        
        lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_sum, b=oii_sum, 
                                                a_err=nii_err_sum, 
                                                b_err=oii_err_sum)
        
        metal_i, metal_err_nodust = calculate_z_and_error(x=lognii_oii, 
                                                   x_err=lognii_oii_err, 
                                                   y=-3.17)
        
        metal_array.append(metal_map_n2o2_nodustc)
        metal_err_array.append(metal_err_nodust)
    
    metal_mean_n2o2wodustc = np.array(metal_array)
    metal_mean_err_n2o2wodustc = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    #print('n2o2 wo dust')
    #print(metal_mean.shape)
    #print(metal_mean_err.shape)
    #print(radius_array.shape)
    
    # center of the bin
    radius_array = radius_array + 0.5
    
    
    
    
    
    ##### n2o2 with dust correction
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        #query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        #if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
        #    break
        
        if model_flux_sn:
            if nii_sn_cut is not None: # as we use ha gist to do dust corr, so use gist sn for ha and hb
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn_model>=nii_sn_cut)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn_model>=3)&(ha_sn>=3)&(hb_sn>=3)
        else:
            if nii_sn_cut is not None:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn>=nii_sn_cut)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(oii_sn>=3)&(ha_sn>=3)&(hb_sn>=3)
        # use radius_fill_perc=0.5 to ensure up to that range have reasonable
        # blob measurement
        # the range for gradient measurement will be decide later (err<0.5dex)
        #if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
        if len(nii_map[query_all]) < 1:
            break
        
        
        
        
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_gist[query_all]) # use gist to calculate balmer
        hb_sum = np.nansum(hb_gist[query_all]) # use gist to calculate balmer
        oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_gist_err[query_all])**2)) # use gist err, as use gist flux  
        hb_err_sum = np.sqrt(np.nansum((hb_gist_err[query_all])**2))
        oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        ###
        
        nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_sum,
                                                         flux_obs_err=nii_err_sum,
                                                         wave=emi_wave.nii_wave,
                                                         ha=ha_sum,hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        # OII correct 
        
        oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_sum,
                                                         flux_obs_err=oii_err_sum,
                                                         wave=emi_wave.oii_wave,
                                                         ha=ha_sum,
                                                         hb=hb_sum,
                                                         ha_err=ha_err_sum,
                                                         hb_err=hb_err_sum,
                                    foreground_E_B_V=foreground_E_B_V,z=z)
        
        logR_map = np.log10(nii_corr/oii_corr)
        metal_map_n2o2_gistdust = metal_k19(logR_map)
        
        # nii_corr/oii_corr with error
        lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                                a_err=nii_err_corr, 
                                                b_err=oii_err_corr)
        
        metal_i, metal_err_withdust = calculate_z_and_error(x=lognii_oii, 
                                                   x_err=lognii_oii_err, 
                                                   y=-3.17)
        
        metal_array.append(metal_map_n2o2_gistdust[0])
        metal_err_array.append(metal_err_withdust[0])
    
    
    
    
    
    metal_mean_n2o2_dustc = np.array(metal_array)
    metal_mean_err_n2o2_dustc = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    
    
    #center of the bin
    radius_array = radius_array + 0.5
    
    #print('n2o2 with dust')
    ###print(metal_mean)
    #print(metal_mean_err)
    #print(radius_array)
    #print(metal_mean.shape)
    #print(metal_mean_err.shape)
    #print(radius_array.shape)
    
    
    
    ##### n2ha 
    
    
    radius_array_new = []
    metal_array = []
    metal_err_array = []
    radius_array = np.arange(r_set)
    for r in radius_array:
        query = (radius_kpc>=r)&(radius_kpc<r+1)
        #query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        #if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
        #    break
        
        if model_flux_sn:
            if nii_sn_cut is not None: 
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn_model>=nii_sn_cut)&(ha_sn_model>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn_model>=3)
        else:
            if nii_sn_cut is not None:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(nii_sn>=nii_sn_cut)&(ha_sn>=3)
            else:
                query_all = (radius_kpc>=r)&(radius_kpc<r+1)&(ha_sn>=3)
        
        
        
        #if len(nii_map[query_all])/len(nii_map[query])<radius_fill_perc:
        if len(nii_map[query_all]) < 1:
            break
        
        radius_array_new.append(r)
        
        ha_sum = np.nansum(ha_map[query_all]) # use b3d to calculate metal!!
        #hb_sum = np.nansum(hb_gist[query_all]) 
        #oii_sum = np.nansum(oii_map[query_all])
        nii_sum = np.nansum(nii_map[query_all])
        
        ha_err_sum = np.sqrt(np.nansum((ha_err[query_all])**2))
        #hb_err_sum = np.sqrt(np.nansum((hb_err[query_all])**2))
        #oii_err_sum = np.sqrt(np.nansum((oii_err[query_all])**2))
        nii_err_sum = np.sqrt(np.nansum((nii_err[query_all])**2))
        
        
        lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_sum, b=ha_sum, 
                                                a_err=nii_err_sum, 
                                                b_err=ha_err_sum)
        
        metal_map_n2ha, metal_err_n2ha = calculate_z_and_error(x=lognii_ha, 
                                                   x_err=lognii_ha_err, 
                                                   y=-3.17,R='N2Ha')
    
        
        
        metal_array.append(metal_map_n2ha)
        metal_err_array.append(metal_err_n2ha)
    
    metal_mean_n2ha = np.array(metal_array)
    metal_mean_err_n2ha = np.array(metal_err_array)
    radius_array = np.array(radius_array_new)
    
    
    
    
    # center of the bin
    radius_array = radius_array + 0.5
    
    #print('n2ha')
    #print(metal_mean)
    #print(metal_mean_err)
    #print(radius_array)
    #print(metal_mean.shape)
    #print(metal_mean_err.shape)
    #print(radius_array.shape)
    
    
    #### decide the radius to fit
    
    r1 = find_half_dex_radius(metal_mean_err_n2o2wodustc)
    r2 = find_half_dex_radius(metal_mean_err_n2o2_dustc)
    r3 = find_half_dex_radius(metal_mean_err_n2ha)
    r_min = min(r1,r2,r3)
    if r_min<3:
        r_min=3
    
    
    
    fit_radius_array = np.array(radius_array_new)+0.5
    if savecsv:
    
        df = pd.DataFrame({'r_kpc':fit_radius_array[:r_min],
                           'n2o2_nodustc_metal':metal_mean_n2o2wodustc[:r_min],
                           'n2o2_nodustc_metal_err':metal_mean_err_n2o2wodustc[:r_min],
                           'n2o2_dustc_metal':metal_mean_n2o2_dustc[:r_min],
                           'n2o2_dustc_metal_err':metal_mean_err_n2o2_dustc[:r_min],
                           'n2ha_metal':metal_mean_n2ha[:r_min],
                           'n2ha_metal_err':metal_mean_err_n2ha[:r_min]})
        df.to_csv(csvpath)


def data_annularbin(ha_map,hb_map,oii_map,nii_map,
                                 ha_sn,hb_sn,oii_sn,nii_sn,
                                 ha_err,hb_err,oii_err,nii_err,
                                 ha_gist_err,hb_gist_err,oii_gist_err,nii_gist_err,
                                 pa,x_cen_pix,y_cen_pix,ellip,
                                 foreground_E_B_V,
                                 z,re_kpc=None,savepath=None,
                                 ha_gist=None,hb_gist=None,oii_gist=None,
                                 nii_gist=None,
                                 r_option='kpc',plot_title=None,
                                 limit_metal_range=False,
                                 savecsv=False,csvpath=None,
                                 nii_sn_cut=True,model_flux_sn=False,
                                 radius_fill_perc=0.8):
    

    ha_sn_model = ha_map/ha_err
    hb_sn_model = hb_map/hb_err
    nii_sn_model = nii_map/nii_err
    oii_sn_model = oii_map/oii_err
    
    
    plt.rcParams.update({'font.size': 25})
    
    dist_arr = ellip_distarr(size=ha_map.shape, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa*np.pi/180,angle_type='WTN')
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    
    
    
    
    ##############################
    # NII/OII map
    
    
    ##############################
    # balmer decrement of GIST
    #ha_gist = ha_sn * ha_err
    #hb_gist = hb_sn * hb_err
    #nii_gist = nii_sn * nii_err
    #oii_gist = oii_sn * oii_err
    
    
    
    ##############################
    # metal n2o2 as func of radius, without dust correction
    
    logR_map = np.log10(nii_map/oii_map)
    metal_map_n2o2_nodustc = metal_k19(logR_map)
    
    if model_flux_sn:
        if nii_sn_cut:
            query = (ha_sn_model>=3) & (hb_sn_model>=3) & (oii_sn_model>=3) & (nii_sn_model>=3)
        else:
            query = (ha_sn_model>=3) & (hb_sn_model>=3) & (oii_sn_model>=3)
    else:
        if nii_sn_cut:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3)
        else:
            query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map_n2o2_nodustc[mask_region] = np.nan
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    # get metal error
    #nii_err_map = nii_map/nii_sn # wrong !!! can't divide by nii_sn, could be 0
    #oii_err_map = oii_map/oii_sn
    
    
    
    
    lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_map, b=oii_map, 
                                            a_err=nii_err, 
                                            b_err=oii_err)
    
    metal_i, metal_err_nodust = calculate_z_and_error(x=lognii_oii, 
                                               x_err=lognii_oii_err, 
                                               y=-3.17)
    
    
    
    
    ##############################
    # metal n2o2 as func of radius, with gist dust correction and err calculation
    
    # NII correct
    nii_corr, nii_err_corr = intrinsic_flux_with_err(flux_obs=nii_map,
                                                     flux_obs_err=nii_err,
                                                     wave=emi_wave.nii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z)
    
    # OII correct 
    
    oii_corr, oii_err_corr = intrinsic_flux_with_err(flux_obs=oii_map,
                                                     flux_obs_err=oii_err,
                                                     wave=emi_wave.oii_wave,
                                                     ha=ha_gist,hb=hb_gist,
                                                     ha_err=ha_gist_err,hb_err=hb_gist_err,
                                foreground_E_B_V=foreground_E_B_V,z=z)
    
    logR_map = np.log10(nii_corr/oii_corr)
    metal_map_n2o2_gistdust = metal_k19(logR_map)
    
    #if nii_sn_cut:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3) & (nii_sn>=3)
    #else:
    #    query = (ha_sn>=3) & (hb_sn>=3) & (oii_sn>=3)
    
    mask_region = ~query
    
    metal_map_n2o2_gistdust[mask_region] = np.nan
    
    radius_kpc = pix_to_kpc(radius_in_pix=dist_arr, z=z)
    radius_re = radius_kpc/re_kpc
    
    
    # nii_corr/oii_corr with error
    lognii_oii,lognii_oii_err = log10_ratio_with_error(a=nii_corr, b=oii_corr, 
                                            a_err=nii_err_corr, 
                                            b_err=oii_err_corr)
    
    metal_i, metal_err_withdust = calculate_z_and_error(x=lognii_oii, 
                                               x_err=lognii_oii_err, 
                                               y=-3.17)
    
    
    ##############################
    # metal gradient - n2ha
    
    if model_flux_sn:
        metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn_model,
                                       nii_sn=nii_sn_model)
    else:
        metal_map_n2ha = get_metal_map_n2ha(ha_map=ha_map,nii_map=nii_map,ha_sn=ha_sn,
                                       nii_sn=nii_sn)
    
    #nii_err_map = nii_map/nii_sn
    #ha_err_map = ha_map/ha_sn
    
    lognii_ha,lognii_ha_err = log10_ratio_with_error(a=nii_map, b=ha_map, 
                                            a_err=nii_err, 
                                            b_err=ha_err)
    
    metal_i, metal_err_n2ha = calculate_z_and_error(x=lognii_ha, 
                                               x_err=lognii_ha_err, 
                                               y=-3.17,R='N2Ha')
    
    
    
    savedata_bin_metal_with_err(metal_map=metal_map_n2o2_nodustc,
                              metal_err_map=metal_err_nodust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 w/o dust c',R='N2O2',
                     radius_fill_perc=radius_fill_perc,
                     savecsvpath=csvpath+'n2o2_nodustc.csv')
    
    savedata_bin_metal_with_err(metal_map=metal_map_n2o2_gistdust,
                              metal_err_map=metal_err_withdust,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2o2 with gist dust c',R='N2O2',
                     radius_fill_perc=radius_fill_perc,
                     savecsvpath=csvpath+'n2o2_dustc.csv')
    
    savedata_bin_metal_with_err(metal_map=metal_map_n2ha,
                              metal_err_map=metal_err_n2ha,
                     ha_map=ha_map,radius_map=radius_kpc,re_kpc=re_kpc,
                     title='n2ha',R='N2Ha',
                     radius_fill_perc=radius_fill_perc,
                     savecsvpath=csvpath+'n2ha.csv')
    
    
    
    
def savedata_bin_metal_with_err(metal_map,metal_err_map,ha_map,radius_map,
                              title,re_kpc,R='N2O2',
                     legend=False,get_arraylenth=False,overplotax=None,
                     radius_fill_perc=0.8,plot_indivspax=False,savefitspath=None,
                     savecsvpath=None):
    '''
    
    savefitspath: save the fits file of metallicity map
    savecsvpath: save the radius vs metallicity data as csv file
    
    '''
    if savefitspath is not None:
        # to do
        
        todo=0
    
    
    
    metal_mean, metal_mean_err, radius_array = get_bin_metal_with_err(
        metal_map=metal_map,metal_err_map=metal_err_map,ha_map=ha_map,
        radius_map=radius_map, radius_fill_perc=radius_fill_perc,weight_mode='no')
    #print('unweighted: mean_array, err_aray')
    #print(metal_mean)
    #print(metal_mean_err)
    
    metal_inv_var, metal_inv_var_err, radius_array = get_bin_metal_with_err(
        metal_map=metal_map,metal_err_map=metal_err_map,ha_map=ha_map,
        radius_map=radius_map, radius_fill_perc=radius_fill_perc,weight_mode='inv_var')
    #print('inverse variance weighted')
    #print(metal_inv_var)
    #print(metal_inv_var_err)
    
    metal_ha_ave, metal_ha_ave_err, radius_array = get_bin_metal_with_err(
        metal_map=metal_map,metal_err_map=metal_err_map,ha_map=ha_map,
        radius_map=radius_map, radius_fill_perc=radius_fill_perc,weight_mode='ha')
    
    if savecsvpath is not None:
        df = pd.DataFrame({'radius_kpc':radius_array,
                           'metal_mean':metal_mean,
                           'metal_mean_err':metal_mean_err,
                           'metal_inv_var':metal_inv_var,
                           'metal_inv_var_err':metal_inv_var_err,
                           'metal_ha_ave':metal_ha_ave,
                           'metal_ha_ave_err':metal_ha_ave_err})
        df.to_csv(savecsvpath)

def get_combined_csv(path,display=False):
    import glob
    #parent_path = '/Users/ymai0110/Documents/Blobby3D_metal/v221/'
    #csv_path = parent_path + 'plots/metalgradient/'+path+'/'
    # Specify the path to your CSV files
    csv_files_path = path+'*.csv'
    
    # Get a list of all CSV files in the specified path
    csv_files = glob.glob(csv_files_path)
    
    # Initialize an empty list to store individual DataFrames
    dataframes = []
    
    # Loop through the list of CSV files and read each into a DataFrame
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Display the combined DataFrame
    if display:
        print(combined_df)
    return combined_df