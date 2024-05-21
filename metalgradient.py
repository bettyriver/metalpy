#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:56:24 2024

for metalicity gradient plot

part of code from Di Wang

@author: ymai0110
"""

import sys
sys.path.insert(0,'/project/blobby3d/Blobby3D/pyblobby3d/src/pyblobby3d/')
#sys.path.insert(0,'/Users/ymai0110/Documents/Blobby3D_develop/Blobby3D/pyblobby3d/src/pyblobby3d/')

import numpy as np
import matplotlib.pyplot as plt

from post_blobby3d import PostBlobby3D
from moments import SpectralModel
 
import matplotlib as mpl
from astropy.io import fits
import pandas as pd

from astropy.cosmology import LambdaCDM
lcdm = LambdaCDM(70,0.3,0.7)


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
                             ellip=ellip, pa=pa,angle_type='NTE')
    
    
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
                             ellip=ellip, pa=pa,angle_type='NTE')
    
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
                             ellip=ellip, pa=pa,angle_type='NTE')
    
    if plot_data:
        # mask some very low metal sapxels...
        metal_map[metal_map<-10]=np.nan
        ax.scatter(dist_arr, metal_map,label='data',alpha=0.4)
    else:
        ax.scatter(dist_arr, metal_map,label=label,alpha=0.4)
    
    
    
    
    
    
    
    
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
