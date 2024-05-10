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
sys.path.insert(0,'/Users/ymai0110/Documents/Blobby3D_develop/Blobby3D/pyblobby3d/src/pyblobby3d/')

import numpy as np
import matplotlib.pyplot as plt

from pyblobby3d.post_blobby3d import PostBlobby3D
from pyblobby3d.moments import SpectralModel
 
import matplotlib as mpl
from astropy.io import fits

from astropy.cosmology import LambdaCDM
lcdm = LambdaCDM(70,0.3,0.7)


def plot_oii_gradient(post_b3d,global_param,ellip,savepath=None):
    '''
    post_b3d: class PostBlobby3D
    global_param: pandas DataFrame, output of global parameter of Blobby3D
    ellip=ellipticity of the ellipse=1-b/a
    
    '''
    
    wave = post_b3d.metadata.get_axis_array('r')
    x_array = post_b3d.metadata.get_axis_array('x')
    y_array = post_b3d.metadata.get_axis_array('y')
    dr = post_b3d.metadata.dr
    dx = post_b3d.metadata.dx
    dy = post_b3d.metadata.dy
    
    x_cen_arcs = global_param['XC'].median()
    y_cen_arcs = global_param['YC'].median()
    pa = global_param['PA'].median() # for PA in b3d output, use angletype='NTE'
    x_cen_pix = find_closest_index(x_array,x_cen_arcs)
    y_cen_pix = find_closest_index(y_array, y_cen_arcs)
    
    sample = 0
    precon_flux_oii_3727 = post_b3d.maps[sample, 0]
    precon_flux_oii_3729 = post_b3d.maps[sample,1]
    
    oii_sum_map = precon_flux_oii_3727 + precon_flux_oii_3729
    
    dist_arr = ellip_distarr(size=oii_sum_map, centre=(x_cen_pix,y_cen_pix),
                             ellip=ellip, pa=pa,angle_type='NTE')
    
    fig, ax = plt.subplots()
    ax.scatter(dist_arr, oii_sum_map)
    ax.set_xlabel('radius')
    ax.set_ylabel('log OII sum')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath,bbox_inches='tight',dpi=300)
    
    plt.show()
    
    
    
    


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
