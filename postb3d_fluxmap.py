#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:14:22 2024

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

def b3d_fluxmap(post_b3d,line_names,savepath=None):
    '''
    generate the median and error of flux of all posterior samples of b3d
    save into fits file into different extensions

    Parameters
    ----------
    post_b3d : PostBlobby3D class
        DESCRIPTION.
    line_names : list of string
        list of str of name of emission line. It should match the 
        lines fitted in post_b3d in order
    savepath : str, optional
        end with .fits . The default is None.

    Returns
    -------
    hdul : TYPE
        DESCRIPTION.

    '''
    
    # Create Primary HDU (empty or with data, typically primary HDU doesn't have data)

    primary_hdu = fits.PrimaryHDU()
    
    image_list = [primary_hdu]
    
    i = 0
    for line_name in line_names:
        flux_maps_all_samples = post_b3d.maps[:, i] # i=0 for the first line
        
        # median is probably better...
        flux_all_samples_med = np.median(flux_maps_all_samples,axis=0)
        flux_all_samples_err = np.std(flux_maps_all_samples,axis=0)
        flux_hdu = fits.ImageHDU(data=flux_all_samples_med, name=line_name+'_flux')
        flux_err_hdu = fits.ImageHDU(data=flux_all_samples_err, name=line_name+'flux_err')
        image_list.append(flux_hdu)
        image_list.append(flux_err_hdu)
        
        i = i + 1
    
    # Create an HDU list and write to a new FITS file
    hdul = fits.HDUList(image_list)
    
    if savepath is not None:
        # Write to a FITS file
        hdul.writeto(savepath, overwrite=True)
    
    return hdul