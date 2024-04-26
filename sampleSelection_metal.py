#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:23:17 2024

@author: ymai0110
"""

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from os import walk
import pandas as pd

def snpixel(flux_data,fluxerr_data,snlim,mask=True,dilated_mask=None):
    '''
    calculate the number of spaxels which sn of given emission line > snlim
    default: only calculate spaxels not been masked (use the dilated mask)

    '''
    
    
    
    
    
    emi_sn = flux_data/fluxerr_data
    if mask == True:
        emi_sn_1d = emi_sn[dilated_mask==0]
        npixel = sum(emi_sn_1d >= snlim)
        return npixel
        
        
    emi_sn_1d = emi_sn.flatten()
    npixel = sum(emi_sn_1d >= snlim)
    
    return npixel

def inclination(semimaj,semimin,correction=False,q0=0.2):
    '''in rad'''
    inc = np.empty_like(semimaj)
    inc[:] = np.nan
    if correction==False:
        inc = np.arccos(semimin/semimaj)
    if correction==True:
        
        index1 = ((semimin/semimaj) >= q0)
        
        inc[index1] = np.arccos(np.sqrt(((semimin[index1]/semimaj[index1])**2 - q0**2)/(1 - q0**2)))
        
        index2 = ((semimin/semimaj) < q0)
        
        inc[index2] = np.arccos(0)
        
        
        
        
    
    return inc

def createsnpixelcsv(field,emidir,emi_flux_list,emi_fluxerr_list,savepath,
                     snlim):
    '''
    create the csv file which give the number of pixel have sn of given emi 
    line greater than given snlim

    Parameters
    ----------
    field : TYPE
        DESCRIPTION.
    emidir : TYPE
        DESCRIPTION.
    savepath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    maplist = next(walk(emidir), (None, None, []))[2]
    magpiidlist = []
    
    df = pd.DataFrame()
    
    for emimap in maplist:
        if emimap[0]== '.':
            maplist.remove(emimap)
            continue
        if emimap[-1]!='s':
            maplist.remove(emimap)
            continue
    
    galaxy_num = len(maplist)
    emi_line_num = len(emi_flux_list)
    snpix_array = np.zeros((galaxy_num,emi_line_num))

    
    for i, emimap in enumerate(maplist): # go through all galaxies
        if emimap[0]== '.':
            continue
        if emimap[-1]!='s':
            continue
        magpiid = emimap[5:15]
        magpiidlist.append(int(magpiid))
        fits_file = fits.open(emidir+emimap)
        dilated_mask = fits_file['dilated_mask'].data
        # go through all emission lines
        j = 0
        for emi_flux_j, emi_fluxerr_j in zip(emi_flux_list,emi_fluxerr_list):
            if emi_flux_j =='OII_sum':
                emi_fits = fits_file
                oii_3727_f = emi_fits['OII_3727_F'].data
                oii_3727_ferr = emi_fits['OII_3727_FERR'].data
                
                
            
                oii_3730_f = emi_fits['OII_3730_F'].data
                oii_3730_ferr = emi_fits['OII_3730_FERR'].data
                
                
                oii_sum_f = oii_3727_f + oii_3730_f
                oii_sum_ferr = np.sqrt(oii_3727_ferr**2+oii_3730_ferr**2)
                
                emi_flux_data = oii_sum_f
                emi_fluxerr_data = oii_sum_ferr
                
                
            else:
                
                emi_flux_data = fits_file[emi_flux_j].data
                emi_fluxerr_data = fits_file[emi_fluxerr_j].data
            
            
            
            snpixel_j = snpixel(emi_flux_data,emi_fluxerr_data,snlim,mask=True,
                    dilated_mask=dilated_mask)
            
            snpix_array[i,j] = snpixel_j
            
            j = j + 1
        
        
        
        

    
    df['MAGPIID'] = magpiidlist
    for i, emi_flux_i in enumerate(emi_flux_list):
        df[emi_flux_i] = snpix_array[:,i]
    
    df.to_csv(savepath+field+'_snpixel.csv')
    print(field+' is done.')



def plot_snpix_bar(emi_list, snpix_array):
    '''
    plot the number of pixel in log 

    Parameters
    ----------
    emi_list : str
        list of emission line name.
    snpix_array : np.array
        (row=n_galaxy,column=n_emi).

    Returns
    -------
    None.

    '''
    n_emi = len(emi_list)
    n_gala = snpix_array.shape[0]
    
    ypos = np.array(range(n_gala))
    w = (1-0.2)/n_emi # width
    adjusted_ypos = np.add(ypos, w)
    
    # unfinished...

def plot_snpix_scatter(oiipix,hbpix,hapix):
    plt.scatter(oiipix,hbpix,c=hapix,vmin=50,vmax=200)
    plt.xlabel('oii pix')
    plt.ylabel('Hbeta pix')
    plt.show()
    