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



def plot_cubecompare(datapath,emi_line,emi_sn_path,mode='random'):
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
    plt.show()
    
    
def plot_ha_oii_cubecompare(datapath_ha,emi_sn_path_ha,datapath_oii,
                            emi_sn_path_oii,mode):
    
    plot_cubecompare(datapath_ha,'Ha',emi_sn_path_ha,mode)
    plot_cubecompare(datapath_oii,'Oii',emi_sn_path_oii,mode)
    

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
    