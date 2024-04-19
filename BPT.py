#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:14:37 2022

bpt

originate from Yong Shi; provide by Xiaoling Yu

@author: ymai0110
"""


import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
import numpy as np
import astropy.units as u
import astropy.constants as const
#from skimage import io
from scipy import misc
from matplotlib import colors
import matplotlib.patches as mpatches
###define bpt region function########
from numpy.ma import is_masked
from matplotlib.gridspec import GridSpec

def bptregion(x, y, mode='N2'):
    '''
    Args:
    lines: dictionary contains all the needed for bpt
    ['Hb-4862', 'OIII-5008','Ha-6564','NII-6585']
    x: log10(NII/Ha) or log10(SII/Ha) or log10(OI/Ha)
    y: log10(OIII/Hb)
    mode: mode "N2" -> x = log10(NII/Ha)
    mode "S2" -> x = log10(SII/Ha)
    mode "O1" -> x = log10(OI/Ha) ! not surpport yet
    Note:
    x, y should be masked array,
    example: x = np.ma.array(x)
    '''
    # check starforming, composite or AGN
    # region = np.zeros_like(lines[0].data)
    if mode == 'N2':
        ke01 = 0.61/(x-0.47)+1.19
        ka03 = 0.61/(x-0.05)+1.3
        schawinski_line = 1.05*x+0.45
        region_AGN = np.logical_or(np.logical_and(x<0.47, y>ke01), x>0.47)
        region_composite = np.logical_and(y<ke01, y>ka03)
        region_starforming = np.logical_and(x<0.05, y<ka03)
        # depleted
        #region_seyfert = np.logical_and(x>np.log10(0.6), y>np.log10(3.))
        #region_liner = np.logical_and(region_AGN, np.logical_and(x>np.log10(0.,!6), y<np.log10(3.)))
        # adapted from Schawinski2007
        region_seyfert = np.logical_and(region_AGN, y>schawinski_line)
        region_liner = np.logical_and(region_AGN, y<schawinski_line)
        if is_masked(x) or is_masked(y):
            return region_AGN.filled(False), region_composite.filled(False),region_starforming.filled(False), region_seyfert.filled(False), region_liner.filled(False)
        else:
            return region_AGN, region_composite, region_starforming,region_seyfert, region_liner
    
    if mode == 'S2':
        ke01_line = 0.72/(x-0.32)+1.3
        seyfert_liner_line = 1.89*x+0.76
        region_seyfert = np.logical_and(np.logical_or(y>ke01_line, x>0.32),y>seyfert_liner_line)
        region_liner = np.logical_and(np.logical_or(y>ke01_line, x>0.32),y<seyfert_liner_line)
        region_starforming = np.logical_and(y<ke01_line, x<0.32)
        if is_masked(x) or is_masked(y):
            return region_seyfert.filled(False), region_liner.filled(False),region_starforming.filled(False)
        else:
            return region_seyfert, region_liner, region_starforming

'''
fitsdoc = fits.open('D:/dissertation/data/galaxytype_criteria_1.fits')
fits_data = fitsdoc[1].data
plates = fits_data['plate']
ifus = fits_data['ifu']
plateifus = fits_data['plateifu']
galaxytypes = fits_data['galaxytype']

for i in range(len(plates)):

    plate = plates[i]
    ifu = ifus[i]
    if galaxytypes[i] == 0:
        figpath = 'D:/dissertation/picture/BPT_criteria_1_SF_AGN_other/other/'
    elif galaxytypes[i] == 1:
        figpath = 'D:/dissertation/picture/BPT_criteria_1_SF_AGN_other/SF/'
    elif galaxytypes[i] == 2:
        figpath = 'D:/dissertation/picture/BPT_criteria_1_SF_AGN_other/AGN/'
    else:
        continue
''' 
def plot_BPT(magpiid,fitspath,figpath):
    '''
    

    Parameters
    ----------
    magpiid : str
        DESCRIPTION.
    fitspath : str
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ####Read Data########
    #info = fits.getdata('/Users/xiaoling/Desktop/project2/changing-look-AGN/data/changing-look-agn-info2.fits',1)
    #pa = fits.getdata('/Users/xiaoling/Desktop/project2/changing-look-AGN/code-python/data/pa.fits',1)\
    
    ####construction the plot panal######
    #fig, axs = plt.subplots(1, 4,figsize=(29,21))
    #####parameters of plot#########
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    mpl.rcParams['axes.linewidth'] = 1.7
    ###########################
    #plate = info['PLATE'][i].strip()
    #ifu = info['IFU'][i].strip()
    #axs[i,0].set_title(str(i),fontsize=25)
    ######read sdss image###########
    #loc = './sdss-image/'
    #im=plt.imread(loc + plate +'-'+ ifu +'.png')
    #axs[i,0].imshow(im)
    #axs[i,0].axis('off')
    
    
    ################################
    #fileNameofFits = 'manga-'+plateifus[i]+'-MAPS-HYB10-GAU-MILESHC.fits.gz'
    dmap = fits.open(fitspath)
    
    
    
    dilated_mask = dmap['DILATED_MASK'].data
    dilated_mask_1 = (dilated_mask==1)
    
    ### read data ####
    ha = dmap['Ha_F'].data
    ha_err = dmap['Ha_FERR'].data
    ha[dilated_mask_1] = np.nan
    ha_err[dilated_mask_1] = np.nan
    ha_snr = ha/ha_err
    
    nii = dmap['NII_6585_F'].data
    nii_err = dmap['NII_6585_FERR'].data
    nii[dilated_mask_1] = np.nan
    nii_err[dilated_mask_1] = np.nan
    nii_snr = nii/nii_err
    
    oiii = dmap['OIII_5008_F'].data
    oiii_err = dmap['OIII_5008_FERR'].data
    oiii[dilated_mask_1] = np.nan
    oiii_err[dilated_mask_1] = np.nan
    oiii_snr = oiii/oiii_err
    
    hb = dmap['Hb_F'].data
    hb_err = dmap['Hb_FERR'].data
    hb[dilated_mask_1] = np.nan
    hb_err[dilated_mask_1] = np.nan
    hb_snr = hb/hb_err
    
    
    
    ###########################################
    #crit_mask = (ha_mask == 0) & (nii_mask == 0)&(oiii_mask == 0)&(hb_mask==0)
    crit_err = (ha_err > 0) & (nii_err > 0)&(oiii_err > 0)&(hb_err > 0)
    crit_snr = (ha_snr > 3) &(hb_snr>3)&(nii_snr>3)&(oiii_snr>3)
    indplot =  crit_err & crit_snr

    ############################################
    ##constrction construction coordinates###
    nx = (np.arange(ha.shape[1]) - ha.shape[1]/2)/5.
    ny = (np.arange(ha.shape[0]) - ha.shape[0]/2)/5.
    # ! note change indexing from 'xy' to 'ij', not sure if it's correct, but it works.
    xpos, ypos = np.meshgrid(nx, ny, sparse=False, indexing='xy')
    ##########caculate bpt region##################
    x = np.log10(nii/ha)
    y = np.log10(oiii/hb)
    radd = np.sqrt(xpos**2 + ypos**2)
    rad = np.median(radd)+2
    ##########################
    ##########################
    ##########################
    #########plot bpt diagram##################
    AGN, CP, SF, *not_need= bptregion(x, y, mode='N2')
    x_type = np.full_like(xpos, np.nan)
    y_type = np.full_like(xpos, np.nan)
    x_type[indplot] = x[indplot]
    y_type[indplot] = y[indplot]
    AGN, CP, SF, *not_need= bptregion(x_type, y_type, mode='N2')
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    fig.suptitle(magpiid,fontsize=25)
    #axs[0].tick_params(direction='in', labelsize = 20, length = 5, width=1.0)
    axs[0].set_title('BPT-Diagram',fontsize=25)
    axs[0].set_xlim(-1.5,0.5)
    axs[0].set_ylim(-1.0,1.5)
    axs[0].plot(x_type[SF],y_type[SF],'.',color = 'blue')##SF
    axs[0].plot(x_type[CP],y_type[CP],'.',color = 'lime')##CP
    axs[0].plot(x_type[AGN],y_type[AGN],'.',color = 'red')##AGN
    axs[0].set_xlabel(r"$\rm{log_{10}(NII/H\alpha)}$",fontsize=25)
    axs[0].set_ylabel(r"$\rm{log_{10}(OIII/H\beta)}$",fontsize=25)
    x1 = np.linspace(-1.5, 0.2, 100)
    y_ke01 = 0.61/(x1-0.47)+1.19
    axs[0].plot(x1,y_ke01)
    x2 = np.linspace(-1.5, -.2, 100)
    y_ka03 = 0.61/(x2-0.05)+1.3
    axs[0].plot(x2,y_ka03,'--')
    axs[0].legend(('SF','Comp','AGN','Ke 01','Ka 03'),loc='best',fontsize = 15.0,markerscale = 2)
    #plt.savefig(figpath+'BPT-'+str(plate)+'-'+str(ifu)+'-Diagram.png',bbox_inches='tight',dpi=300)
    #plt.show()
    ###########################
    ########bpt diagram map#########################
    #fig, axs = plt.subplots()
    axs[1].tick_params(direction='in', labelsize = 20, length = 5, width=1.0)
    axs[1].set_title(r'resolved BPT-map',fontsize=25)
    region_type = np.full_like(xpos, np.nan)
    region_color = ['red','lime','blue']
    region_name = ['AGN', 'Comp', 'HII']
    # AGN, CP, SF, *not_need= bptregion(x, y, mode='N2')
    region_type[AGN] = 1
    region_type[CP] = 2
    region_type[SF] = 3
    new_region = np.full_like(xpos, np.nan)
    new_region[indplot] = region_type[indplot]
    bounds = [0.5, 1.5, 2.5, 3.5] # set color for imshow
    cmap = colors.ListedColormap(region_color)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # the map data was up-down inverse as the optical image from sdss
    axs[1].pcolormesh(xpos, ypos, new_region, cmap=cmap, norm=norm)
    axs[1].set_xlabel('arcsec',fontsize=25)
    axs[1].set_ylabel('arcsec',fontsize=25)
    plt.savefig(figpath+magpiid+'_BPT.pdf',bbox_inches='tight',dpi=300)
    plt.show()
    
    



def plot_BPT_and_emi(magpiid,fitspath,figpath):
    '''
    

    Parameters
    ----------
    magpiid : str
        DESCRIPTION.
    fitspath : str
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ####Read Data########
    #info = fits.getdata('/Users/xiaoling/Desktop/project2/changing-look-AGN/data/changing-look-agn-info2.fits',1)
    #pa = fits.getdata('/Users/xiaoling/Desktop/project2/changing-look-AGN/code-python/data/pa.fits',1)\
    
    ####construction the plot panal######
    #fig, axs = plt.subplots(1, 4,figsize=(29,21))
    #####parameters of plot#########
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    mpl.rcParams['axes.linewidth'] = 1.7
    ###########################
    #plate = info['PLATE'][i].strip()
    #ifu = info['IFU'][i].strip()
    #axs[i,0].set_title(str(i),fontsize=25)
    ######read sdss image###########
    #loc = './sdss-image/'
    #im=plt.imread(loc + plate +'-'+ ifu +'.png')
    #axs[i,0].imshow(im)
    #axs[i,0].axis('off')
    
    
    ################################
    #fileNameofFits = 'manga-'+plateifus[i]+'-MAPS-HYB10-GAU-MILESHC.fits.gz'
    dmap = fits.open(fitspath)
    
    
    dilated_mask = dmap['DILATED_MASK'].data
    dilated_mask_1 = (dilated_mask==1)
    
    ### read data ####
    ha = dmap['Ha_F'].data
    ha_err = dmap['Ha_FERR'].data
    ha[dilated_mask_1] = np.nan
    ha_err[dilated_mask_1] = np.nan
    ha_snr = ha/ha_err
    
    nii = dmap['NII_6585_F'].data
    nii_err = dmap['NII_6585_FERR'].data
    nii[dilated_mask_1] = np.nan
    nii_err[dilated_mask_1] = np.nan
    nii_snr = nii/nii_err
    
    oiii = dmap['OIII_5008_F'].data
    oiii_err = dmap['OIII_5008_FERR'].data
    oiii[dilated_mask_1] = np.nan
    oiii_err[dilated_mask_1] = np.nan
    oiii_snr = oiii/oiii_err
    
    hb = dmap['Hb_F'].data
    hb_err = dmap['Hb_FERR'].data
    hb[dilated_mask_1] = np.nan
    hb_err[dilated_mask_1] = np.nan
    hb_snr = hb/hb_err
    
    
    
    
    ###########################################
    #crit_mask = (ha_mask == 0) & (nii_mask == 0)&(oiii_mask == 0)&(hb_mask==0)
    crit_err = (ha_err > 0) & (nii_err > 0)&(oiii_err > 0)&(hb_err > 0)
    crit_snr = (ha_snr > 3) &(hb_snr>3)&(nii_snr>3)&(oiii_snr>3)
    indplot =  crit_err & crit_snr

    ############################################
    ##constrction construction coordinates###
    nx = (np.arange(ha.shape[1]) - ha.shape[1]/2)/5.
    ny = (np.arange(ha.shape[0]) - ha.shape[0]/2)/5.
    # ! note change indexing from 'xy' to 'ij', not sure if it's correct, but it works.
    xpos, ypos = np.meshgrid(nx, ny, sparse=False, indexing='xy')
    ##########caculate bpt region##################
    x = np.log10(nii/ha)
    y = np.log10(oiii/hb)
    radd = np.sqrt(xpos**2 + ypos**2)
    rad = np.median(radd)+2
    ##########################
    ##########################
    ##########################
    #########plot bpt diagram##################
    AGN, CP, SF, *not_need= bptregion(x, y, mode='N2')
    x_type = np.full_like(xpos, np.nan)
    y_type = np.full_like(xpos, np.nan)
    x_type[indplot] = x[indplot]
    y_type[indplot] = y[indplot]
    AGN, CP, SF, *not_need= bptregion(x_type, y_type, mode='N2')
    
    fig = plt.figure(constrained_layout=True,figsize=(15,15))
    gs = GridSpec(3, 3, figure=fig)
    
    
    fig.suptitle(magpiid,fontsize=25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, 0])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[2, 2])
    
    ax1.set_title('BPT-Diagram',fontsize=25)
    ax1.set_xlim(-1.5,0.5)
    
    ax1.set_ylim(-1.0,1.5)
    ax1.plot(x_type[SF],y_type[SF],'.',color = 'blue')##SF
    ax1.plot(x_type[CP],y_type[CP],'.',color = 'lime')##CP
    ax1.plot(x_type[AGN],y_type[AGN],'.',color = 'red')##AGN
    ax1.set_xlabel(r"$\rm{log_{10}(NII/H\alpha)}$",fontsize=25)
    ax1.set_ylabel(r"$\rm{log_{10}(OIII/H\beta)}$",fontsize=25)
    x1 = np.linspace(-1.5, 0.2, 100)
    y_ke01 = 0.61/(x1-0.47)+1.19
    ax1.plot(x1,y_ke01)
    x2 = np.linspace(-1.5, -.2, 100)
    y_ka03 = 0.61/(x2-0.05)+1.3
    ax1.plot(x2,y_ka03,'--')
    ax1.legend(('SF','Comp','AGN','Ke 01','Ka 03'),loc='best',fontsize = 15.0,markerscale = 2)
    #plt.savefig(figpath+'BPT-'+str(plate)+'-'+str(ifu)+'-Diagram.png',bbox_inches='tight',dpi=300)
    #plt.show()
    ###########################
    ########bpt diagram map#########################
    #fig, axs = plt.subplots()
    ax2.tick_params(direction='in', labelsize = 20, length = 5, width=1.0)
    ax2.set_title(r'resolved BPT-map',fontsize=25)
    region_type = np.full_like(xpos, np.nan)
    region_color = ['red','lime','blue']
    region_name = ['AGN', 'Comp', 'HII']
    # AGN, CP, SF, *not_need= bptregion(x, y, mode='N2')
    region_type[AGN] = 1
    region_type[CP] = 2
    region_type[SF] = 3
    new_region = np.full_like(xpos, np.nan)
    new_region[indplot] = region_type[indplot]
    bounds = [0.5, 1.5, 2.5, 3.5] # set color for imshow
    cmap = colors.ListedColormap(region_color)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # the map data was up-down inverse as the optical image from sdss
    ax2.pcolormesh(xpos, ypos, new_region, cmap=cmap, norm=norm)
    ax2.set_xlabel('arcsec',fontsize=25)
    ax2.set_ylabel('arcsec',fontsize=25)
    
    ### ha
    ha_plot = np.full_like(ha, np.nan)
    ha_plot[ha_snr>3] = ha[ha_snr>3]
    if sum(sum(ha_snr>3))!=0:
        ax3.imshow(ha_plot,origin='lower',norm=mpl.colors.LogNorm())
    else:
        ax3.imshow(ha_plot,origin='lower')
    ax3.set_title(r'$\rm{H\alpha\ flux\ log-scale}$',fontsize=25)
    
    ## nii
    
    nii_plot = np.full_like(nii, np.nan)
    nii_plot[nii_snr>3] = nii[nii_snr>3]
    if sum(sum(nii_snr>3))!=0:
        ax4.imshow(nii_plot,origin='lower',norm=mpl.colors.LogNorm())
    else:
        ax4.imshow(nii_plot,origin='lower')
    ax4.set_title(r'$\rm{NII\ flux\ log-scale}$',fontsize=25)
    
    ## nii/ha
    niiha_plot = np.full_like(ha,np.nan)
    niiha_index = (ha_snr>3) & (nii_snr>3)
    if sum(sum(niiha_index))!=0:
        niiha_plot[niiha_index] = np.log10(nii/ha)[niiha_index]
        im5=ax5.imshow(niiha_plot,origin='lower')
        cb5 = plt.colorbar(im5,ax=ax5,fraction=0.047)
        ax5.set_title(r'$\rm{log(NII/H\alpha)}$',fontsize=25)
    else:
        im5=ax5.imshow(niiha_plot,origin='lower')
        ax5.set_title(r'$\rm{log(NII/H\alpha)}$',fontsize=25)
    
    ## hb
    hb_plot = np.full_like(hb, np.nan)
    hb_plot[hb_snr>3] = hb[hb_snr>3]
    if sum(sum(hb_snr>3))!=0:
        ax6.imshow(hb_plot,origin='lower',norm=mpl.colors.LogNorm())
    else:
        ax6.imshow(hb_plot,origin='lower')
    ax6.set_title(r'$\rm{H\beta\ flux\ log-scale}$',fontsize=25)
    
    ## oiii
    oiii_plot = np.full_like(oiii, np.nan)
    oiii_plot[oiii_snr>3] = oiii[oiii_snr>3]
    if sum(sum(oiii_snr>3))!=0:
        ax7.imshow(oiii_plot,origin='lower',norm=mpl.colors.LogNorm())
    else:
        ax7.imshow(oiii_plot,origin='lower')
    ax7.set_title(r'$\rm{OIII\ flux\ log-scale}$',fontsize=25)   
    
    ## oiii/hb
    oiiihb_plot = np.full_like(oiii, np.nan)
    oiiihb_index = (hb_snr>3) & (oiii_snr>3)
    if sum(sum(oiiihb_index))!=0:
        
        oiiihb_plot[oiiihb_index] = np.log10(oiii/hb)[oiiihb_index]
        im8 = ax8.imshow(oiiihb_plot,origin='lower')
        cb8 = plt.colorbar(im8,ax=ax8,fraction=0.047)
        ax8.set_title(r'$\rm{log(OIII/H\beta)}$',fontsize=25)
    else:
        im8 = ax8.imshow(oiiihb_plot,origin='lower')
        ax8.set_title(r'$\rm{log(OIII/H\beta)}$',fontsize=25)
    
    
    
    
    plt.savefig(figpath+magpiid+'_BPT_and_emi.pdf',bbox_inches='tight',dpi=300)
    plt.show()
    












def plot_BPT_and_emi_err(magpiid,fitspath,figpath=None,savefigure=True):
    '''
    
    bpt-diagram with error bar

    Parameters
    ----------
    magpiid : str
        DESCRIPTION.
    fitspath : str
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ####Read Data########
    #info = fits.getdata('/Users/xiaoling/Desktop/project2/changing-look-AGN/data/changing-look-agn-info2.fits',1)
    #pa = fits.getdata('/Users/xiaoling/Desktop/project2/changing-look-AGN/code-python/data/pa.fits',1)\
    
    ####construction the plot panal######
    #fig, axs = plt.subplots(1, 4,figsize=(29,21))
    #####parameters of plot#########
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    mpl.rcParams['axes.linewidth'] = 1.7
    ###########################
    #plate = info['PLATE'][i].strip()
    #ifu = info['IFU'][i].strip()
    #axs[i,0].set_title(str(i),fontsize=25)
    ######read sdss image###########
    #loc = './sdss-image/'
    #im=plt.imread(loc + plate +'-'+ ifu +'.png')
    #axs[i,0].imshow(im)
    #axs[i,0].axis('off')
    
    
    ################################
    #fileNameofFits = 'manga-'+plateifus[i]+'-MAPS-HYB10-GAU-MILESHC.fits.gz'
    dmap = fits.open(fitspath)
    
    
    
    dilated_mask = dmap['DILATED_MASK'].data
    dilated_mask_1 = (dilated_mask==1)
    
    ### read data ####
    ha = dmap['Ha_F'].data
    ha_err = dmap['Ha_FERR'].data
    ha[dilated_mask_1] = np.nan
    ha_err[dilated_mask_1] = np.nan
    ha_snr = ha/ha_err
    
    nii = dmap['NII_6585_F'].data
    nii_err = dmap['NII_6585_FERR'].data
    nii[dilated_mask_1] = np.nan
    nii_err[dilated_mask_1] = np.nan
    nii_snr = nii/nii_err
    
    oiii = dmap['OIII_5008_F'].data
    oiii_err = dmap['OIII_5008_FERR'].data
    oiii[dilated_mask_1] = np.nan
    oiii_err[dilated_mask_1] = np.nan
    oiii_snr = oiii/oiii_err
    
    hb = dmap['Hb_F'].data
    hb_err = dmap['Hb_FERR'].data
    hb[dilated_mask_1] = np.nan
    hb_err[dilated_mask_1] = np.nan
    hb_snr = hb/hb_err
    
    
    
    
    
    ###########################################
    #crit_mask = (ha_mask == 0) & (nii_mask == 0)&(oiii_mask == 0)&(hb_mask==0)
    crit_err = (ha_err > 0) & (nii_err > 0)&(oiii_err > 0)&(hb_err > 0)
    crit_snr = (ha_snr > 3) &(hb_snr>3)&(nii_snr>3)&(oiii_snr>3)
    indplot =  crit_err & crit_snr

    ############################################
    ##constrction construction coordinates###
    nx = (np.arange(ha.shape[1]) - ha.shape[1]/2)/5.
    ny = (np.arange(ha.shape[0]) - ha.shape[0]/2)/5.
    # ! note change indexing from 'xy' to 'ij', not sure if it's correct, but it works.
    xpos, ypos = np.meshgrid(nx, ny, sparse=False, indexing='xy')
    ##########caculate bpt region##################
    x = np.log10(nii/ha)
    y = np.log10(oiii/hb)
    radd = np.sqrt(xpos**2 + ypos**2)
    rad = np.median(radd)+2
    ##########################
    ##########################
    
    
    ### calculate the error
    err_log10_niiha = 1/np.log(10)*np.sqrt((nii_err/nii)**2 + (ha_err/ha)**2)
    err_log10_oiiihb = 1/np.log(10)*np.sqrt((oiii_err/oiii)**2 + (hb_err/hb)**2)
    
    
    
    
    ##########################
    #########plot bpt diagram##################
    AGN, CP, SF, *not_need= bptregion(x, y, mode='N2')
    x_type = np.full_like(xpos, np.nan)
    y_type = np.full_like(xpos, np.nan)
    x_type[indplot] = x[indplot]
    y_type[indplot] = y[indplot]
    AGN, CP, SF, *not_need= bptregion(x_type, y_type, mode='N2')
    
    fig = plt.figure(constrained_layout=True,figsize=(15,15))
    gs = GridSpec(3, 3, figure=fig)
    
    
    fig.suptitle(magpiid,fontsize=25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, 0])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[2, 2])
    
    # bpt-diagram with error
    
    ax9 = fig.add_subplot(gs[0,1])
    
    ax1.set_title('BPT-Diagram',fontsize=25)
    ax1.set_xlim(-1.5,0.5)
    
    ax1.set_ylim(-1.0,1.5)
    ax1.plot(x_type[SF],y_type[SF],'.',color = 'blue')##SF
    ax1.plot(x_type[CP],y_type[CP],'.',color = 'lime')##CP
    ax1.plot(x_type[AGN],y_type[AGN],'.',color = 'red')##AGN
    ax1.set_xlabel(r"$\rm{log_{10}(NII/H\alpha)}$",fontsize=25)
    ax1.set_ylabel(r"$\rm{log_{10}(OIII/H\beta)}$",fontsize=25)
    x1 = np.linspace(-1.5, 0.2, 100)
    y_ke01 = 0.61/(x1-0.47)+1.19
    ax1.plot(x1,y_ke01)
    x2 = np.linspace(-1.5, -.2, 100)
    y_ka03 = 0.61/(x2-0.05)+1.3
    ax1.plot(x2,y_ka03,'--')
    ax1.legend(('SF','Comp','AGN','Ke 01','Ka 03'),loc='best',fontsize = 15.0,markerscale = 2)
    #plt.savefig(figpath+'BPT-'+str(plate)+'-'+str(ifu)+'-Diagram.png',bbox_inches='tight',dpi=300)
    #plt.show()
    ###########################
    ########bpt diagram map#########################
    #fig, axs = plt.subplots()
    ax2.tick_params(direction='in', labelsize = 20, length = 5, width=1.0)
    ax2.set_title(r'resolved BPT-map',fontsize=25)
    region_type = np.full_like(xpos, np.nan)
    region_color = ['red','lime','blue']
    region_name = ['AGN', 'Comp', 'HII']
    # AGN, CP, SF, *not_need= bptregion(x, y, mode='N2')
    region_type[AGN] = 1
    region_type[CP] = 2
    region_type[SF] = 3
    new_region = np.full_like(xpos, np.nan)
    new_region[indplot] = region_type[indplot]
    bounds = [0.5, 1.5, 2.5, 3.5] # set color for imshow
    cmap = colors.ListedColormap(region_color)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    # the map data was up-down inverse as the optical image from sdss
    ax2.pcolormesh(xpos, ypos, new_region, cmap=cmap, norm=norm)
    ax2.set_xlabel('arcsec',fontsize=25)
    ax2.set_ylabel('arcsec',fontsize=25)
    
    ### ha
    ha_plot = np.full_like(ha, np.nan)
    ha_plot[ha_snr>3] = ha[ha_snr>3]
    if sum(sum(ha_snr>3))!=0:
        ax3.imshow(ha_plot,origin='lower',norm=mpl.colors.LogNorm())
    else:
        ax3.imshow(ha_plot,origin='lower')
    ax3.set_title(r'$\rm{H\alpha\ flux\ log-scale}$',fontsize=25)
    
    ## nii
    
    nii_plot = np.full_like(nii, np.nan)
    nii_plot[nii_snr>3] = nii[nii_snr>3]
    if sum(sum(nii_snr>3))!=0:
        ax4.imshow(nii_plot,origin='lower',norm=mpl.colors.LogNorm())
    else:
        ax4.imshow(nii_plot,origin='lower')
    ax4.set_title(r'$\rm{NII\ flux\ log-scale}$',fontsize=25)
    
    ## nii/ha
    niiha_plot = np.full_like(ha,np.nan)
    niiha_index = (ha_snr>3) & (nii_snr>3)
    if sum(sum(niiha_index))!=0:
        niiha_plot[niiha_index] = np.log10(nii/ha)[niiha_index]
        im5=ax5.imshow(niiha_plot,origin='lower')
        cb5 = plt.colorbar(im5,ax=ax5,fraction=0.047)
        ax5.set_title(r'$\rm{log(NII/H\alpha)}$',fontsize=25)
    else:
        im5=ax5.imshow(niiha_plot,origin='lower')
        ax5.set_title(r'$\rm{log(NII/H\alpha)}$',fontsize=25)
    
    ## hb
    hb_plot = np.full_like(hb, np.nan)
    hb_plot[hb_snr>3] = hb[hb_snr>3]
    if sum(sum(hb_snr>3))!=0:
        ax6.imshow(hb_plot,origin='lower',norm=mpl.colors.LogNorm())
    else:
        ax6.imshow(hb_plot,origin='lower')
    ax6.set_title(r'$\rm{H\beta\ flux\ log-scale}$',fontsize=25)
    
    ## oiii
    oiii_plot = np.full_like(oiii, np.nan)
    oiii_plot[oiii_snr>3] = oiii[oiii_snr>3]
    if sum(sum(oiii_snr>3))!=0:
        ax7.imshow(oiii_plot,origin='lower',norm=mpl.colors.LogNorm())
    else:
        ax7.imshow(oiii_plot,origin='lower')
    ax7.set_title(r'$\rm{OIII\ flux\ log-scale}$',fontsize=25)   
    
    ## oiii/hb
    oiiihb_plot = np.full_like(oiii, np.nan)
    oiiihb_index = (hb_snr>3) & (oiii_snr>3)
    if sum(sum(oiiihb_index))!=0:
        
        oiiihb_plot[oiiihb_index] = np.log10(oiii/hb)[oiiihb_index]
        im8 = ax8.imshow(oiiihb_plot,origin='lower')
        cb8 = plt.colorbar(im8,ax=ax8,fraction=0.047)
        ax8.set_title(r'$\rm{log(OIII/H\beta)}$',fontsize=25)
    else:
        im8 = ax8.imshow(oiiihb_plot,origin='lower')
        ax8.set_title(r'$\rm{log(OIII/H\beta)}$',fontsize=25)
    
    #### ax9 bpt-diagram with errorbar
    
    err_x = np.full_like(xpos, np.nan)
    err_y = np.full_like(xpos, np.nan)
    err_x[indplot] = err_log10_niiha[indplot]
    err_y[indplot] = err_log10_oiiihb[indplot]
    
    
    
    ax9.set_title('BPT-Diagram-err',fontsize=25)
    ax9.set_xlim(-1.5,0.5)
    
    ax9.set_ylim(-1.0,1.5)
    
    
    

    ax9.errorbar(x_type[SF],y_type[SF],xerr=err_x[SF],yerr=err_y[SF],fmt=' ',color = 'blue',label='SF')##SF
    ax9.errorbar(x_type[CP],y_type[CP],xerr=err_x[CP],yerr=err_y[CP],fmt=' ',color = 'lime',label='Comp')##CP
    ax9.errorbar(x_type[AGN],y_type[AGN],xerr=err_x[AGN],yerr=err_y[AGN],fmt=' ',color = 'red',label='AGN')##AGN
    ax9.set_xlabel(r"$\rm{log_{10}(NII/H\alpha)}$",fontsize=25)
    ax9.set_ylabel(r"$\rm{log_{10}(OIII/H\beta)}$",fontsize=25)
    x1 = np.linspace(-1.5, 0.2, 100)
    y_ke01 = 0.61/(x1-0.47)+1.19
    ax9.plot(x1,y_ke01,label='Ke 01')
    x2 = np.linspace(-1.5, -.2, 100)
    y_ka03 = 0.61/(x2-0.05)+1.3
    ax9.plot(x2,y_ka03,'--',label='Ka 03')
    ax9.legend(loc='best',fontsize = 15.0,markerscale = 2)
    
    if savefigure==True:
        plt.savefig(figpath+magpiid+'_BPT_and_emi_err.pdf',bbox_inches='tight',dpi=300)
    plt.show()
    




























