#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pre-process magpi data

@author: Yifan Mai
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pandas
from meta import Metadata
from reconstructLSF import reconstruct_lsf
import fit_two_gaussian_from_moffat as mofgauFit
from BPT import bptregion
import copy
import matplotlib as mpl
from matplotlib import colors
from numpy.ma import is_masked
from matplotlib.gridspec import GridSpec
import pandas as pd
import math


class PreBlobby3D:
    
    def __init__(
            self, dirpath, fitsname, emipath, eminame, redshift_path, save_path,
            emi_line, conti_path=None, wave_axis=0):
        """
        

        Parameters
        ----------
        dirpath : TYPE
            DESCRIPTION.
        fitsname : TYPE
            DESCRIPTION.
        redshift_path : TYPE
            DESCRIPTION.
        save_path : TYPE
            DESCRIPTION.
        wave_axis : TYPE, optional
            DESCRIPTION. The default is 0.
        emi_line: Ha or Oii or Hb
        
        
        
        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.dirpath = dirpath
        self.fitsname = fitsname
        self.fitsdata = fits.open(dirpath + fitsname)
        self.emidata = fits.open(emipath+eminame)
        self.emipath_w = emipath+eminame
        self.ha_flux = self.emidata['Ha_F'].data # extension 47
        self.ha_err = self.emidata['Ha_FERR'].data# extension 48
        
        # oii flux and ferr from emidata
        oii_3727_f = self.emidata['OII_3727_F'].data
        oii_3727_ferr = self.emidata['OII_3727_FERR'].data
        oii_3730_f = self.emidata['OII_3730_F'].data
        oii_3730_ferr = self.emidata['OII_3730_FERR'].data
        self.oii_sum_flux = oii_3727_f + oii_3730_f
        self.oii_sum_ferr = np.sqrt(oii_3727_ferr**2+oii_3730_ferr**2)
        self.oii_sum_sn = self.oii_sum_flux/self.oii_sum_ferr
        self.dilated_mask = self.emidata['DILATED_MASK'].data
        self.ha_sn = self.ha_flux/self.ha_err
        
        
        self.nii_flux = self.emidata['NII_6585_F'].data
        self.nii_ferr = self.emidata['NII_6585_FERR'].data
        self.nii_sn = self.nii_flux/self.nii_ferr
        
        
        self.hb_flux = self.emidata['Hb_F'].data
        self.hb_err = self.emidata['Hb_FERR'].data
        self.hb_sn = self.hb_flux/self.hb_err
        
        self.data1 = self.fitsdata[1].data
        # datacube is (wavelength, yaxis, xaxis)
        self.flux = self.data1
        
        self.data2 = self.fitsdata[2].data
        self.var = self.data2
        
        self.cubeshape = self.flux.shape
        self.magpiid = fitsname[5:15]
        self.wave_axis = wave_axis
        self.nw = self.cubeshape[wave_axis]
        self.save_path = save_path
        #if wave_axis == 0:
        #    self.nx , self.ny = self.cubeshape[1:]
        #elif wave_axis == 2:
        #    self.nx , self.ny = self.cubeshape[:2]
        #else:
        #    raise ValueError('Wave axis needs to be 0 or 2.')
            
        redshift_data =  pandas.read_csv(redshift_path)
        index = np.where(redshift_data['MAGPIID']==int(self.magpiid))[0][0]
        self.magpiredshift = redshift_data['Z'][index]
        
        self.HD1 = self.fitsdata[1].header
        HD1 = self.HD1
        self.nx = HD1['NAXIS1']
        self.x_del = HD1['CD1_1']
        x_crpix = HD1['CRPIX1']
        self.ny = HD1['NAXIS2']
        self.y_del = HD1['CD2_2']
        y_crpix = HD1['CRPIX2']

        wavelength_crpix = HD1['CRPIX3']
        wavelength_crval = HD1['CRVAL3']
        self.wavelength_del_Wav = HD1['CD3_3']
        self.nw = HD1['NAXIS3']
        
        self.Wavelengths = wavelength_crval + (np.arange(0,self.nw)+1-wavelength_crpix)*self.wavelength_del_Wav
        self.Wavelengths_deredshift = self.Wavelengths/(1+self.magpiredshift)
        # The data is assumed to be de-redshifted 
        # and centred about (0, 0) spatial coordinates.

        self.x_pix_range_sec = (np.arange(0,self.nx)+1-x_crpix)*self.x_del*3600
        self.x_pix_range = self.x_pix_range_sec - (self.x_pix_range_sec[0]+self.x_pix_range_sec[-1])/2

        self.y_pix_range_sec = (np.arange(0,self.ny)+1-y_crpix)*self.y_del*3600
        self.y_pix_range = self.y_pix_range_sec - (self.y_pix_range_sec[0]+self.y_pix_range_sec[-1])/2
        self.flux_scale_factor = 1
        
        # todo 
        self.conti_path = conti_path
        conti_fits = fits.open(conti_path)
        self.conti_spec = conti_fits[3].data - conti_fits[4].data
        self.emi_line = emi_line
            
    
    def plot_interg_flux(self,snlim,xlim=None, ylim=None, mask_dilated_mask=True, **kwargs):
        """
        

        Parameters
        ----------
        xlim : TYPE, optional
            DESCRIPTION. The default is None.
        ylim : TYPE, optional
            DESCRIPTION. The default is None.
        wavelim : TYPE, optional
            deredshifted wavelength range. The default is None.

        Returns
        -------
        None.

        """
        cutoutdata, cutoutvar, metadata = self.cutout_data(xlim=xlim, ylim=ylim,
                                                           mask_dilated_mask=mask_dilated_mask,
                                                        **kwargs)
        
        interg_flux = np.nansum(cutoutdata.reshape(int(metadata[0]),int(metadata[1]),int(metadata[2])),axis=2)
        
        ha_sn_masked = self.ha_sn
        if mask_dilated_mask==True:
            ha_sn_masked[self.dilated_mask==1] = np.nan
        
        ha_sn_cut = self.cutout(data = ha_sn_masked,dim=2,xlim=xlim,ylim=ylim)
        
        
        #interg_flux[self.dilated_mask==1] = np.nan
        
        ha_sn_low = ha_sn_cut < snlim
        ha_sn_high = ha_sn_cut >= snlim
        
        
        
        sn_diagram = np.zeros(ha_sn_cut.shape)
        sn_diagram[ha_sn_high] = 1
        
        fig, axs = plt.subplots(1,3)
        
        print(ha_sn_cut.shape)
        fig.suptitle(self.magpiid)
        axs[0].imshow(interg_flux,origin='lower')
        axs[0].set_title('interg_flux')
        
        
        im1 = axs[1].imshow(ha_sn_cut,origin='lower')
        axs[1].set_title('Halpha_sn')
        
        #fig.colorbar(im1)
        axs[2].imshow(sn_diagram,origin='lower')
        axs[2].set_title('Halpha_sn > ' + str(snlim))
        
        
        plt.show()
        return cutoutdata, cutoutvar, metadata
    
    
    #def plot_allinfo(self,snlim,xlim=None, ylim=None, **kwargs):
    #    cutoutdata, cutoutvar, metadata = self.plot_interg_flux(snlim,xlim=None,
    #                                                            ylim=None, **kwargs)
    #    self.plot_BPT_cut(snlim=snlim,xlim=xlim,ylim=ylim,**kwargs)
    #    
    #    return cutoutdata, cutoutvar, metadata
    
    
    
    
    def cutout(self,data,dim, xlim=None, ylim=None, wavelim=None):
        x_mask = np.full(self.nx,True)
        y_mask = np.full(self.ny,True)
        if dim == 3:
            wavelength_mask = np.ones(self.nw,dtype=bool)
            
            
            
            if wavelim is not None:
                wavelength_mask = (self.Wavelengths_deredshift > wavelim[0]) & (self.Wavelengths_deredshift < wavelim[1])
            if xlim is not None:
                x_values = np.arange(self.nx)
                x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
            if ylim is not None:
                y_values = np.arange(self.ny)
                y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
            
            if self.wave_axis == 0:
                cutoutdata = data[wavelength_mask,:,:][:, y_mask, :][:, :, x_mask]
                
            # I think it's wrong... not need to think about it at this time...
            #if self.wave_axis == 2:
            #    cutoutdata = data[:, x_mask, :][:, :, y_mask][wavelength_mask,:,:]
        if dim == 2:
            if xlim is not None:
                x_values = np.arange(self.nx)
                x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
            if ylim is not None:
                y_values = np.arange(self.ny)
                y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
            
            cutoutdata = data[ y_mask, :][ :, x_mask]
        
        
        return cutoutdata
    
    def sn_cut(self,snlim,data,var):
        mask = (self.ha_sn < snlim) | (np.isnan(self.ha_sn))
        data[:,mask] = 0
        var[:,mask] = 0
        return data, var
    
    def sn_cut_2d_for_bpt(self,snlim,data):
        mask = (self.ha_sn < snlim) | (np.isnan(self.ha_sn))
        data[mask] = np.nan
        
        return data
    
    def cutout_data(self, snlim=None,xlim=None, ylim=None, wavelim=None, scale_flux=False,
                    subtract_continuum=False,mask_center=False,mask_center_pix=None,mask_radius=None,
                    AGN_mask=False,AGN_mask_exp=None,comp_mask=False,niiha_mask=False,mask_dilated_mask=True):
        
        ''' mask_center, mask_center_pix, mask_radius is for mask a circular region at the center
        AGN_mask is mask AGN region based on bpt map
        AGN_mask_exp is expand the AGN mask region, int
        comp_mask is mask composite region based on bpt map
        niiha_mask mask log(nii/ha) > 0.1 , these pixels definitely AGN
        
        
        
        '''
        
        
        
        if subtract_continuum == True:
            if self.conti_path == None:
                print("No continuum available")
                return 0
            flux_subtracted = self.flux - self.conti_spec
        
        flux_mask,var_mask = self.sn_cut(snlim=3, data=flux_subtracted, var=self.var)
        if mask_dilated_mask==True:
            dilated_mask_1 = (self.dilated_mask==1)
            
            # mask pixels where dilated mask == 1
            flux_mask[:,dilated_mask_1] = 0
            var_mask[:,dilated_mask_1] = 0
        
        
        if AGN_mask or comp_mask:
            
            ### read data ####
            dmap = self.emidata
            ha = dmap['Ha_F'].data
            ha_err = dmap['Ha_FERR'].data
            ha_snr = ha/ha_err
            
            nii = dmap['NII_6585_F'].data
            nii_err = dmap['NII_6585_FERR'].data
            nii_snr = nii/nii_err
            
            oiii = dmap['OIII_5008_F'].data
            oiii_err = dmap['OIII_5008_FERR'].data
            oiii_snr = oiii/oiii_err
            
            hb = dmap['Hb_F'].data
            hb_err = dmap['Hb_FERR'].data
            hb_snr = hb/hb_err
            
            crit_err = (ha_err > 0) & (nii_err > 0)&(oiii_err > 0)&(hb_err > 0)
            crit_snr = (ha_snr > 3) &(hb_snr>3)&(nii_snr>3)&(oiii_snr>3)
            indplot =  crit_err & crit_snr
            
            x = np.log10(nii/ha)
            y = np.log10(oiii/hb)
            
            ##constrction construction coordinates###
            nx = (np.arange(ha.shape[1]) - ha.shape[1]/2)/5.
            ny = (np.arange(ha.shape[0]) - ha.shape[0]/2)/5.
            xpos, ypos = np.meshgrid(nx, ny, sparse=False, indexing='xy')
            
            x_type = np.full_like(xpos, np.nan)
            y_type = np.full_like(xpos, np.nan)
            x_type[indplot] = x[indplot]
            y_type[indplot] = y[indplot]
            AGN, CP, SF, *not_need= bptregion(x_type, y_type, mode='N2')
            
            if AGN_mask:
                ## mask AGN region
                flux_mask[:,AGN] = 0
                var_mask[:,AGN] = 0
            if comp_mask:
                ## mask composite region
                flux_mask[:,CP] = 0
                var_mask[:,CP] = 0
            if AGN_mask_exp is not None:
                yyy, xxx = np.meshgrid(np.arange(ha.shape[1]),np.arange(ha.shape[0]))
                AGN_E = copy.copy(AGN)
                for xx, yy in zip(xxx[AGN],yyy[AGN]):
                    AGN_E[xx-AGN_mask_exp:xx+AGN_mask_exp,yy-AGN_mask_exp:yy+AGN_mask_exp] = True
                flux_mask[:,AGN_E] = 0
                var_mask[:,AGN_E] = 0
        
        if niiha_mask:
            ### read data ####
            dmap = self.emidata
            ha = dmap['Ha_F'].data
            ha_err = dmap['Ha_FERR'].data
            ha_snr = ha/ha_err
            
            nii = dmap['NII_6585_F'].data
            nii_err = dmap['NII_6585_FERR'].data
            nii_snr = nii/nii_err
            
            crit_err = (ha_err > 0) & (nii_err > 0)
            crit_snr = (ha_snr > 3) &(nii_snr>3)
            indplot =  crit_err & crit_snr
            log_niiha = np.log10(nii/ha)
            
            niiha_mask_region = (log_niiha > 0.1) & (indplot)
            
            ## mask 
            flux_mask[:,niiha_mask_region] = 0
            var_mask[:,niiha_mask_region] = 0
        
        
        
        if mask_center==True:
            mask_outflow = np.full_like(self.ha_sn,False,dtype=bool)
            total_rows, total_cols = mask_outflow.shape
            #center_row, center_col = total_rows/2, total_cols/2
            integ_flux = np.nansum(flux_mask,axis=0)
            center_row, center_col = np.unravel_index(np.argmax(integ_flux),integ_flux.shape)
            
            if mask_center_pix is not None:
                center_col, center_row = mask_center_pix
            Y, X = np.ogrid[:total_rows, :total_cols]
            dist_from_center = np.sqrt((Y - center_row)**2 + (X-center_col)**2)
            circular_mask = (dist_from_center < mask_radius)
            flux_mask[:,circular_mask] = 0
            var_mask[:,circular_mask] = 0
            
        
        
        
        flux_scale_factor = 1
        if scale_flux is True:
            if self.wave_axis == 0:
                flux_scale_factor = (np.nanmedian(self.flux[:,int(self.ny/2),int(self.nx/2)])*10)
            if self.wave_axis == 2:
                flux_scale_factor = (np.nanmedian(self.flux[int(self.ny/2),int(self.nx/2),:])*10)
        
        
        cutoutdata = self.cutout(data=flux_mask,dim=3,xlim=xlim,ylim=ylim,wavelim=wavelim)
        cutoutvar = self.cutout(data=var_mask,dim=3,xlim=xlim,ylim=ylim,wavelim=wavelim)
        cutoutdata = cutoutdata/flux_scale_factor
        cutoutvar = cutoutvar/flux_scale_factor**2
        
        
        wavelength_mask = np.ones(self.nw,dtype=bool)
        x_mask = np.full(self.nx,True)
        y_mask = np.full(self.ny,True)
        
        
        if wavelim is not None:
            wavelength_mask = (self.Wavelengths_deredshift > wavelim[0]) & (self.Wavelengths_deredshift < wavelim[1])
        if xlim is not None:
            x_values = np.arange(self.nx)
            x_mask = (x_values >= xlim[0]) & (x_values <= xlim[1])
        if ylim is not None:
            y_values = np.arange(self.ny)
            y_mask = (y_values >= ylim[0]) & (y_values <= ylim[1])
        
        
        cutout_nx = np.sum(x_mask)
        cutout_ny = np.sum(y_mask)
        cutout_nw = np.sum(wavelength_mask)
        
        if self.wave_axis == 0:
            data_for_save = cutoutdata.reshape(cutout_nw,cutout_nx*cutout_ny).T
            var_for_save = cutoutvar.reshape(cutout_nw,cutout_nx*cutout_ny).T
        if self.wave_axis == 2:
            data_for_save = cutoutdata.reshape(cutout_nx*cutout_ny,cutout_nw)
            var_for_save = cutoutvar.reshape(cutout_nx*cutout_ny,cutout_nw)
        
        #if subtract_continuum == True:
        #    for i in range(data_for_save.shape[0]):
        #        data_for_save[i] = self.subtract_continuum_spaxel(data_for_save[i])
        
        
        

        metadata = np.array([cutout_ny,cutout_nx,cutout_nw,
                             -self.x_pix_range[x_mask][0]+1/2*self.x_del*3600,-self.x_pix_range[x_mask][-1]-1/2*self.x_del*3600,
                             self.y_pix_range[y_mask][0]-1/2*self.y_del*3600,self.y_pix_range[y_mask][-1]+1/2*self.y_del*3600,
                             self.Wavelengths_deredshift[wavelength_mask][0]-1/2*self.wavelength_del_Wav/(1+self.magpiredshift),
                             self.Wavelengths_deredshift[wavelength_mask][-1]+1/2*self.wavelength_del_Wav/(1+self.magpiredshift)])
        
        return data_for_save, var_for_save, metadata
        
    
    
    def subtract_continuum_spaxel(self, spectrum):
        """simple fit of continuum and subtract the continuum from a spectrum
        
        Parameters
        ----------
        spectrum : array-like
        
        
        Returns
        -------
        spectrum_subtracted : np.array
        
        """
        wave = np.arange(len(spectrum))
        wave_to_fit = np.concatenate((wave[:5],wave[-5:]))
        spec_to_fit = np.concatenate((spectrum[:5],spectrum[-5:]))
        coefficients= np.polyfit(wave_to_fit,spec_to_fit,deg=1)
        poly = np.poly1d(coefficients)
        straight_line_values = poly(wave)
        spectrum_subtracted = spectrum - straight_line_values
        return spectrum_subtracted
    
    def savedata(self,data,var,metadata):
        

        metafile = open(self.save_path+"metadata.txt","w")
        metafile.write("%d %d %d %.3f %.3f %.3f %.3f %.3f %.3f"%(metadata[0],metadata[1],
                                                                 metadata[2],metadata[3],
                                                                 metadata[4],metadata[5],
                                                                 metadata[6],metadata[7],
                                                                 metadata[8]))

        metafile.close()

        np.savetxt(self.save_path+"data.txt",np.nan_to_num(data))
        np.savetxt(self.save_path+"var.txt",np.nan_to_num(var))
    
    def plot_txt(self,xpix,ypix):
        metadata = Metadata(self.save_path+"metadata.txt")
        data = np.loadtxt(self.save_path+"data.txt")
        var = np.loadtxt(self.save_path+"var.txt")
        nx = metadata.naxis[1]
        ny = metadata.naxis[0]
        nw = metadata.naxis[2]
        data3d = data.reshape(ny,nx,nw)
        plt.plot(data3d[ypix,xpix,:])
        plt.show()

    def model_options(self,inc_path,flat_vdisp=True,psfimg=True,gaussian=2,
                      constrain_kinematics=None,constrain_vdisp=False,
                      constrain_pa=None,vsys_set=None):
        
        
        '''emi_line: Ha or Oii
            constrain_kinematics: path of kinematic parameter csv file
        
        constrain_kinematics: constrain vel profile, pa
        constrain_vdisp: when constrain_kinematics is not None, constrain
                         vel profile, pa, vdisp
        constrain_pa: constrain pa only
        
        
        '''
        
        if self.emi_line =='Ha':
            emi_line_wave = 6563
            # z-band, is most close to Halpha line at z~0.3
            emi_line_psfband = 3
        elif self.emi_line == 'Oii':
            # oii 3727 and oii 3729, choose a wavelength in between
            emi_line_wave = 3728
            # g-band, is most close to OII line at z~0.3
            emi_line_psfband = 0
        elif self.emi_line == 'Hb':
            # Hbeta line wavelength in vacuum 4862.683 
            emi_line_wave = 4862.683
            # r-band, is the most close to Hbeta line at z~0.3
            emi_line_psfband = 1
        
        
        
        
        
        modelfile = open(self.save_path+"MODEL_OPTIONS","w")
        # first line
        modelfile.write('# Specify model options\n')
        # LSFFWHM
        # reconstruct return sigma
        lsf_recon = reconstruct_lsf(wavelengths=emi_line_wave*(1+self.magpiredshift), 
                        resolution_file=self.dirpath + self.fitsname)
        lsf_deredshift = lsf_recon/(1+self.magpiredshift)
        lsf_fwhm = 2*np.sqrt(2*np.log(2))*lsf_deredshift
        modelfile.write('LSFFWHM\t%.4f\n'%(lsf_fwhm))
        
        if psfimg==True:
            # default z-band
            img = self.fitsdata[4].data[emi_line_psfband]
            
            if gaussian==2:
                weight1, weight2, fwhm1, fwhm2 = mofgauFit.psf_img_to_gauss(img)
                
                modelfile.write('PSFWEIGHT\t%f %f\n'%(weight1,weight2))
                modelfile.write('PSFFWHM\t%f %f\n'%(fwhm1,fwhm2))
                
            elif gaussian==1:
                
                weight1, fwhm1 = mofgauFit.psf_img_to_gauss(img)
                
                modelfile.write('PSFWEIGHT\t%f\n'%(weight1))
                modelfile.write('PSFFWHM\t%f\n'%(fwhm1))
                
                
                
                
            else:
                print('error... Gaussian component num no supported')
            #if gaussian==3:
            #    weight1, weight2, weight3, fwhm1, fwhm2 ,fwhm3= mofgauFit.psf_img_to_gauss_three(img)
            #    
            #    modelfile.write('PSFWEIGHT\t%f %f %f\n'%(weight1,weight2,weight3))
            #    modelfile.write('PSFFWHM\t%f %f %f\n'%(fwhm1,fwhm2,fwhm3))
                
                
                
                
        else:
            print('error... PSF fitting method not supported')
            #PSF
            #psfhdr = self.fitsdata[4].header
            # note that the datacubes have mistake, alpha is beta , beta is alpha
            #beta = psfhdr['MAGPI PSF ZBAND MOFFAT ALPHA']
            #alpha = psfhdr['MAGPI PSF ZBAND MOFFAT BETA']
            
            #weight1, weight2, fwhm1, fwhm2 = mofgauFit.mof_to_gauss(alpha=alpha, 
            #                                                        beta=beta)
            #modelfile.write('PSFWEIGHT\t%f %f\n'%(weight1,weight2))
            #modelfile.write('PSFFWHM\t%f %f\n'%(fwhm1,fwhm2))
        
        # inclination
        inc_data =  pandas.read_csv(inc_path)
        index = np.where(inc_data['MAGPIID']==int(self.magpiid))[0][0]
        semimaj = inc_data['semimaj'][index]
        semimin = inc_data['semimin'][index]
        q0 = 0.2
        if (semimin/semimaj) >= q0:
            inc = np.arccos(np.sqrt(((semimin/semimaj)**2 - q0**2)/(1 - q0**2)))
        elif (semimin/semimaj) < q0:
            inc = np.arccos(0)
        
        modelfile.write('INC\t%f\n'%(inc))
        if self.emi_line =='Ha':
            modelfile.write('LINE\t6562.81\n')
            modelfile.write('LINE\t6583.1\t6548.1\t0.333\n')
        elif self.emi_line == 'Oii':
            modelfile.write('LINE\t3727.092\n')
            modelfile.write('LINE\t3729.875\n')
        elif self.emi_line == 'Hb':
            modelfile.write('LINE\t4862.683\n')
        
        
        if flat_vdisp == True:
            modelfile.write('VDISPN_SIGMA\t1.000000e-09\n')
        
        if constrain_pa is not None:
            kinematics_df = pd.read_csv(constrain_pa)
            
            pa_array = kinematics_df['PA'].to_numpy()
            pa_mean, pa_std = self.circular_mean_and_std(pa_array)
            
            if pa_std>0.000001: # if too small, then pa_min would = pa_max when round to 6 decimal places
                modelfile.write('PA_MIN\t{:.6f}\n'.format(pa_mean - pa_std))
                modelfile.write('PA_MAX\t{:.6f}\n'.format(pa_mean + pa_std))
            else:
                modelfile.write('PA_MIN\t{:.6f}\n'.format(pa_mean - np.power(0.1,5-self.get_order(pa_mean))))
                modelfile.write('PA_MAX\t{:.6f}\n'.format(pa_mean + np.power(0.1,5-self.get_order(pa_mean))))
            
        
        if constrain_kinematics is not None:
            kinematics_df = pd.read_csv(constrain_kinematics)
            # vsys to be free
            i = 0
            vc = kinematics_df['VMAX'][i]
            rt = kinematics_df['VSLOPE'][i]
            beta = kinematics_df['VBETA'][i]
            gamma = kinematics_df['VGAMMA'][i]
            pa_array = kinematics_df['PA'].to_numpy()
            #vsys = kinematics_df['VSYS'][i]
            
            modelfile.write('VC_MIN\t{:.6f}\n'.format(vc - np.power(0.1,5-self.get_order(vc))))
            modelfile.write('VC_MAX\t{:.6f}\n'.format(vc + np.power(0.1,5-self.get_order(vc))))
            
            modelfile.write('VSLOPE_MIN\t{:.6f}\n'.format(rt - np.power(0.1,5-self.get_order(rt))))
            modelfile.write('VSLOPE_MAX\t{:.6f}\n'.format(rt + np.power(0.1,5-self.get_order(rt))))
            
            modelfile.write('VBETA_MIN\t{:.6f}\n'.format(beta - np.power(0.1,5-self.get_order(beta))))
            modelfile.write('VBETA_MAX\t{:.6f}\n'.format(beta + np.power(0.1,5-self.get_order(beta))))
            
            modelfile.write('VGAMMA_MIN\t{:.6f}\n'.format(gamma - np.power(0.1,5-self.get_order(gamma))))
            modelfile.write('VGAMMA_MAX\t{:.6f}\n'.format(gamma + np.power(0.1,5-self.get_order(gamma))))
            
            pa_mean, pa_std = self.circular_mean_and_std(pa_array)
            
            if pa_std>0.000001:
                modelfile.write('PA_MIN\t{:.6f}\n'.format(pa_mean - pa_std))
                modelfile.write('PA_MAX\t{:.6f}\n'.format(pa_mean + pa_std))
            else:
                modelfile.write('PA_MIN\t{:.6f}\n'.format(pa_mean - np.power(0.1,5-self.get_order(pa_mean))))
                modelfile.write('PA_MAX\t{:.6f}\n'.format(pa_mean + np.power(0.1,5-self.get_order(pa_mean))))
            
            
            if constrain_vdisp == True:
                # vdisp take median value of all effective samples
                vdisp = kinematics_df['VDISP0'].median()
                modelfile.write('LOGVDISP0_MIN\t{:.6f}\n'.format(vdisp - np.power(0.1,5-self.get_order(vdisp))))
                modelfile.write('LOGVDISP0_MAX\t{:.6f}\n'.format(vdisp + np.power(0.1,5-self.get_order(vdisp))))
        if vsys_set is not None:
            modelfile.write('VSYS_MAX\t{:.1f}\n'.format(vsys_set))
            
            
        
        modelfile.close()
        
    def get_order(self,num):
        # Handling zero separately because log10(0) is undefined
        if num == 0:
            raise ValueError("log10(0) is undefined, so the order of 0 is not defined.")
        
        # Calculate the order using log10 and floor function
        order = math.floor(math.log10(abs(num)))
        
        if order<0:
            order = -1
        
        return order

    # Test cases
    #print(get_order(321))  # Output: 2
    #print(get_order(67))   # Output: 1
    #print(get_order(3))    # Output: 0
    #print(get_order(0.3))  # Output: -1
    
    def circular_mean_and_std(self,angles_rad):
        x = np.cos(angles_rad)
        y = np.sin(angles_rad)
        
        # Compute the mean direction
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Mean angle in radians
        mean_angle_rad = np.arctan2(mean_y, mean_x)
        
        # the above func return [-pi,pi], but I prefer[0,2pi], so plus
        # 2pi for negative angles.....
        if mean_angle_rad<0:
            mean_angle_rad = mean_angle_rad + 2*np.pi
        
        # Resultant vector length
        R = np.sqrt(mean_x**2 + mean_y**2)
        
        # Circular standard deviation in radians
        circular_std_rad = np.sqrt(-2 * np.log(R))
        
        return mean_angle_rad, circular_std_rad
        
    def dn4_options(self,mask_dilated_mask=True):
        modelfile = open(self.save_path+"OPTIONS","w")
        
        ha_sn_masked = self.ha_sn
        if mask_dilated_mask==True:
            ha_sn_masked[self.dilated_mask==1] = np.nan
        
        ha_sn_1d = ha_sn_masked.flatten()
        npixel = sum(ha_sn_1d >= 3)
        
        
        if self.emi_line == 'Ha':
            if npixel <= 300:
                iterations = 5000
            elif npixel <= 400:
                iterations = 7000
            elif npixel <= 500:
                iterations = 9000
            elif npixel <= 800:
                iterations = 11000
            elif npixel <= 1000:
                iterations = 15000
            elif npixel <= 3000:
                iterations = 20000
            else:
                iterations = 25000
        elif self.emi_line=='Oii' or self.emi_line=='Hb':
            if npixel <= 300:
                iterations = 5000
            elif npixel <= 400:
                iterations = 7000
            elif npixel <= 500:
                iterations = 9000
            elif npixel <= 800:
                iterations = 11000
            elif npixel <= 1000:
                iterations = 13000
            elif npixel <= 3000:
                iterations = 15000
            else:
                iterations = 20000
        
        modelfile.write('# File containing parameters for DNest4\n')
        modelfile.write('# Put comments at the top, or at the end of the line.\n')
        modelfile.write('1	# Number of particles\n')
        modelfile.write('10000	# New level interval\n')
        modelfile.write('10000	# Save interval\n')
        modelfile.write('100	# Thread steps - how many steps each thread should do independently before communication\n')
        modelfile.write('0	# Maximum number of levels\n')
        modelfile.write('10	# Backtracking scale length (lambda in the paper)\n')
        modelfile.write('100	# Strength of effect to force histogram to equal push (beta in the paper)\n')
        modelfile.write('%d	# Maximum number of saves (0 = infinite)\n'%(iterations))
        modelfile.write('sample.txt	# Sample file\n')
        modelfile.write('sample_info.txt	# Sample info file\n')
        modelfile.write('levels.txt	# Sample file\n')
        modelfile.close()
        
        
    def try_sth(self,w):
        print('hello')
        
    def get_flux_scale_factor(self):
        flux_scale_factor_1 = (np.nanmedian(self.flux[:,int(self.ny/2),int(self.nx/2)])*10)
        return flux_scale_factor_1
    
    def get_fwhm_highweight(self):
        #PSF
        psfhdr = self.fitsdata[4].header
        # note that the datacubes have mistake, alpha is beta , beta is alpha
        beta = psfhdr['MAGPI PSF ZBAND MOFFAT ALPHA']
        alpha = psfhdr['MAGPI PSF ZBAND MOFFAT BETA']
        
        weight1, weight2, fwhm1, fwhm2 = mofgauFit.mof_to_gauss(alpha=alpha, 
                                                                beta=beta)
        if weight1 > weight2:
            return fwhm1
        else:
            return fwhm2
    
    
    def plot_BPT_cut(self,snlim=None,xlim=None, ylim=None,mask_center=False,
                     mask_center_pix=None,mask_radius=None,mask_dilated_mask=True,
                     BPT_save=None):
        '''
        
        bpt-diagram with error bar, and cut minicube

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
        magpiid = self.magpiid
        fitspath = self.emipath_w
        
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
        dilated_mask_1 = np.full_like(dilated_mask,False,dtype=bool)
        if mask_dilated_mask==True:
            dilated_mask_1 = (dilated_mask==1)
        
        ### read data ####
        ha = dmap['Ha_F'].data
        ha_err = dmap['Ha_FERR'].data
        ha[dilated_mask_1] = np.nan
        ha_err[dilated_mask_1] = np.nan
        ha = self.sn_cut_2d_for_bpt(snlim=3, data=ha)
        ha = self.cutout(data=ha,dim=2,xlim=xlim,ylim=ylim)
        ha_err = self.cutout(data=ha_err,dim=2,xlim=xlim,ylim=ylim)
        ha_snr = ha/ha_err
        
        nii = dmap['NII_6585_F'].data
        nii_err = dmap['NII_6585_FERR'].data
        nii[dilated_mask_1] = np.nan
        nii_err[dilated_mask_1] = np.nan
        nii = self.sn_cut_2d_for_bpt(snlim=3, data=nii)
        nii = self.cutout(data=nii,dim=2,xlim=xlim,ylim=ylim)
        nii_err = self.cutout(data=nii_err,dim=2,xlim=xlim,ylim=ylim)
        nii_snr = nii/nii_err
        
        oiii = dmap['OIII_5008_F'].data
        oiii_err = dmap['OIII_5008_FERR'].data
        oiii[dilated_mask_1] = np.nan
        oiii_err[dilated_mask_1] = np.nan
        oiii = self.sn_cut_2d_for_bpt(snlim=3, data=oiii)
        oiii = self.cutout(data=oiii,dim=2,xlim=xlim,ylim=ylim)
        oiii_err = self.cutout(data=oiii_err,dim=2,xlim=xlim,ylim=ylim)
        oiii_snr = oiii/oiii_err
        
        hb = dmap['Hb_F'].data
        hb_err = dmap['Hb_FERR'].data
        hb[dilated_mask_1] = np.nan
        hb_err[dilated_mask_1] = np.nan
        hb = self.sn_cut_2d_for_bpt(snlim=3, data=hb)
        hb = self.cutout(data=hb,dim=2,xlim=xlim,ylim=ylim)
        hb_err = self.cutout(data=hb_err,dim=2,xlim=xlim,ylim=ylim)
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
        
        
        if BPT_save is not None:
            AGN_num = AGN.sum()
            CP_num = CP.sum()
            SF_num = SF.sum()
            df = pd.DataFrame({'AGN_num':[AGN_num],
                               'CP_num':[CP_num],
                               'SF_num':[SF_num]})
            df.to_csv(BPT_save)
        
        
        fig = plt.figure(constrained_layout=True,figsize=(15,15))
        gs = GridSpec(3, 3, figure=fig)
        
        
        fig.suptitle(magpiid,fontsize=25)
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[2, 0])
        ax7 = fig.add_subplot(gs[2, 1])
        ax8 = fig.add_subplot(gs[2, 2])
        
        # bpt-diagram with error
        
        ax9 = fig.add_subplot(gs[0,0])
        
        
        
        
        ax1.tick_params(direction='in', labelsize = 20, length = 5, width=1.0)
        ax1.set_title(r'resolved BPT-map',fontsize=25)
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
        ax1.pcolormesh(xpos, ypos, new_region, cmap=cmap, norm=norm)
        ax1.set_xlabel('arcsec',fontsize=25)
        ax1.set_ylabel('arcsec',fontsize=25)
        
        
        
        #plt.savefig(figpath+'BPT-'+str(plate)+'-'+str(ifu)+'-Diagram.png',bbox_inches='tight',dpi=300)
        #plt.show()
        ###########################
        ########bpt diagram map#########################
        #fig, axs = plt.subplots()
        
        
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
        
        
        
        ###########
        ###########
        
        ### read data ####
        ha = dmap['Ha_F'].data
        ha_err = dmap['Ha_FERR'].data
        ha[dilated_mask_1] = np.nan
        ha_err[dilated_mask_1] = np.nan
        
        if mask_center==True:
            mask_outflow = np.full_like(self.ha_sn,False,dtype=bool)
            total_rows, total_cols = mask_outflow.shape
            #center_row, center_col = total_rows/2, total_cols/2
            #integ_flux = np.nansum(flux_mask,axis=0)
            center_row, center_col = int(mask_outflow.shape[0]/2),int(mask_outflow.shape[1]/2)
            
            if mask_center_pix is not None:
                center_col, center_row = mask_center_pix
            Y, X = np.ogrid[:total_rows, :total_cols]
            dist_from_center = np.sqrt((Y - center_row)**2 + (X-center_col)**2)
            circular_mask = (dist_from_center < mask_radius)
            ha[circular_mask] = np.nan
            
        
        
        ha = self.sn_cut_2d_for_bpt(snlim=3, data=ha)
        ha = self.cutout(data=ha,dim=2,xlim=xlim,ylim=ylim)
        ha_err = self.cutout(data=ha_err,dim=2,xlim=xlim,ylim=ylim)
        ha_snr = ha/ha_err
        
        nii = dmap['NII_6585_F'].data
        nii_err = dmap['NII_6585_FERR'].data
        nii[dilated_mask_1] = np.nan
        nii_err[dilated_mask_1] = np.nan
        nii = self.sn_cut_2d_for_bpt(snlim=3, data=nii)
        nii = self.cutout(data=nii,dim=2,xlim=xlim,ylim=ylim)
        nii_err = self.cutout(data=nii_err,dim=2,xlim=xlim,ylim=ylim)
        nii_snr = nii/nii_err
        
        oiii = dmap['OIII_5008_F'].data
        oiii_err = dmap['OIII_5008_FERR'].data
        oiii[dilated_mask_1] = np.nan
        oiii_err[dilated_mask_1] = np.nan
        oiii = self.sn_cut_2d_for_bpt(snlim=3, data=oiii)
        oiii = self.cutout(data=oiii,dim=2,xlim=xlim,ylim=ylim)
        oiii_err = self.cutout(data=oiii_err,dim=2,xlim=xlim,ylim=ylim)
        oiii_snr = oiii/oiii_err
        
        hb = dmap['Hb_F'].data
        hb_err = dmap['Hb_FERR'].data
        hb[dilated_mask_1] = np.nan
        hb_err[dilated_mask_1] = np.nan
        hb = self.sn_cut_2d_for_bpt(snlim=3, data=hb)
        hb = self.cutout(data=hb,dim=2,xlim=xlim,ylim=ylim)
        hb_err = self.cutout(data=hb_err,dim=2,xlim=xlim,ylim=ylim)
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
        
        
        ax2.tick_params(direction='in', labelsize = 20, length = 5, width=1.0)
        ax2.set_title(r'resolved BPT-map masked',fontsize=25)
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
        
        plt.show()




    def plot_BPT_cut_lowsn(self,snlim=None,xlim=None, ylim=None,mask_center=False,
                     mask_center_pix=None,mask_radius=None,mask_dilated_mask=True,
                     BPT_save=None):
        '''
        
        bpt-diagram with error bar, and cut minicube
        
        for spxels with low sn emission lines (sn<3), 
        use three times of error as value of emission line flux...
        
    
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
        magpiid = self.magpiid
        fitspath = self.emipath_w
        
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
        dilated_mask_1 = np.full_like(dilated_mask,False,dtype=bool)
        if mask_dilated_mask==True:
            dilated_mask_1 = (dilated_mask==1)
        
        ### read data ####
        ha = dmap['Ha_F'].data
        ha_err = dmap['Ha_FERR'].data
        ha[dilated_mask_1] = np.nan
        ha_err[dilated_mask_1] = np.nan
        ha = self.sn_cut_2d_for_bpt(snlim=3, data=ha)
        ha = self.cutout(data=ha,dim=2,xlim=xlim,ylim=ylim)
        ha_err = self.cutout(data=ha_err,dim=2,xlim=xlim,ylim=ylim)
        ha_snr = ha/ha_err
        
        nii = dmap['NII_6585_F'].data
        nii_err = dmap['NII_6585_FERR'].data
        nii[dilated_mask_1] = np.nan
        nii_err[dilated_mask_1] = np.nan
        nii = self.sn_cut_2d_for_bpt(snlim=3, data=nii)
        nii = self.cutout(data=nii,dim=2,xlim=xlim,ylim=ylim)
        nii_err = self.cutout(data=nii_err,dim=2,xlim=xlim,ylim=ylim)
        nii_snr = nii/nii_err
        
        nii_lowsn = nii_snr<=3
        nii[nii_lowsn] = nii_err[nii_lowsn]*3.01
        nii_snr = nii/nii_err
        
        
        
        oiii = dmap['OIII_5008_F'].data
        oiii_err = dmap['OIII_5008_FERR'].data
        oiii[dilated_mask_1] = np.nan
        oiii_err[dilated_mask_1] = np.nan
        oiii = self.sn_cut_2d_for_bpt(snlim=3, data=oiii)
        oiii = self.cutout(data=oiii,dim=2,xlim=xlim,ylim=ylim)
        oiii_err = self.cutout(data=oiii_err,dim=2,xlim=xlim,ylim=ylim)
        oiii_snr = oiii/oiii_err
        
        oiii_lowsn = oiii_snr<=3
        oiii[oiii_lowsn] = oiii_err[oiii_lowsn]*3.01
        oiii_snr = oiii/oiii_err
        
        
        
        hb = dmap['Hb_F'].data
        hb_err = dmap['Hb_FERR'].data
        hb[dilated_mask_1] = np.nan
        hb_err[dilated_mask_1] = np.nan
        hb = self.sn_cut_2d_for_bpt(snlim=3, data=hb)
        hb = self.cutout(data=hb,dim=2,xlim=xlim,ylim=ylim)
        hb_err = self.cutout(data=hb_err,dim=2,xlim=xlim,ylim=ylim)
        hb_snr = hb/hb_err
        
        hb_lowsn = hb_snr<=3
        hb[hb_lowsn] = hb_err[hb_lowsn]*3.01
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
        
        
        if BPT_save is not None:
            AGN_num = AGN.sum()
            CP_num = CP.sum()
            SF_num = SF.sum()
            df = pd.DataFrame({'AGN_num':[AGN_num],
                               'CP_num':[CP_num],
                               'SF_num':[SF_num]})
            df.to_csv(BPT_save)
        
        
        fig = plt.figure(constrained_layout=True,figsize=(15,15))
        gs = GridSpec(3, 3, figure=fig)
        
        
        fig.suptitle(magpiid,fontsize=25)
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[2, 0])
        ax7 = fig.add_subplot(gs[2, 1])
        ax8 = fig.add_subplot(gs[2, 2])
        
        # bpt-diagram with error
        
        ax9 = fig.add_subplot(gs[0,0])
        
        
        
        
        ax1.tick_params(direction='in', labelsize = 20, length = 5, width=1.0)
        ax1.set_title(r'resolved BPT-map',fontsize=25)
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
        ax1.pcolormesh(xpos, ypos, new_region, cmap=cmap, norm=norm)
        ax1.set_xlabel('arcsec',fontsize=25)
        ax1.set_ylabel('arcsec',fontsize=25)
        
        
        
        #plt.savefig(figpath+'BPT-'+str(plate)+'-'+str(ifu)+'-Diagram.png',bbox_inches='tight',dpi=300)
        #plt.show()
        ###########################
        ########bpt diagram map#########################
        #fig, axs = plt.subplots()
        
        
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
        
        
        
        ###########
        ###########
        
        ### read data ####
        ha = dmap['Ha_F'].data
        ha_err = dmap['Ha_FERR'].data
        ha[dilated_mask_1] = np.nan
        ha_err[dilated_mask_1] = np.nan
        
        if mask_center==True:
            mask_outflow = np.full_like(self.ha_sn,False,dtype=bool)
            total_rows, total_cols = mask_outflow.shape
            #center_row, center_col = total_rows/2, total_cols/2
            #integ_flux = np.nansum(flux_mask,axis=0)
            center_row, center_col = int(mask_outflow.shape[0]/2),int(mask_outflow.shape[1]/2)
            
            if mask_center_pix is not None:
                center_col, center_row = mask_center_pix
            Y, X = np.ogrid[:total_rows, :total_cols]
            dist_from_center = np.sqrt((Y - center_row)**2 + (X-center_col)**2)
            circular_mask = (dist_from_center < mask_radius)
            ha[circular_mask] = np.nan
            
        
        
        ha = self.sn_cut_2d_for_bpt(snlim=3, data=ha)
        ha = self.cutout(data=ha,dim=2,xlim=xlim,ylim=ylim)
        ha_err = self.cutout(data=ha_err,dim=2,xlim=xlim,ylim=ylim)
        ha_snr = ha/ha_err
        
        nii = dmap['NII_6585_F'].data
        nii_err = dmap['NII_6585_FERR'].data
        nii[dilated_mask_1] = np.nan
        nii_err[dilated_mask_1] = np.nan
        nii = self.sn_cut_2d_for_bpt(snlim=3, data=nii)
        nii = self.cutout(data=nii,dim=2,xlim=xlim,ylim=ylim)
        nii_err = self.cutout(data=nii_err,dim=2,xlim=xlim,ylim=ylim)
        nii_snr = nii/nii_err
        
        nii_lowsn = nii_snr<=3
        nii[nii_lowsn] = nii_err[nii_lowsn]*3.01
        nii_snr = nii/nii_err
        
        oiii = dmap['OIII_5008_F'].data
        oiii_err = dmap['OIII_5008_FERR'].data
        oiii[dilated_mask_1] = np.nan
        oiii_err[dilated_mask_1] = np.nan
        oiii = self.sn_cut_2d_for_bpt(snlim=3, data=oiii)
        oiii = self.cutout(data=oiii,dim=2,xlim=xlim,ylim=ylim)
        oiii_err = self.cutout(data=oiii_err,dim=2,xlim=xlim,ylim=ylim)
        oiii_snr = oiii/oiii_err
        
        oiii_lowsn = oiii_snr<=3
        oiii[oiii_lowsn] = oiii_err[oiii_lowsn]*3.01
        oiii_snr = oiii/oiii_err
        
        
        hb = dmap['Hb_F'].data
        hb_err = dmap['Hb_FERR'].data
        hb[dilated_mask_1] = np.nan
        hb_err[dilated_mask_1] = np.nan
        hb = self.sn_cut_2d_for_bpt(snlim=3, data=hb)
        hb = self.cutout(data=hb,dim=2,xlim=xlim,ylim=ylim)
        hb_err = self.cutout(data=hb_err,dim=2,xlim=xlim,ylim=ylim)
        hb_snr = hb/hb_err
        
        hb_lowsn = hb_snr<=3
        hb[hb_lowsn] = hb_err[hb_lowsn]*3.01
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
        
        
        ax2.tick_params(direction='in', labelsize = 20, length = 5, width=1.0)
        ax2.set_title(r'resolved BPT-map masked',fontsize=25)
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
        
        plt.show()
    
    
    def ha_flux_compare(self,snlim=None,xlim=None, ylim=None,mask_center=False,
                     mask_center_pix=None,mask_radius=None,mask_dilated_mask=True,
                     BPT_save=None):
        '''
        
        bpt-diagram with error bar, and cut minicube
        
        for spxels with low sn emission lines (sn<3), 
        use three times of error as value of emission line flux...
        
        compare the ha_flux in sf region, and ha_flux in sf+composite region
    
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
        magpiid = self.magpiid
        fitspath = self.emipath_w
        
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
        dilated_mask_1 = np.full_like(dilated_mask,False,dtype=bool)
        if mask_dilated_mask==True:
            dilated_mask_1 = (dilated_mask==1)
        
        ### read data ####
        ha = dmap['Ha_F'].data
        ha_err = dmap['Ha_FERR'].data
        ha[dilated_mask_1] = np.nan
        ha_err[dilated_mask_1] = np.nan
        ha = self.sn_cut_2d_for_bpt(snlim=3, data=ha)
        ha = self.cutout(data=ha,dim=2,xlim=xlim,ylim=ylim)
        ha_err = self.cutout(data=ha_err,dim=2,xlim=xlim,ylim=ylim)
        ha_snr = ha/ha_err
        
        nii = dmap['NII_6585_F'].data
        nii_err = dmap['NII_6585_FERR'].data
        nii[dilated_mask_1] = np.nan
        nii_err[dilated_mask_1] = np.nan
        nii = self.sn_cut_2d_for_bpt(snlim=3, data=nii)
        nii = self.cutout(data=nii,dim=2,xlim=xlim,ylim=ylim)
        nii_err = self.cutout(data=nii_err,dim=2,xlim=xlim,ylim=ylim)
        nii_snr = nii/nii_err
        
        nii_lowsn = nii_snr<=3
        nii[nii_lowsn] = nii_err[nii_lowsn]*3.01
        nii_snr = nii/nii_err
        
        
        
        oiii = dmap['OIII_5008_F'].data
        oiii_err = dmap['OIII_5008_FERR'].data
        oiii[dilated_mask_1] = np.nan
        oiii_err[dilated_mask_1] = np.nan
        oiii = self.sn_cut_2d_for_bpt(snlim=3, data=oiii)
        oiii = self.cutout(data=oiii,dim=2,xlim=xlim,ylim=ylim)
        oiii_err = self.cutout(data=oiii_err,dim=2,xlim=xlim,ylim=ylim)
        oiii_snr = oiii/oiii_err
        
        oiii_lowsn = oiii_snr<=3
        oiii[oiii_lowsn] = oiii_err[oiii_lowsn]*3.01
        oiii_snr = oiii/oiii_err
        
        
        
        hb = dmap['Hb_F'].data
        hb_err = dmap['Hb_FERR'].data
        hb[dilated_mask_1] = np.nan
        hb_err[dilated_mask_1] = np.nan
        hb = self.sn_cut_2d_for_bpt(snlim=3, data=hb)
        hb = self.cutout(data=hb,dim=2,xlim=xlim,ylim=ylim)
        hb_err = self.cutout(data=hb_err,dim=2,xlim=xlim,ylim=ylim)
        hb_snr = hb/hb_err
        
        hb_lowsn = hb_snr<=3
        hb[hb_lowsn] = hb_err[hb_lowsn]*3.01
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
        
        
        ha_sf = np.nansum(ha[SF])
        
        ha_cp = np.nansum(ha[CP])
        
        ha_sfandcp = ha_sf + ha_cp
        
        return ha_sf, ha_sfandcp