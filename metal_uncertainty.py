#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:15:31 2024

calculate the uncertainty of flux applied dust correction

use Fitzpatrick+19 extinction curve

@author: yifan mai
"""

import numpy as np
import astropy.units as u
from dust_extinction.parameter_averages import F19
from scipy.optimize import curve_fit
from scipy import interpolate
from dust_extinction.averages import G03_SMCBar
from dust_extinction.averages import G03_LMCAvg


class emi_wave():
    # just for reference ...
    oii_wave = 3728.4835
    hb_wave = 4862.683
    ha_wave = 6564.61
    nii_wave = 6585.28
    oiii_wave = 5007
    sii_6716_wave = 6716
    sii_6731_wave = 6731
    

def log10_ratio_with_error(a, b, a_err, b_err):
    # Calculate the value of log10(a/b)
    log10_ratio = np.log10(a / b)
    
    # Calculate the error using error propagation
    ln10 = np.log(10)
    log10_ratio_err = np.sqrt((a_err / (a * ln10))**2 + (b_err / (b * ln10))**2)
    
    return log10_ratio, log10_ratio_err

def E_B_V_with_error(ha, hb, ha_err, hb_err,model='F19'):
    
    '''
    E(B-V) = 2.5/(k(lambda_hb) - k(lambda_ha)) * log10((ha/hb)_obs/(ha/hb)_int)
    calculate E(B-V) and its error
    
    Parameters:
        ha: array-like
            flux of Halpha
        hb: array-like
            flux of Hbeta
        ha_err: array-like
            error of Halpha flux
        hb_er: array-like
            error of Hbeta flux
    Return:
        E_B_V: array-like
            E(B-V)
        E_B_V_err: array-like
            error of E(B-V)
    
    
    '''
    hb_wave = 4862.683
    ha_wave = 6564.61
    Rv = 3.1
    # calculate k(lambda_hb) - k(lambda_ha)
    # F19.evaluate return A(x)/A(V), 
    # we know A(x)/A(V) = k(x-V) / Rv + 1.0
    # so k(lambda_hb) - k(lambda_ha) = formula below
    if model=='F19':
        ext = F19(Rv=3.1)
        khb_kha = (ext.evaluate(in_x=hb_wave*u.AA, Rv=Rv) - \
            ext.evaluate(in_x=ha_wave*u.AA, Rv=Rv)) * Rv
    elif model=='SMC':
        ext = G03_SMCBar()
        khb_kha = (ext.evaluate(in_x=hb_wave*u.AA) - \
            ext.evaluate(in_x=ha_wave*u.AA)) * Rv
    elif model=='LMC':
        ext = G03_LMCAvg()
        khb_kha = (ext.evaluate(in_x=hb_wave*u.AA) - \
            ext.evaluate(in_x=ha_wave*u.AA)) * Rv
    
    
    
    balmer_decre = ha/hb
    
    # no negative correction
    if balmer_decre.size == 1:
        if balmer_decre < 2.86:
            balmer_decre = 2.86
    else:
        balmer_decre[(ha/hb)<2.86] = 2.86
    
    E_B_V = 2.5/khb_kha * np.log10(balmer_decre/(2.86))
    
    log10ratio, log10ratio_err = log10_ratio_with_error(ha, hb, ha_err, hb_err)
    
    E_B_V_err = 2.5/khb_kha * log10ratio_err
    
    return E_B_V, E_B_V_err

def intrinsic_flux_with_err(flux_obs,flux_obs_err,wave,ha,hb,ha_err,hb_err,
                            foreground_E_B_V,z,dust_model='F19'):
    '''
    this functioin do two steps of dust correction and calculate the flux err
    1. foreground correction for the dust in the Milky Way
    2. dust correction for the dust in the source galaxy
    
    flux_int = flux_obs * 10^(0.4 * k(lambda) * E(B-V))
    
    Parameters:
        flux_obs: array-like
            observed flux
        flux_obs_err: array-like
            error of observed flux
        wave: float
            is the rest-frame wavelength for the emission line flux we want to correct
        ha: array-like
            observed ha flux
        hb: array-like
            observed hb flux
        ha_err: array-like
            error of the observed ha flux
        hb_err: array-like
            error of the observed hb flux
        foreground_E_B_V: float
            E(B-V) for the foreground dust correction for the dust in MW
        z: float
            redshift of the source galaxy
    
    Return:
        flux_int: array-like
            intrinsic flux (corrected flux)
        flux_int_err: array-like
            error of the intrinsic flux
    
    '''
    Rv = 3.1
    
    ha_correct_MW,ha_err_correct_MW = dust_correction(flux=ha, flux_err=ha_err,
                                               E_B_V=foreground_E_B_V, 
                                               wavelength=emi_wave.ha_wave* (1+z),
                                               model='F19')
    
    hb_correct_MW,hb_err_correct_MW = dust_correction(flux=hb, flux_err=hb_err,
                                               E_B_V=foreground_E_B_V, 
                                               wavelength=emi_wave.hb_wave* (1+z),
                                               model='F19')
    
    # MW correct for the flux we want to correct
    # assume no error in foreground_E_B_V
    flux_correct_MW,flux_err_correct_MW = dust_correction(flux=flux_obs, flux_err=flux_obs_err,
                                               E_B_V=foreground_E_B_V,
                                               wavelength=wave* (1+z),
                                               model='F19')
    
    # E_B_V for the source galaxy
    E_B_V_intrin, E_B_V_intrin_err = E_B_V_with_error(ha=ha_correct_MW, 
                                                      hb=hb_correct_MW, 
                                                      ha_err=ha_err_correct_MW, 
                                                      hb_err=hb_err_correct_MW,
                                                      model=dust_model)
    # k(x) = A(x)/A(V) * R(V)
    # F19.evaluate return A(x)/A(V)
    
    if dust_model=='F19':
        ext = F19(Rv=Rv)
        k_lambda = ext.evaluate(in_x=wave*u.AA, Rv=Rv) * Rv
    elif dust_model=='SMC':
        ext = G03_SMCBar()
        k_lambda = ext.evaluate(in_x=wave*u.AA) * Rv
    elif dust_model=='LMC':
        ext = G03_LMCAvg()
        k_lambda = ext.evaluate(in_x=wave*u.AA) * Rv
    
    
    
    
    
    flux_int = flux_correct_MW * np.power(10,0.4*k_lambda*E_B_V_intrin)
    flux_int_err = np.sqrt((flux_err_correct_MW*np.power(10,0.4*k_lambda*E_B_V_intrin))**2+
                           (E_B_V_intrin_err*flux_correct_MW*np.log(10)*np.power(10,0.4*k_lambda*E_B_V_intrin)*0.4*k_lambda)**2)
    
    
    return flux_int, flux_int_err



def dust_correction_2d(flux_map, flux_err_map, E_B_V_map, wavelength):
    '''
    do the dust correction for E(B-V) map (each spaxel have different E(B-V))
    assuming no err in E_B_V
    
    Parameters:
        flux: array-like
            flux to be correct
        flux_err: array-like
            error of the flux
        E_B_V: array-like
            E(B-V) map
        wavelength: float
            wavelength for the emission line
    
    
    
    return: 
        flux_correct: array-like
            dust corrected flux
        flux_err_correct: array-like
            error of corrected flux
    
    '''
    flux_map_correct = np.full_like(flux_map,np.nan)
    flux_err_map_correct = np.full_like(flux_map,np.nan)
    
    ny, nx = flux_map.shape
    for i in range(ny):
        for j in range(nx):
            flux_map_correct[i,j],flux_err_map_correct[i,j] = dust_correction(flux=flux_map[i,j], 
                                                    flux_err=flux_err_map[i,j],
                                                    E_B_V=E_B_V_map[i,j],
                                                    wavelength=wavelength)
    
    return flux_map_correct, flux_err_map_correct
    
    

def dust_correction(flux, flux_err, E_B_V,wavelength,model='F19'):
    '''
    do the dust correction for a given E(B-V)
    assuming no err in E_B_V
    
    Parameters:
        flux: array-like
            flux to be correct
        flux_err: array-like
            error of the flux
        E_B_V: float
            E(B-V)
        wavelength: float
            wavelength for the emission line
        model: str, default is F19
            option: F19, SMC, LMC
    
    
    
    return: 
        flux_correct: array-like
            dust corrected flux
        flux_err_correct: array-like
            error of corrected flux
    '''
    flux = flux * u.erg/(u.s * u.cm**2 * u.AA)
    flux_err = flux_err * u.erg/(u.s * u.cm**2 * u.AA)
    wavelength = wavelength * u.AA
    if model=='F19':
        ext = F19(Rv=3.1)
    elif model=='SMC':
        ext = G03_SMCBar()
    elif model=='LMC':
        ext = G03_LMCAvg()
    flux_correct = flux / ext.extinguish(wavelength, Ebv=E_B_V)
    flux_err_correct = flux_err / ext.extinguish(wavelength, Ebv=E_B_V)
    flux_correct = flux_correct.value
    flux_err_correct = flux_err_correct.value
    return flux_correct, flux_err_correct