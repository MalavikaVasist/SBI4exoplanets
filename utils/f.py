import torch
from torch import Tensor
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

import os
scratch = os.environ.get('SCRATCH') 
home = os.environ.get('HOME')

import sys
sys.path.insert(0, str(Path(home) / 'HighRes'))
from simulations.DataProcuring import Data 

import astropy.units as u
import astropy.constants as const
from simulations.parameter_set_script import param_set_ext

# param_set, param_list_ext, deNormVal

from code2explore.observations import load_observations
from added_scripts.pt_plotting import levels_and_creds, PT_plot as PT_plot_jwst
# from code2explore.NPE_new import NPEWithEmbedding_sepEmb , NPEWithEmbedding_sepEmb_diffdim, NPEWithEmbedding_oneEmb
# from added_scripts.AverageEstimator_chatgpt import avgestimator
from code2explore.flexible_simulator import pt_profile_theta, pt_profile_parameters, pressureCGS, \
                                            pressureBAR

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from scipy.interpolate import interp1d

import pandas as pd

from petitRADTRANS import nat_cst as nc, Radtrans
import petitRADTRANS as prt



def return_tchvalues_from_files(path):
    df_v = pd.read_csv(path)
    v = df_v.values
    v = torch.from_numpy(v)
    
    return v

def save_tchvalues_to_files(path, v):
    df_v = pd.DataFrame(v) #convert to a dataframe
    df_v.to_csv(path,index=False) #save to file
    
    return print('saved')

def consistency_plot_full(xmhg_, thetamhg_, sigmaM, xhhg_, thetahhg_, sigmaH, simulator, obs_wlen_miri, obs_miri, \
                     obs_wlen_hst, obs_hst, full= False, xhhg_short= None, sigmaH_short= None ):
    
    if full :
        _, _, _, \
                obs_wlen_hstf, obs_hstf, _, \
                _, \
                _ = load_observations(data = 'WISEJ1828')
        
    else:
        obs_wlen_hstf, obs_hstf = obs_wlen_hst, obs_hst
        
    def noisybfactor( x: Tensor, b: Tensor, sigma: Tensor) -> Tensor:
        b = torch.unsqueeze(b, 1)
        sigma_new = torch.sqrt(torch.Tensor(sigma)**2 + 10**b)
        error_new = sigma_new * torch.randn_like(x) * simulator.scale    
        return x + error_new , sigma_new

    _, sigmam_new = noisybfactor(xmhg_, thetamhg_[:,-1], sigmaM)
    residualsm_new = (xmhg_ - obs_miri) / (sigmam_new * simulator.scale)

    _, sigmahhg_new = noisybfactor(xhhg_, thetahhg_[:,-2], sigmaH)   
    residualsh_new = (xhhg_ - obs_hst) / (sigmahhg_new * simulator.scale)
    
    if full:
        _, sigmahhg_new_short = noisybfactor(xhhg_short, thetahhg_[:,-2], sigmaH_short)   
        residualsh_new_short = (xhhg_short - obs_hstf[:83]) / (sigmahhg_new_short * simulator.scale)
    

    params = {
            'axes.labelsize': 12,
              'legend.fontsize': 10, 
              'xtick.labelsize': 8, 'ytick.labelsize': 8}
    plt.matplotlib.rcParams.update(params)


    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,5), gridspec_kw={'height_ratios': [3, 1]})

    creds= [0.997, 0.955, 0.683]
    alpha = (0.0, 0.9)
    levels, creds = levels_and_creds(creds= creds, alpha = alpha)
    cmap= LinearAlphaColormap('steelblue', levels=creds, alpha=alpha)
    cmap1= LinearAlphaColormap('orange', levels=creds, alpha=alpha)

    for q, l in zip(creds[:-1], levels):
        lower1, upper1 = np.quantile(xhhg_, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        lower2, upper2 = np.quantile(xmhg_.numpy(), [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax1.fill_between(obs_hst, lower1, upper1, color= cmap(l), linewidth=0) 
        ax1.fill_between(obs_wlen_miri, lower2, upper2, color= cmap(l), linewidth=0) 
        if full:
            lower1_short, upper1_short = np.quantile(xhhg_short, [0.5 - q / 2, 0.5 + q / 2], axis=0)
            ax1.fill_between(obs_hstf[:83], lower1_short, upper1_short, color= cmap1(l), linewidth=0) 

    ## the upper plot
    ## plotting observation
    ax1.plot(obs_wlen_miri, obs_miri, color='black', label = r'$ x_{obs}$', linewidth = 0.4)
    ax1.plot(obs_wlen_hstf, obs_hstf, color='black') 

    handles, texts = legends(axes= ax1, alpha=alpha) 
    texts = [r'$ x_{obs}$', r'$p_{\phi}(f(\theta)|x_{obs})$']

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel(r'Planet flux $F_\nu$ (10$^{-5}$) Jy')
    ax1.set_xlim(0, 18.2)
    ax1.legend(handles, texts, bbox_to_anchor=(1,1)) 
    ax1.set_ylim(-10, 60)

    ax_inset1 = inset_axes(ax2, 1, 0.4 , loc=2, bbox_to_anchor=(0.22, 0.28), bbox_transform=ax2.figure.transFigure) # no zoom
    ax_inset1.hlines(0, obs_wlen_hst[0], obs_wlen_hst[-1], color= 'black')

    ## the lower plot
    for q, l in zip(creds[:-1], levels):
        lower11, upper11 = np.quantile(residualsm_new.numpy(), [0.5 - q / 2, 0.5 + q / 2], axis=0)
        lower22, upper22 = np.quantile(residualsh_new, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax2.fill_between(obs_wlen_miri, lower11, upper11, color= cmap(l) , linewidth=0) 
        ax2.fill_between(obs_wlen_hst, lower22, upper22, color= cmap(l) , linewidth=0)
        ax2.hlines(0, obs_wlen_hst[0], obs_wlen_hst[-1], color= 'black')
        ax2.hlines(0, obs_wlen_miri[0], obs_wlen_miri[-1], color= 'black')
        ax_inset1.fill_between(obs_wlen_hst, lower22, upper22, color= cmap(l), linewidth=0) 
        if full:
            lower22_short, upper22_short = np.quantile(residualsh_new_short, [0.5 - q / 2, 0.5 + q / 2], axis=0)
            ax2.fill_between(obs_wlen_hstf[:83], lower22_short, upper22_short, color= cmap1(l) , linewidth=0)
            ax_inset1.fill_between(obs_wlen_hstf[:83], lower22_short, upper22_short, color= cmap1(l), linewidth=0) 

    ax2.set_ylabel(r'Residuals')
    ax2.set_xlabel( r'Wavelength ($\mu$m)') 
    ax2.set_xlim(0, 18.2)

    ax2.set_ylim(-25, 25)

    ax_inset = inset_axes(ax1, 1.8, 0.9 , loc=2, bbox_to_anchor=(0.14, 0.87), bbox_transform=ax1.figure.transFigure) # no zoom
    ax_inset.plot(obs_wlen_hstf, obs_hstf, color = 'black')

    for q, l in zip(creds[:-1], levels):
        lower1, upper1 = np.quantile(xhhg_, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax_inset.fill_between(obs_wlen_hst, lower1, upper1, color= cmap(l), linewidth=0) 
        if full:
            lower1_short, upper1_short = np.quantile(xhhg_short, [0.5 - q / 2, 0.5 + q / 2], axis=0)
            ax_inset.fill_between(obs_wlen_hstf[:83], lower1_short, upper1_short, color= cmap1(l), linewidth=0) 

    ## the upper plot
    ax_inset.set_xlim(0.85, 1.75) 
    ax_inset.set_ylim(-0.25, 3)
    #     ax_inset.set_ylim(-5, 5)

    mark_inset(ax1, ax_inset, loc1=4, loc2=3, fc="none", ec="0.7")  

    ## the lower plot
    ax_inset1.set_xlim(0.85, 1.75) 
    ax_inset1.set_ylim(-10, 20) 
    mark_inset(ax2, ax_inset1, loc1=4, loc2=3, fc="none", ec="0.7")  

    return fig


def consistency_plot(xmhg_, thetamhg_, sigmaM, xhhg_, thetahhg_, sigmaH, simulator, obs_wlen_miri, obs_miri, obs_wlen_hst, obs_hst, full= False):
    
    if full :
        _, _, _, \
                obs_wlen_hstf, obs_hstf, _, \
                _, \
                _ = load_observations(data = 'WISEJ1828')
        
    else:
        obs_wlen_hstf, obs_hstf = obs_wlen_hst, obs_hst
        
    def noisybfactor( x: Tensor, b: Tensor, sigma: Tensor) -> Tensor:
        b = torch.unsqueeze(b, 1)
        sigma_new = torch.sqrt(torch.Tensor(sigma)**2 + 10**b)
        error_new = sigma_new * torch.randn_like(x) * simulator.scale    
        return x + error_new , sigma_new

    _, sigmam_new = noisybfactor(xmhg_, thetamhg_[:,-1], sigmaM)
    residualsm_new = (xmhg_ - obs_miri) / (sigmam_new * simulator.scale)

    _, sigmahhg_new = noisybfactor(xhhg_, thetahhg_[:,-2], sigmaH)   
    residualsh_new = (xhhg_ - obs_hst) / (sigmahhg_new * simulator.scale)

    params = {
            'axes.labelsize': 12,
              'legend.fontsize': 10, 
              'xtick.labelsize': 8, 'ytick.labelsize': 8}
    plt.matplotlib.rcParams.update(params)


    fig, (ax1, ax2) = plt.subplots(2, figsize=(10,5), gridspec_kw={'height_ratios': [3, 1]})

    creds= [0.997, 0.955, 0.683]
    alpha = (0.0, 0.9)
    levels, creds = levels_and_creds(creds= creds, alpha = alpha)
    cmap= LinearAlphaColormap('steelblue', levels=creds, alpha=alpha)

    for q, l in zip(creds[:-1], levels):
        lower1, upper1 = np.quantile(xhhg_, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        lower2, upper2 = np.quantile(xmhg_.numpy(), [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax1.fill_between(obs_hst, lower1, upper1, color= cmap(l), linewidth=0) 
        ax1.fill_between(obs_wlen_miri, lower2, upper2, color= cmap(l), linewidth=0) 

    ## the upper plot
    ## plotting observation
    ax1.plot(obs_wlen_miri, obs_miri, color='black', label = r'$ x_{obs}$', linewidth = 0.4)
    ax1.plot(obs_wlen_hstf, obs_hstf, color='black') 

    handles, texts = legends(axes= ax1, alpha=alpha) 
    texts = [r'$ x_{obs}$', r'$p_{\phi}(f(\theta)|x_{obs})$']

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel(r'Planet flux $F_\nu$ (10$^{-5}$) Jy')
    ax1.set_xlim(0, 18.2)
    ax1.legend(handles, texts, bbox_to_anchor=(1,1)) 
    ax1.set_ylim(-10, 60)

    ax_inset1 = inset_axes(ax2, 1, 0.4 , loc=2, bbox_to_anchor=(0.22, 0.28), bbox_transform=ax2.figure.transFigure) # no zoom
    ax_inset1.hlines(0, obs_wlen_hst[0], obs_wlen_hst[-1], color= 'black')

    ## the lower plot
    for q, l in zip(creds[:-1], levels):
        lower11, upper11 = np.quantile(residualsm_new.numpy(), [0.5 - q / 2, 0.5 + q / 2], axis=0)
        lower22, upper22 = np.quantile(residualsh_new, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax2.fill_between(obs_wlen_miri, lower11, upper11, color= cmap(l) , linewidth=0) 
        ax2.fill_between(obs_wlen_hst, lower22, upper22, color= cmap(l) , linewidth=0)
        ax2.hlines(0, obs_wlen_hst[0], obs_wlen_hst[-1], color= 'black')
        ax2.hlines(0, obs_wlen_miri[0], obs_wlen_miri[-1], color= 'black')

        ax_inset1.fill_between(obs_wlen_hst, lower22, upper22, color= cmap(l), linewidth=0) 

    ax2.set_ylabel(r'Residuals')
    ax2.set_xlabel( r'Wavelength ($\mu$m)') 
    ax2.set_xlim(0, 18.2)

    ax2.set_ylim(-25, 25)

    ax_inset = inset_axes(ax1, 1.8, 0.9 , loc=2, bbox_to_anchor=(0.14, 0.87), bbox_transform=ax1.figure.transFigure) # no zoom
    ax_inset.plot(obs_wlen_hstf, obs_hstf, color = 'black')

    for q, l in zip(creds[:-1], levels):
        lower1, upper1 = np.quantile(xhhg_, [0.5 - q / 2, 0.5 + q / 2], axis=0)
        ax_inset.fill_between(obs_wlen_hst, lower1, upper1, color= cmap(l), linewidth=0) 

    ## the upper plot
    ax_inset.set_xlim(0.85, 1.75) 
    ax_inset.set_ylim(-0.25, 3)
    mark_inset(ax1, ax_inset, loc1=4, loc2=3, fc="none", ec="0.7")  

    ## the lower plot
    ax_inset1.set_xlim(0.85, 1.75) 
    ax_inset1.set_ylim(-10, 20) 
    mark_inset(ax2, ax_inset1, loc1=4, loc2=3, fc="none", ec="0.7")  

    return fig


def calc_temp(simulator, theta, R):
    
    x = simulator(theta)

    s_ = simulator.wavelength[simulator.wavelength>1]
    p_ =  simulator.wavelength[simulator.wavelength>0.8]
    
    ## 0.8 < w < 250
    flll = x[simulator.wavelength>0.8]*1e-5 * nc.c  /(1e7 * 1e23 * (p_*1e-4)**2.)  
    Teffff = teff_calc( p_, flll/(D/9.9)**2 , dist=7.34, r_pl= 2.3168979e-9* R)
    
    return Teffff #, bol2 #Tefff, bol1, #

def teff_calc_and_luminosity(waves, model, dist=1.0, r_pl=1.0):
    """
    This function calculates the effective temperature by integrating the model and 
    using the Stefan-Boltzmann law. It also calculates the bolometric luminosity 
    using the energy derived from the flux and the Stefan-Boltzmann law.
    
    Args:
        waves : numpy.ndarray
            Wavelength grid in units of microns.
        model : numpy.ndarray
            Flux density grid in units of W/m²/μm.
        dist : Optional[float]
            Distance to the object. Must have the same units as r_pl.
        r_pl : Optional[float]
            Object radius. Must have the same units as dist.
    
    Returns:
        teff : float
            Effective temperature in Kelvin.
        bol_lum_solar : float
            Bolometric luminosity in units of solar luminosity.
        energy : float
            Integrated energy in Watts.
    """
    import astropy.units as u
    import astropy.constants as c
    import numpy as np

    def integ(waves, model):
        """Function to integrate the flux and get energy."""
        return np.sum(model[:-1] * ((dist / r_pl) ** 2) * (u.W / u.m**2 / u.micron) * np.diff(waves) * u.micron)

    # Calculate the integrated energy
    energy = integ(waves, model)

    # Stefan-Boltzmann law: L = 4πR²σTₑₓₖ⁴
    # Calculate the effective temperature (in Kelvin)
    summed = energy / c.sigma_sb
    teff = (summed.value) ** 0.25

    # Calculate the bolometric luminosity
    # L = 4πR²σTₑₓₖ⁴
    surface_area = 4 * np.pi * (r_pl * 3.086e16 * u.m) ** 2  # Surface area in m²
    bol_lum = surface_area * c.sigma_sb * teff ** 4  # Bolometric luminosity in Watts

    # Convert the bolometric luminosity to solar units
    L_sun = 3.846e26 * u.W  # Solar luminosity in Watts
    bol_lum_solar = bol_lum / L_sun  # Bolometric luminosity in solar units

    # Return temperature, luminosity in solar units, and energy
    return teff, np.log10(bol_lum_solar.value), energy.value

def plot_spectrum(wlen, spectrum):
    ### spectrum
    plt.figure(figsize = (15,7))
    plt.plot(wlen, spectrum)
    plt.xlabel('Wavelength (microns)')
    plt.ylabel(r'Planet flux $F_\nu$ Jy')
    plt.show()
    plt.clf()

def plot_pt_profile(temperatures, pressures):    
    ## PT profile
    plt.figure(figsize = (15,7))
    plt.plot(temperatures, pressures)
    plt.yscale('log')
    plt.ylim([1e2, 1e-5])
    plt.xlabel('T (K)')
    plt.ylabel('P (bar)')
    plt.show()
    plt.clf()

def plot_cloud_particle_size(atmosphere, labelsize, fontsize, xtick_labelsize, ytick_labelsize):
    fig = plt.figure(figsize=(5, 5))
    plt.yscale('log')
    plt.xscale('log')
    plt.gca().invert_yaxis()
    plt.ylabel('Pressure (bar)' , fontsize = labelsize)  # Increase ylabel font size
    plt.xlabel('Average particle size (microns)', fontsize= labelsize)  # Increase xlabel font size

    plt.xticks(fontsize=xtick_labelsize)  # Increase x-axis tick label size
    plt.yticks(fontsize=ytick_labelsize)  # Increase y-axis tick label size

    plt.plot(atmosphere.r_g[:, atmosphere.cloud_species.index('H2O(c)')] / 1e-4, 
             atmosphere.press / 1e6, label='H2O(c)')

    plt.legend(fontsize=fontsize)  # Increase legend font size fontsize=16

    plt.show()
    plt.clf()
    return fig


def calculateAplot_spectral_weights(wlen, spectrum, atmosphere, plot= False):
    fig, ax = plt.subplots(figsize=(15,7))
    bf_contribution = atmosphere.contr_em
    bf_wlen = wlen / 1e4
    bf_spectrum = spectrum
    nu = nc.c/bf_wlen
    mean_diff_nu = -np.diff(nu)
    diff_nu = np.zeros_like(nu)
    diff_nu[:-1] = mean_diff_nu
    diff_nu[-1] = diff_nu[-2]
    spectral_weights = bf_spectrum * diff_nu / np.sum(bf_spectrum * diff_nu)
    if plot:
        plt.clf()
        plt.title('spectral weights')
        plt.plot(bf_wlen / 1e-4, spectral_weights) 
        plt.show()
    return bf_spectrum, bf_wlen, bf_contribution, spectral_weights

def calculateAplot_contribution_function(wlen, spectral_weights, atmosphere_contr_em, press, plot= False):
    pressure_weights = np.diff(np.log10(press))
    weights = np.ones_like(press)
    weights[:-1] = pressure_weights
    weights[-1] = weights[-2]
    weights = weights / np.sum(weights)
    weights = weights.reshape(len(weights), 1)
    bf_contribution = atmosphere_contr_em
    contr_em0 = bf_contribution / weights 
    if plot:
        contr_em = np.sum(contr_em0 * spectral_weights, axis=1) / np.sum(contr_em0)
        plt.clf()
        plt.title('Contribution Emission')
        plt.yscale('log')
        plt.ylim([press[-1], press[0]])
        plt.plot(contr_em, press)
        plt.grid(True, which='both', alpha=0.5)
        plt.show()

        plt.rcParams['figure.figsize'] = (15, 7)
        X, Y = np.meshgrid(wlen, press)
        plt.contourf(X,Y,contr_em0,40,cmap=plt.cm.bone_r)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim([1e2,1e-5])
        plt.xlim([np.min(wlen),np.max(wlen)])
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('P (bar)')
        plt.title('Emission contribution function')
        plt.show()
        plt.clf()
    return weights, contr_em0

def vapour_pressure_curves(simulator, pressures, temperatures, c_o, metallicity, cloud= False):
        ## equilibrium abundance of water interpolated from a precalculated grid
        ## cf
        if not cloud:
            abundances_interp = pm.interpol_abundances(c_o * np.ones_like(pressures), \
                                                    metallicity * np.ones_like(pressures), \
                                                    temperatures, \
                                                    pressures, ## has to be in bar
                                                    Pquench_carbon = None)
            XH2O = 10**abundances_interp['H2O']
        else:
            ## cl, mix, pat
            XH2O = 10**thetam[index, 23] * np.ones_like(pressures)

        MMW = 2.33
        T = np.linspace(100.,10000.,1000)
        Tc = T-273
        # Taken from Ackerman & Marley (2001) including their erratum
        
        ## water crys clouds
        P_vap = lambda x: 6111.5 * np.exp(((23.036*x) - (x**2/333.7))/(x+279.82))
        m_h2o =  2* 1 + 16
        PP, TT = (P_vap(Tc)*1e-6)/(XH2O*MMW/m_h2o), T  #*1e-6- cgs to bar 

        ## water liq clouds
        P_vap = lambda x: 6112.1 * np.exp(((18.729*x) - (x**2/227.3))/(x+257.87))
        m_h2o =  2* 1 + 16
        PPl, TTl = (P_vap(Tc)*1e-6)/(XH2O*MMW/m_h2o), T  #*1e-6
        
        return TT, PP, TTl, PPl
    
def combined_emission_contrbn_fn_with_PT(index, thetam, xlim, temperatures, atmosphere, \
                                         contr_em0, spectral_weights, press_log, \
                                         color= 'steelblue', invert = True, ax= None, fig= None,\
                                         plot_legend= True, frac= 0.6, simulated_sample = False, theta_star= None, \
                                        legend_fontsize = 12, fontsize= 14, \
                                       xtick_labelsize = 12 , ytick_labelsize = 12, \
                                        creds= [0.997, 0.955, 0.683], alpha = [0, 0.9], \
                                         labl = True,  lw= 0, feh = 0, PT_plot = None):
    if PT_plot is None:
        PT_plot = PT_plot_jwst
        
    ## only for simulations
    if simulated_sample:  
        fig_pt = PT_plot(fig, ax, thetam[:2**8], atmosphere.press/1e6, theta_star, invert = invert, color = color, \
                    legend_fontsize = legend_fontsize, fontsize= fontsize, \
                   xtick_labelsize = xtick_labelsize , ytick_labelsize = ytick_labelsize, \
                        creds= creds, alpha = alpha, labl = labl,  lw= lw) 
    else:
        ## only for real obs
        fig_pt = PT_plot(fig, ax, thetam[:2**8], atmosphere.press/1e6, invert = invert, color = color, \
                         legend_fontsize = legend_fontsize, fontsize= fontsize, \
                       xtick_labelsize = xtick_labelsize , ytick_labelsize = ytick_labelsize, \
                        creds= creds, alpha = alpha, labl = labl,  lw= lw) 

    # Condensation curves
    temp_water_sol = 10**4 / (38.84 - 3.93 * feh - 3.83 * np.log10(press_log) - 0.2 * feh * np.log10(press_log))
    temp_ammonia_sol = 10**4 / (68.02 - 6.19 * feh - 6.31 * np.log10(press_log))
    temp_na2s_sol = 10**4 / (10.045 - 0.72 * np.log10(press_log) - 1.08 * feh)
    temp_kcl_sol = 10**4 / (12.479 - 0.879 * np.log10(press_log) - 0.879 * feh)
    temp_fe_sol = 10**4 / (5.44 - 0.48 * np.log10(press_log) - 0.48 * feh)
    temp_mgsio3_sol = 10**4 / (6.26 - 0.35 * np.log10(press_log) - 0.70 * feh)
    temp_mns_sol = 10**4 / (7.45 - 0.42 * np.log10(press_log) - 0.84 * feh)


    ax.plot(temp_water_sol, press_log, lw=2, color='blue', label='H$_2$O condensation (solar)')
    ax.plot(temp_ammonia_sol, press_log, lw=2, color='purple', label='NH$_3$ condensation (solar)')
    ax.plot(temp_na2s_sol, press_log, lw=2, color='green', label='Na$_2$S condensation (solar)')
    ax.plot(temp_kcl_sol, press_log, lw=2, color='red', label='KCl condensation (solar)')   
    #     ax.plot(temp_fe_sol, press_log, lw=2, color='orange', label='Fe condensation (solar)')
    #     ax.plot(temp_mgsio3_sol, press_log, lw=2, color='pink', label='MgSiO$_3$ condensation (solar)') 
    #     ax.plot(temp_mns_sol, press_log, lw=2, color='grey', label='MnS condensation (solar)') 
    ax.set_ylim(1e2, 1e-5)
    
    contr_em = np.sum(contr_em0 * spectral_weights, axis=1) / np.sum(contr_em0)
    contr_em_weigh = contr_em / np.max(contr_em)
    contr_em_weigh_intp = interp1d(atmosphere.press/1e6, contr_em_weigh)

    tlims = (np.min(temperatures)*0.97,np.max(temperatures)*1.03)
    yborders = atmosphere.press/1e6
    for i_p in range(len(yborders) - 1):
        mean_press = (yborders[i_p + 1] + yborders[i_p]) / 2.  
        
        ax.fill_between(tlims,
                        yborders[i_p + 1],
                        yborders[i_p],
                        color='white',
                        alpha=min(1. - contr_em_weigh_intp(mean_press), frac),
                        linewidth=0,
                        rasterized=True,
                        zorder=4)

    ax.plot(contr_em_weigh * (
            tlims[1] - tlims[0])
            + tlims[0],
            atmosphere.press/1e6, '--',
            color= color,
            linewidth=1,
    #             label='Spectrally weighted contribution',
            zorder=5)

    ax.set_yscale('log')
    ax.set_ylim([(atmosphere.press[-1]/1e6) *1.03, (atmosphere.press[0]/1e6)/1.03])
    # Labelling and output
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel('Pressure [bar]')
    # ax.set_ylim(1e2, 1e-3)
    plt.xlim([0, xlim])
    if plot_legend:
        ax.legend(loc='upper right')
        
    return contr_em, contr_em_weigh, fig

def plotting_contribution(thetam, xlim, index, simulator, frac=0.6, simulated_sample=False, theta_star= False, cloud=False, \
                         legend_fontsize = 12, fontsize= 14, \
                                       xtick_labelsize = 12 , ytick_labelsize = 12, \
                                        creds= [0.997, 0.955, 0.683], alpha = [0, 0.9], \
                                         labl = True,  lw= 0, feh = 0, PT_plot= None, temperatures=None): 
    if temperatures is None:
        pressures =  pressureBAR(AMR=True)  # len(1000) between -6 to 1000  in bar - np.logspace(-6, 3, 100/1000) is bar
        press_log = np.logspace(-6, 3, 100)  # Pressure in logarithmic scale
        #     logP = np.log10(atmosphere.press/1e6)
        if simulated_sample:
            wlen,spectrum, atmosphere = simulator(theta_star) 
            temperatures = pt_profile_theta(theta_star, pressures)
        else:
            wlen,spectrum, atmosphere = simulator(thetam[index]) 
            temperatures = pt_profile_theta(thetam[index], pressures)
        ## spectral weights 
        bf_spectrum, bf_wlen, bf_contribution, spectral_weights = calculateAplot_spectral_weights(wlen, spectrum, atmosphere, plot=True)
        weights, contr_em0 = calculateAplot_contribution_function(wlen, spectral_weights, atmosphere.contr_em, press = atmosphere.press/1e6, plot= True) 

    
    else:
        wlen, spectrum, atmosphere = simulator(thetam[index])
        spectrum = spectrum[0]
        press_log = np.logspace(-6, 2, 80)
        bf_spectrum, bf_wlen, bf_contribution, spectral_weights = calculateAplot_spectral_weights(wlen, spectrum, atmosphere, plot=True)
        d = Data()
        wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9
       
        #### reshaping contrb em 
        atmosphere_contr_em = np.array([np.interp(d.model_wavelengths, wl, row) for row in atmosphere.contr_em])        
        param_dict = param_set_ext.param_dict(thetam[index,-4:])
        shifted_wavelengths = (1+param_dict['rv']/const.c.to(u.km/u.s).value) * d.model_wavelengths
        atmosphere_contr_em_interp = np.array([np.interp(d.data_wavelengths, shifted_wavelengths, row) for row in atmosphere_contr_em])
    
        #flux at data_wavelengths
        
        weights, contr_em0 = calculateAplot_contribution_function(wlen, spectral_weights, atmosphere_contr_em_interp, press = atmosphere.press/1e6, plot= True) 
        
        thetam  = torch.from_numpy(thetam)

           
    if cloud:
        plot_cloud_particle_size(atmosphere)
        
    
    fig, ax = plt.subplots(figsize=(15,7))
    contr_em, contr_em_weigh, fig = combined_emission_contrbn_fn_with_PT(index, thetam[:512], xlim, temperatures, atmosphere,\
                                              contr_em0 , spectral_weights, press_log, ax=ax, fig=fig,\
                                               frac=frac,simulated_sample=simulated_sample, theta_star = theta_star, \
        #                                                 legend_fontsize = legend_fontsize, fontsize= fontsize, \
        #                    xtick_labelsize = xtick_labelsize , ytick_labelsize = ytick_labelsize, \
        #                         creds= creds, alpha = alpha, labl = labl,  lw= lw, \
                                                                         feh = feh, PT_plot= PT_plot)


    return wlen, atmosphere, temperatures, bf_contribution, bf_wlen, bf_spectrum, spectral_weights, weights, \
            contr_em0, contr_em, contr_em_weigh, fig


def pt_plotting_wcondcurv( colors, legnds, invert, plot_legends,  temperatures, index, atmosphere, contr_em0, \
                        spectral_weights, theta, xlim, frac = 0.8, simulated_sample= False, theta_star = False,\
                          legend_fontsize = 12, fontsize= 14, \
                                       xtick_labelsize = 12 , ytick_labelsize = 12, \
                          creds= [[0.997, 0.955, 0.683]], alpha = [[0, 0.9]], labl = [True],  lw= [0] , PT_plot= None, \
                          press_log = None):
    
    if press_log is None:
        press_log = np.logspace(-6, 3, 100)  # Pressure in logarithmic scale

    params = {
            'axes.labelsize': 14,
    #           'axes.titlesize':20, 
    #           'font.size': 14, 
    #           'legend.fontsize': 10, 
    #           'xtick.labelsize': 10, 'ytick.labelsize': 10}
    }
    plt.matplotlib.rcParams.update(params)

    

    fig, ax = plt.subplots(figsize=(5,5))
    print(np.shape(atmosphere))
    


    for i, th in enumerate(theta):

        mask = torch.any(torch.isinf(th), dim=-1)
        print('No of inf in theta (post) is ', mask.sum())
        th = th[~mask]
        
        
        _, _, fig = combined_emission_contrbn_fn_with_PT(index[0], th, xlim[i], temperatures[i], \
                                         atmosphere[i], contr_em0[i], spectral_weights[i] , press_log, \
                                           color= colors[i], ax = ax, fig= fig, plot_legend= plot_legends[i]\
                                        , frac= frac, simulated_sample= simulated_sample, theta_star = theta_star, \
                                        legend_fontsize = legend_fontsize, fontsize= fontsize, \
                   xtick_labelsize = xtick_labelsize , ytick_labelsize = ytick_labelsize, \
                                        creds= creds[i], alpha = alpha[i], labl = labl[i],  lw= lw[i], PT_plot= PT_plot)
    
    return fig, ax