import numpy as np
from pathlib import Path
import pandas as pd
import os
from petitRADTRANS import nat_cst as nc
import torch 
from torch import Tensor

'''
FutureWarning: The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version. Use ``sep='\s+'`` instead
  obs_hst = pd.read_csv(obsHST_file + 'WISE1828.fl.txt', delim_whitespace= True, header=1)
/home/mvasist/WISEJ1738/sbi_ear/code2explore/observations.py:132: FutureWarning: The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version. Use ``sep='\s+'`` instead
  obs_hst_pd = pd.read_csv(obsHST_file, delim_whitespace= True, header=1)
'''

home = os.environ.get('HOME')

def read_csv_file(file_path, **kwargs):
    '''
    Read CSV file as pandas DataFrame.
    '''
    obs_pd = pd.read_csv(file_path, **kwargs,  header=0, delimiter = ',')
    obs_wlen = np.array(obs_pd.iloc[:,0])
    obs_fl = np.array(obs_pd.iloc[:, 1])
    sigma = np.array(obs_pd.iloc[:,2])
    return obs_wlen, obs_fl, sigma

def scale_fluxAsigma2D(obs, sigma, D, D_sim, scale_factor):
    '''
    Scale flux and sigma based on distance.
    scaling the observations as if they were seen from D_sim pc to use simulations that were simulated for D_sim pc 
    '''
    obs_scaled = obs * scale_factor * (D / D_sim) ** 2
    sigma_scaled = sigma * (D / D_sim) ** 2
    return obs_scaled, sigma_scaled

class load_observations():
    '''
    ['WISE0855', 'WISE0458', 'ROSS458C', 'WISEJ1828', 'WISEJ1738']
    [2.28, 14, 11.5, 9.9, 7.34]

    #apply whatever source, instruments to return as a dictionary as 
    
    when source "WISEJ1738" and instrument is ["miri", "hst"] and tag is [None] 
    observation= {
                    "WISEJ1738": {
                                "miri" : {
                                        "wlen" : obs_wlen_miri,
                                        "flux" : obs_miri,
                                        "err" : sigmaM,
                                        },

                                "hst" : {
                                        "wlen" : obs_wlen_hst,
                                        "flux" : obs_hst,
                                        "err" : sigmaH,
                                        },
                                }
            }
    OR 
    when source "WISEJ1738" and instrument is ["miri"] and tag is ["Helena_rebinning_all_ch_together"]
    observation= {
                    "WISEJ1738": {
                                "Helena_rebinning_all_ch_together" : {
                                                                    "miri" : {
                                                                            "wlen" : obs_wlen_miri,
                                                                            "flux" : obs_miri,
                                                                            "err" : sigmaM,
                                                                    },
                                                            }
                                    }
                                }
    '''
    def __init__(self, source, instruments, tag, D, D_sim, scale_factor):
        self.source = source #"WISEJ1738"or "WISEJ1828", .....
        self.instruments = instruments #["miri", "hst", "gemini"]
        self.tag = tag #"None" or "Helena_rebinning_all_ch_together" or "mostprobCLavg_simulation3"
        self.D = D
        self.D_sim = D_sim
        self.scale_factor = scale_factor

    def _load_simulation_data(self, sim_file, base_path):
        '''
        [mostprobCF_simulation0_noisefree, "mostprobCF_simulation1", "mostprobCF_simulation2", "mostprobCF_simulation3",
        mostprobCLavg_simulation1, mostprobCLavg_simulation2, mostprobCLavg_simulation3, mostprobCLavg_simulation0_noisefree]

        file_map = {
#             'simulation0': 'mostprobCLavg_simulation0_noisefree.csv',
#             'simulation1': 'mostprobCLavg_simulation1.csv',
#             'simulation2': 'mostprobCLavg_simulation2.csv',
#             'simulation3': 'mostprobCLavg_simulation3.csv',
#             'simulation0_cf': 'mostprobCF_simulation0_noisefree.csv',
#             'simulation1_cf': 'mostprobCF_simulation1.csv',
#             'simulation2_cf': 'mostprobCF_simulation2.csv',
#             'simulation3_cf': 'mostprobCF_simulation3.csv'
#         }
        '''
        
        x_star_pd = pd.read_csv(sim_file)
        x_star = x_star_pd.iloc[0]

        # Common paths
        obsHST_file = base_path / "HST" /  "spectrum.csv"
        obs_MIRI_file = base_path / "MIRI" / "spectrum.csv"
        obs_Gemini_file = base_path / "Gemini" / "spectrum.csv"

        # Load base wavelength and sigma
        obs_hst_pd = pd.read_csv(obsHST_file)
        obs_wlen_hst = np.array(obs_hst_pd.iloc[:, 0])
        sigmaH = np.array(obs_hst_pd.iloc[:, 2])

        obs_miri_pd = pd.read_csv(obs_MIRI_file)
        obs_wlen_miri = np.array(obs_miri_pd.iloc[:, 0])
        sigmaM = np.array(obs_miri_pd.iloc[:, 2])

        obs_gemini_pd = pd.read_csv(obs_Gemini_file)
        obs_wlen_gemini = np.array(obs_gemini_pd.iloc[:, 0])
        sigmaG = np.array(obs_gemini_pd.iloc[:, 2])

        # Split simulation data
        obs_wlen_inst = np.append(obs_wlen_hst, obs_wlen_gemini)
        index_argsort = np.argsort(obs_wlen_inst)
        index_getback = np.argsort(index_argsort)

        obs_hst = np.array(x_star.iloc[:-1298].iloc[index_getback].iloc[:129])
        obs_gemini = np.array(x_star.iloc[:-1298].iloc[index_getback].iloc[129:])
        obs_miri = np.array(x_star.iloc[-1298:])

        # Scale sigmas for actual distance
        sigmaH *= (self.D/self.D_sim)**2
        sigmaG *= (self.D/self.D_sim)**2
        sigmaM *= (self.D/self.D_sim)**2

        return {
            "MIRI": {"wlen": obs_wlen_miri, "flux": obs_miri, "err": sigmaM},
            "Gemini": {"wlen": obs_wlen_gemini, "flux": obs_gemini, "err": sigmaG},
            "HST": {"wlen": obs_wlen_hst, "flux": obs_hst, "err": sigmaH},
            }
        
    def forward(self):
        base_path = Path(home) / "SBI4exoplanets" / "observations" / self.source

        observation = {self.source: {}}

        # If we have a tag (rebinned data), create nested dictionary
        if self.tag and self.tag != "None":
            observation[self.source][self.tag] = {}
            target_dict = observation[self.source][self.tag]
        else:
            target_dict = observation[self.source]

        ##########################################################################################
        ## Handle simulated observations
        ##########################################################################################
        if "simulation" in self.tag:
            sim_file = base_path / "simulations" / (self.tag+".csv") 
            sim_data = self._load_simulation_data(sim_file, base_path)
            target_dict.update(sim_data)
            return observation

        ##########################################################################################
        ## Handle real observations
        ##########################################################################################
        else: 
            for inst in self.instruments:

                # Determine file path based on tag
                if self.tag and self.tag != "None":
                    file_path = base_path / inst / self.tag / "spectrum.csv"
                else:
                    file_path = base_path / inst / "spectrum.csv"

                # Read the CSV file
                obs_wlen, obs_flux, obs_sigma = read_csv_file(file_path)

                # Scale to reference distance (9.9 pc)
                obs_flux, obs_sigma = scale_fluxAsigma2D(obs_flux, obs_sigma, self.D, self.D_sim, self.scale_factor)

                # Store in dictionary
                target_dict[inst] = {
                    "wlen": obs_wlen,
                    "flux": obs_flux,
                    "err": obs_sigma
                }

            return observation
    
