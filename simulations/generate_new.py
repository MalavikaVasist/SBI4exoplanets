#!/usr/bin/env python
import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import Tensor, Size
from torch.distributions import Distribution

from dawgz import job, after, ensure, schedule
from itertools import starmap
from pathlib import Path
from typing import *
from tqdm import tqdm


from lampe.data import H5Dataset, IterableDataset, JointLoader
from lampe.distributions import BoxUniform

import os
scratch = os.environ.get('SCRATCH') 
home = os.environ.get('HOME')

import sys
sys.path.insert(0, str(Path(home) / 'WISEJ1738/sbi_ear'))# from ees_MIRIinst import Simulator, emission_spectrum
from ees_MIRIinsts import Simulator as Simulator_cloudfree, LOWER as LOWER_cloudfree, UPPER as UPPER_cloudfree
from eesmetalclouds_MIRIinsts import Simulator as Simulator_cloudy, LOWER as LOWER_cloudy, UPPER as UPPER_cloudy

from petitRADTRANS import nat_cst as nc, Radtrans
from petitRADTRANS.retrieval.data import Data
from petitRADTRANS.retrieval.rebin_give_width import rebin_give_width


import astropy.units as u
from astropy import constants as c


path_miri_cloudy= Path(scratch) / 'JWST/simulations/sim_15NH3HCNTiOVOXH2OcLPbasefsedkzzslnbfacM_unregPT/cloudy/MIRI/Na2SKCl/'
path_miri_cloudy.mkdir(parents=True, exist_ok=True)
path_hst_1_cloudy= Path(scratch) / 'JWST/simulations/sim_15NH3HCNTiOVOXH2OcLPbasefsedkzzslnbfacM_unregPT/cloudy/HST/Na2SKCl/simulation/1/'
path_hst_1_cloudy.mkdir(parents=True, exist_ok=True)
path_hst_cloudy= Path(scratch) / 'JWST/simulations/sim_15NH3HCNTiOVOXH2OcLPbasefsedkzzslnbfacM_unregPT/cloudy/HST/Na2SKCl/simulation/'
path_hst_cloudy.mkdir(parents=True, exist_ok=True)
path_hst_convrebin_cloudy= Path(scratch) / 'JWST/simulations/sim_15NH3HCNTiOVOXH2OcLPbasefsedkzzslnbfacM_unregPT/cloudy/HST/Na2SKCl/conv_rebinned/'
path_hst_convrebin_cloudy.mkdir(parents=True, exist_ok=True)

path_miri_cloudfree= Path(scratch) / 'JWST/simulations/sim_15NH3HCNTiOVOXH2OcLPbasefsedkzzslnbfacM_unregPT/cloudfree/MIRI/'
path_hst_1_cloudfree= Path(scratch) / 'JWST/simulations/sim_15NH3HCNTiOVOXH2OcLPbasefsedkzzslnbfacM_unregPT/cloudfree/HST/simulation/1/'
path_hst_cloudfree= Path(scratch) / 'JWST/simulations/sim_15NH3HCNTiOVOXH2OcLPbasefsedkzzslnbfacM_unregPT/cloudfree/HST/simulation/'
path_hst_convrebin_cloudfree= Path(scratch) / 'JWST/simulations/sim_15NH3HCNTiOVOXH2OcLPbasefsedkzzslnbfacM_unregPT/cloudfree/HST/conv_rebinned/'

i = int(sys.argv[1]) 
print(i, 'here')
# i = 200
# @job(array=300, cpus=1, ram='16GB', time='2-00:00:00') 
def simulate(i:int):

    ### Simulating from scratch 
    # prior = BoxUniform(torch.tensor(LOWER_cloudy), torch.tensor(UPPER_cloudy))
    simulator_miri_cloudfree = Simulator_cloudfree(a=4.9, b= 19, noisy=False)
    simulator_hst_cloudfree = Simulator_cloudfree(a=0.98, b= 2.2, noisy=False)

    # loader = JointLoader(prior, simulator_miri_cloudy, batch_size=16, numpy=True)

    # # def filter_M50(theta, x):
    # #     M = (10**theta[:,1]*u.cm*u.s**-2 * theta[:,0]**2*u.R_jup.cgs**2) / (c.G * u.M_jup).cgs
    # #     mask = M>50
    # #     return theta[~mask], x[~mask]
    
    def filter_nanandinf(theta, x):
        # theta, x = filter_M50(theta,x)
        mask = ~torch.any(torch.isnan(x), dim=-1)
        mask1 = ~torch.any(torch.isinf(x[mask]), dim=-1)
        return theta[mask][mask1], x[mask][mask1]   
    
    # H5Dataset.store(
    #     starmap(filter_nanandinf, loader),
    #     # loader,
    #     path_miri_cloudy / f'samples_{i:06d}.h5',
    #     size=4096*3,  
    # )

#-----------------------------------------------------------------------------------------
## cloudless miri models with the same theta as above

    LABELSclouds, LOWERclouds, UPPERclouds = zip(*[
            [                  r'$log\_X\_cb\_Na_2S(c)$',  -10,   0],  # old   
            [                  r'$log\_X\_cb\_KCl(c)$',  -10,   0],  # constrained clouds    
            [                  r'$log\_P_{\rm base, Na_2S}$',  -6,   3], # NH3_mol_scale
            [                  r'$log\_P{\rm base, KCl}$',  -6,   3], # NH3_mol_scale
            [                  r'$f_{\rm sed,Na_2S}$',  0,   10], # NH3_mol_scale
            [                  r'$f_{\rm sed,KCl}$',  0,   10], # NH3_mol_scale
            [                  r'$log\_k_{\rm zz}$',  5,   13], # NH3_mol_scale
            [                  r'$sgln$',  1.05,   3], # sigma\_lnorm   
            ])
    

    priorc = BoxUniform(torch.tensor(LOWERclouds), torch.tensor(UPPERclouds))
    simulator_miri_cloudy = Simulator_cloudy(a=4.9, b= 19, noisy=False)
    
    loader_cloudfree = H5Dataset(path_miri_cloudfree / f'samples_{i:06d}.h5') 

    def simulator_cloudy(theta,x):
        theta_cloudfree = theta.detach().cpu() #.numpy().astype(np.double)
        batch_size = len(theta_cloudfree)
        theta_cl= priorc.sample((1,))
        # print(np.shape(theta_cloudfree), np.shape(theta_cl))
        theta_cloudy = torch.hstack((theta_cloudfree[:23], theta_cl.squeeze(), theta_cloudfree[-2:]))
        x_miri_cloudy = simulator_miri_cloudy(theta_cloudy.numpy().astype(np.double))
        x_miri_cloudy = torch.from_numpy(x_miri_cloudy).to(theta_cloudy)
        theta_cloudy, x_miri_cloudy = filter_nanandinf(theta_cloudy, x_miri_cloudy)
        return theta_cloudy, x_miri_cloudy

    # H5Dataset.store(
    #         starmap(simulator_cloudy, loader_cloudfree),
    #         path_miri_cloudy / f'samples_{i:06d}.h5',
    #         size=4096*3,
    #     )

#-----------------------------------------------------------------------------------------

    loader0 = H5Dataset(path_miri_cloudy / f'samples_{i:06d}.h5') 
    simulator_hst_cloudy = Simulator_cloudy(a=0.98, b= 2.2, noisy=False)


    def simulator_h_cloudy(theta,x):
        x_hst = simulator_hst_cloudy(theta.detach().cpu().numpy().astype(np.double))
        x_hst = torch.from_numpy(x_hst).to(theta)
        theta, x_hst = filter_nanandinf(theta, x_hst)
        return theta, x_hst

    # H5Dataset.store(
    #         starmap(simulator_h_cloudy, loader0),
    #         path_hst_1_cloudy / f'samples_{i:06d}.h5',
    #         size=4096*3,
    #     )

#-----------------------------------------------------------------------------------------

    # ## HST removing mike line b parameter and adding csf and cushing b

    LABELS1, LOWER1, UPPER1 = zip(*[
[                    r'$Cushing_scale_factor$',   0.5, 1.5],  # Cushing_scale_factor
[                    r'$Mike\_Line\_b\_Cushing$', -17, -11], #Mike_Line_b_Cushing 
])
    
    prior1 = BoxUniform(torch.tensor(LOWER1), torch.tensor(UPPER1))
    loader1 = H5Dataset(path_hst_1_cloudy / f'samples_{i:06d}.h5', batch_size=16)

    def simulator_new(theta,x):
        # print(theta.size(), 'before')
        theta = theta[:, :-1]
        batch_size = len(theta)
        theta_b = prior1.sample((batch_size,))
        theta = torch.hstack((theta, theta_b))
        # print(theta.size(), 'after')
        return theta, x

    # H5Dataset.store(
    #         starmap(simulator_new, loader1),
    #         path_hst_cloudy / f'samples_{i:06d}.h5',
    #         size=4096*3,
    #     )
    
    # ## HST removing mike line b parameter and adding csf and cushing b same

    # loader1_cloudfree = H5Dataset(path_hst_cloudy / f'samples_{i:06d}.h5') #, batch_size=16)

    # def simulator_new_cloudfree(theta,x):
    #     theta_cloudfree = theta.detach().cpu() #.numpy().astype(np.double)
    #     theta_cloudfree = torch.hstack((theta_cloudfree[:23], theta_cloudfree[-3:]))
    #     x_hst_cloudfree = simulator_hst_cloudfree(theta_cloudfree.numpy().astype(np.double))
    #     x_hst_cloudfree = torch.from_numpy(x_hst_cloudfree).to(theta_cloudfree)
    #     theta_cloudfree, x_hst_cloudfree = filter_nanandinf(theta_cloudfree, x_hst_cloudfree)
    #     return theta_cloudfree, x_hst_cloudfree

    # H5Dataset.store(
    #         starmap(simulator_new_cloudfree, loader1_cloudfree),
    #         path_hst_cloudfree / f'samples_{i:06d}.h5',
    #         size=4096*3,
    #     )

# ##-----------------------------------------------------
    ## Rebinning HST 
    obsHST_file = Path(home) / 'WISEJ1738/observation/HST/WISEJ1738_HST.txt'
    obs_hst_pd = pd.read_csv(obsHST_file, header=0, delimiter = ',')
    obs_wlen_hst = np.array(obs_hst_pd.iloc[:,0])
    obs_hst_fnu = np.array(obs_hst_pd.iloc[:, 1]*1e30 * (obs_wlen_hst*1e-4)**2/nc.c)
    obs_error_hst_fnu = np.array(obs_hst_pd.iloc[:,2]*1e30 * (obs_wlen_hst*1e-4)**2/nc.c)

    loader2 = H5Dataset(path_hst_cloudy / f'samples_{i:06d}.h5', batch_size = 16)
        
    def wlenbins(w1):
        wlen_bins = np.zeros_like(w1)
        wlen_bins[:-1] = np.diff(w1)
        wlen_bins[-1] = wlen_bins[-2]
        return wlen_bins
    
    def rebinit(thetah, xh):
        wlen_hstsim = simulator_hst_cloudy.wavelength
        xh = xh[:,115:552]
        wlen_hstsim = wlen_hstsim[115:552]
        wlen_bins = wlenbins(obs_wlen_hst)
        xx = np.stack([Data.convolve(wlen_hstsim, x, 130) for x in xh])
        flux_rebinned = torch.stack([torch.from_numpy(rebin_give_width(wlen_hstsim, x, obs_wlen_hst, wlen_bins)) for x in xx])
    #     print(thetah.size(), xh.size(), flux_rebinned.size(), '3')
        return thetah, flux_rebinned
    
    H5Dataset.store(
            starmap(rebinit, loader2),
            path_hst_convrebin_cloudy / f'samples_{i:06d}.h5',
            size=len(loader2),
        )
    
    ## rebinned cloudless HST models with the same theta as above
    # loader2_cloudfree = H5Dataset(path_hst_cloudfree / f'samples_{i:06d}.h5', batch_size = 16)
    # H5Dataset.store(
    #         starmap(rebinit, loader2_cloudfree),
    #         path_hst_convrebin_cloudfree / f'samples_{i:06d}.h5',
    #         size=len(loader2_cloudfree),
    #     )

# def aggregate(path):
#     files = list(path.glob('samples_*.h5'))
#     length = len(files)
#     files.sort()

#     i = int(0.9 * length) 
#     j = int(0.99 * length) 
#     splits = {
#         'train': files[:i],
#         'valid': files[i:j],
#         'test': files[j:],
#     }
    
#     #     def filter_largeAsmall(theta, x):
#     #         mask = (x.mean(dim=-1) < 30) & (x.mean(dim=-1) >10)
#     #         mask1 = x[mask].var(dim=-1) <1000
#     # #         return theta[mask], x[mask]
#     #         return theta[mask][mask1], x[mask][mask1]

#     for name, files in splits.items():
#         dataset = H5Dataset(*files, batch_size=4096*3) 

#         H5Dataset.store(
#             dataset,
#             # starmap(filter_largeAsmall, dataset),
#             path / f'{name}.h5',
#             size=len(dataset),
#         )


from contextlib import suppress

def aggregate_two_loaders(path_cf: Path, path_cl: Path, index_limit_cf=-2, index_limit_cl = -10):
    files_cf = sorted(path_cf.glob('samples_*.h5'))
    files_cl = sorted(path_cl.glob('samples_*.h5'))
    length = min(len(files_cf), len(files_cl))
    i, j = int(0.9 * length), int(0.99 * length)

    splits = {
        'train': (files_cf[:i], files_cl[:i]),
        'valid': (files_cf[i:j], files_cl[i:j]),
        'test':  (files_cf[j:], files_cl[j:]),
    }

    def match_pairs(thcf, thcl):
        thcf_trim = thcf[:, :index_limit_cf]
        thcl_trim = thcl[:, :index_limit_cl]
        mask = (thcf_trim.unsqueeze(1) == thcl_trim.unsqueeze(0)).all(dim=2)
        idx_cf, idx_cl = torch.nonzero(mask, as_tuple=True)
        return idx_cf, idx_cl

    for name, (f_cf, f_cl) in splits.items():
        print(f"Processing split: {name} — {len(f_cf)} files each")
        ds_cf = H5Dataset(*f_cf, batch_size=4096 * 3)
        ds_cl = H5Dataset(*f_cl, batch_size=4096 * 3)

        out_cf = path_cf / f"{name}_metalcloudmatches.h5"
        out_cl = path_cl / f"{name}_metalcloudmatches.h5"

        for p in (out_cf, out_cl):
            with suppress(FileNotFoundError):
                p.unlink()

        # Function to yield matching pairs of (theta, x)
        def generate_matches():
            for (thcf, xcf), (thcl, xcl) in tqdm(zip(ds_cf, ds_cl)):
                idx_cf, idx_cl = match_pairs(thcf, thcl)
                if len(idx_cf) > 0:
                    # print(thcf[idx_cf].size(), idx_cf)
                    yield (thcf[idx_cf], xcf[idx_cf]), (thcl[idx_cl], xcl[idx_cl])

        # Flatten generator to individual samples
        def file_cf():
            for (thcf_match, xcf_match), _ in generate_matches():
                # print(thcf_match.size(), ' here')
                yield thcf_match, xcf_match
                
#                 for theta, x in zip(thcf_match, xcf_match):
#                     yield theta, x

        def file_cl():
            for _, (thcl_match, xcl_match) in generate_matches():
                # print(thcl_match.size(), ' here too')
                yield thcl_match, xcl_match
                
#                 for theta, x in zip(thcl_match, xcl_match):
#                     print(theta.size(), ' here too')
#                     yield theta, x

        # Count total number of matches for size argument
        total_matches = sum(len(thcf_match) for (thcf_match, _), _ in generate_matches())
        print(f"Total matches in split '{name}': {total_matches}")

        if total_matches == 0:
            print(f"No matches found for split {name}, skipping saving.")
            continue

        # Store using H5Dataset.store()
        H5Dataset.store(file_cf(), out_cf, size=total_matches)
        H5Dataset.store(file_cl(), out_cl, size=total_matches)

        print(f"Finished saving {total_matches} matches for split: {name}")

        break


def truncate_empty_rows(filepath, key="theta"):
    """
    Truncates 'theta' and 'x' datasets in an HDF5 file by removing trailing rows that are exactly zeros.

    Args:
        filepath (str or Path): Path to the HDF5 file.
        key (str): Dataset to use for checking (default is 'theta').
    """
    with h5py.File(filepath, "r+") as f:
        dset = f[key]
        data = dset[()]  # load entire dataset

        if data.ndim == 1:
            nonzero_rows = np.flatnonzero(data != 0)
        else:
            nonzero_rows = np.flatnonzero(np.any(data != 0, axis=tuple(range(1, data.ndim))))
        
        if len(nonzero_rows) == 0:
            print(f"No non-empty rows found in {filepath}. Skipping.")
            return
        
        new_size = nonzero_rows[-1] + 1

        for k in ["theta", "x"]:
            f[k].resize((new_size,) + f[k].shape[1:])
        
        f.attrs["size"] = new_size
        print(f"Truncated '{filepath}' to {new_size} rows based on exact trailing zeros.")

def match_files(miri_file: Path, gemini_file: Path, index_limit_miri=-2, index_limit_gem=-3):
    print(f"Matching:\n  MIRI:   {miri_file}\n  Gemini: {gemini_file}")

    # Create dataset loaders
    ds_miri = H5Dataset(miri_file, batch_size=4096 * 3)
    ds_gemini = H5Dataset(gemini_file, batch_size=4096 * 3)

    # Matching function
    def match_pairs(th_miri, th_gemini):
        th_miri_trim = th_miri[:, :index_limit_miri]
        th_gemini_trim = th_gemini[:, :index_limit_gem]
        mask = (th_miri_trim.unsqueeze(1) == th_gemini_trim.unsqueeze(0)).all(dim=2)
        idx_miri, idx_gemini = torch.nonzero(mask, as_tuple=True)
        return idx_miri, idx_gemini

    # Generate matched pairs
    def generate_matches():
        for (th_miri, x_miri), (th_gemini, x_gemini) in tqdm(zip(ds_miri, ds_gemini)):
            idx_miri, idx_gemini = match_pairs(th_miri, th_gemini)
            if len(idx_miri) > 0:
                yield (th_miri[idx_miri], x_miri[idx_miri]), (th_gemini[idx_gemini], x_gemini[idx_gemini])

    matched_pairs = list(generate_matches())
    total_matches = sum(len(th_miri) for (th_miri, _), _ in matched_pairs)
    print(f"Total matches found: {total_matches}")

    if total_matches == 0:
        print("No matches found. Skipping overwrite.")
        return

    def file_miri():
        for (th_miri_match, x_miri_match), _ in matched_pairs:
            yield th_miri_match, x_miri_match

    # Construct a *deterministic* output filename
    tmp_path = miri_file.with_name(miri_file.stem + "_temp.h5")

    # Remove tmp file if it exists already to avoid FileExistsError
    if tmp_path.exists():
        tmp_path.unlink()

    # Write to the new temp file with overwrite=True to be sure
    H5Dataset.store(file_miri(), tmp_path, size=total_matches, overwrite=True)
    print(f"Wrote matches to: {tmp_path}")

    # Safely replace the original file
    with suppress(FileNotFoundError):
        miri_file.unlink()

    tmp_path.rename(miri_file)
    print(f"Replaced original file with matches: {miri_file}")

if __name__ == '__main__':
    # simulate(i)


    # # filterbigM(i)
    # aggregate_two_loaders(path_miri_cloudfree, path_miri_cloudy, index_limit_cf=-2, index_limit_cl=-10)
    # aggregate_two_loaders(path_hst_convrebin_cloudfree, path_hst_convrebin_cloudy, index_limit_cf=-3, index_limit_cl=-11)
    aggregate_two_loaders(path_hst_cloudfree, path_hst_cloudy, index_limit_cf=-3, index_limit_cl=-11)


    ## Cloudfree
    # match_files(path_miri_cloudfree / 'train_metalcloudmatches.h5', \
    #             path_hst_cloudfree / 'train_metalcloudmatches.h5', index_limit_miri= -2, index_limit_gem= -3)
    # match_files(path_hst_cloudfree / 'train_metalcloudmatches.h5', \
    #             path_miri_cloudfree / 'train_metalcloudmatches.h5', index_limit_miri= -3, index_limit_gem= -2)
    # match_files(path_miri_cloudfree / 'valid_metalcloudmatches.h5', \
    #             path_hst_cloudfree / 'valid_metalcloudmatches.h5', index_limit_miri= -2, index_limit_gem= -3)
    # match_files(path_miri_cloudfree / 'test_metalcloudmatches.h5', \
    #             path_hst_cloudfree / 'test_metalcloudmatches.h5', index_limit_miri= -2, index_limit_gem= -3)

    # ## Cloudy
    # match_files(path_miri_cloudy / 'train_metalcloudmatches.h5', \
    #             path_hst_cloudy / 'train_metalcloudmatches.h5', index_limit_miri= -10, index_limit_gem= -11)
    # match_files(path_hst_cloudfree / 'train_metalcloudmatches.h5', \
    #             path_miri_cloudfree / 'train_metalcloudmatches.h5', index_limit_miri= -11, index_limit_gem= -10)
    # match_files(path_miri_cloudy / 'valid_metalcloudmatches.h5', \
    #             path_hst_cloudy / 'valid_metalcloudmatches.h5', index_limit_miri= -10, index_limit_gem= -11)
    # match_files(path_miri_cloudy / 'test_metalcloudmatches.h5', \
    #             path_hst_cloudy / 'test_metalcloudmatches.h5', index_limit_miri= -10, index_limit_gem= -11)



    # schedule(
    #     simulate, # event,
    #     name='Data generation',
    #     backend='slurm',
    #     prune=True,
    #     env=[
    #         'source ~/.bashrc',
    #         'conda activate WISEJ1828',
    #     ]
    # )

    # remove_extra(i)

