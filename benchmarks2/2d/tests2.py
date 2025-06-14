import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from scipy.sparse import csr_matrix,coo_matrix,kron,identity #optimizes H . v operations. to check if H already row sparse, do  isspmatrix_csr(H)
from scipy.sparse.linalg import eigsh
import time
import pickle
import argparse
from itertools import product,combinations
from math import comb,log10
from collections import Counter
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import random
from scipy.special import logsumexp
from scipy.sparse import SparseEfficiencyWarning
import warnings
from numba import njit
import random
from functools import lru_cache
import h5py
from collections import defaultdict
import ast
sys.path.append('/mnt/users/kotssvasiliou/ED/core_scripts')
from ED_chains_final_final import chain_configs,chains
parameters = {'L':3,
                  'geometry':'square',
                  't':1,
                  'mu':0,
                  'U':4,
                  'V':1,
                  'partial':False,
                  'projection':False,
                  'sign':False,
                  'JW string':True,
                  'mode':'full'}
def config_exe(parameters,target_dir='/mnt/users/kotssvasiliou/ED/benchmarks2/2d/data',verbose=1):
    '''
    This checks the symmetry sector generator part of the code.

    Input:
        target_dir(str)         :The location of the directory  to put output in
        parameters(dict)        :A dictionary of input parameters. Must contain:
                                                                    'L','geometry': Size and geometry of lattice
                                                                    't','U','V','mu': Hamiltonian parameters    
                                                                    'sign','JW string': treat hardcore bosons or fermions? sign is in hamiltonian while JW string is in greens function
                                                                    'mode': Full or Lanczos diagonalization
                                                                    'partial','projection': creation of Hilebrt space
                                                                
    Output:
        combined_data(dict)     :A dictionary containing information on the symmetry sectors and the eigenstates of the representative sectors
                                    Keys: The representative sectors
                                    Values: A dictionary with
                                                Keys: 'es': Spectra
                                                      'vs': Eigenstates
                                                      'ns': Occupation numbers of eigenstates
                                                      'equivalent sectors': Two lists; One with all sectors in equivalence class and one with all permutations
                                                      'weight': How many equivalent sectors there are
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    L = parameters['L']
    t = parameters['t']
    U = parameters['U']
    V = parameters['V']
    mu = parameters['mu']
    geometry = parameters['geometry']
    partial = parameters['partial']
    projection = parameters['projection']
    sign = parameters['sign']
    JWstring = parameters['JW string']
    mode = parameters['mode']
    #mu = 0 #half-filling; for 1electron per site, mu=-3U and for two, mu=-1.5U
    system_params = {'geometry':geometry,'L':L,'partial':partial,'projection':projection}
    sector_params = {'L':L,'geometry':geometry,'sign':sign,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':mode}}
    if verbose>0:print('='*100+'\n GENERATING ALL SYMMETRY SECTORS \n'+'='*100)
    timei = time.time()
    CCs = chain_configs(params=system_params)
    CCs.sector_check()
    config_data = CCs.compressed_data
    timef = time.time()
    if verbose>0:
        print('time to generate sectors:',timef-timei,' secs')
        if geometry=='square': secs_tot = (L+1)**(4*L); order = 2*4*L**2
        elif geometry=='triangular': secs_tot = (L+1)**(6*L); order = 2*3*L**2
        rep_secs = len(config_data)
        print('information on symmetry sectors:\n Total sectors: '+str(secs_tot)+'\n Number of representative sectors: '+str(rep_secs)+'\n Optimal number of representative sectors: '+str(int(secs_tot/order)))
    with open(target_dir+f'/L_{L}_{geometry}_symmetry_sectors.pkl','wb') as f:
        pickle.dump(config_data,f)
    
    ##
    #ED
    ##
    if verbose>0:print('='*100+'\n PERFORMING EXACT DIAGONALIZATION ON ALL REPRESENTATIVE SYMMETRY SECTORS \n'+'='*100)
    itime = time.time()
    combined_data = {}
    dim_H_stats = {}
    for k in config_data.keys():
        sector_params['config'] = k
        chain_instance = chains(sector_params)
        dimH = chain_instance.dim
        if dimH not in dim_H_stats.keys():
            dim_H_stats[dimH] = 0
        dim_H_stats[dimH] += 1
    #############################
    for dim,num in dim_H_stats.items():
        print(f'Theres {num} inequivalent symmetry sectors with dimension {dim}')
    return

##############
if __name__ == "__main__":
    config_exe(parameters=parameters)