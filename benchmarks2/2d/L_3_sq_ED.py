'''
In this code I will perform ED on a 3x3 square lattice system,
with the ultimate goal of calculating some energies, maybe N vs mu etc.
Calculating dynamical quantities would be tricky since the Hilbert spaces are too large to solve without Lanczos

I have the option to either load the representative symmetry sectors or generate them on the spot
'''
import os
import sys
import time
import pickle
sys.path.append('/mnt/users/kotssvasiliou/ED/core_scripts')
from ED_chains_final_final import chain_configs,chains,thermodynamics
from numba import njit
params = {'L':3,
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
def generate_symmetry_sectors(parameters,target_dir = '/mnt/users/kotssvasiliou/ED/benchmarks2/2d/data',load = True,analyse = True,verbose = 1):
    '''
    Either generates or loads all the symmetry configurations to be fed onto ED.
    Input:
        target_dir(str)         :The location of the directory  to put output in
        parameters(dict)        :A dictionary of input parameters. Must contain:
                                                                    'L','geometry': Size and geometry of lattice
                                                                    't','U','V','mu': Hamiltonian parameters    
                                                                    'sign','JW string': treat hardcore bosons or fermions? sign is in hamiltonian while JW string is in greens function
                                                                    'mode': Full or Lanczos diagonalization
                                                                    'partial','projection': creation of Hilebrt space NOTE doesn't work
        load(Bool)              :If True, just load pre-existing data
        analyse(Bool)           :Parses through the symmetry configs and checks all the hilbert sector bases dimensions
                                                                
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
    L = parameters['L'];t = parameters['t'];U = parameters['U'];V = parameters['V'];mu = parameters['mu'];geometry = parameters['geometry'];partial = parameters['partial']
    projection = parameters['projection'];sign = parameters['sign'];JWstring = parameters['JW string'];mode = parameters['mode']
    #mu = 0 #half-filling; for 1electron per site, mu=-3U and for two, mu=-1.5U
    system_params = {'geometry':geometry,'L':L,'partial':partial,'projection':projection}
    sector_params = {'L':L,'geometry':geometry,'sign':sign,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':mode}}
    if load == True:
        if verbose>0:print('LOADING CONFIG DATA')
        with open(target_dir+f'/L_{L}_{geometry}_symmetry_sectors.pkl',"rb") as f:
            config_data = pickle.load(f)
    else:
        if verbose>0:print('GENERATING CONFIG DATA')
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
    #Check symmetry sectors
    ##
    if analyse:
        if verbose>0:print('='*100+'\n ANALYZING REPRESENTATIVE SYMMETRY SECTORS \n'+'='*100)
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
    return config_data

def test_sectors(config_data,number = 10):
    for index,rep_sec in enumerate(config_data.keys()):
        if index > number:break
        print('representative symmetry config:',rep_sec)
        print('symmetry related sectors:')
        num_of_secs = len(config_data[rep_sec][0])
        print('num of secs',num_of_secs)
        for i in range(num_of_secs):
            print(f'sec:{config_data[rep_sec][0][i]},operator={config_data[rep_sec][1][i]}')
        print('\n \n \n','='*100)
    return

def diagonalize_sectors():
    '''
    The main part of the code. I want to go through 
    '''
    return
#############
if __name__ == "__main__":
    time_i = time.time()
    config_data = generate_symmetry_sectors(params,analyse=False)
    time_f = time.time()
    print(f'time to load and analyze sectors:{time_f-time_i}:.2f')
    test_sectors(config_data)