'''
Sets up runs that only calculate <H>,<H^2>,<N>,<N^2>; not greens functions
'''
import sys
import numpy as np
sys.path.append('/mnt/users/kotssvasiliou/ED/core_scripts')
sys.path.append('/mnt/users/kotssvasiliou/ED/utils')
import ED_chains_final_final_anis
import ED_chains_final_final
import time
import pickle
import os
from scipy.sparse import csr_matrix,coo_matrix,kron,identity #optimizes H . v operations. to check if H already row sparse, do  isspmatrix_csr(H)
def ED_exe(parameters,verbose = 1):
    '''
    This is a single ED run.

    Diagonalizes the system for a single point in parameter space, if the flag is up.

    Input:
        target_dir(str)         :The location of the directory  to put output in
        parameters(dict)        :A dictionary of input parameters. Must contain:
                                                                    'L','geometry': Size and geometry of lattice
                                                                    't','U','V','mu': Hamiltonian parameters    
                                                                    'sign': treat hardcore bosons or fermions?
                                                                    'mode': Full or Lanczos diagonalization
                                                                    'partial','projection': creation of Hilbert space
                                                                
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
    mode = parameters['mode']
    #mu = 0 #half-filling; for 1electron per site, mu=-3U and for two, mu=-1.5U
    system_params = {'geometry':geometry,'L':L,'partial':partial,'projection':projection}
    sector_params = {'L':L,'geometry':geometry,'sign':sign,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':mode}}
    #if verbose>0:print('='*100+'\n GENERATING ALL SYMMETRY SECTORS \n'+'='*100)
    #timei = time.time()
    #CCs = ED_chains_final_final_anis.chain_configs(params=system_params)
    #CCs.sector_check()
    #config_data = CCs.compressed_data
    #timef = time.time()
    #if verbose>0:
    #    print('time to generate sectors:',timef-timei,' secs')
    #    if geometry=='square': secs_tot = (L+1)**(4*L); order = 2*4*L**2
    #    elif geometry=='triangular': secs_tot = (L+1)**(6*L); order = 2*3*L**2
    #    rep_secs = len(config_data)
    #    print('information on symmetry sectors:\n Total sectors: '+str(secs_tot)+'\n Number of representative sectors: '+str(rep_secs)+'\n Optimal number of representative sectors: '+str(int(secs_tot/order)))
    ##
    #ED
    ##
    sector = ((0, 0, 0, 1, 2, 1), (2, 2, 2, 2, 2, 2))
    sector_params['config'] = sector
    chain_instance = ED_chains_final_final_anis.chains(sector_params)
    diag_states = chain_instance.diagonalization()
    chain_instance_iso = ED_chains_final_final.chains(sector_params)
    diag_states_iso = chain_instance_iso.diagonalization()
    print('Energies anisotropic:',diag_states['es'])
    print('Energies isotropic:',diag_states_iso['es'])
    print('Energies anisotropic:',diag_states['vs'])
    print('Energies isotropic:',diag_states_iso['vs'])
    quit()
    print('basis:',chain_instance.basis)
    num_chains = len(chain_instance.chain_hamiltonians)
    total_dim = np.prod([h.shape[0] for h in chain_instance.chain_hamiltonians])
    H = csr_matrix((total_dim, total_dim), dtype=np.float64)
    print('num chains',num_chains)
    print('total dim',total_dim)

    basis_dec = [int(el,2) for el in chain_instance.basis]
    for i in range(total_dim):
        state = chain_instance.basis[i]
        #print('state dec',s,chain_instance.basis[m])
        n_tot = len(chain_instance.basis[i])
        n_chain = n_tot // (2*parameters['L'])
        out = []
        for m in range(n_chain):
            start = m * 2 * L
            spins_up = int(state[start:start+L], 2)
            spins_down  = int(state[start+L:start+2*L], 2)
            #where on the chain there's single or double occupancy
            ones = spins_up ^ spins_down
            twos = spins_up & spins_down
            # Build digits from MSB→LSB
            row = []
            for i in range(L-1, -1, -1):
                if (twos >> i) & 1:
                    row.append('2')
                elif (ones >> i) & 1:
                    row.append('1')
                else:
                    row.append('0')
            out.append(''.join(row))
        print('state',state)
        print('summed:')
        occ_state = ''.join(out)
        print(''.join(out))
        locations = [1,3,2,4,1,2,3,4,1,4,2,3]
        occ_matrix = np.zeros((4,3),dtype=int)
        for loc in range(12):
            valley = loc // 4 #what valley this data point belongs to
            occ_matrix[locations[loc]-1,valley] = int(occ_state[loc])
        print('matrix',occ_matrix)
        print('')
    #diag_states = chain_instance.diagonalization()
    
if __name__ == "__main__":
    U = 10.
    V = 1.23
    mu = 0.14
    parameters = {'L':2,
                  'geometry':'triangular',
                  't':1.,
                  'mu':mu,
                  'U':U,
                  'V':V,
                  'partial':False,
                  'projection':False,
                  'sign':False,
                  'mode':'full'}
    print('params',parameters)
    results = ED_exe(parameters=parameters) 