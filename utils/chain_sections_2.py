import os
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
from chain_sections import chains
import random

warnings.simplefilter("ignore", SparseEfficiencyWarning) #supress warning???

class chain_configs():
    def __init__(self, params):
        self.L = params['L']
        self.geometry = params['geometry']
        self.projection = params['projection'] #boolean
        self.Nchains = 3 * self.L if self.geometry == 'triangular' else quit()
        self.configurations = []
        self.equivalence_classes = {}  # Maps a representative configuration to all symmetry-related ones
        self.permutation_map = {}  # Maps a representative to its symmetry operations (in cycle notation)
        self.weights = {}
        if 'Nel_max' in params: #if there's no projection, no need to define min and max number of electrons
            self.Nel_max = params['Nel_max']
        else:
            self.Nel_max = None
        if 'Nel_min' in params:
            self.Nel_min = params['Nel_min']
        else:
            self.Nel_min = None

        if self.projection:
            self.generate_projected_configurations()
        else:
            self.generate_all_configurations()
        ########################################
        #########save in compact way############
        ########################################
        self.compressed_data = {}
        #can also generate symmetry-related configs from permutations!

        for rep_config in self.equivalence_classes:
            self.compressed_data[rep_config] = (
                self.equivalence_classes[rep_config],  # All symmetry-related configurations
                self.permutation_map[rep_config]  # Associated permutations (cycle notation)
            )


    def cycle_notation(self, perm):
        """Convert a permutation list to cycle notation."""
        seen = set()
        cycles = []
        for i in range(len(perm)):
            if i not in seen:
                cycle = []
                x = i
                while x not in seen:
                    seen.add(x)
                    cycle.append(x)
                    x = perm[x]
                if len(cycle) > 1:
                    cycles.append(tuple(cycle))
        return cycles

    def apply_permutation(self, config, perm):
        """Apply a permutation to the full configuration (treat as a single sequence)."""
        c_up, c_down = config
        full_config = c_up + c_down  # Concatenate both spin-up and spin-down parts

        # Debugging statements
        #print(f"DEBUG: Applying permutation")
        #print(f" - Length of perm: {len(perm)} (Expected: {2 * self.Nchains})")
        #print(f" - Full config length: {len(full_config)}")
        #print(f" - Permutation: {perm}")

        if len(perm) != len(full_config):
            raise ValueError(f"Permutation length {len(perm)} does not match full config length {len(full_config)}")

        # Apply permutation to the full configuration
        permuted_config = tuple(full_config[i] for i in perm)

        # Split back into spin-up and spin-down components
        new_c_up = permuted_config[:self.Nchains]
        new_c_down = permuted_config[self.Nchains:]

        return (new_c_up, new_c_down)



    def track_permutation(self, perm, mapping):
        """Update a permutation using a given mapping."""
        return [perm[i] for i in mapping]

    def C3_permutation(self):
        """Returns the correct C3 permutation list."""
        L = self.L
        mapping = list(range(2 * self.Nchains))  # Identity mapping initially

        # First L chains (1,x) → (2,x)
        for x in range(L):
            mapping[x] = x + L  # Move to second set of chains

        # Second L chains (2,x) → (3,L-x)
        for x in range(L):
            mapping[x + L] = 2 * L + (L - 1 - x)  # Reverse order

        # Third L chains (3,x) → (1,L-x)
        for x in range(L):
            mapping[x + 2 * L] = (L - 1 - x)  # Reverse order back to first set

        # Apply same transformation to spin-down (offset by Nchains)
        for x in range(3 * L):
            mapping[self.Nchains + x] = self.Nchains + mapping[x]

        return mapping



    def translation_permutation(self, n, m):
        """Returns the cycle notation for translation (T1^n, T2^m)."""
        L = self.L
        mapping = list(range(2 * self.Nchains))

        for i in range(L):
            mapping[i] = (i - m) % L
            mapping[i + L] = ((i - n) % L) + L
            mapping[i + 2 * L] = ((i - (n + m)) % L) + 2 * L

        for i in range(3*L):
            #repeat for spin_down
            mapping[self.Nchains + i] = self.Nchains + mapping[i]

        return mapping

    def spin_inversion_permutation(self):
        """Returns cycle notation for spin inversion."""
        mapping = list(range(2 * self.Nchains))

        for i in range(self.Nchains):
            mapping[i] = i + self.Nchains
            mapping[i + self.Nchains] = i

        return mapping

    def generate_all_configurations(self):
        """Generate all configurations one by one, classifying them by symmetry equivalence."""
        electron_count = range(self.L + 1)
        configs_spinup = list(product(electron_count, repeat=self.Nchains))
        configs_spindown = list(product(electron_count, repeat=self.Nchains))

        for c_up in configs_spinup:
            for c_down in configs_spindown:
                config = (c_up, c_down)

                self.process_configuration(config)

    def generate_projected_configurations(self):
        """
        Generate *some* configurations one by one, classifying them by symmetry equivalence.
        
        """
        electron_count = range(self.L + 1)
        configs_spinup = list(product(electron_count, repeat=self.Nchains))
        configs_spindown = list(product(electron_count, repeat=self.Nchains))
        ###
        proj_dim = 0
        for el in range(self.Nel_min,self.Nel_max+1):
            proj_dim += self.combinatorics(boxes=2*self.Nchains,capacity=self.L,balls = el)
        ###
        for c_up in configs_spinup:
            for c_down in configs_spindown:
                config = (c_up, c_down)
                Ne = sum(c_up)+ sum(c_down)
                while len(self.equivalence_classes) < proj_dim:
                    if Ne <= self.Nel_max and Ne >= self.Nel_min:
                        self.process_configuration(config)

    def process_configuration(self, config):
        """Determine if a configuration is already accounted for or should be stored as a representative,
        and calculate its symmetry weight."""
        
        if config in self.equivalence_classes:
            return  # Already classified

        transformed_configs = []
        permutations = []

        # Generate all symmetry-related configurations
        for n in range(self.L):
            for m in range(self.L):
                for r in range(3):  # 0 or 1 or 2 C3 rotations (square lattice)
                    for s in range(2):  # 0 or 1 spin inversions
                        perm = self.identity_permutation()
                        perm = self.track_permutation(perm, self.translation_permutation(n, m))
                        for _ in range(r):
                            perm = self.track_permutation(perm, self.C3_permutation())
                        if s:
                            perm = self.track_permutation(perm, self.spin_inversion_permutation())

                        transformed = self.apply_permutation(config, perm)
                        #cycle_notation_perm = self.cycle_notation(perm)

                        transformed_configs.append(transformed)
                        #permutations.append(cycle_notation_perm) # !  not in cyclic notation
                        permutations.append(perm)

        # Find the lexicographically smallest configuration to serve as the representative
        rep_config = min(transformed_configs)
        
        # Debugging: Check how many symmetry-related elements exist
        weight = len(set(transformed_configs))  # Number of unique symmetry-related states

        # Store the equivalence class and weight
        if rep_config not in self.equivalence_classes:
            #is the minimum one the original one? it should be
            if rep_config != config:
                print('representative config is not the expected one:',config,rep_config)
            ########################################
            self.equivalence_classes[rep_config] = []
            self.permutation_map[rep_config] = []
            self.weights[rep_config] = weight  # Store weight

        for trans_config, perm in zip(transformed_configs, permutations):
            if trans_config not in self.equivalence_classes[rep_config]:
                if rep_config != config:
                    print('found previously unknown perms...',config,rep_config)
                self.equivalence_classes[rep_config].append(trans_config)
                self.permutation_map[rep_config].append(perm)


    def identity_permutation(self):
        """Returns the identity permutation."""
        return list(range(2 * self.Nchains))
    @staticmethod
    def combinatorics(boxes,capacity,balls):
        '''
        calculates ways to distribute x balls in N boxes gives each box fits M balls
        '''
        raise NotImplementedError
################################
def reindex_permutation(perm):
    """
    Reorder a permutation of indices from 0,1,...,6L-1 into the new order:
    0, 3L, 1, 3L+1, 2, 3L+2, ..., 3L-1, 6L-1.
    Parameters:
        perm (list): A list of length 6L representing the permutation.
    Returns:
        list: The reordered permutation.
    """
    n = len(perm)
    if n % 6 != 0:
        raise ValueError("The length of the permutation must be a multiple of 6.")
    
    L = n // 6
    half = 3 * L  # since n = 6L, half of the list is 3L elements.
    
    new_perm = []
    for i in range(half):
        new_perm.append(perm[i])
        new_perm.append(perm[i + half])
    
    return new_perm
################################
def save_L_2_configs():
    params = {}
    L = 2
    params['L'] = L
    params['geometry'] = 'triangular'
    params['projection'] = False
    configs = chain_configs(params)
    DATA = configs.compressed_data
    with open('/mnt/users/kotssvasiliou/ED/utils/configs/'+'triangle_2_full.pkl', 'wb') as f:
            pickle.dump(DATA, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_configs(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def test_configurations(repeats = 1):
    '''
    does some sanity checks on the configurations:
    1)check that they sum up to the correct number of sectors
    2) CHeck spectra of symmetry related configs match agfter permuting!
    '''
    DATA = load_configs('configs/triangle_2_full.pkl')
    L = 2
    geometry = 'triangular'
    if geometry == 'triangular':
        sec_full = (L+1)**(6*L)
        G_order = 2*3*L**2
    else:
        sec_full = (L+1)**(4*L)
        G_order = 2*4*L**2
    sec = 0
    sectors = len(DATA)
    for config in DATA.keys():
        w = len(DATA[config][1])
        sec += w
    print('~'*60)
    print('TOTAL # OF SECTORS vs THOSE WE HAVE:',sec_full,sec)
    print('NUMBER OF INEQUIVALENT SECTORS:',sectors)
    print('GAIN BY UTILIZING SYMMETRY vs MAX GAIN:',sec_full/sectors,'vs',G_order)
    print('~'*60)
    print('CHECKING THAT SYMMETRY RELATED CONFIGURATIONS HAVE IDENTICAL SPECTRA:')
    #############################
    t = 1
    U = 6
    V = 0
    mu = 0
    loc = [1,3,1,3,2,4,2,4,1,2,1,2,3,4,3,4,1,4,1,4,2,3,2,3]
    H_params = {'L':L,'loc':loc,'sign':True,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    ################################
    for repeat in range(int(repeats)):
        Es = []
        Vs = []
        config = random.choice(list(DATA.keys()))
        config = ((0,0,0,0,0,1),(0,0,0,2,0,1))
        weight = len(DATA[config][0])
        for s in range(weight):
            config_s,perm_s = DATA[config][0][s],DATA[config][1][s]
            print(s,':',config_s,perm_s,reindex_permutation(perm_s))
            #continue
            H_params['config'] = config_s
            chain_instance = chains(H_params)
            if chain_instance.dim>7:
                break
            diag_states = chain_instance.diagonalization()
            Es.append(diag_states['es'])
            Vs.append(diag_states['vs'])
            print('eig',diag_states['vs'][:,0])
            print('basis',chain_instance.basis)
            print(chain_instance.configuration_Hamiltonian())
            #quit()

    print(chain_instance.location_mapping())
    state = '110100000000000000000000'
    state_dec = int(state,2)
    print('state',state,state_dec)
    for eta in range(3): #edit for square
            for s in range(2):
                for j in range(1,L**2+1):
                    J,sgn = chain_instance.creation_operator(j,eta,s,I=state_dec)
                    if J!= None:
                        print('transition',bin(J),sgn)


def test_spectra(repeats=1):
    params = {}
    L = 2
    params['L'] = L
    params['geometry'] = 'triangular'
    params['projection'] = False
    DATA = load_configs('configs/triangle_2_full.pkl')
    #############################
    t = 1
    U = 6
    V = 1
    mu = 0
    loc = [1,3,1,3,2,4,2,4,1,2,1,2,3,4,3,4,1,4,1,4,2,3,2,3]
    H_params = {'L':L,'loc':loc,'sign':True,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    for repeat in range(int(repeats)):
        config = random.choice(list(DATA.keys()))
        H_params['config'] = config
        chain_instance = chains(H_params)
        diag_states = chain_instance.diagonalization()
        print(config,DATA[config][1],len(DATA[config][1]))


###############
def generate_partitions(L, N):
        """
        Generate all possible partitions of N electrons in a chain with L sites as binary strings.

        Parameters:
            L (int): Number of sites.
            N (int): Number of electrons of certain spin.

        Returns:
            list: List of binary *strings* representing the partitions.
        """

        if N > L:
            raise ValueError("Number of electrons (N) cannot exceed number of sites (L).")

        # Generate all combinations of N positions from L sites
        partitions = []
        for positions in combinations(range(L), N):
            # Create a binary representation of the partition
            binary = ['0'] * L
            for pos in positions:
                binary[pos] = '1'
            partitions.append(''.join(binary))
        return partitions
def c2i(c,perm):
    L = len(int(c[0]/3))
    basis = []
    for chain in range(int(3*L)):
        chain_up = c[0][chain]
        chain_down = c[1][chain]
        basis_up = generate_partitions(L,chain_up)
        basis_down = generate_partitions(L,chain_down)
        if len(basis) == 0: 
            basis = basis_up
            basis = [old + new for old in basis for new in basis_down]
        else:
            basis = [old + new for old in basis for new in basis_up]
            basis = [old + new for old in basis for new in basis_down]
    return basis
#######
if __name__ == "__main__":
    #save_L_2_configs()
    #quit()
    #print(reindex_permutation([2,3,5,4,1,0,8,9,11,10,7,6]))
    #quit()
    test_configurations(1)
    #test_spectra()
    #print(len(chain.equivalence_classes))
    #print(chain.weights.values())
    #print('configs',chain.configurations)
    #for k in chain.equivalence_classes.keys():
    #    print('key',k)
    #    print('configs',chain.equivalence_classes[k])
    #print('permutation map',chain.permutation_map)