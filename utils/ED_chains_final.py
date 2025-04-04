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
from functools import lru_cache
import h5py
from collections import defaultdict
import ast

warnings.simplefilter("ignore", SparseEfficiencyWarning) #supress warning???

#TODO: PROJECTION CURRENTLY HAS SOME BUG IT CANT EVEN BUILD CORRECT NUMBER OF CONFIGURATIONS. FIX

####################
####################
#   CHAIN CONFIGS  #
####################
####################

class chain_configs():
    '''
    An instance of this class contains information about all possible chain configurations.
    The output will be a set of  equivalence classes that are not related to eachother by the symmetries of the system
    Within the equivalence class, we have info on
        0)
        1)all permutation(symmetry) related configs in the equivalence class


    ---------------------------------------------------------------------
    A given configuration will have the form of a nested tuple

    c = ((c_up),(c_down))

    with c_spin = (c_1,c_2,.....,c_{{2,3}L})'
    '''
    def __init__(self, params):
        self.L = params['L']
        self.geometry = params['geometry']
        self.projection = params['projection'] #boolean
        if self.geometry == 'triangular':
            self.Nchains = 3 * self.L
        elif self.geometry == 'square':
            self.Nchains = 2 * self.L
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
        #generate permutations of basis elements
        #self.generate_permutations()
        #save in compact way
        self.compressed_data = {}
        self.save_compressed()
    ###################################################
    # GENERATE AND PROCESS ALL/PROJECTED CONFIGURATIONS
    ###################################################
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
        print('projected dimension',proj_dim)
        for c_up in configs_spinup:
            for c_down in configs_spindown:
                config = (c_up, c_down)
                Ne = sum(c_up)+ sum(c_down)
                while len(self.equivalence_classes) < proj_dim:
                    print(len(self.equivalence_classes))
                    # uhhh gets stuck at zero equivalence classes
                    if Ne <= self.Nel_max and Ne >= self.Nel_min:
                        self.process_configuration(config)
    
    def process_configuration(self, config):
        """Determine if a configuration is already accounted for or should be stored as a representative,
        and calculate its symmetry weight.
        
        Symmetry operations: (s,r,n,m) for TR^2 Rot^r T_1^n T_2^m {|configs>}
        """
        
        if config in self.equivalence_classes:
            return  # Already classified

        transformed_configs = []
        permutations = []
        symmetry_operations = [] 

        if self.geometry == 'square':
            Rot_num = 4                                #TODO check this works fine.....
            Rot_permutation = self.C4_permutation
        elif self.geometry == 'triangular':
            Rot_num = 3
            Rot_permutation = self.C3_permutation
        

        # Generate all symmetry-related configurations
        for n in range(self.L):
            for m in range(self.L):
                for r in range(Rot_num):  # 0 or 1 or 2 C3 rotations (square lattice)
                    for s in range(2):  # 0 or 1 spin inversions
                        perm = self.identity_permutation()
                        perm = self.track_permutation(perm, self.translation_permutation(n, m))
                        for _ in range(r):
                            perm = self.track_permutation(perm, Rot_permutation())
                        if s:
                            perm = self.track_permutation(perm, self.spin_inversion_permutation())

                        transformed = self.apply_permutation(config, perm)
                        sym_op = (s, r, n, m)  # Store the symmetry operation tuple instead of permutation

                        transformed_configs.append(transformed)
                        symmetry_operations.append(sym_op)
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
        ###
        #for trans_config, perm in zip(transformed_configs, permutations):
        #    if trans_config not in self.equivalence_classes[rep_config]:
        #        if rep_config != config:
        #            print('found previously unknown perms...',config,rep_config)
        #        self.equivalence_classes[rep_config].append(trans_config)
        #        self.permutation_map[rep_config].append(perm)
        for trans_config, sym_op in zip(transformed_configs, symmetry_operations):
            if trans_config not in self.equivalence_classes[rep_config]: # adds only new permutations
                if rep_config != config: print('found previously unknown permutations...shouldnt happen',config,rep_config)
                self.equivalence_classes[rep_config].append(trans_config)
                self.permutation_map[rep_config].append(sym_op)  # Store the symmetry operation

    ######################
    # symmetry operations#
    ######################
    def C3_permutation(self):
        """
        Apply C3 rotation on triangular lattice
        (1,x)-->(2,x)
        (2,x)-->(3,L-x)
        (3,x)-->(1,L-x)
        """
        L = self.L
        mapping = list(range(2 * self.Nchains)) 
        for x in range(L):
            mapping[x] = x + L 
            mapping[x + L] = 2 * L + (L - 1 - x) 
            mapping[x + 2 * L] = (L - 1 - x) 
        # Apply same transformation to spin-down (offset by Nchains)
        for x in range(3 * L):
            mapping[self.Nchains + x] = self.Nchains + mapping[x]
        return mapping
    def C4_permutation(self):
        """
        Apply C4 rotation on square lattice (counter-clockwise!)
        (1,x)-->(2,L-x)
        (2,x)-->(1,x)
        """
        L = self.L
        mapping = list(range(2 * self.Nchains)) 
        for x in range(L):
            mapping[x] = x + L  # (1, x) → (2, x)     
            #this may look like the inverse of what we defined but its not:
            # f(0) = i(L)
            # f(1) = i(L+1) etc. so the 1st flavor of the transformed sector is a copy of the 2nd flavor of the  initial sector, or (2, x) → (1, x)  and (1, x) → (2,L - 1 - x). The -1 is a python thing.
            mapping[x + L] = L - 1 - x
        # Apply same transformation to spin-down (offset by Nchains)
        for x in range(2 * L):
            mapping[self.Nchains + x] = self.Nchains + mapping[x]
        return mapping
    def translation_permutation(self, n, m):
        """
        Apply generic translation element (T1^n, T2^m) on the lattice
        """
        L = self.L
        mapping = list(range(2 * self.Nchains))

        if self.geometry == 'triangular':
            for i in range(L):
                mapping[i] = (i - m) % L
                mapping[i + L] = ((i - n) % L) + L
                mapping[i + 2 * L] = ((i - (n + m)) % L) + 2 * L
        elif self.geometry == 'square':
            for i in range(L):
                mapping[i] = (i - m) % L
                mapping[i + L] = ((i - n) % L) + L
        for i in range(self.Nchains):
            #repeat for spin down
            mapping[self.Nchains + i] = self.Nchains + mapping[i]
        return mapping
    def spin_inversion_permutation(self):
        """
        Apply time-reversal operator on system
        """
        mapping = list(range(2 * self.Nchains))

        for i in range(self.Nchains):
            mapping[i] = i + self.Nchains
            mapping[i + self.Nchains] = i

        return mapping
    #########################
    #helper & test functions#
    #########################
    def sector_check(self):
        '''
        checks that total_sectors = rep_sectors x weight_of_rep_class
        '''
        print('.'*50+' \n TESTING NUMBER OF SYMMETRY SECTORS \n'+'*'*50)
        num_secs = (self.L+1)**(2*self.Nchains)
        symmetry_secs = 0
        if hasattr(self,'compressed_data'):
            for i,representative_sector in enumerate(self.compressed_data.keys()):
                symmetry_secs += len(set(self.compressed_data[representative_sector][0]))
        if symmetry_secs == num_secs:
            print('.'*50+' \n TEST SUCCESFULL \n'+'*'*50)
        else:
            print('.'*50+' \n TEST UNSUCCESFULL \n'+'*'*50)
            print(symmetry_secs,num_secs,i)
            raise ValueError
        return
    def save_compressed(self):
        self.compressed = {}
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
    @staticmethod
    def combinatorics(boxes, capacity, balls):
        """
        Calculates the number of ways to distribute 'balls' into 'boxes',
        given each box can hold at most 'capacity' balls.
        """
        
        @lru_cache(None)
        def count_ways(remaining_balls, current_box):
            # Base case: all balls distributed among boxes
            if current_box == boxes:
                return 1 if remaining_balls == 0 else 0

            # Recursive case: Distribute balls in the current box (0 to capacity)
            ways = 0
            for b in range(min(remaining_balls, capacity) + 1):
                ways += count_ways(remaining_balls - b, current_box + 1)

            return ways

        return count_ways(balls, 0)
    def identity_permutation(self):
        """Returns the identity permutation."""
        return list(range(2 * self.Nchains))

#########################
class chains():
    '''
    An instance of this class represents a Hamiltonian associated with a particular configuration of chains.
    It contains the Hilbert space and Hamiltonian of individual chains, as well as the total hilbert space and Hamiltonian, plus its eigenstuff

    The underlying lattice is only inputed through:
        1) the self.loc property, mapping the chain numbers to sites on the lattice
        2) the self.geometry property. May depreciate it in the future.
        3) Implicitly throught the number of chans for a n LxL system.
    
    ------------------------------------------------------------------------
    ORDER OF HILBERT SPACE
    ()
    '''
    def __init__(self,params):
        '''
        Initializes the system for a given config(symmetry sector)

        Input:
        params(dict)            :Input parameters. Keys:
                                                        config
                                                        H_params: t,mu,V,U,SGN
                                                        diag_params:mode (full/lanczos)
                                                        geometry
                                                        L
        
        '''
        self.config = params['config']
        self.H_params = params['H_params']
        self.geometry = params['geometry']
        self.L = params['L']
        if self.geometry == 'triangular':
            self.Nflav = 3
        elif self.geometry == 'square':
            self.Nflav = 2
        if self.L == 2 and self.geometry == 'triangular':
            self.loc = [1,3,1,3,2,4,2,4,1,2,1,2,3,4,3,4,1,4,1,4,2,3,2,3]
        elif self.L == 2 and self.geometry == 'square':
            self.loc = [1,3,1,3,2,4,2,4,1,2,1,2,3,4,3,4]
        elif self.L == 3 and self.geometry == 'square':
            self.loc = [1,4,7,1,4,7,2,5,8,2,5,8,3,6,9,3,6,9,1,2,3,1,2,3,4,5,6,4,5,6,7,8,9,7,8,9]
        else:
            print('not implemented geometry or size')
            raise NotImplementedError
        #print('flavor',self.Nflav,'L',self.L,'config',self.config)
        self.sign = params['sign']# boolean T/F
        self.basis()#generates basis
        self.diag_params = params['diag_params']
        self.sites = {} #uhhhhh forgot what this does exactly....
        for i,v in enumerate(self.loc):
            if v not in self.sites.keys():
                self.sites[v] = 2**(((2*self.Nflav)*self.L**2)-i-1)
            else:
                self.sites[v] += 2**(((2*self.Nflav)*self.L**2)-i-1)
    def basis_old(self):
        '''
        Generates the basis in terms of binaries. An LxL system will have a basis with 3L*2*L=6L**2 sites. Each site is 0 or 1 so full hilbert space is ofcourse 2**(6L**2)=64**L**2
        The ordering of the basis is: chain_1_up,chain_1_down,chain_2_up,chain_2_down,......,chain_3L_up,chain_3L_down
        Args:
            c(nested tuple):        The chain configuration
        Returns:
            basis(list):            A list of all states in the hilbert space spanned by the configuration
            length_basis(int):      The size of the HIlbert space
            Hamiltonians(list):     List of 6L**2 npcarray Hamiltonians to be combined into a full hamiltonian
        '''
        c = self.config
        L = self.L
        basis = []
        Hamiltonians = []
        for chain in range(int(3*L)):
            chain_up = c[0][chain]
            chain_down = c[1][chain]
            basis_up = generate_partitions(L,chain_up)
            Hamiltonians.append(self.chain_Hamiltonian(basis_up))
            basis_down = self.generate_partitions(L,chain_down)
            Hamiltonians.append(self.chain_Hamiltonian(basis_down))
            if len(basis) == 0: 
                basis = basis_up
                basis = [old + new for old in basis for new in basis_down]
            else:
                basis = [old + new for old in basis for new in basis_up]
                basis = [old + new for old in basis for new in basis_down]
        self.basis = basis
        self.dim = len(basis)
        self.chain_hamiltonians = Hamiltonians
        return 
    def basis(self):
        '''
        Generates the basis in terms of binaries. An LxL system will have a basis with 3L*2*L=6L**2 sites. Each site is 0 or 1 so full hilbert space is ofcourse 2**(6L**2)=64**L**2
        
        Args:
            c(nested tuple):        The chain configuration
        Returns:
            basis(list):            A list of all states in the hilbert space spanned by the configuration
            length_basis(int):      The size of the HIlbert space
            Hamiltonians(list):     List of 6L**2 npcarray Hamiltonians to be combined into a full hamiltonian
            ns(list):               A list of the occupation numbers of the basis of the configuration (will be same occupation for all )
        '''
        c = self.config
        L = self.L

        basis = []
        Hamiltonians = []
        for chain in range(int(self.Nflav*L)):
            chain_up = c[0][chain]
            chain_down = c[1][chain]
            basis_up = self.generate_partitions(L,chain_up)
            Hamiltonians.append(self.chain_Hamiltonian(basis_up))
            basis_down = self.generate_partitions(L,chain_down)
            Hamiltonians.append(self.chain_Hamiltonian(basis_down))
            if len(basis) == 0: 
                basis = basis_up
                basis = [old + new for old in basis for new in basis_down]
            else:
                basis = [old + new for old in basis for new in basis_up]
                basis = [old + new for old in basis for new in basis_down]
        self.basis = basis
        self.dim = len(basis)
        self.chain_hamiltonians = Hamiltonians
        #########################################
        #count occupation number of each basis element of the hilbert space
        self.nstates = []
        basis_dec = [int(el,2) for el in self.basis]
        for m in range(self.dim):
            s = basis_dec[m]
            self.nstates.append(self.countBits(s))
        #print('~'*100)
        #print(self.nstates)
        #print('~'*100)
        return 
    def chain_Hamiltonian(self,chain_basis):
        '''
        Generates the small non-interacting Hamiltonian for a single chain and spin. The dimension is read from the basis.
        The full Hilbert space for a single chain and spin is 2**L but in our case the Hilbert space will be L Choose N_spin. For L=2, dim =1 or 2 while for L=3, dim = 1 or 3 

        Parameters:
            Basis(list):        A list containing all states (represented by binaries) in the chain's Hilbert space
            t(float):           Hopping strength
        Returns:
            H(npc array):       A (densely constructed) Hamiltonian
        
        '''
        #step1) Build lookup table for all states(very small table). That can just be the basis_up/down list
        #step2) go through basis size, associate index with state and check hopping, mapping it back to a new state.
        #step3) done#
        dim = len(chain_basis)
        basis_dec = [int(el,2) for el in chain_basis]
        L_chain = self.L
        t = self.H_params['t']
        H = np.zeros((dim,dim),dtype=float)
        for m in range(dim):
            s = basis_dec[m]
            if L_chain == 2: L_max = 1
            else:
                L_max = L_chain
            #in L=2 case there is overcounting of processes since 0<-->1 and 1<--->0 . compare to eg L=3: 0<-->1,1<-->2,2<-->0 
            for i in range(L_max):
                j=(i+1)%L_chain
                s2 = self.hop(s,i,j)
                if s2 != -1:
                    try:
                        n = basis_dec.index(s2)
                    except ValueError:
                        print('Index not found...quitting!')
                        quit()
                    sgn = 1
                    if self.sign == True:
                        sgn = self.fermion_sgn(self.binp(s,length=L_chain),self.binp(s2,length=L_chain))
                        if sgn == -1 and L_chain == 2:
                            print('negative sign? Shouldnt happen for L=2')
                    H[n,m] -= t*sgn
        return H
    #@staticmethod
    def count_occupancies(self,s):
        '''
        s is the configuration as a binary string. it has size 6*L**2
        v has same length as s and holds the location of s 
        '''
        #s = int(s,2) #convert to integer from string
        #for each site (0 to L**2-1) create a mask based on v
        masks = self.sites
        state = int(s,2)
        occupancy = {}
        occupancy_tot = 0
        for i in masks.keys():
            print('--')
            print(self.binp(masks[i],length=6*self.L**2))
            print(self.binp(state,length=6*self.L**2))
            print(self.binp(masks[i]&state,length=6*self.L**2))
            print(self.countBits(masks[i] & state))
            print('--')
            occupancy[i] = self.countBits(masks[i] & state)
            occupancy_tot += occupancy[i]
        return occupancy,occupancy_tot
    def configuration_Hamiltonian(self):
        '''
        Returns the *full* Hamiltonian of the configuration
        Steps:
        1)Creates tensor product for hopping Hamiltonians sparsely
        2)Adds interactions and chemical potential (all are diagonal terms)
        '''
        U = self.H_params['U']
        V = self.H_params['V']
        mu = self.H_params['mu']
        num_chains = len(self.chain_hamiltonians)
        total_dim = np.prod([h.shape[0] for h in self.chain_hamiltonians])
        H = csr_matrix((total_dim, total_dim), dtype=np.float64)
        # Loop through each local Hamiltonian and embed it in the tensor product space
        for i, h_local in enumerate(self.chain_hamiltonians):
            h_local_sparse = csr_matrix(h_local)
            # Identity operators for spaces before and after the current subspace
            identity_before = identity(np.prod([self.chain_hamiltonians[j].shape[0] for j in range(i)]), format="csr") if i > 0 else 1
            identity_after = identity(np.prod([self.chain_hamiltonians[j].shape[0] for j in range(i + 1, num_chains)]), format="csr") if i < num_chains - 1 else 1
            # Embed the local Hamiltonian in the full tensor product space
            term = kron(kron(identity_before, h_local_sparse), identity_after, format="csr")
            H += term
        ######################################
        ##### interactions#######
        ##########################
        basis_dec = [int(el,2) for el in self.basis]
        for m in range(total_dim):
            s = basis_dec[m]
            occupations = []
            for site in range(1,self.L**2+1): # keyyyyyyyy need to go all the way from 1 to L**2 not L**2 -1!!!!! BUG!!! 
                occ_site = self.countBits(self.sites[site] & s)
                occupations.append(occ_site)
                H[m,m] += -mu*occ_site + 0.5*U*(occ_site-self.Nflav)**2  #half filling is at \mu = (2,3)*U+(8,18)*V for rect & triangle respectively
            #n.n. electron repulsion
            #print('U term',m,H[m,m]/U)
            H[m,m] += V*self.nn_repulsion(occupations)
            #print('V term',m,self.nn_repulsion(occupations))
        ###################
        diff = H - H.getH()
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        if max_diff != 0:
            print('Hamiltonian is not hermitian!!!!',max_diff)
        return H
    
    def diagonalization(self):
        '''
        Sorts out the diagonalization of this configuration. The output contains the full information necessary to calculate thermodynamic properties.

        Returns:
            diag_states(dict):      A dictionary with keys: 'params':           Holds minimal system information such as configuration and weight of the configuration
                                                            'es':               An 1xM array of dtype=float containing the eigenstates
                                                            'ns':               An 1xM array of dtype=int containing the occupation number of the eigenstates
                                                            'vs':               An MxM array of dtype=complex containing the eigenstates. Its the main memory bottleneck by far
        '''
        #lanczos or full?
        H = self.configuration_Hamiltonian()

        if self.diag_params['mode'] == 'full':
            H_dense = H.toarray()
            e,v = np.linalg.eigh(H_dense)
            
        elif self.diag_params['mode'] == 'Lanczos':
            k = self.diag_params['k']
            dim = self.dim
            if 1<dim<10:
                k = min(dim-1,k)
            e,v = eigsh(H,k=k, which='SA', tol=1e-10)
        else:
            print('no mode added')
            quit()
        diag_states = {}
        #diag_states['config'] = self.config
        diag_states['es'] = e
        diag_states['vs'] = v
        diag_states['ns'] = self.nstates 
        return diag_states
    @staticmethod
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
    @staticmethod
    def hop(s,i,j):
        '''
        CHecks if hopping is allowed between sites i and j for state s and if it is,
        it outputs the resulting state

        Args:
            s(bin):         A binary number with L digits(L=length of chain) signifying the state of the chain
            i(int),j(int):  0 =<i,j<L Integers representing sites on the chain

        Returns:
            s2:             Either -1 to signify no allowed hopping or a binary to denote the resulting state after the hopping
        '''
        mask = 2**(i)+2**(j)
        K = s & mask #bitwise AND.
        P = K ^ mask #bitwise XOR.
        # L will have structure 0000000[i]0000[j]00000 and there's four cases:
        #1) L = mask means I1[i]=I1[j]=0 -> hopping is not allowed
        #2) L = 000..00 means I1[i]=I1[j]=1 -> hopping is not allowed
        #3&4) L = ...[1]...[0]... or L = ...[0]...[1]... means hopping is allowed, in which case new integer is 
        if P == mask or P == 0:
            s2 = -1#flag to signify no hopping
        else:
            s2 = s - K + P
        return s2
    @staticmethod
    def countBits(x):
        '''Counts number of 1s in bin(n)'''
        #From Hacker's Delight, p. 66
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        x = x + (x >> 8)
        x = x + (x >> 16)
        return x & 0x0000003F 
    
    def nn_repulsion(self,occupations):
        '''
        Calculates the n.n. repulsion
        Input:
            occupations(list)       :A list of size L**2 labelling the 
        Output:
            count(int)              :\sum_{nn pairs} (n_i - Nflav)(n_j - Nflav)
        ------------------------------
        TODO Check this works for L=2 triangular array
        TODO ADD L=3 triangular thing
        NOTE Difference between labelling sites from 0 to L**2-1 vs 1 to L**2+1
        '''
        if self.geometry == 'square':
            if self.L == 2:
                pairs = [(1,2),(1,3),(2,4),(3,4)]
            elif self.L == 3:
                pairs = [(1,2),(1,3),(1,4),(1,7),(2,3),(2,5),(2,8),(3,6),(3,9),(4,5),(4,6),(4,7),(5,6),(5,8),(6,9),(7,8),(7,9),(8,9)]
            else:
                print('dencdbj',self.L)
                raise NotImplementedError
        elif self.geometry == 'triangular':
            if self.L == 2:
                pairs = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
            else:
                raise NotImplementedError
        else:
            raise ValueError
        count = 0
        for pair in pairs:
           pair_i = pair[0] - 1
           pair_j = pair[1] - 1#turnng site to index with the minus one
           count += (occupations[pair_i]-self.Nflav)*(occupations[pair_j]-self.Nflav)  
        return count
    def nn_repulsion_old(self,occupations):
        '''
        NOTE: Replaced with nn_repulsion to generalize this.
        ---------------
        calculates n.n. repulsion for L=2 where all sites are nn with all others
        ---------
        Not True for square!
        '''
        if self.L != 2:
            print('n.n. interaction not implemented for L != 2')
            return 0
            #quit()
        count = 0
        for i,occ1 in enumerate(occupations):
            for j,occ2 in enumerate(occupations):
                if i < j:
                    count += (occ1-self.Nflav)*(occ2-self.Nflav) 
        return count
    @staticmethod
    def binp(num, length=4):
        '''
        print a binary number without python 0b and appropriate number of zeros
        regular bin(x) returns '0bbinp(x)' and the 0 and b can fuck up other stuff
        '''
        return format(num, '#0{}b'.format(length + 2))[2:]
    @staticmethod
    def fermion_sgn(binary1,binary2):
        '''
        Modified from 'count_ones_between_flips' function in hubbard_chains.py
        --------------------
        Takes two binary strings that are meant to be related by a flip, ie they only differ in two sites, s1=xxx0xxx1xxx and s2=xxx1xxx0xxx.
        It then counts the number of 1's that separate these flipped sites, and outputs (-1)**count
        This accounts for the anticommutaative relations of the fermions
        
        Input:
            binary1(str):       A binary string representing a spinless fermion state on a chain
            binaryw(str):       A binary string representing a spinless fermion state on a chain
        Return:
            sgn(int):           The sign relating these two states
        '''
        # Ensure both binaries are of the same length
        if len(binary1) != len(binary2):
            raise ValueError("Both binary strings must have the same length.")
        # Find the XOR of the two binary strings
        xor_result = ''.join(str(int(b1) ^ int(b2)) for b1, b2 in zip(binary1, binary2))
        # Identify the positions of '1's in the XOR result
        flip_positions = [i for i, bit in enumerate(xor_result) if bit == '1']
        # Check if there are exactly two flipped positions
        if len(flip_positions) != 2:
            raise ValueError("There must be exactly two flipped bits.")
        # Get the range between the two flipped positions
        start, end = flip_positions
        between_segment = binary1[start + 1:end]
        # Count the number of '1's in the segment between the flipped positions
        ones_count = between_segment.count('1')
        return (-1)**ones_count
    ################
    #bit permutations
    # figure out how to have bit permutations
############################
############################
############################
class thermodynamics():
    '''
    A class that takes as input the configurations and their weights from the chain_configs class instance and also the eigenbasis of a chains class instance.
    An instance of this class then calculates information about 
        1) Partition function
        2) <E>,<E^2>
        3) other non-diagonal operators like <N>,<N^2>
        4) The time displaced Green's function G(τ)

    The instance is initialized by:
    1)Shifting the spectrum to >=0
    2)Changing the basis on occupation operator expectations
    3)Calculating the 
    '''
    def __init__(self,parameters,combined_data):
        '''
        Input:
        
        parameters(dict)        :Contains info on lattice geometry, projection options and Hamiltonian parameters
                                Keys: 
                                    verbose: How much stuff to print out
                                    loc: a list mapping each digit of a binary rep of a state to a physical lattice site
                                    L: system size
                                    geometry: The geometry ('square'/'triangular')
                                    calc_greens_func: Boolean (T/F) To calculate or not the Green's function. It precomputes some quantities

        combined_data(dict)     : Combined data with information on which sectors each equivalence class contains, 
                                as well as ED data for each of these representative sectors
                                Keys: The representative sectors
                                Values: A dictionary with
                                        Keys:
                                            'es': Spectra
                                            'vs': Eigenstates
                                            'ns': Occupation numbers of eigenstates***
                                            'equivalent sectors': Two lists; One with all sectors in equivalence class and one with all permutations
                                            'weight': How many equivalent sectors there are
        ------------------------------------------------------------------------------------------------------------------------------------------------------
        ** Note:
                The 'ns' justr calculates <i|N|i> for {|i>} in each sector. so, really all |i> in each
                sector have same N = N_sector, and thus so will the eigenstates: <α|N|α>=N_sector within each sector.
                So this representation of a vector of 'ns' per sector is wasteful, a number would do.
        '''
        self.data = combined_data
        self.verbose = parameters['verbose']
        self.L = parameters['L']
        self.t = parameters['t']
        self.V = parameters['V']
        self.U = parameters['U']
        self.mu = parameters['mu']
        self.sign = parameters['sign']
        self.geometry = parameters['geometry']
        self.loc = parameters['loc']
        self.calc_greens_func = parameters['greens function']
        if self.geometry == 'square':
            self.Nflav = 2
            self.Nchains = 2*self.Nflav*self.L
            self.group_order = 2*2*self.Nflav*self.L**2
        elif self.geometry == 'triangular':
            self.Nflav = 3
            self.Nchains = 2*self.Nflav*self.L
            self.group_order = 2*self.Nflav*self.L**2
        if self.verbose > 0:print('='*100+ " \n CALCULATING THERMODYNAMICAL PROPERTIES \n "+'='*100)
        self.digits = 2*self.Nflav*self.L**2 #number of binary digits on a state
        #######################################
        # find ground state and shift energies#
        #######################################
        self.shift_spectrum()
        #####################################
        #some helper stuff for c_dagger calc#
        #####################################
        if self.calc_greens_func == True:
            self.map = self.location_mapping()
            #create reverse lookup table for our sectors. I.e. given a sector c, find its representative \tilde{c}
            lookup = {}
            timei = time.time()
            for key in self.data.keys():
                equivalent_sectors = self.data[key]['equivalent sectors'][0]
                for sec in equivalent_sectors:
                    lookup[sec] = key
            self.sector_lookup = lookup
            timef = time.time()
            if self.verbose > 0:print('='*100+'\n SECTOR LOOKUP TABLE CREATED IN ',timef-timei,' SECONDS \n'+'='*100)
    ###################################
    #         BASIC OBSERVABLES       #
    ###################################
    def partition_function(self,beta):
        '''
        Computes the partition function using logsumexp for numerical stability
        ------------------------------------------------------------------------
        Parameters:
        data (dict):            Dictionary where keys label fundamental sectors, and each sector contains:
                                    - 'weight': Number of symmetry-equivalent sectors
                                    - 'es': Mx1 array of eigenvalues (spectrum)

        beta (float):               Inverse temperature (1/kT).
        
        Returns:
        float:                      The log of the partition function Z.
        '''
        log_terms = []
        for sector_data in self.data.values():
            weight = sector_data['weight']
            energies = sector_data['es']
            log_terms.append(np.log(weight) + (-beta * energies))
        log_terms = np.concatenate(log_terms)
        log_Z = logsumexp(log_terms)
        return log_Z
    def H_moments(self,beta):
        '''
        Computes the <H> and <H^2> observables using logsumexp for numerical stability.
        Takes care to filter out E=0 (GS) as their log is ill defined and they shouldn't contribute to shifted expectation values
        ------------------------------------------------------------------------------
        Parameters:
        data_dict (dict):           Dictionary where keys label fundamental sectors, and each sector contains:
                                    - 'weight': Number of symmetry-equivalent sectors
                                    - 'es': Mx1 array of eigenvalues (spectrum)
                                    - 'vs': MxM array of eigenvectors

        beta (float):               Inverse temperature (1/kT).
        
        Returns:
        (float, float, float, float): 
                                    - H_avg (shifted)
                                    - H_avg_unshifted (original energy scale)
                                    - H_sq_avg (shifted)
                                    - H_sq_avg_unshifted (original energy scale
        '''
        t0 = time.time()
        log_Z =  self.partition_function(beta)#changed pervious code so that it returns logsumexp() rather than its exponential
        tf = time.time()
        if self.verbose>1:print('time to calculate log_Z:',np.round(tf-t0,2))
        log_terms_H = []
        log_terms_H_sq = []
        for sector_data in self.data.values():
            weight = sector_data['weight']
            energies = sector_data['es']
            valid_mask = energies > 0
            valid_energies = energies[valid_mask]
            #
            if valid_energies.size > 0:
                log_terms_H.append(np.log(weight) + (-beta * valid_energies) + np.log(valid_energies))  
                log_terms_H_sq.append(np.log(weight) + (-beta * valid_energies) + np.log(valid_energies**2))
                log_H = logsumexp(np.concatenate(log_terms_H)) - log_Z
                log_H_sq = logsumexp(np.concatenate(log_terms_H_sq)) - log_Z
        ##################################
        #return also the values with shifted energy.
        H_avg = np.exp(log_H)
        H_avg_unshifted = H_avg + self.GS_energy
        H_sq_avg = np.exp(log_H_sq)
        H_sq_avg_unshifted = H_sq_avg +2*self.GS_energy*H_avg+self.GS_energy**2
        return H_avg,H_avg_unshifted,H_sq_avg,H_sq_avg_unshifted 
    def N_moments(self, beta):
        """
        Computes the thermal expectation value of \hat{N} and \hat{N}^2, using logsumexp for numerical stability.
        Takes care to filter out N=0 states as their log is ill defined and they shouldn't contribute to shifted expectation values
        Parameters:
        data_dict (dict):               Dictionary where keys label fundamental sectors, and each sector contains:
                                        - 'weight': Number of symmetry-equivalent sectors
                                        - 'es': Mx1 array of eigenvalues (spectrum)
                                        - 'ns': Mx1 array of the <a|N|a> expectation value of the occupation in each eigenstate

        beta (float):                   Inverse temperature (1/kT).
        
        Returns:
        (float,float):                  Thermal expectation value <N>,<N^2>.
        """
        log_Z = self.partition_function(beta)  # log of partition function
        log_terms_N = []
        log_terms_N_sq = []

        for sector_data in self.data.values():
            weight = sector_data['weight']
            energies = sector_data['es']
            N_a = np.array(sector_data['ns'])  # 1xM (Mx1?) array of occupation numbers
            if len(set(N_a))!= 1:
                print('!!!! errr')
                raise ValueError

            # Filter out zero-occupation cases to prevent log(0) issues
            valid_mask = N_a > 0
            valid_N_a = N_a[valid_mask]
            valid_energies = energies[valid_mask]

            if valid_N_a.size > 0:
                log_terms_N.append(np.log(weight) + np.log(valid_N_a) - beta * valid_energies)
                log_terms_N_sq.append(np.log(weight) + 2*np.log(valid_N_a) - beta * valid_energies)

        # Compute the final thermal average <N>
        log_N = logsumexp(np.concatenate(log_terms_N)) - log_Z
        log_N_sq = logsumexp(np.concatenate(log_terms_N_sq)) - log_Z

        return np.exp(log_N),np.exp(log_N_sq)
    ###################################
    #         GREENS FUNC MAIN       #
    ###################################       
    def greens_func_element(self,index_1,index_2,g,taus,beta):
        '''
        Calculates an element of the Green's function G_{α,β}(τ) for a single group action:
        G_{α,β}(τ) ~ Σ_g <c^\dagger_{gα}><c^\dagger_{gβ}>.
        Since most of G_{α,β} elements will vanish (only L x 2*Nflav*L**2 out of (2*Nflav*L**2)**2  are non-zero), 
        but if <c^\dagger_{α}><c^\dagger_{β}> !=0, then also <c^\dagger_{gα}><c^\dagger_{gβ}> != 0,
        we can say:
        0)How many elements do not vanish? 1/8th,1/12th for L=2 square,triangle and 1/12th for L=2 square
        1) Since most elements of the Green's function vanish, most G_{α,β} will be calculated very quickly. (Time~0)
        2) Those that do not vanish will take a while because all g \in G will contribute, none giving zero.

        Thus we want to make it easy to parallelize at the level of g.

        L=2 square: Takes 60-100s per g
        --------------------------------------------------------------------------------------------------
        Input:
            index_1(2)(int)             :Indices of G_{α,β}. They take values from 0 to 2*Nflav*L**2 - 1. They are mapped to
                                        a tuple (op_1/2) of the form (r,eta,s) with r=1,...,L**2 & eta=1,...,Nflav & s=0(up) and 1(dn)                            
            g(???)                      :The group element that will map our operators to new operators and also potentially give them a sign.
                                        Each og these g's contributes to the same Green's function matrix element
            taus(np.array)              :A 1D array for the τ where the Green's function will be measured. Must have τ \in [0,β] and really,0 means 0^{+}  and β means β^{-},
                                        as otherwise the Green's function form changes
            beta(float)                 :The inverse temperature of the system
        Output:
            Gs(np.array)                :An array matching the shape of the taus. It gives a contribution for this g to the component G_{op_1,op_2}
        --------------------------------------------------------------------------------------------------
        TODO: MAP OPERATORS TO INDICES ON THE GREENS FUNCTION
        TODO: MAP TO CORRECT PERMUTATION OF THE EIGENSTATES
        TODO: SAVE REP BASIS MAYBE INSTEAD OF GENERATING ON THE SPOT????
        '''
        H_params_temp = {'L':self.L,'geometry':self.geometry,'sign':self.sign,'H_params':{'t':self.t,'mu':self.mu,'U':self.U,'V':self.V},'diag_params':None}
        ###########
        # setup
        ###########
        for op,index in self.map.items():
            if index == index_1:
                op_1_init = op
            if index == index_2:
                op_2_init = op
        op_1,factor1 = self.group_action(action=g,operator=op_1_init)
        op_2,factor2 = self.group_action(action=g,operator=op_2_init)
        print('OPERATORS:',g,op_1_init,op_1,op_2_init,op_2)
        ###########
        #helper func
        def F(group,v_i,v_f,perm):
            '''
            Calculates
            <m|c^\dagger_{\\alpha}|n> = \sum_{i,j} m_i n_j^* <j|c^\\dagger_{\\alpha}|i>
            Input:
                group(hd5)          :The part of the c_dagger operator for a given operator and a given initial (& final) sector
                v_i(np.array)       :The initial eigenstate, which lies in a representative sector
                v_f(np.array)       :The final eigenstate, which also lies in a representative sector
                perm(list)          :The permutation to be applied in the final eigenstate, to map v_f to the actual (in general non-representative) sector of relevance
            -----------------------------------------------
            TODO: RN I am MAPPING TO WRONG PERMUTATION OF EIGENSTATES, MAKE SURE YOU INCLUDE THE CORRECT PERMUTATION
            '''
            ij_pairs = group['ij_pairs'][:]
            signs = group['signs'][:]
            prod = 0
            for k,pair in enumerate(ij_pairs):
                i_index = pair[0]
                j_index = pair[1]
                #prod += v_i[i_index]*v_f[j_index].conj()*signs[k]
                prod += v_i[i_index]*v_f[perm[j_index]].conj()*signs[k]
            return prod

        ###########
        log_Z =  self.partition_function(beta)
        log_terms_G = []
        op_group_name_1 = f"cdagger_op_{op_1[0]}_{op_1[1]}_{op_1[2]}"
        op_group_name_2 = f"cdagger_op_{op_2[0]}_{op_2[1]}_{op_2[2]}"
        group1 = self.cdagger_map[op_group_name_1]
        group2 = self.cdagger_map[op_group_name_2]
        #right now, consider only a single group element...
        for representative_sector in self.data.keys():
            #what is relevant sector?
            sector_f_1 = None
            sector_f_2 = None
            for name in group1:
                if name.startswith(str(representative_sector)):
                    try:
                        rhs = name.split('_to_')[-1]
                        sector_f_1 = ast.literal_eval(rhs)
                        break
                    except Exception as e:
                        print(f"Failed parsing sector name: {name}", e)
            for name in group2:
                if name.startswith(str(representative_sector)):
                    try:
                        rhs = name.split('_to_')[-1]
                        sector_f_2 = ast.literal_eval(rhs)
                        break
                    except Exception as e:
                        print(f"Failed parsing sector name: {name}", e)
            if ((sector_f_1 == None) and (sector_f_2 == None)) :
                #print('full chain')
                continue
            elif sector_f_1 != sector_f_2:
                #print('non compatible operators')
                continue
            #print('sectors new',representative_sector,sector_f_1,sector_f_2)
            sec_f = sector_f_1
            #####################
            #at this point find the representative sec corresponding to sec_f, and generate the relevant permutation between the two sector's basis.
            sec_f_rep = self.sector_lookup[sec_f]
            rep_sector_period = int(self.group_order/self.data[representative_sector]['weight'])      #TODO CHECK
            permutation = None
            for i,sec in enumerate(self.data[sec_f_rep]['equivalent sectors'][0]):
                if sec == sec_f:
                    permutation = self.data[sec_f_rep]['equivalent sectors'][1][i]
                    continue
            if permutation == None:
                raise ValueError
            H_params_temp['config'] = sec_f_rep
            sec_f_rep_system = chains(H_params_temp)
            sec_f_rep_basis = [int(el,2) for el in sec_f_rep_system.basis]
            transformed_basis = compound_transform(permutation,sec_f_rep_basis,geometry=self.geometry)
            basis_perm = np.argsort(transformed_basis)[::-1] #TODO: ARE WE SURE THIS IS CORRECT PERM OR IS IT LIKE ITS INVERSE????
            #####################
            sector_group_name = f"{representative_sector}_to_{sec_f}"
            subgroup1 = group1[f"{sector_group_name}"]
            subgroup2 = group2[f"{sector_group_name}"]
            ############################################
            # now go through eigenstates in these two sectors
            for i,vm in enumerate(self.data[representative_sector]['vs'].T): #NOTE: This is transposed cause v[i,j] is i'th component of j'th eigenstate so we want to sum over the eigenstates. ie vm is the i'th eigenstate
                for j,vn in enumerate(self.data[sec_f_rep]['vs'].T):
                    Em = self.data[representative_sector]['es']
                    En = self.data[sec_f_rep]['es']
                    F1 = F(subgroup1,vm,vn,basis_perm)*factor1
                    F2 = F(subgroup2,vm,vn,basis_perm).conj()*factor2
                    if F1*F2 == 0:
                        print('uhhhh')
                        #print(F1,F2,vm,vn)
                        #raise ValueError
                    #print('F1F2',F1,F2,factor1,factor2,np.log(F1*F2),np.log(1/rep_sector_period))
                    else:
                        log_term = np.log(1/rep_sector_period)+ (-beta*Em[i])-taus *(En[j]-Em[i])+ np.log(F1*F2+0j) #force log to take complex (and negative) arguments when F1*F2 is negative, which i don't see why it can't be.
                        log_terms_G.append(log_term)
                    #print(log_term)
        log_G = logsumexp(log_terms_G,axis=0) - log_Z
        #################################
        if not hasattr(self,'Gfunc'):
            self.Gfunc = {}
            self.Gfunc[beta] = np.zeros((taus.shape[0],2*self.Nflav*self.L**2,2*self.Nflav*self.L**2),dtype=complex) 
            if self.verbose>0:print('Initialized Greens function')
        self.Gfunc[beta][:,index_1,index_2] += np.exp(log_G)
        return np.exp(log_G)
    
    ###################################
    #         GREENS FUNC STUFF       #
    ################################### 
    def location_mapping(self):
        """
        Maps each tuple (j, eta, s) to the corresponding x'th binary place in self.loc.
        self.loc is a list/array of length 6*L**2 
        """
        mapping = {}
        for eta in range(self.Nflav):
            for s in range(2):
                for j in range(1,self.L**2+1):
                    # Only consider indices in the current eta block.
                    start = eta * 2 * self.L**2
                    end = (eta + 1) * 2 * self.L**2
                    # Find the first index in the block with the correct parity and j value.
                    for i in range(start, end):
                        if ((i // self.L) % 2 == s) and (self.loc[i] == j):
                            mapping[(j, eta, s)] = i
                            break
                    else:
                        raise ValueError(f"No index found for (j={j}, eta={eta}, s={s})")
        return mapping
    def cdagger(self,loc,state):
        '''
        Applies c^\{dagger} on binary representation of state.
        
        Input:
        state(int)      :The integer I representing basis state |i>
        loc(int)        :The location on bin(I) that c^\dagger acts on. Equivalent to a tuple (r,eta,s)
        
        Output:
        state_out(int)  :The integer J of the output state (None if c^\dagger kills state)
        sign(\pm 1)     :The sign associated with the fermionic ordering of Fock space(is equal to number of electrons to the right of loc)
        '''
        #need to add sign as well
        if (state >> loc) & 1:  # check if bit at loc is 1
            return None,None
        # Compute sign factor (count fermions to the right)
        sign = (-1) **(self.binp(state & ((1 << loc) - 1),length=2*self.L**2*self.Nflav).count('1'))
        return state ^ (1 << loc),sign
    def state2sector(self,state):
        ''''
        Given a state, find the sector its in. brings out the configuration (symmetry sector) it lies in

        Input:
        state(int)      :The integer I representing basis state |i>

        Output:
        sector(tuple)   :The (generically not representative) sector the state belongs to. This tuple looks like c=(c_up,c_down) with c_s a tuple of L*Nflav integers denoting the occupation number of each chain
        '''
        string = self.binp(state,length= 2*self.L**2*self.Nflav)
        chain_length = self.L
        num_chains = len(string) // chain_length
        config = [string[i*chain_length:(i+1)*chain_length].count('1') for i in range(num_chains)]
        config_up,config_dn = tuple(config[::2]),tuple(config[1::2])
        return (config_up,config_dn)
    def gen_creation_mapping(self,filename='greens_mapping.h5',load=False):
        '''
        generate or load <i|c^\dagger|j> data as HD5 with structure:
        greens_mapping.h5
        └── c^{\dagger}_{r0,eta0,s0}
            ├── sector0_to_sector_new0
            |   ├── sectors  : [rep_sector,sector_new,rep_sector_new]
            │   ├── ij_pairs : (i, j) pairs for basis states in sec0 and sec_new0
            │   ├── signs    : Fermionic signs for each pair
            ├── sector1_to_sector_new1
            │   ├── ...
        └── cdagger_r1_eta2_s1
            ├── ...
        ------------------------------------------------
        Input:
            load(Bool)              : Load data or generate it?
            filename(str)           : File name of hd5 object, to either load or save as
        Output:
            <i|c^\dagger|j>(hd5)    : Data structure holding matrix elements of c^\dagger
        ------------------------------------------------
        TODO        0) Have it get the bases of new and old sector in smarter way, utilizing ED data
                    1) Implement projection
        '''
        ######################################
        if load == True:
            self.cdagger_map =  h5py.File(filename, 'r')
            return
        ######################################
        #some temporary H_params thing to be able to calculate sector basis states
        H_params_temp = {'L':self.L,'geometry':self.geometry,'sign':self.sign,'H_params':{'t':self.t,'mu':self.mu,'U':self.U,'V':self.V},'diag_params':None}
        ######################################
        DATA = self.data
        Sites = range(1,self.L**2+1)
        Flavors = range(self.Nflav)
        Spin = range(2)
        #####
        with h5py.File(filename, 'w') as f:
            for x, eta, s in ((x, eta, s) for x in Sites for eta in Flavors for s in Spin):
                operator_loc = (self.digits-1) - self.map[(x,eta,s)] #NOTE: location in self.cdagger function starts from right-most binary digit while self.map starts from left, hence the shift.
                op_group = f.create_group(f"cdagger_op_{x}_{eta}_{s}")
                for rep_sector in DATA.keys():
                    new_sector = None#initialized value
                    new_rep_sector = None
                    H_params_temp['config'] = rep_sector
                    rep_sector_system = chains(H_params_temp)
                    rep_sector_basis = [int(el,2) for el in rep_sector_system.basis]
                    ij_pairs = [] #where ill store the mappings
                    signs = [] #where ill store the signs
                    for i,I in enumerate(rep_sector_basis):
                        J,sgn = self.cdagger(loc=operator_loc,state=I) # apply creation operator 
                        if J == None:
                            continue
                        else:
                            new_sector_temp = self.state2sector(J) #within a sector, all i should map to a j that lies in the same new_sector
                            if new_sector == None:
                                new_sector = new_sector_temp
                            elif new_sector != new_sector_temp:
                                print("Sector mismatch!",new_sector,new_sector_temp)
                                print('....')
                                print('ABORTING')
                                quit()
                            #find representative secotr related to new sector
                            #one method is regular binary search and the other is with a lookup table.
                            #at L=2 square, you get about 10% faster performance with lookup table.
                            #as sanity check one can make sure the mappings agree
                            new_rep_sector_temp = self.sector_lookup.get(new_sector)
                            #new_rep_sector_temp_2 = find_principle_sec(new_sector,DATA)
                            #if new_rep_sector_temp != new_rep_sector_temp_2:
                            #    print('lookup methods disagrreee')
                            #    quit() 
                            if new_rep_sector == None:
                                new_rep_sector = new_rep_sector_temp
                            elif new_rep_sector != new_rep_sector_temp:
                                print("Sector mismatch!",new_sector,new_sector_temp)
                                print('....')
                                print('ABORTING')
                                quit()
                            H_params_temp['config'] = new_sector
                            new_sector_sys = chains(H_params_temp)
                            new_sector_basis = [int(el,2) for el in new_sector_sys.basis]
                            try:
                                j = new_sector_basis.index(J)
                            except ValueError:
                                print(f"J={J} not found in new sector basis!")
                                print('....')
                                print('ABORTING')
                                #continue
                                quit()
                            ij_pairs.append((i, j))
                            signs.append(sgn)
                    if ij_pairs:  # only save non-empty transitions
                        group_name = f"{rep_sector}_to_{new_sector}"
                        trans_group = op_group.create_group(group_name)
                        trans_group.create_dataset('sectors',data=[rep_sector,new_sector,new_rep_sector])
                        trans_group.create_dataset("ij_pairs", data=np.array(ij_pairs, dtype=np.int32), compression="gzip")
                        trans_group.create_dataset("signs", data=np.array(signs, dtype=np.int8), compression="gzip")
        self.cdagger_map =  h5py.File(filename, 'r')
        return 

    ###################################
    #       HELPER FUNCTIONS          # 
    ###################################
    def shift_spectrum(self):
        '''
        Goes through all symmetry sectors, retrieves the keys where the energy is minimum
        '''    
        GS_energy = 1e10
        GS_secs = []
        for symm_sector,symm_sector_data in self.data.items():
            sector_GS = np.min(symm_sector_data['es'])
            if sector_GS < GS_energy:
                GS_energy = sector_GS
                GS_secs = [symm_sector] #reset list with new minimum sector
            elif sector_GS == GS_energy:
                GS_secs.append(symm_sector) #append list if we have degeneracy
        if self.verbose>0:
            print('GROUND STATE ENERGY IS ',GS_energy,' AND IS IN SYMMETRY SECTOR(S)',GS_secs)
            print('SHIFTING SPECTRA')
        for symm_sector,symm_sector_data in self.data.items():
            symm_sector_data['es'] -= GS_energy
        Es_flat = np.concatenate([symm_sector_data['es'] for symm_sector_data in self.data.values()])
        if self.verbose>0:print('AFTER SHIFTING, EXTREMAL VALUES OF ENERGY ARE',np.min(Es_flat),np.max(Es_flat))
        self.GS_energy = GS_energy
        return
    @staticmethod
    def binp(num, length=4):
        '''
        Returns the binary representation of an integer num, preserving the correct length (ie not disgarding any zeros in the front)
        '''
        if num == None:
            return None
        return format(num, '#0{}b'.format(length + 2))[2:]
    ######################################
    # action of group element on operator#
    ######################################
    def init_group_action(self):
        '''
        initializes the group action on a creation operator. Ie this is a map g: c^\dagger_{α}----->(factor)xc^\dagger_{β} for g  \in the generators of the group: {T1,T2,C_4,TR}.
        Composite group element maps are calculated with function *group_action*
        '''
        if self.L !=2 and self.geometry != 'square':
            raise NotImplementedError
        if self.L == 2 and self.geometry == 'square':
            map_x = {1:3,2:4,3:1,4:2}
            map_y = {1:2,2:1,3:4,4:3}
            map_c4_r = {1:1,2:3,3:2,4:4}
            map_c4_eta = {0:1,1:0}
            map_TR_s = {0:1,1:0}
            self.map_T1 = {}
            self.map_T2 = {}
            self.map_C4 = {}
            self.map_TR = {}
            self.map_id = {}
            for site in range(1,self.L**2+1):
                for eta in range(self.Nflav):
                    for s in range(2):
                        index_in = (site,eta,s)
                        index_out_T1 = (map_x[site],eta,s)
                        factor_out_T1 = 1
                        index_out_T2 = (map_y[site],eta,s)
                        factor_out_T2 = 1
                        index_out_C4 = (map_c4_r[site],map_c4_eta[eta],s)
                        factor_out_C4 = 1
                        index_out_TR = (site,eta,map_TR_s[s])
                        factor_out_TR = (-1)**s
                        self.map_T1[index_in] = [index_out_T1,factor_out_T1]
                        self.map_T2[index_in] = [index_out_T2,factor_out_T2]
                        self.map_C4[index_in] = [index_out_C4,factor_out_C4]
                        self.map_TR[index_in] = [index_out_TR,factor_out_TR]
                        self.map_id[index_in] = [index_in,1]
    def group_action(self,action,operator):
        '''
        Calculates a map g: c^\dagger_{α}----->(factor) x c^\dagger_{β}
        by composing multiple of the generating maps.
        ----------------------------------------------------------------------------
        Input:
            action(tuple)                   :A tuple (s,r,n,m) representing the group element g= (TR)^s   x   (C_4)^r   x   (T_2)^m   x   (T_1)^n  
            operator(tuple)                 :A tuple (r,eta,s) representing the creation operator.
        Output:
            operator(tuple)                 :A tuple (r,eta,s) representing the *new* creation operator.
            factor(complex)                 :complex factor (all but TR give fCTOR 1)
        '''
        if not hasattr(self,'map_id'):
            self.init_group_action()
        factor = 1
        operator_temp = operator
        (s,r,m,n) = action
        for _ in range(n):
            operator_temp,f = self.map_T1[operator_temp]
            factor *= f
        for _ in range(m):
            operator_temp,f = self.map_T2[operator_temp]
            factor *= f
        for _ in range(r):
            operator_temp,f = self.map_C4[operator_temp]
            factor *= f
        for _ in range(s):
            operator_temp,f = self.map_TR[operator_temp]
            factor *= f
        return [operator_temp,factor]
############################
############################
def time_reversal_transform(number, L, N):
    """
    Apply time-reversal symmetry transformation on a binary number.

    Parameters:
    - number: The integer representing the original binary state.
    - L: The system size.
    - N: The total number of bits (4L^2 or 6L^2).

    Returns:
    - The transformed integer after swapping L-bit pairs.
    """
    result = 0  # Store the transformed number
    
    for i in range(0, N, 2 * L):
        # Extract two L-bit blocks
        block1 = (number >> i) & ((1 << L) - 1)  # Extract L bits from position i
        block2 = (number >> (i + L)) & ((1 << L) - 1)  # Extract L bits from position i + L
        
        # Swap their positions in the result
        result |= (block1 << (i + L)) | (block2 << i)

    return result

def translation_y_transform_old(number, L):
    """
    Apply T2 translation (y-direction) symmetry transformation on a 4L²-bit binary number.

    Parameters:
    - number: The integer representing the original binary state.
    - L: The system size.

    Returns:
    - The transformed integer after applying the translation along x.
    """
    N = 4 * L**2  # Total bit length
    half_N = N // 2  # Each half has 2L² bits
    result = 0

    # Step 1: Process First Half (Shifts inside each L-bit chain)
    for i in range(0, half_N, L):
        # Extract L-bit block
        block = (number >> (half_N - (i + L))) & ((1 << L) - 1)
        # Circular shift: abcdef -> fabcde
        shifted_block = ((block & 1) << (L - 1)) | (block >> 1)
        # Store the shifted bits in result
        result |= (shifted_block << (half_N - (i + L)))

    # Step 2: Process Second Half (Permutes entire chains)
    for i in range(0, half_N, 2 * L):
        # Extract two (2L-bit) groups
        group_A = (number >> (i + half_N)) & ((1 << (2 * L)) - 1)
        group_B = (number >> (i + half_N + 2 * L)) & ((1 << (2 * L)) - 1)
        # Swap groups
        result |= (group_B << (i + half_N)) | (group_A << (i + half_N + 2 * L))

    return result
def translation_x_transform_old(number, L):
    """
    Apply T1 translation (x-direction) symmetry transformation on a 4L²-bit binary number,
    structured in the same way as the provided translation_y_transform function.

    Parameters:
    - number: The integer representing the original binary state.
    - L: The system size.

    Returns:
    - The transformed integer after applying the translation along x.
    """
    N = 4 * L**2  # Total bit length
    half_N = N // 2  # Each half has 2L² bits
    result = 0

    # Step 1: Process First Half (Permutes entire chains)
    for i in range(0, half_N - 4 * L + 1, 4 * L):
        # Extract two (2L-bit) groups
        group_A = (number >> (half_N - (i + 2 * L))) & ((1 << (2 * L)) - 1)
        group_B = (number >> (half_N - (i + 4 * L))) & ((1 << (2 * L)) - 1)
        # Swap groups
        result |= (group_B << (half_N - (i + 2 * L))) | (group_A << (half_N - (i + 4 * L)))

    # Step 2: Process Second Half (Shifts within chains)
    for i in range(0, half_N, L):  # Each group is L bits
        # Extract L-bit block
        block = (number >> (i + half_N)) & ((1 << L) - 1)
        # Circular shift within L-bit block: abcdef -> fabcde
        shifted_block = ((block & 1) << (L - 1)) | (block >> 1)
        # Store the shifted bits in result
        result |= (shifted_block << (i + half_N))

    return result
def translation_x_transform_new(number, L):
    '''
    this code, for the first half of the binary (the least important digits) (flavor = 1)
    it permutes blocks of size 2L. ie the rightmost 2L block will at the end occupy the digits (2L**2 - 2L,2L**2)

    For the second half of the binary, within each block of length L it permutes by one. so eg digit 2L**2 +1 will go to 2L**2 + (L-1)
    '''   
    N = 4 * L**2  # Total bit length
    half_N = N // 2  # Each half has 2L² bits
    result = 0

    # Step 1: Process First Half (Permutes entire chains)
    for i in range(0,half_N,2*L):
        # Extract two (2L-bit) groups
        chain_i = (number >> (i)) & ((1 << (2 * L)) - 1)
        if i == 0:
            chain_i_shifted = chain_i << (half_N - 2*L)
        else:
            chain_i_shifted = chain_i << (i - 2*L)
        result |= chain_i_shifted
    # Step 2: Process Second Half (Shifts within chains)
    for i in range(0, half_N, L):  # Each group is L bits
        # Extract L-bit block
        block = (number >> (i + half_N)) & ((1 << L) - 1)
        # Circular shift within L-bit block: abcdef -> fabcde
        shifted_block = ((block & 1) << (L - 1)) | (block >> 1)
        #print(bin(block),'--->',bin(shifted_block))
        # Store the shifted bits in result
        result |= (shifted_block << (i + half_N))
        #print(bin(result>>18))
    return result
def translation_y_transform_new(number, L):
    '''
    this code, for the first half of the binary (the least important digits) (flaor = 2)
    within each block of length L it permutes by one. so eg digit 0 will go to 2L**2 -1

    For the second half of the binary,  it permutes blocks of size 2L. ie the rightmost 2L block will at the end occupy the digits (2L**2 - 2L,2L**2)
    '''   
    N = 4 * L**2  # Total bit length
    half_N = N // 2  # Each half has 2L² bits
    result = 0

    # Step 1: Process 2nd  Half (Permutes entire chains)
    for i in range(0,half_N,2*L):
        # Extract two (2L-bit) groups
        chain_i = (number >> (i+half_N)) & ((1 << (2 * L)) - 1)
        if i == 0:
            chain_i_shifted = chain_i << (N - 2*L)
        else:
            chain_i_shifted = chain_i << (half_N + i - 2*L)
        result |= chain_i_shifted
        #print(bin(result>>18))
    # Step 2: Process 1st Half (Shifts within chains)
    #print('----')
    for i in range(0, half_N, L):  # Each group is L bits
        # Extract L-bit block
        block = (number >> (i)) & ((1 << L) - 1)
        # Circular shift within L-bit block: abcdef -> fabcde
        shifted_block = ((block & 1) << (L - 1)) | (block >> 1)
        #print(bin(block),'--->',bin(shifted_block))
        # Store the shifted bits in result
        result |= (shifted_block << (i))
        #print(bin(result))
    return result
def C4_rotation_new(number,L):
    '''
    implements C4 rotation.
    1) first half of binary (eta=2) has permutations within each chain  (each block of size L) and then this gets shifted by Nhalf digits to go to eta=1
    2) the second half (eta=1) has chain permutations
    '''
    N = 4 * L**2  # Total bit length
    half_N = N // 2  # Each half has 2L² bits
    result = 0
    #eta=2 part
    for i in range(0,half_N,L):
        # Extract L-bit block
        block = (number >> (i)) & ((1 << L) - 1)
        #reverse block. L=2: xy-->yx and L=3: xyz-->zyx
        reversed_block = 0
        for x in range(L):
            if (block >> x) & 1:
                reversed_block |= (1 << (L - 1 - x))
        shifted_reversed_block = reversed_block << (half_N+i)
        result |= shifted_reversed_block
        #print(bin(result>>18))
    #eta=1 part
    for i in range(0,half_N,2*L):
        # Extract 2L-bit block
        block = (number >> (half_N + i)) & ((1 << 2*L) - 1)
        #shift block
        shifted_block = block << (half_N - 2*L -i)
        result |= shifted_block
        #print(bin(result))
    return result
def compound_transform(action,basis,geometry):
    '''
    given a tuple (s,r,n,m) for a group action act on representative basis states with

    T_1^n T_2^m C_rot^r TR^s

    hence:
    s=0,1 Time reversal
    r=0,1,2,(3) Rotation
    n,m=0,1,2,...,L Translations

    '''
    L = 2
    if geometry == 'square':
        s,r,n,m = action
        for _ in range(m):
            basis = [translation_y_transform_new(state,L) for state in basis]
        for _ in range(n):
            basis = [translation_x_transform_new(state,L) for state in basis]       
        for _ in range(r):
            basis = [C4_rotation_new(state,L) for state in basis]
        if s == 1:
            basis = [time_reversal_transform(state,L=L,N=4**L**2) for state in basis]
    elif geometry == 'triangular':
        raise NotImplementedError
    return basis
##############
##############
##############
def check_spectra(equiv_classes_max = 60):
    '''
    Performs a sanity check that ensures that symmetry sectors in the same equivalence class have eigenstates related by a permutation.
    This requirement can only be failed in the case of eigenstates in a degenerate subspace because in there different numerical diagonalizations will generically give different eigenvectors

    Input:
    equiv_classes_max(int)         :How many equivalence classes to check?
    Output:
    Success/Failure(str)
    '''
    L = 2
    params = {'L':L,'geometry':'square','projection':False}
    if params['geometry'] == 'square':
        config_number = (L+1)**(4*L)
        group_order = 2*2*L**2
    elif params['geometry'] == 'triangular':
        config_number = (L+1)**(6*L)
        group_order = 2*3*L**2
    if params['projection'] == True:
        params['Nel_min'] = 2*(L**2) - 1
        params['Nel_max'] = 2*(L**2) + 1
    ###################################
    #generate all sectors#
    ###################################
    timei = time.time()
    configs = chain_configs(params)
    timef = time.time()
    print('time to generate sectors:',timef-timei,' secs')
    ####################################
    #initialize hamiltonian params for system#
    ####################################
    DATA = configs.compressed_data
    t = 1
    U = 6
    V = 1
    mu = 0
    H_params = {'L':L,'sign':True,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    H_params['geometry'] = params['geometry']
    ####################################
    #in each equivalence class go on and diagonalize all sectors#
    ####################################
    equiv_classes = 0
    for representative_config in DATA:
        equiv_classes +=1
        sectors_list, permutations_list = DATA[representative_config]
        #original basis
        H_params['config'] = representative_config
        representative_chain_instance = chains(H_params)
        representative_basis = [int(el,2) for el in representative_chain_instance.basis]
        eigenstuff = representative_chain_instance.diagonalization()
        representative_eigstates = eigenstuff['vs']
        representative_spectra = eigenstuff['es']
        for (sector, permutation) in zip(sectors_list, permutations_list):
            #if permutation[2] != 0 or permutation[3]!= 0: #i think i have an issue with C4 symmetry????
            #    continue
            H_params['config'] = sector
            chain_instance = chains(H_params)
            basis = [int(el,2) for el in chain_instance.basis]
            transformed_basis = compound_transform(permutation,representative_basis,geometry=params['geometry'])
            basis_perm = np.argsort(transformed_basis)[::-1]
            eigenstuff = chain_instance.diagonalization()
            eigstates = eigenstuff['vs']
            spectra = eigenstuff['es']
            permuted_eigstates = eigstates[np.argsort(np.array(basis_perm)),:]
            #if permutation[2] != 0 or permutation[3]!= 0: #i think i have an issue with C4 symmetry????
            #    if not np.allclose(basis,np.sort(transformed_basis)[::-1]):
            #        print('wrong transform',basis,np.sort(transformed_basis)[::-1])
                    #quit()
                
            if not np.allclose(np.abs(permuted_eigstates),np.abs(representative_eigstates)): #check if eigenstates dont match
                if np.allclose(spectra,representative_spectra):#check if 
                    if not has_duplicates_with_tol(spectra):
                        #print(spectra,permutation)
                        print('spectra matchbut not eigenstates....',permutation,'...do bases match?',np.allclose(basis,np.sort(transformed_basis)[::-1]),'...involves translation?',(permutation[2]==1)or(permutation[3]==1))
                else:
                    print('spectra dont match???')
                    quit()
                        

                        #print('spectra identical, no degeneracies',spectra,len(spectra),np.unique(spectra),len(np.unique(spectra)))
                #else:
                #    print('secs',representative_config,sector,'spectra not identical')
                #print('secs',representative_config,sector,' are spectra identical?',np.allclose(spectra,representative_spectra),'is there degeneracy in spectra?',len(np.unique(spectra)) != len(spectra))
            #if not np.allclose(permuted_eigstates,representative_eigstates):
                #for i in range(permuted_eigstates.shape[0]):
                #    print(permuted_eigstates[:,i],representative_eigstates[:,i])
                #    print('eig dont match')
                #    print(basis==transformed_basis)
                #    print(permutation)
                #    print('*')
                #quit()
        if equiv_classes >= equiv_classes_max:
            quit()
def check_perms(repeat = 10):
    '''
    projection has some bug in it....
    '''
    L = 2
    params = {'L':L,'geometry':'square','projection':False}
    if params['geometry'] == 'square':
        config_number = (L+1)**(4*L)
        group_order = 2*2*L**2
    elif params['geometry'] == 'triangular':
        config_number = (L+1)**(6*L)
        group_order = 2*3*L**2
    if params['projection'] == True:
        params['Nel_min'] = 2*(L**2) - 1
        params['Nel_max'] = 2*(L**2) + 1

    timei = time.time()
    configs = chain_configs(params)
    timef = time.time()
    ####################################
    config_count = 0
    rep_config_count = 0
    for representative_config in configs.compressed_data:
        rep_config_count += 1
        config_count += len(configs.compressed_data[representative_config][0])
    ####################################
    print('TOTAL CONFIGURATIONS:')
    print(config_count,' VS EXPECTED ',config_number)
    print('TOTAL INEQUIVALENT CONFIGURATIONS:',rep_config_count,' vs optimal ', config_number/group_order)
    print('time taken:',timef-timei,' secs')
    print(timef-timei)
    ####################################
    ######symmetry related spectra######
    ####################################
    DATA = configs.compressed_data
    t = 1
    U = 6
    V = 1
    mu = 0
    H_params = {'L':L,'sign':True,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    H_params['geometry'] = params['geometry']
    for _ in range(repeat):
        config = random.choice(list(DATA.keys()))
        config = ((0, 1, 0, 2), (0, 2, 1, 2))
        H_params['config'] = config
        chain_instance = chains(H_params)
        #################
        basis = [int(el,2) for el in chain_instance.basis]
        basis_1000 = [time_reversal_transform(state,L,N=4**L**2) for state in basis]
        basis_0001 = [translation_y_transform_new(state,L) for state in basis]
        basis_0010 = [translation_x_transform_new(state,L) for state in basis]
        basis_0011 = [translation_x_transform_new(state,L) for state in basis_0001]
        basis_0100 = [C4_rotation_new(state,L) for state in basis]
        basis_1001 = [time_reversal_transform(state,L,N=4**L**2) for state in basis_0001]
        basis_1010 = [time_reversal_transform(state,L,N=4**L**2) for state in basis_0010]
        basis_1011 = [time_reversal_transform(state,L,N=4**L**2) for state in basis_0011]
        basis_0111 = [C4_rotation_new(state,L) for state in basis_0011]
        basis_1111 = [C4_rotation_new(state,L) for state in basis_1011]
        print('(0, 0, 0, 1)',basis_0001)
        print('(0, 0, 1, 0)',basis_0010)
        print('(0, 0, 1, 1)',basis_0011)
        print('(0, 1, 0, 0)',basis_0100)
        print('(1, 0, 0, 0)',basis_1000)
        print('(1, 0, 1, 0)',basis_1010)
        print('(1, 0, 1, 1)',basis_1011)
        print('(1, 0, 0, 1)',basis_1001)
        print('(0, 1, 1, 1)',basis_0111)
        print('(1, 1, 1, 1)',basis_1111)
        #################
        if chain_instance.dim > 4:
            continue
        ##############################################
        # generate all symmetry equivalent configs####
        ############## and check spectra #############
        symm_rel_configs = DATA[config][0]
        for i,c_dash in enumerate(symm_rel_configs):
            H_params['config'] = c_dash
            chain_instance = chains(H_params)
            print('------')
            print('config:',c_dash,' and g operator',DATA[config][1][i])
            print(chain_instance.dim,[int(el,2) for el in chain_instance.basis])
            diag_states = chain_instance.diagonalization()
            print(diag_states['vs'])
            print('------')
        quit()
        #print(config,DATA[config][1],len(DATA[config][1]))
    return
#
def ED_exe(parameters,target_dir='/mnt/users/kotssvasiliou/ED/RUNS/BENCHMARKS/',verbose=1):
    '''
    This is a single ED run.

    Diagonalizes the system for a single point in parameter space, if the flag is up.

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
    ##
    #ED
    ##
    if verbose>0:print('='*100+'\n PERFORMING EXACT DIAGONALIZATION ON ALL REPRESENTATIVE SYMMETRY SECTORS \n'+'='*100)
    itime = time.time()
    combined_data = {}
    for k in config_data.keys():
        sector_params['config'] = k
        chain_instance = chains(sector_params)
        diag_states = chain_instance.diagonalization()
        combined_data[k] = {}
        combined_data[k]['es'] = diag_states['es']
        combined_data[k]['vs'] = diag_states['vs']
        combined_data[k]['ns'] = diag_states['ns']
        combined_data[k]['equivalent sectors'] = config_data[k]
        combined_data[k]['weight'] = len(config_data[k][0])
    ftime = time.time()
    if verbose>0:print('time to perform ED:',ftime-itime,' seconds \n'+'='*100)
    ###############################
    #   Initialize thermodynamic  #
    #   calculation instance      #
    ###############################
    parameters['loc'] = chain_instance.loc
    parameters['verbose'] = 1
    parameters['greens function'] = True
    thermo = thermodynamics(parameters=parameters,combined_data=combined_data)
    timei = time.time()
    if verbose>0: print('calculating H & N moments as test')
    thermo.H_moments(beta=8)
    thermo.N_moments(beta=8)
    timef = time.time()
    if verbose>0: print('Time taken:',timef-timei)
    ##################################
    #       GREENS FUNCTION CALC     #
    ##################################
    timei = time.time()
    if verbose>0: print('CALCULATING OR LOADING CREATION OPERATOR MAPPING')
    thermo.gen_creation_mapping(load=True)
    timef = time.time()
    if verbose>0: print('Time taken:',timef-timei)
    timei = time.time()
    print(thermo.map)
    beta = 10
    taus = np.linspace(0,1,num=20)*beta
    print(thermo.greens_func_element(index_1=0,index_2=1,g=(1,0,0,0),taus = taus,beta=beta))
    timef = time.time()
    print(timef-timei)
    quit()
    def print_h5_structure(h5obj, indent=0):
        for key in h5obj:
            if indent == 0 and key != 'cdagger_op_1_0_0': quit()
            item = h5obj[key]
            print("  " * indent + f"- {key}")
            if isinstance(item, h5py.Group):
                print_h5_structure(item, indent + 1)
    print_h5_structure(thermo.cdagger_map)
def Greens_function_calculation(index_1,index_2,parameters,beta,Ntau,target_dir='/mnt/users/kotssvasiliou/ED/RUNS/BENCHMARKS/',verbose=1):
    '''
    This is a single ED run that also caluclates a *single* Green's function component

    Input:
        index_1/2(int)          :The indices of the Greens function to be calculated.
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
        greens_func(np.array)   :An np.array of shape Ntau x (2*Nflav*L**2) x (2*Nflav*L**2)
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
    ##
    #ED
    ##
    if verbose>0:print('='*100+'\n PERFORMING EXACT DIAGONALIZATION ON ALL REPRESENTATIVE SYMMETRY SECTORS \n'+'='*100)
    itime = time.time()
    combined_data = {}
    for k in config_data.keys():
        sector_params['config'] = k
        chain_instance = chains(sector_params)
        diag_states = chain_instance.diagonalization()
        combined_data[k] = {}
        combined_data[k]['es'] = diag_states['es']
        combined_data[k]['vs'] = diag_states['vs']
        combined_data[k]['ns'] = diag_states['ns']
        combined_data[k]['equivalent sectors'] = config_data[k]
        combined_data[k]['weight'] = len(config_data[k][0])
    ftime = time.time()
    if verbose>0:print('time to perform ED:',ftime-itime,' seconds \n'+'='*100)
    ###############################
    #   Initialize thermodynamic  #
    #   calculation instance      #
    ###############################
    parameters['loc'] = chain_instance.loc
    parameters['verbose'] = 1
    parameters['greens function'] = True
    thermo = thermodynamics(parameters=parameters,combined_data=combined_data)
    ##################################
    #       GREENS FUNCTION CALC     #
    ##################################
    timei = time.time()
    if verbose>0: print('CALCULATING OR LOADING CREATION OPERATOR MAPPING')
    thermo.gen_creation_mapping(load=True)
    timef = time.time()
    if verbose>0:
        print('Time taken:',timef-timei)
        print('MAPPPPPPP',thermo.map)
    taus = np.linspace(0,1,num=Ntau)*beta
    #quit()
    #loop through group elements
    for t1 in range(L):#t1 translations
        for t2 in range(L):#t2 translations
            for r in range(4):#rotations
                for s in range(2):#tr
                    group_element = (s,r,t1,t2)
                    timei = time.time()
                    if verbose>0:print('CACLULATING GREENS FUNCTION TERM FOR GROUP ELEMENT',group_element)
                    thermo.greens_func_element(index_1=index_1,index_2=index_2,g=group_element,taus=taus,beta=beta)
                    timef = time.time()
                    if verbose>0:print('Time(s):',timef-timei,'\n')
    print('GFUNK',thermo.Gfunc[beta][:,index_1,index_2])
    ##########
    #save green's functions
    ##########
    picklefile = target_dir +'greens_function_000.pkl'
    if os.path.exists(picklefile):
        with open(picklefile, 'rb') as f:
            G = pickle.load(f)
    else:
        print('new greends func')
        G =  np.zeros((taus.shape[0],2*thermo.Nflav*thermo.L**2,2*thermo.Nflav*thermo.L**2),dtype=complex) 
    G[:,index_1,index_2] = thermo.Gfunc[beta][:,index_1,index_2]
    with open(picklefile,'wb') as f:
        pickle.dump(G,f)
    
    return
def has_duplicates_with_tol(arr, tol=1e-8):
    '''
    True: Spectrum has degeneracy
    False: Doens't have degeneracy
    '''
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if np.isclose(arr[i], arr[j], atol=tol):
                return True
    return False

def translation_spectra(equiv_classes_max=60):
    '''
    Performs a sanity check that ensures that symmetry sectors in the same equivalence class have eigenstates related by a permutation.
    This requirement can only be failed in the case of eigenstates in a degenerate subspace because in there different numerical diagonalizations will generically give different eigenvectors

    Input:
    equiv_classes_max(int)         :How many equivalence classes to check?
    Output:
    Success/Failure(str)
    '''
    L = 2
    params = {'L':L,'geometry':'square','projection':False}
    if params['geometry'] == 'square':
        config_number = (L+1)**(4*L)
        group_order = 2*2*L**2
    elif params['geometry'] == 'triangular':
        config_number = (L+1)**(6*L)
        group_order = 2*3*L**2
    if params['projection'] == True:
        params['Nel_min'] = 2*(L**2) - 1
        params['Nel_max'] = 2*(L**2) + 1
    ###################################
    #generate all sectors#
    ###################################
    timei = time.time()
    configs = chain_configs(params)
    timef = time.time()
    print('time to generate sectors:',timef-timei,' secs')
    ####################################
    #initialize hamiltonian params for system#
    ####################################
    DATA = configs.compressed_data
    t = 1
    U = 6
    V = 1
    mu = 0
    H_params = {'L':L,'sign':True,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    H_params['geometry'] = params['geometry']
    ####################################
    #in each equivalence class go on and diagonalize all sectors#
    ####################################
    equiv_classes = 0
    for representative_config in DATA:
        equiv_classes +=1
        sectors_list, permutations_list = DATA[representative_config]
        #original basis
        H_params['config'] = representative_config
        representative_chain_instance = chains(H_params)
        representative_basis = [int(el,2) for el in representative_chain_instance.basis]
        eigenstuff = representative_chain_instance.diagonalization()
        representative_eigstates = eigenstuff['vs']
        representative_spectra = eigenstuff['es']
        for (sector, permutation) in zip(sectors_list, permutations_list):
            #if permutation[1] != 0:
            #    continue
            H_params['config'] = sector
            chain_instance = chains(H_params)
            basis = [int(el,2) for el in chain_instance.basis]
            transformed_basis = compound_transform(permutation,representative_basis,geometry=params['geometry'])
            basis_perm = np.argsort(transformed_basis)[::-1]
            eigenstuff = chain_instance.diagonalization()
            eigstates = eigenstuff['vs']
            spectra = eigenstuff['es']
            permuted_eigstates = eigstates[np.argsort(np.array(basis_perm)),:]
            if not np.allclose(basis,np.sort(transformed_basis)[::-1]):
                print('bases dont agree',permutation)
        if equiv_classes > equiv_classes_max:
            quit()

##############
def plot_mu_vs_N(parameters,verbose=1):
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
        
    betas = [4,8]
    mus = np.linspace(-3,3,num=120)*U
    N = np.zeros((2,120),dtype=float)
    for i,mu in enumerate(mus):
        ##
        #ED
        ##
        sector_params = {'L':L,'geometry':geometry,'sign':sign,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':mode}}
        if verbose>0:print('='*100+'\n PERFORMING EXACT DIAGONALIZATION ON ALL REPRESENTATIVE SYMMETRY SECTORS \n'+'='*100)
        itime = time.time()
        combined_data = {}
        for k in config_data.keys():
            sector_params['config'] = k
            chain_instance = chains(sector_params)
            diag_states = chain_instance.diagonalization()
            combined_data[k] = {}
            combined_data[k]['es'] = diag_states['es']
            combined_data[k]['vs'] = diag_states['vs']
            combined_data[k]['ns'] = diag_states['ns']
            combined_data[k]['equivalent sectors'] = config_data[k]
            combined_data[k]['weight'] = len(config_data[k][0]) 
        ftime = time.time()
        if verbose>0:print('time to perform ED:',ftime-itime,' seconds \n'+'='*100)
        ###############################
        #   Initialize thermodynamic  #
        #             #   calculation instance      #
        ###############################
        parameters['loc'] = chain_instance.loc
        parameters['verbose'] = 1
        parameters['greens function'] = False
        thermo = thermodynamics(parameters=parameters,combined_data=combined_data)
        N[0,i] = thermo.N_moments(beta=betas[0])[0]
        N[1,i] = thermo.N_moments(beta=betas[1])[0]
    plt.plot(mus,N[0,:],'.-',color='blue',label='$\\beta=4$')
    plt.plot(mus,N[1,:],'.-',color='red',label='$\\beta=8$')
    plt.legend()
    plt.xlabel('$\mu/U$')
    plt.ylabel('$\langle N \\rangle$')
    plt.savefig('/mnt/users/kotssvasiliou/ED/figures/mN_vs_mu_newwww.png')
    return                   
def check_hamiltonian(parameters,config):
    '''
    code to explicitly check the construction of a Hamiltonian
    Input:
        parameters(dict)        :The system parameters
        config(tuple)           :The symmetry sector to test
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
    system_params = {'geometry':geometry,'L':L,'partial':partial,'projection':projection}
    sector_params = {'L':L,'geometry':geometry,'sign':sign,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':mode}}
    print('HAMILTONIAN PARAMETERS(t,U,V,mu)',t,U,V,mu)
    timei = time.time()
    CCs = chain_configs(params=system_params)
    CCs.sector_check()
    config_data = CCs.compressed_data
    timef = time.time()
    sector_params['config'] = config
    chain_instance = chains(sector_params)
    print('BASIS',chain_instance.basis)
    H = chain_instance.configuration_Hamiltonian().toarray()
    print(H)
    plt.imshow(H)
    plt.colorbar()
    plt.savefig('tttttttt.png')
    return
##############
if __name__ == "__main__":
    parameters = {'L':2,
                  'geometry':'square',
                  't':1,
                  'mu':2,
                  'U':4,
                  'V':1,
                  'partial':False,
                  'projection':False,
                  'sign':True,
                  'JW string':True,
                  'mode':'full'}
    #check_hamiltonian(parameters=parameters,config=((1,0,0,0),(1,0,1,0)))
    index_1 = int(sys.argv[1])
    index_2 = int(sys.argv[2])
    print('doing indices',index_1,index_2)
    Greens_function_calculation(index_1=index_1,index_2=index_2,parameters=parameters,beta=10,Ntau=10,verbose=1)
    quit()
    #plot_mu_vs_N(parameters=parameters)
    ED_exe(parameters=parameters)
    #########################################
    #check_spectra()

    #####################################
    #checking symmetry action on *states*
    #In = 2863
    #Out = C4_rotation_new(In,L=2)
    #print(bin(In))
    #print(bin(Out),Out)
    #In1 = '0b111111111111111111010001000000111111'
    #In2 = '0b010001000000111111111111111111111111'
    #Out = C4_rotation(int(In1,2),L=3)
    #Out = translation_y_transform_new(int(In1,2),L=3)
    #Out = time_reversal_transform(int(In1,2), L=3, N=4*3**2)
    #print(In2)
    #print(bin(Out))
    #print(bin(Out&(1<<18 -1)))
    #quit()

    ################################
    #checking permutations
    #check_perms()
