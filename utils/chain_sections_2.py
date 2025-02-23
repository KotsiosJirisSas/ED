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

warnings.simplefilter("ignore", SparseEfficiencyWarning) #supress warning???

class chain_configs():
    def __init__(self, params):
        self.L = params['L']
        self.geometry = params['geometry']
        self.Nchains = 3 * self.L if self.geometry == 'triangular' else quit()
        self.configurations = []
        self.equivalence_classes = {}  # Maps a representative configuration to all symmetry-related ones
        self.permutation_map = {}  # Maps a representative to its symmetry operations (in cycle notation)
        self.weights = {}

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
                        cycle_notation_perm = self.cycle_notation(perm)

                        transformed_configs.append(transformed)
                        permutations.append(cycle_notation_perm)

        # Find the lexicographically smallest configuration to serve as the representative
        rep_config = min(transformed_configs)
        
        # Debugging: Check how many symmetry-related elements exist
        weight = len(set(transformed_configs))  # Number of unique symmetry-related states

        # Store the equivalence class and weight
        if rep_config not in self.equivalence_classes:
            self.equivalence_classes[rep_config] = []
            self.permutation_map[rep_config] = []
            self.weights[rep_config] = weight  # Store weight

        for trans_config, perm in zip(transformed_configs, permutations):
            if trans_config not in self.equivalence_classes[rep_config]:
                self.equivalence_classes[rep_config].append(trans_config)
                self.permutation_map[rep_config].append(perm)


    def identity_permutation(self):
        """Returns the identity permutation."""
        return list(range(2 * self.Nchains))

#######
if __name__ == "__main__":
    params = {}
    params['L'] = 2
    params['geometry'] = 'triangular'
    chain = chain_configs(params)
    #print(chain.identity_permutation())
    #print(chain.spin_inversion_permutation())
    ##print(chain.C3_permutation())
    #print(chain.translation_permutation(1,1))
    #print(chain.cycle_notation())
    #c3_perm = chain.C3_permutation()
    #print("DEBUG: C3 Permutation:", c3_perm)
    #print("C3 Rotation (Cycle Notation):", chain.cycle_notation(c3_perm))
    #tr_perm = chain.spin_inversion_permutation()
    #print("DEBUG: TR Permutation:", tr_perm)
    #print("TR Rotation (Cycle Notation):", chain.cycle_notation(tr_perm))
    #t10_perm = chain.translation_permutation(1,0)
    #print("T10  (Cycle Notation):", chain.cycle_notation(t10_perm))
    #t01_perm = chain.translation_permutation(0,1)
    #print("T01  (Cycle Notation):", chain.cycle_notation(t01_perm))
    #t11_perm = chain.translation_permutation(1,1)
    #print("T11  (Cycle Notation):", chain.cycle_notation(t11_perm))
    #quit()
    time0 = time.time()
    chain.generate_all_configurations()
    timef = time.time()
    print(timef-time0)
    print(len(chain.equivalence_classes))
    print(chain.weights.values())
    #print('configs',chain.configurations)
    #for k in chain.equivalence_classes.keys():
    #    print('key',k)
    #    print('configs',chain.equivalence_classes[k])
    #print('permutation map',chain.permutation_map)