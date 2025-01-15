import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from scipy.sparse import csr_matrix,coo_matrix #optimizes H . v operations. to check if H already row sparse, do  isspmatrix_csr(H)
from scipy.sparse.linalg import eigsh
import time
import pickle
import argparse
from itertools import product
from math import comb,log10
from collections import Counter
'''
Some test code for a system of 1D chains:


|   |   |   |               |   |   |   |               |   |   |   |       
|   |   |   |               |   |   |   |               |   |   |   | 
|   |   |   |               |   |   |   |               |   |   |   | 
|   |   |   |               |   |   |   |               |   |   |   | 
|   |   |   |       x       |   |   |   |       x       |   |   |   | 
|   |   |   |               |   |   |   |               |   |   |   | 
|   |   |   |               |   |   |   |               |   |   |   | 
|   |   |   |               |   |   |   |               |   |   |   | 

   flavor 1                    flavor 2                      flavor 3

These are permutted by C3 and each one has its own action under T1,T2

For L1 != L2 things are tricky because how does the C3 action make sense? chains along one direction differ in amount and in how many electrons they fit depending on the direction (flavor)


A given sector is labelled by {n_i,s} with i the chain number and s the spin. we have 0=<n_{i,s} <= L (length of each chain)

This system has
1) T1,T2
2) C_3
3) Spin Inversion

So what i want to do is enumerate all possible combos and count the inequivalent configurations.
ie if g \in G does g( {n_i,s} ) =  {m_i,s} then in a sense  {n_i,s} ~ {m_i,s}.
Further, a state that you get multiple times after the action of g 's has to be weighed differently

We are not using translational symmetry the usual way because of the way we are build the basis.

The total number of sectors here is (1+L)**(6*L)
and i want to see the effective number of sectors, ie those sectors not related by any g \in G.
------------------------------------------------------------------------------------------------
'''
def section_generator(L):
    '''
    Args:
        L(int): The number of chains in the system (per flavor)
        L(int): The sites in each chain
        Equal in this case. Need to generalize
    Returns:
        All possible configurations
        A given configuration(section) has the form (n_1,n_2,.....,n_3L)
    Convention: First L numbers are for the first valley parallel to a1, [L->2L] are second valley parallel to a2, [2L->3L] are third valley parallel to a3.
    '''
    electron_count = range(L + 1)
    configs_spinup = list(product(electron_count, repeat=3*L))
    configs_spindown = list(product(electron_count, repeat=3*L))
    configs =[(c_up,c_down) for c_up in configs_spinup for c_down in configs_spindown]
    return configs
def T(s,t1,t2):
    '''
    Takes a configuration s and acts on it with the tranlsation operator

    T^t2_y   T^t1_x 

    Translates it by m sites.

    Args:
        s(tuple): configuration
        m(int): number of sites to translate m \in [0,L1)
    Returns:
    '''
    s_up,s_down = s[0],s[1]
    s_up_t = tuple([s_up[i-m] for i in range(len(s_up))])
    s_down_t = tuple([s_down[i-m] for i in range(len(s_down))])
    return (s_up_t,s_down_t)