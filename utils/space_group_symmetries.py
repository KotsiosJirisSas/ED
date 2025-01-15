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


|   |   |   |
|   |   |   |
|   |   |   |
|   |   |   |
|   |   |   |

A given sector is labelled by {n_i,s} with i the chain number and s the spin. we have 0=<n_{i,s} <= L_2 (length of each chain)

This system only has translation and inversion.
Under translation, T^{m} n_{i,s} = n_{i+m,s} (m=0...L_1 and i+m is really i+m % L_1) (L_1 is number of chains)

So what i want to do is enumerate all possible combos and count the inequivalent configurations.
ie if g \in G does g( {n_i,s} ) =  {m_i,s} then in a sense  {n_i,s} ~ {m_i,s}.
Further, a state that you get multiple times after the action of g 's has to be weighed differently

We are not using translational symmetry the usual way because of the way we are build the basis.

The total number of sectors here is (1+L_2)**(2*L_1)
and i want to see the effective number of sectors, ie those sectors not related by any g \in G.
------------------------------------------------------------------------------------------------
1) Generate all tuples.
2) Have maps for mapping of tuples
'''
def section_generator(L1,L2):
    '''
    Args:
        L1(int): The number of chains in the system
        L2(int): The sites in each chain
    Returns:
        All possible configurations
        A given configuration(section) has the form [n_1,n_2,.....,n_L1]
    '''
    electron_count = range(L2 + 1)
    configs_spinup = list(product(electron_count, repeat=L1))
    configs_spindown = list(product(electron_count, repeat=L1))
    configs =[(c_up,c_down) for c_up in configs_spinup for c_down in configs_spindown]
    return configs
def T1(s,m):
    '''
    Takes a configuration s and Translates it by m sites.

    Args:
        s(tuple): configuration
        m(int): number of sites to translate m \in [0,L1)
    Returns:
    '''
    s_up,s_down = s[0],s[1]
    s_up_t = tuple([s_up[i-m] for i in range(len(s_up))])
    s_down_t = tuple([s_down[i-m] for i in range(len(s_down))])
    return (s_up_t,s_down_t)
def Inv(s):
    '''
    Takes configfuration s and inverts it about the centre
    Could just done [::-1]
    Args:

    Returns:
    '''
    s_up,s_down = s[0],s[1]
    s0 = len(s_up)
    s_up_inv = tuple([s_up[s0-i-1] for i in range(s0)])
    s_down_inv = tuple([s_down[s0-i-1] for i in range(s0)])
    return (s_up_inv,s_down_inv)
def SpinInv(s):
    '''
    exchanges the two tuples
    '''
    s_up,s_down = s[0],s[1]
    return (s_down,s_up)
def group_action(s,m_t,m_inv,m_spin_inv):
    '''
    
    Args:
        s(tuple): Initial state
        m_t(int): How many times to apply translation; m \in [0,L1)
        m_inv(int): How many times to apply inversion; m \in [0,1]
        m_spin_inv(int): How many times to apply spin inversion; m \in [0,1]
    Returns:
        s'(tuple):The state after acting on it with the group
    '''

    s_temp = s
    s_temp = T1(s_temp,m=m_t)
    ind_inv = 0
    while ind_inv < m_inv:
        s_temp = Inv(s_temp)
        ind_inv += 1

    ind_spin_inv = 0
    while ind_spin_inv < m_spin_inv:
        s_temp = SpinInv(s_temp)
        ind_spin_inv += 1

    return s_temp

def c2i(config,L1,L2):
    """
    Find the index of a given configuration.

    Args:
        config (tuple): A configuration in the form [(red_config), (blue_config)].
        N (int): Maximum number of balls of each color per box.
        L (int): Number of boxes in the chain.

    Returns:
        int: The index of the configuration.
    """
    # Unpack the red and blue configurations
    red_config, blue_config = config

    # Convert the tuple to a unique index
    base = L2 + 1
    index_red = sum(r * (base ** i) for i, r in enumerate(reversed(red_config)))
    index_blue = sum(b * (base ** i) for i, b in enumerate(reversed(blue_config)))
    
    # Combine the two indices
    total_index = index_red * (base ** L1) + index_blue
    
    return total_index
def reweight(dict):
    '''
    
    '''
    dict_out = {}
    for k in dict.keys():
        lst = dict[k]
        element_counts = Counter(lst)
        unique_elements = list(element_counts.keys())
        multiplicities = list(element_counts.values())
        if not multiplicities.count(multiplicities[0]) == len(multiplicities): 
            print('uhhhhhhhhhhhhhhhhh')
            quit()
        weight = multiplicities[0]**2
        dict_out[k] = (unique_elements,weight)
    return dict_out
def section_reduction(L1,L2):
    configs = section_generator(L1,L2)
    if len(configs) == (1+L2)**(2*L1):
        Tot_number = (1+L2)**(2*L1)
    else:
        print('?')
        quit()
    Equivalence_classes = {}
    mask = [True]*len(configs) # checks if this state has been 'met' before
    for i,c in enumerate(configs):
        if mask[i] == False:
            continue
        Equivalence_classes[i] = []
        for m_t in range(L1):
            for m_inv in range(2):
                for m_spin_inv in range(2):
                    c_new = group_action(c,m_t,m_inv,m_spin_inv)
                    i_new = c2i(c_new,L1,L2)
                    Equivalence_classes[i].append(i_new)
                    mask[i_new] = False
    Equivalence_classes = reweight(Equivalence_classes)
    Reduced_number = len(Equivalence_classes)
    print(Tot_number,Reduced_number,Tot_number/Reduced_number)
    return Equivalence_classes
################################
def feature1():
    L1 = 3
    L2 = 8
    #L1 = 2
    #L2 = 1
    configs = section_generator(L1=L1,L2=L2)
    secs = section_reduction(L1=L1,L2=L2)
    Tot_number = (1+L2)**(2*L1)
    multiplicity = 0
    #for i in secs.keys():
    #    print(configs[i])
    #    multiplicity += secs[i][1]
    #print(Tot_number)
    #print(multiplicity)
def feature2():
    for ell1 in range(2,5):
        for ell2 in range(2,5):
            tot,red = section_reduction(L1=ell1,L2=ell2)
            plt.scatter(ell1*ell2,tot/red,c='r')
            plt.scatter(ell1*ell2,2*ell1,c='b')
    plt.savefig('ttt.png')
    return
################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multiple features.")
    parser.add_argument("feature", type=str, choices=["feature1", "feature2","feature3","feature4","feature5", "feature6","feature7","feature8"], help="Feature to run")

    args = parser.parse_args()

    if args.feature == "feature1":
        feature1()
    elif args.feature == "feature2":
        feature2()

