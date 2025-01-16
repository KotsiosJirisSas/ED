import os
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
16 Jan Log:
    To do:
            0) Just a bit more testing
            1) Section reduction function
            2) configuration to index
            3) Account for weights
            4) Sketch solution for L=3 or (2x3) system...
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
    #################################
    L=1: #~64      time = 0s
    L=2: #~5*1e5   time = 0.06s
    L=3: #~7*1e10  time ~ 1.5 hours process is killed, maybe too much memory. would require about 24TB of memory woOoOah

    memory= # x 4 bytes per integer x 2*3*L integers for a configuration. + overhead for tuples (overhead is 3x memory to store a tuple so its bad but its order 1 bad)
    L=1: ~1e-6 GB
    L=2: ~1e-2 GB
    L=3: ~1e+4 GB
    L=4: ~1e+10 GB

    Clearly for L=3 to stand a chance we have to dynamically create the configurations. Start from lowest, map all the ones related by symmetry, then generate second lowest (not related by symmetry etc etc)
    So at each process, we are only storing integers. Also after we are done at each step, don't sort them but save them in a dictionary or something, because just storing 4**18 numbers needs ~100GB.
    
    The result we actually care about will be about the 4**18/ 10 ~ 10**9 unique configurations and their weights.
    If we do this smartly we can get final memory of ~<10 GBs
    '''
    electron_count = range(L + 1)
    configs_spinup = list(product(electron_count, repeat=3*L))
    configs_spindown = list(product(electron_count, repeat=3*L))
    configs =[(c_up,c_down) for c_up in configs_spinup for c_down in configs_spindown]
    return configs
def section_generator_partial(L, n):
    """
    Generates the first n configurations for a given L, saves them temporarily to a file,
    and returns them as a list of tuples. The temporary file is deleted after use.

    Args:
        L (int): The system size (number of chains per valley and sites per chain).
        n (int): The number of configurations to generate.

    Returns:
        list: A list of configurations (tuples) equivalent to section_generator(L).
    """
    # Temporary file to store configurations
    temp_file = "configurations_temp.txt"

    # Step 1: Generate configurations and save to file
    electron_count = range(L + 1)
    count = 0
    with open(temp_file, 'w') as f:
        for c_up in product(electron_count, repeat=3 * L):
            for c_down in product(electron_count, repeat=3 * L):
                config = (c_up, c_down)
                f.write(str(config) + '\n')
                count += 1
                if count >= n:
                    break
            if count >= n:
                break

    # Step 2: Read configurations back into a list of tuples
    configurations = []
    with open(temp_file, 'r') as f:
        for line in f:
            configurations.append(eval(line.strip()))  # Convert string back to tuple

    # Step 3: Delete the temporary file
    os.remove(temp_file)

    return configurations
def T(s,n,m,L):
    '''
    Takes a configuration s and acts on it with the translation operator
    T^m_2   T^n_1 
    Translates it by n sites along a_1 and m sites along a_2.

    Args:
        s(tuple):               The configuration of chains
        n(int \in [0,L)):       Translation about a_1
        m(int \in [0,L)):       Translation about a_2
    Returns:
        s'(tuple):              The transformed configuration
    '''
    s_up,s_down = s[0],s[1]
    s_up_1 = s_up[:L]
    s_up_2 = s_up[L:2*L]
    s_up_3 = s_up[2*L:3*L]
    s_down_1 = s_down[:L]
    s_down_2 = s_down[L:2*L]
    s_down_3 = s_down[2*L:3*L]
    #check len(s_up_1) == L

    s_up_1 = [s_up_1[i-m] for i in range(L)]
    s_up_2 = [s_up_2[i-n] for i in range(L)]
    s_up_3 = [s_up_3[i-(n+m)] for i in range(L)]
    s_down_1 = [s_down_1[i-m] for i in range(L)]
    s_down_2 = [s_down_2[i-n] for i in range(L)]
    s_down_3 = [s_down_3[i-(n+m)] for i in range(L)]
    s_up_t = tuple(s_up_1+s_up_2+s_up_3)
    s_down_t = tuple(s_down_1+s_down_2+s_down_3)
    return (s_up_t,s_down_t)
def T_old(s, n, m, L):
    """
    supposedly faster but at least for 10**5 elements, my code is slightly faster.
    #############
    Applies the translation operator to a configuration.
    
    Args:
        s (tuple): configuration, containing two tuples of integers (s_up, s_down).
        n (int): number of sites to translate in the x-direction.
        m (int): number of sites to translate in the y-direction.
        L (int): number of sites in one chain direction.

    Returns:
        tuple: Translated configuration (s_up_t, s_down_t).
    """
    s_up, s_down = s
    
    # Extract slices for s_up and s_down
    s_up_1, s_up_2, s_up_3 = s_up[:L], s_up[L:2*L], s_up[2*L:3*L]
    s_down_1, s_down_2, s_down_3 = s_down[:L], s_down[L:2*L], s_down[2*L:3*L]
    
    # Translate each segment using modular arithmetic
    s_up_1 = [s_up_1[(i - m) % L] for i in range(L)]
    s_up_2 = [s_up_2[(i - n) % L] for i in range(L)]
    s_up_3 = [s_up_3[(i - (n + m)) % L] for i in range(L)]
    s_down_1 = [s_down_1[(i - m) % L] for i in range(L)]
    s_down_2 = [s_down_2[(i - n) % L] for i in range(L)]
    s_down_3 = [s_down_3[(i - (n + m)) % L] for i in range(L)]

    # Combine translated segments and return as tuples
    s_up_t = tuple(s_up_1 + s_up_2 + s_up_3)
    s_down_t = tuple(s_down_1 + s_down_2 + s_down_3)

    return (s_up_t, s_down_t)
def C3(s,L):
    '''
    Implements C3 rotation which cycles through the chain directions while also permuting the order withing a chain type

    (1,x)----->(2,x)
    (2,x)----->(3,L-x)
    (3,x)----->(1,L-x)

    Args:
        s(tuple):               The configuration of chains
        L(int):                 System size
    Returns:
        s'(tuple):              The transformed configuration
    '''
    s_up,s_down = s[0],s[1]
    s_up_1 = s_up[:L]
    s_up_2 = s_up[L:2*L]
    s_up_3 = s_up[2*L:3*L]
    s_down_1 = s_down[:L]
    s_down_2 = s_down[L:2*L]
    s_down_3 = s_down[2*L:3*L]

    # Rotate the chains:
    # a_1 -> a_2 (unchanged order)
    # a_2 -> a_3 (reversed order)
    # a_3 -> a_1 (reversed order)
    #s_up_1_rot = s_up_3[::-1]

    s_up_1_rot = tuple([s_up_3[L-i-1] for i in range(L)]) #could do tuple(s_up[::-1]) instead... but for such small lists difference is marginal...
    s_up_2_rot = s_up_1
    s_up_3_rot = tuple([s_up_2[L-i-1] for i in range(L)])
    s_down_1_rot = tuple([s_down_3[L-i-1] for i in range(L)])
    s_down_2_rot = s_down_1
    s_down_3_rot = tuple([s_down_2[L-i-1] for i in range(L)])

    s_up_rot = tuple(s_up_1_rot+s_up_2_rot+s_up_3_rot)
    s_down_rot = tuple(s_down_1_rot+s_down_2_rot+s_down_3_rot)

    return (s_up_rot,s_down_rot)
def S_inv(s):
    '''
    exchanges the two tuples
    Equivalent to S_z -> S_(-z)
    '''
    return (s[1],s[0])
def GroupAction(s,n,m,m_c3,m_spin,L):
    '''
    Acts on a configuration with a generic element from the symmetry group

    Args:
        s(tuple):               Original configuration
        L(int):                 System size
        n(int \in [0,L)):       Translation about a_1
        m(int \in [0,L)):       Translation about a_2
        m_c3(int \in [0,1,2]):  C3 rotation
        m_spin(int \in [0,1]):  Spin inversion


    Returns:
        s'(tuple):              The transformed configuration
    '''
    s_temp = s
    s_temp = T(s_temp,n,m,L)
    for x in range(m_c3):
        s_temp = C3(s_temp,L)
    for y in range(m_spin):
        s_temp = S_inv(s_temp)
    return s_temp

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#################################################### TERMINAL FUNCTIONS #############################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
def feature1():
    for L in [1,2,3]:
        stime = time.time()
        configs = section_generator(L)
        print(len(configs))
        print((L+1)**(6*L))
        ftime = time.time()
        print(f"elapsed time: {ftime-stime:.2f} seconds")
    return
def feature2():
    '''
    translation test that two functions are the same
    '''
    L = 2
    configs = section_generator(L)
    itime = time.time()
    for i,config in enumerate(configs):
        if i % 10000 ==0:
            print(i)
        for m in range(2):
            for n in range(2):
                config1 = T(config,n,m,L)
                config2 = T_old(config,n,m,L)
                if config2!=config1:
                    print('uhuhh')
                    quit()
                
    return
def feature3():
    '''
    translation test that times two translation functions
    '''
    L = 2
    configs = section_generator(L)
    itime = time.time()
    for i,config in enumerate(configs):
        for m in range(2):
            for n in range(2):
                T(config,n,m,L)

    ftime = time.time()
    print(f"elapsed time: {ftime-itime:.2f} seconds")
    itime = time.time()
    for i,config in enumerate(configs):
        for m in range(2):
            for n in range(2):
                T_old(config,n,m,L)
    ftime = time.time()
    print(f"elapsed time: {ftime-itime:.2f} seconds")
    return
def feature4():
    '''
    For L=2 How long does it take to apply all g's on all configurations?
    Takes about ~1.5 mins.
    '''
    L = 2
    configs = section_generator(L)
    itime = time.time()
    for i,config in enumerate(configs):
        for m in range(2):
            for n in range(2):
                for m_c3 in range(3):
                        for m_spin in range(2):
                            GroupAction(config,n,m,m_c3,m_spin,L)

    ftime = time.time()
    print(f"elapsed time: {ftime-itime:.2f} seconds")
    return
def feature5():
    '''
    This is to test the operators and their commutator relationships
    '''
    L = 12
    n = 10000
    partial = True
    if partial == True:
        configs = section_generator_partial(L,n)
    else:
        configs = section_generator(L)
    itime = time.time()
    '''
    checks C3T1C3^-1 == T2^-1
    '''
    print('='*50)
    print('='*20,'Testing','='*21)
    print('='*50)
    for i,config in enumerate(configs):
        config_temp = config
        config_temp = GroupAction(config_temp,n=0,m=0,m_c3=2,m_spin=0,L=L)
        config_temp = GroupAction(config_temp,n=1,m=0,m_c3=1,m_spin=0,L=L)
        config_temp_alt = GroupAction(config,n=0,m=L-1,m_c3=0,m_spin=0,L=L)
        if config_temp != config_temp_alt:
            print('test1:Fail')
            print('config',i)
            print(config_temp)
            print(config_temp_alt)
            quit()
    print('Test 1 Success')
    '''
    checks C3T2C3^-1 == T1 x T2^-1
    '''
    for i,config in enumerate(configs):
        config_temp = config
        config_temp = GroupAction(config_temp,n=0,m=0,m_c3=2,m_spin=0,L=L)
        config_temp = GroupAction(config_temp,n=0,m=1,m_c3=1,m_spin=0,L=L)
        config_temp_alt = GroupAction(config,n=1,m=L-1,m_c3=0,m_spin=0,L=L)
        if config_temp != config_temp_alt:
            print('test2:Fail')
            print('config',i)
            print(config_temp)
            print(config_temp_alt)
            quit()
    print('Test 2 Success')
    '''
    checks C^3_3 == id
    '''
    for i,config in enumerate(configs):
        config_temp = config
        config_temp = GroupAction(config_temp,n=0,m=0,m_c3=3,m_spin=0,L=L)
        if config_temp != config:
            print('test3:Fail')
            print('config',i)
            print(config_temp)
            print(config)
            quit()
    print('Test 3 Success')
    ftime = time.time()
    print('='*50)
    print('='*20,'Done Test','='*19)
    print('='*50)
    print(f"elapsed time: {ftime-itime:.2f} seconds")
    return
def feature6():
    L=3
    n=100
    configurations = section_generator_partial(L, n)
    print(configurations)
    return
def feature7():
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
    elif args.feature == "feature3":
        feature3()
    elif args.feature == "feature4":
        feature4()
    elif args.feature == "feature5":
        feature5()
    elif args.feature == "feature6":
        feature6()
    elif args.feature == "feature7":
        feature7()