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
Code that can generate all symmetry-inequivalent configurations of chain occupations.
These unique configurations can then be fed to a different function to create their Hamiltonian and diagonalize.
For 2x2 system, there are ~ 5*1e5 different configurations, but only ~ 2*1e4 unique ones.
For 3x3 system, there are ~ 7*1e10 different configurations, but ~ 1*1e9 unique ones.

The optimal reduction is by the order of the symmetry group which i think is 6L**2 (3 from C3, 2 from spin-inversion, L**2 from translations)


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
These should scale like ~ (1+L)**(6*L)/(6L**2)
------------------------------------------------------------------------------------------------
17 Jan Log:
   Everything is working great.
   Tested the action of the symmetries on arbitrary L x L systems.
   To do:
            1) Figure out how to consistencly define the chains for a L1 x L2 system with L1 != L2
            2) Figure out how to dynamically generate the configurations for systems larger than 2x2.
'''
def section_generator(L):
    '''
    Generates all configurations for a system of size L
    Args:
        L(int):         The number of chains in the system (per flavor)
        L(int):         The sites in each chain
        (Equal in this case. Need to generalize)
    Returns:
        All possible configurations
        A given configuration(section) has the form (n_1,n_2,.....,n_3L)
    
    
    ###########################################################################################################################################################
    Convention: First L numbers are for the first valley parallel to a1, [L->2L] are second valley parallel to a2, [2L->3L] are third valley parallel to a3.
    
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
        L (int):        The system size (number of chains per valley and sites per chain).
        n (int):        The number of configurations to generate.

    Returns:
        list:           A list of configurations (tuples) equivalent to section_generator(L).
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
def c2i(config,L):
    """
    Finds the index of a given configuration.
    *add explanation for mapping*

    Args:
        config(tuple):          A configuration in the form [(up_spin), (down_spin)].
        L(int):                 Linear system size.

    Returns:
        total_index(int):       The index of the configuration.
    """
    # Unpack the red and blue configurations
    up_config,down_config = config

    # Convert the tuple to a unique index
    base = L + 1
    up_index= sum(r * (base ** i) for i, r in enumerate(reversed(up_config)))
    down_index = sum(b * (base ** i) for i, b in enumerate(reversed(down_config)))
    
    # Combine the two indices
    total_index = up_index * (base ** (3*L)) + down_index
    
    return total_index
def section_reduction(L):
    '''
    Generates all the states, then goes throught them, applies all the group elements and bunches the configs together in equivalence classes: if \exists g \in G: g(c)=c' => c ~c'
    Then it calls the reweighing function that turns equivalence classes into two numbers; Their representative and their order(weight)
    Args:
        dict(dictionary):       Dictionary of the form (key,value):(x_1,x_1,x_2,x_3,.....) with number equal to the order of the group

    Returns:
        Equivalence_c...(dict): Dictionaty of the equivalence classes. Keys are the representative configurations and values their weights.
        Tot_number(int):        Total number of configurations
        Reduced_number(int):    Number of configurations after this reduction
    '''
    configs = section_generator(L)
    Tot_number = (1+L)**(6*L)
    Equivalence_classes = {}
    mask = [True]*Tot_number # checks if this state has been 'met' before
    for i,c in enumerate(configs):
        if mask[i] == False:
            continue
        Equivalence_classes[i] = []
        #######
        for n in range(2):
            for m in range(2):
                for m_c3 in range(3):
                        for m_spin in range(2):
                            c_new = GroupAction(c,n,m,m_c3,m_spin,L)
                            i_new = c2i(c_new,L)
                            Equivalence_classes[i].append(i_new)
                            mask[i_new] = False
    Equivalence_classes = reweight(Equivalence_classes)
    Reduced_number = len(Equivalence_classes)
    print('Total Number of configurations',Tot_number)
    print('Reduced Number of configurations',Reduced_number)
    return Equivalence_classes,Reduced_number,Tot_number
def reweight(dict):
    '''
    Takes the symmetry-reduced configurations and 'reweights' them. Ie for eg [x_1]:(x_1,x_2,x_3) in the same equivalence class, it turns it into [x_1]:(3) 
    Meaning for calculations, x_1 cnofiguration should count 3 times as it really represents three states.

    Args:
        dict(dictionary):       Dictionary of the form (key,value):(x_1,x_1,x_2,x_3,.....) with number equal to the ordger of the group

    Returns:
        dict_out(dictionary):   Dictionary of the form (key,value):(x_1,w_1) with w_1 the weight corresponding to configuration x_1
    
    '''
    dict_out = {}
    for k in dict.keys():
        lst = dict[k]
        element_counts = Counter(lst)
        unique_elements = list(element_counts.keys())
        multiplicities = list(element_counts.values())
        if not multiplicities.count(multiplicities[0]) == len(multiplicities):
            #checks that all items in an equivalence class appear an equal amount of times: 1,2,3,4,6,8,12,24
            print('uhhhhhhhhhhhhhhhhh')
            quit()
        weight = len(unique_elements)
        #dict_out[k] = (unique_elements,weight) # this returns something that holds info of all equivalent states
        dict_out[k] = weight # this only holds info of the *representative* configuration through the dict key
    return dict_out
def Qnumber_printout(c):
    '''
    Outputs the quantum numbers of each configuration

    Args:
        c(nested tuple):        The configuration

    Returns:
        Qs(tuple):              Nested Tuple of quantum numbers:(\\nu,Sz_tot,N_1,N_2,N_3,Sz_1,Sz_2,Sz_3)
    
    '''
    up_c,down_c = c
    L = int(len(up_c)/3)
    N_1_up,N_2_up,N_3_up = sum(up_c[:L]),sum(up_c[L:2*L]),sum(up_c[2*L:3*L])
    N_1_down,N_2_down,N_3_down = sum(down_c[:L]),sum(down_c[L:2*L]),sum(down_c[2*L:3*L])
    N_up = sum(up_c)
    N_down = sum(down_c)
    nu = (N_up+N_down)/L**2
    Sz_tot = (N_up-N_down)/2
    N_1 = N_1_up + N_1_down
    N_2 = N_2_up + N_2_down
    N_3 = N_3_up + N_3_down
    Sz_1 = (N_1_up - N_1_down)/2
    Sz_2 = (N_2_up - N_2_down)/2
    Sz_3 = (N_3_up - N_3_down)/2
    return (nu,Sz_tot,N_1,N_2,N_3,Sz_1,Sz_2,Sz_3)
def fcombinatoric(c):
    '''
    Calculates the Hilbert space dimension of a configuration.
    '''
    c_up = c[0]
    c_down = c[1]
    dim = 1
    L = int(len(c_up)/3)
    for i in range(L):
        dim *= comb(L,c_up[i])*comb(L,c_down[i])*comb(L,c_up[i+L])*comb(L,c_down[i+L])*comb(L,c_up[i+2*L])*comb(L,c_down[i+2*L])
    return dim
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
###########################################  Config to chain mapping ###############################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
def total_hilbert_space_dim():
    L = 2
    configs = section_generator(L)
    dim = 0
    for config in configs:
        dim += fcombinatoric(config)
    print(dim)
    print(64**4)
    Equivalence_classes,Reduced_number,Tot_number = section_reduction(L)
    dim_alt = 0
    for k in Equivalence_classes.keys():
        dim_alt += fcombinatoric(configs[k])*Equivalence_classes[k]
    print(dim_alt)


    return

def Configs_to_chain(L=2,number='full'):
    #need to add lattice information as well
    configs = section_generator(L)
    #Equivalence_classes,Reduced_number,Tot_number = section_reduction(L)
    config = configs[154033]
    print(config)
    H_params = {}
    params = {}
    params['config'] = (config[0][-1],config[1][-1])
    params['L'] = L
    params['loc'] = 0
    c = chain(params,H_params)
    N = c.basis()
    print(N)
    return

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
##########################################  Single chain Hamiltonian ###############################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
'''
Essentially, for each chain the input will only be 
    1) How many sites it has
    2) How many spin up and spin down electrons it has
    3) Its location on the grid.
The relevant properties of the chain will then be its hamiltonian as well as its lattice location.
'''
class chain():
    def __init__(self,params,H_params):
        self.config = params['config']
        self.H_params = H_params
        self.L = params['L']
        self.loc = params['loc']
        self.Ne_up = self.config[0]
        self.Ne_down = self.config[1]
        return
    def basis(self):
        '''
        
        '''
        self.N = comb(self.L,self.Ne_up)*comb(self.L,self.Ne_down)
        return self.N
    #####################
    ####combinatorics####
    #####################
    def partitions(value,parts,max):
        """
        Generates all way to partition a *value* into *parts* of non-negative integers.
        It then filters any partition where any entry exceeds *max*
        ----------------------
        Example of partition:
        value=4,parts=3---->(1,1,2)
        """
        def helper(remaining, parts_left):
            # Base case: If no parts left to fill
            if parts_left == 0:
                if remaining == 0:
                    yield []
                return
            # Generate partitions
            for i in range(remaining + 1):  # Allow any non-negative integer
                for rest in helper(remaining - i, parts_left - 1):
                    yield [i] + rest
        # Generate all partitions
        all_partitions = list(helper(value,parts))
        # Filter out partitions with any entry > L2
        filtered_partitions = [p for p in all_partitions if all(x <= max for x in p)]
        return filtered_partitions
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
##########################################  Multi-chain Hamiltonian ################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#################################################### TERMINAL FUNCTIONS ############################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
def test1():
    '''
    Testing time to generate sections
    '''
    for L in [1,2,3]:
        stime = time.time()
        configs = section_generator(L)
        print(len(configs))
        print((L+1)**(6*L))
        ftime = time.time()
        print(f"elapsed time: {ftime-stime:.2f} seconds")
    return
def test2():
    '''
    Test comparing time to implement translations between two functions
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
def test3():
    '''
    Tests for L=2 how long does it take to apply all g's on all configurations?
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
def test4():
    '''
    Tests the operators and their commutator relationships. Either with full configs or a partial construction (see test5)
    '''
    L = 3
    n = 100
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
        if i< 20:
            print(config)
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
def test5():
    '''
    Tests partial configuration generation for L>2
    '''
    L=3
    n=100
    configurations = section_generator_partial(L, n)
    print(configurations)
    return
def test6():
    ''''
    Tests the config to index function
    '''
    print('=='*50)
    print('=='*15,'Testing Config 2 index function after group action','=='*15)
    print('=='*50)
    itime = time.time()
    L = 2
    configs = section_generator(L)
    for i,c in enumerate(configs):
        for n in range(2):
            for m in range(2):
                for m_c3 in range(3):
                        for m_spin in range(2):
                            c_new = GroupAction(c,n,m,m_c3,m_spin,L)
                            i_new = c2i(c_new,L)
                            #checking:
                            if c_new != configs[i_new]:
                                print('error mapping')
                                print(c_new,configs[i_new])
                                quit()
    ftime = time.time()
    print('=='*50)
    print('=='*40,'Success','=='*40)
    print(f"elapsed time: {ftime-itime:.2f} seconds")
    print('=='*50)
    return
def test7():
    '''
    Tests the Equivalent sector generator
    Specifically tests that the *sum_total* of the reduced configs equals the number of total configs
    '''
    L = 2
    Equivalence_clases,Reduced_number,Tot_number = section_reduction(L = L)
    print(Tot_number)
    print(Reduced_number)
    print(Tot_number/Reduced_number,'vs optimal',3*L*L*2)
    sum_reduced = 0
    sum_total = 0
    for k in Equivalence_clases.keys():
        sum_reduced +=1
        sum_total += Equivalence_clases[k][1]
    print(sum_reduced)
    print(sum_total)
    return
def test8():
    '''
    Tests the function that prints the quantum numbers of a given configuration
    '''
    L = 2
    configs = section_generator(L)
    print(configs[123456])
    print(Qnumber_printout(c=configs[123456]))
    return
def test9():
    #Configs_to_chain()
    total_hilbert_space_dim()
    return
def exe1():
    print('executing....')
    return
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
##########################################  COMMAND LINE RUN #######################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multiple features.")
    parser.add_argument("test", type=str, choices=["no_test","test1", "test2","test3","test4","test5", "test6","test7","test8","test9"], help="test to run")
    parser.add_argument("execute", type=str, choices=["no_exe","exe1"], help="execute ting")

    args = parser.parse_args()

    if args.test == "test1":
        test1()
    elif args.test == "test2":
        test2()
    elif args.test == "test3":
        test3()
    elif args.test == "test4":
        test4()
    elif args.test == "test5":
        test5()
    elif args.test == "test6":
        test6()
    elif args.test == "test7":
        test7()
    elif args.test == "test8":
        test8()
    elif args.test == "test9":
        test9()
    if args.execute == "exe1":
        exe1()