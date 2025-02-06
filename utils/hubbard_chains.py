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
'''


'''
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
##############################################  ONE CHAIN ##########################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################



################################
######  BASIS CREATION #########
################################
def basis_00(L,Q1,Q2):
    ''' 
    generates Lin lookup tables for a given symmetry sector
    Output:
    1)states and number of states and also nstates, ie the occupancy of each basis state. this final one is to be used later for ED
    2) lookup maps J_up,J_down,J
    '''
    states = []
    J_up = {}
    J_down = {}
    J = 0
    for I_down in range(2**L):
        if countBits(I_down) == Q2:
            count = 0
            for I_up in range(2**L):
                if countBits(I_up) == Q1:
                    ###
                    J_up[I_up] = count
                    J_down[I_down] = J - J_up[I_up]
                    ###
                    states.append(I_up+I_down*2**L)
                    count +=1
                    J += 1
    #now given the basis for the chain, also provide a nstates vector that says how many electons there are for each basis state
    nstates = []
    for state in states:
        nstates.append(countBits(state))
    return states, len(states),J_up,J_down,J,nstates
################################
####  HAMILTONIAN CREATION #####
################################
def ham_symm_full(parameters):
    '''
    create dense hamiltonian
    ------------------------
    H = H_T + H_V
    H_T = -t \\sum_{<ij>,s} (c^\\dagger_{i,s}c_{j,s}+h.c.) - \\mu \\sum_{i,s}n_{i,s}
    H_V = (U/2) \\sum_{i} n_{i,\\uparrow}n_{i,\\downarrow}
    -------------------------

    sign(Boolean): IMplements fermionic sifn checks to ensure the anticommutation relations are implemented correctly.
    '''
    #0) read parameters
    Q_up = parameters['Electrons_up']
    Q_down = parameters['Electrons_down']
    L = parameters['L']
    t = parameters['hop']
    mu = parameters['mu']
    U = parameters['int']
    sign = parameters['sign']
    mu_eff = mu+0.5*U # maps between Un_down n_up and Un^2
    mu_eff = mu
    #1) create basis
    states, M,J_up,J_down,J,nstates = basis_00(L=L,Q1=Q_up,Q2=Q_down)
    #2) build Hamiltonian
    H = np.zeros((M,M),dtype=float)
    for m in range(M):
        state = states[m]
        for i in range(L):
            j = (i+1)%L
            #############
            #interaction#
            ######&######
            ##chem pot###
            #############
            occ = occupancy(L,state,i)
            if occ == 2:
                H[m,m] += U
            H[m,m] -= mu_eff*occ
            #############
            ###hopping###
            #############
            psi_up = state%(2**L)
            psi_down = state//(2**L)
            nup = J_up[psi_up]
            ndown = J_down[psi_down]
            psi_up_prime = hop(psi_up,i,j)
            psi_down_prime = hop(psi_down,i,j)
            if psi_up_prime!= -1:
                sgn = +1
                if sign == True:
                    sgn = (-1)**count_ones_between_flips(binp(psi_up,length=L),binp(psi_up_prime,length=L))
                nup_prime = J_up[psi_up_prime]
                H[m,nup_prime + ndown] -= t*sgn
            if psi_down_prime!= -1:
                sgn = +1
                if sign == True:
                    sgn = (-1)**count_ones_between_flips(binp(psi_down,length=L),binp(psi_down_prime,length=L))
                ndown_prime = J_down[psi_down_prime]
                H[m,nup + ndown_prime] -= t*sgn
    if np.all(H==H.T) == False:
        print('hermitian?',np.all(H==H.T))
        quit()
    #sparsity(H)
    return H,nstates 
def ham_symm_sparse(parameters):
    '''
    Creates sparse Hamiltonian
    '''
    dtype = float
    rows = []
    cols = []
    data = []
    #0) read parameters
    Q_up = parameters['Electrons_up']
    Q_down = parameters['Electrons_down']
    L = parameters['L']
    t = parameters['hop']
    mu = parameters['mu']
    U = parameters['int']
    mu_eff = mu+0.5*U
    mu_eff = mu
    #1) create basis
    states, M,J_up,J_down,J,nstates = basis_00(L=L,Q1=Q_up,Q2=Q_down)
    #2) build Hamiltonian
    for m in range(M):
        state = states[m]
        for i in range(L):
            j = (i+1)%L
            #############
            #interaction#
            ######&######
            ##chem pot###
            #############
            occ = occupancy(L,state,i)
            if occ == 2:
                rows.append(m)
                cols.append(m)
                data.append(U-mu_eff*occ)
            else:
                rows.append(m)
                cols.append(m)
                data.append(-mu_eff*occ)
            #############
            ###hopping###
            #############
            psi_up = state%(2**L)
            psi_down = state//(2**L)
            nup = J_up[psi_up]
            ndown = J_down[psi_down]
            psi_up_prime = hop(psi_up,i,j)
            psi_down_prime = hop(psi_down,i,j)
            if psi_up_prime!= -1:
                nup_prime = J_up[psi_up_prime]
                rows.append(m)
                cols.append(nup_prime + ndown)
                data.append(-t)
            if psi_down_prime!= -1:
                ndown_prime = J_down[psi_down_prime]
                rows.append(m)
                cols.append(nup + ndown_prime)
                data.append(-t)
    H_coo = coo_matrix((data, (rows, cols)), shape=(M,M), dtype=dtype)
    H_csr = H_coo.tocsr()
    return H_csr,nstates
################################
###### EIGENSOLVERS FOR H ######
################################
def getSpectrumLanczos(pars,k=2):
    '''
    same as full but only for bottom k values
    '''
    energies = []
    lowestEnergy = 1e10
    L = pars['L']
    for n_up in np.arange(0,L+1):
        for n_down in np.arange(0,L+1):
            if n_up+n_down != L: continue # interested in half-filled case
            if n_up == 0 or n_down == 0: continue #here the hilbert space is one dimensional-> can't do lanczos.
            pars['Electrons_up'] = n_up
            pars['Electrons_down'] = n_down
            H = ham_symm_sparse(pars)[0]
            print('=============')
            print('Lanczos for (n_up,n_down) sector (',n_up,',',n_down,')')
            print('=============')
            lam,v = eigsh(csr_matrix(H), k=k, which='SA', tol=1e-10)
            #keep track of GS
            if min(lam) < lowestEnergy:
                lowestEnergy  = min(lam)
                GSSector      = (n_up,n_down)
                GSEigenvector = v[:,lam.argmin()]    
    print("Energies assembled!")
    print("Lowest energy:",lowestEnergy)
    print("The ground state occured in (n_up,n_down)=",GSSector)
    return (lowestEnergy,GSSector,GSEigenvector,energies)
def getSpectrumFull(pars):
    '''Returns lowestEnergy, 
               (N_up,N_down) sector of the GS, 
               and what else??'''
    '''
    N_up goes from 0 to L
    N_down goes from 0 to L
    '''
    energies = []
    lowestEnergy = 1e10
    L = pars['L']
    for n_up in np.arange(0,L+1):
        for n_down in np.arange(0,L+1):
            if n_up+n_down != L: continue # interested in half-filled case
            pars['Electrons_up'] = n_up
            pars['Electrons_down'] = n_down
            H = ham_symm_full(pars)[0]
            print('=============')
            print('diagonalizing (n_up,n_down) sector (',n_up,',',n_down,')')
            print('=============')
            lam,v = np.linalg.eigh(H)
            energies.append(lam)
            #keep track of GS
            if min(lam) < lowestEnergy:
                lowestEnergy  = min(lam)
                GSSector      = (n_up,n_down)
                GSEigenvector = v[:,lam.argmin()]    
    print("Energies assembled!")
    print("Lowest energy:",lowestEnergy)
    print("The ground state occured in (n_up,n_down)=",GSSector)
    return (lowestEnergy,GSSector,GSEigenvector,energies)
def EDFullSpectrum(pars):
    '''
    Does ED on all sectors and returns
    0) Info on lowest sector and energy
    1) Energies as a list of 1D arrays, one array per sector
    2) Eigenstates as a list of 2D arrays, one per sector
    3) An 'occupation' vector for each sector (that tells us the occupation number of each basis element in a sector)
    N_up goes from 0 to L
    N_down goes from 0 to L
    '''
    energies = []
    eigenstates = []
    eigenoccupation = []
    lowestEnergy = 1e10
    L = pars['L']
    for n_up in np.arange(0,L+1):
        for n_down in np.arange(0,L+1):
            #if n_up+n_down != L: continue # interested in half-filled case
            pars['Electrons_up'] = n_up
            pars['Electrons_down'] = n_down
            H,nstates = ham_symm_full(pars)
            #print('=============')
            #print('diagonalizing (n_up,n_down) sector (',n_up,',',n_down,')')
            #print('=============')
            lam,v = np.linalg.eigh(H)
            energies.append(lam)
            eigenstates.append(v)
            #print(v.dtype,v.shape)
            eigenoccupation.append(nstates)
            #print('****')
            #print(nstates)
            #print('***')
            #keep track of GS
            if min(lam) < lowestEnergy:
                lowestEnergy  = min(lam)
                GSSector      = (n_up,n_down)
                GSEigenvector = v[:,lam.argmin()]    
    #print("Energies assembled!")
    #print("Lowest energy:",lowestEnergy)
    print("The ground state occured in (n_up,n_down)=",GSSector)
    return (lowestEnergy,GSSector,energies,eigenstates,eigenoccupation)
################################
######  HELPER FUNCTIONS #######
################################
def count_ones_between_flips(binary1, binary2):
    '''
    Given two binary strings, calculate the number of 1's between their non-same elements. eg 010010010 and 000010110, we want to count the number of 1's between the 1st and -3 index:0010 so 1
    Used to add a sign after hopping due to fermionic anticommutation relations
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

    return ones_count
def countBits(x):
    '''Counts number of 1s in bin(n)'''
    #From Hacker's Delight, p. 66
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x0000003F 
def binp(num, length=4):
    '''
    print a binary number without python 0b and appropriate number of zeros
    regular bin(x) returns '0bbinp(x)' and the 0 and b can fuck up other stuff
    '''
    return format(num, '#0{}b'.format(length + 2))[2:]
def basisVisualizer(L,psi):
    '''
    Given psi=(#)_10, outputs the state in arrows
    (psi is the decimal number not its binary rep)
    '''
    #ex: |↓|↑|↓|↑|↑|
    psi_up = psi%(2**L)
    psi_down = psi//(2**L)

    s_up = bin(psi_up)[2:]
    s_down = bin(psi_down)[2:]
    N_up  = len(s_up)
    N_down = len(s_down)
    up = (L-N_up)*'0'+s_up
    down = (L-N_down)*'0'+s_down
    configStrUp = ""
    configStrDown = ""
    configStr = ""
    uparrow   = '\u2191'
    downarrow = '\u2193'
    empty = 'o'
    for i in range(L):
        blank = True
        if up[i] == '1' and down[i] == '1':
            configStr += '('+uparrow+downarrow+')'
            blank = False
        if up[i] == '1' and down[i] == '0':
            configStr +=uparrow
            blank = False
        if up[i] == '0' and down[i] == '1':
            configStr +=downarrow
            blank = False
        if up[i] == '0' and down[i] == '0':
            configStr += empty
            blank = False
        if blank:
            configStr+="_"
    print(configStr)
    return
def hop(I1,i,j):
    '''
    if hopping is allowed, outputs integer I2 that it maps to.
    This is for a single spin
    '''
    mask = 2**(i)+2**(j)
    K = I1 & mask #bitwise AND.
    L = K ^ mask #bitwise XOR.
    # L will have structure 0000000[i]0000[j]00000 and there's four cases:
    #1) L = mask means I1[i]=I1[j]=0 -> hopping is not allowed
    #2) L = 000..00 means I1[i]=I1[j]=1 -> hopping is not allowed
    #3&4) L = ...[1]...[0]... or L = ...[0]...[1]... means hopping is allowed, in which case new integer is 
    if L == mask or L == 0:
        #print('no hop')
        I2 = -1#flag to signify no hopping
    else:
        I2 = I1 - K + L
    return I2
def hop_spinful(L,states,m,i,j,J_down,J_up):
    ''' 
    Returns all possible states connected to state m by hopping at sites i,j.
    m: basis index 
    out: n basis index
    '''
    print('og state')
    print('--------------')
    basisVisualizer(L=L,psi=states[m])
    print('--'*10)
    print('derived states:')
    psi =  states[m]
    psi_up = psi%(2**L)
    psi_down = psi//(2**L)
    nup = J_up[psi_up]
    ndown = J_down[psi_down]
    psi_up_prime = hop(psi_up,i,j)
    psi_down_prime = hop(psi_down,i,j)
    ns = []
    if psi_up_prime!= -1:
        #print(psi_up_prime)
        nup_prime = J_up[psi_up_prime]
        ns.append(nup_prime + ndown)
        basisVisualizer(L=L,psi=psi_up_prime+psi_down*2**L)
        print('-'*10)
    if psi_down_prime!= -1:
        #print(psi_down_prime)
        ndown_prime = J_down[psi_down_prime]
        ns.append(nup + ndown_prime)
        basisVisualizer(L=L,psi=psi_up+psi_down_prime*2**L)
        print('-'*10)
    return ns
def occupancy(L,psi,i):
    '''
    Calculates occupancy of site i
    '''
    mask = 2**(i)+2**(L+i)
    occ = countBits(psi & mask)
    return occ
def sparsity(X):
    '''
    sparsity calculator. For eg L=22, sparsity is 99.98% and this should keep growing w/ L
    because basis grows exponentially while states connected via hopping  grow linearly(?)
    note: can't apply this function to sparse(csr) matrix. Have to apply to dense matrix to
    make sense
    '''
    nnz = np.sum(np.abs(X) > 1e-10)  # Non-zero count
    total = X.size
    sparsity_percentage = 100 * (1 - nnz / total)
    print('Sparsity is:', np.round(sparsity_percentage, 2), '%')
    return sparsity_percentage
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#############################################  MULTIPLE CHAINS #####################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
'''
I want to create multiple chains  and create the tensor product of the sparse Hamiltonian. 
At first this will be non-interacting chains with just hoppings
1st step: Enumerate all possible sections with given total N_up,N_down sections and do some histogram of their sizes
2nd step: how to tensor product sparse matrices?

To do:
Fix total filling and enumerate all possible chain configs. Plot hisogram of their sizes
Expect: largest section is fully symmetric one.
Then the next step is to construct bases and diagonalize
'''

################################
########  Combinatorics ########
################################
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
def combos(p,x,y,flav):
    '''
    Given a partition p (list) of possible splits of electrons along N_chain chains of N_site sites, calculate the total number of configurations for this partiton

    f(p) = \\prod_{i=1}^{N_chain} (N_sites) C (p[i])

    Now, N_chain and N_site depend on x,y and the flavor (or valley or type) of chain 

    '''
    if flav == 1:
        chains = y
        sites = x
    elif flav == 2 or flav == 3:
        chains = x
        sites = y
    f = 1
    if len(p) != chains:
        print('Number of chains not what was expected')
        quit()
    for pi in p:
        f *= comb(sites,pi)
    return f
def configs_chains(N_1_up,N_1_down,N_2_up,N_2_down,N_3_up,N_3_down,L1,L2):
    '''
    Generates all partitions of particles in chains given the total number of particles of each spin of each valley
    ----------------
    Possible number of particles in each valley and spin is between 
    0 =< N_{\eta,s} =< L1*L2
    All these combos are possible
    ----------------
    N_TOT = N_1_up + N_1_down + N_2_up + N_2_down + N_3_up + N_3_down
    S_TOT = 0.5*(N_1_up - N_1_down + N_2_up -N_2_down + N_3_up - N_3_down)
    '''
    # setup
    #N_eta_i = number of chains of valley i
    #L_eta_i = number of sites in a chainof valley i
    N_eta_1 = L2
    L_eta_1 = L1

    N_eta_2 = L1
    L_eta_2 = L2
    
    N_eta_3 = L1 # these can be flipped. total hilbert space should be invariant,,,,,
    L_eta_3 = L2

    #generate partitions of each section and combine them in the end
    n_1_up = partitions(value = N_1_up,parts = N_eta_1,max = L_eta_1)
    n_1_down = partitions(value = N_1_down,parts = N_eta_1,max = L_eta_1)

    n_2_up = partitions(value = N_2_up,parts = N_eta_2,max = L_eta_2)
    n_2_down = partitions(value = N_2_down,parts = N_eta_2,max = L_eta_2)

    n_3_up = partitions(value = N_3_up,parts = N_eta_3,max = L_eta_3)
    n_3_down = partitions(value = N_3_down,parts = N_eta_3,max = L_eta_3)

    n_1_up_combinations = {tuple(p): combos(p,L1,L2,flav=1) for p in n_1_up}
    n_1_down_combinations = {tuple(p): combos(p,L1,L2,flav=1) for p in n_1_down}

    n_2_up_combinations = {tuple(p): combos(p,L1,L2,flav=2) for p in n_2_up}
    n_2_down_combinations = {tuple(p): combos(p,L1,L2,flav=2) for p in n_2_down}

    n_3_up_combinations = {tuple(p): combos(p,L1,L2,flav=3) for p in n_3_up}
    n_3_down_combinations = {tuple(p): combos(p,L1,L2,flav=3) for p in n_3_down}

    #partitions_final = product(n_1_up,n_1_down,n_2_up,n_2_down)

    combined_results = {}
    for x,y,z,w,f,g in product(n_1_up,n_1_down,n_2_up,n_2_down,n_3_up,n_3_down):
        key = (tuple(x), tuple(y),tuple(z),tuple(w),tuple(f),tuple(g))
        combined_results[key] = n_1_up_combinations[tuple(x)]*n_1_down_combinations[tuple(y)]*n_2_up_combinations[tuple(z)]*n_2_down_combinations[tuple(w)]*n_3_up_combinations[tuple(f)]*n_3_down_combinations[tuple(g)]
    
    return sum(combined_results.values()),len(combined_results),max(combined_results.values())
def configs_chains_hist(N_1_up,N_1_down,N_2_up,N_2_down,N_3_up,N_3_down,L1,L2):
    '''
    Specifically: saves an array with number of occurances of states
    -----------------------------------------------
    Generates all partitions of particles in chains given the total number of particles of each spin of each valley
    ----------------
    Possible number of particles in each valley and spin is between 
    0 =< N_{\eta,s} =< L1*L2
    All these combos are possible
    ----------------
    N_TOT = N_1_up + N_1_down + N_2_up + N_2_down + N_3_up + N_3_down
    S_TOT = 0.5*(N_1_up - N_1_down + N_2_up -N_2_down + N_3_up - N_3_down)
    '''
    # setup
    #N_eta_i = number of chains of valley i
    #L_eta_i = number of sites in a chainof valley i
    N_eta_1 = L2
    L_eta_1 = L1

    N_eta_2 = L1
    L_eta_2 = L2
    
    N_eta_3 = L1 # these can be flipped. total hilbert space should be invariant,,,,,
    L_eta_3 = L2

    #generate partitions of each section and combine them in the end
    n_1_up = partitions(value = N_1_up,parts = N_eta_1,max = L_eta_1)
    n_1_down = partitions(value = N_1_down,parts = N_eta_1,max = L_eta_1)

    n_2_up = partitions(value = N_2_up,parts = N_eta_2,max = L_eta_2)
    n_2_down = partitions(value = N_2_down,parts = N_eta_2,max = L_eta_2)

    n_3_up = partitions(value = N_3_up,parts = N_eta_3,max = L_eta_3)
    n_3_down = partitions(value = N_3_down,parts = N_eta_3,max = L_eta_3)

    n_1_up_combinations = {tuple(p): combos(p,L1,L2,flav=1) for p in n_1_up}
    n_1_down_combinations = {tuple(p): combos(p,L1,L2,flav=1) for p in n_1_down}

    n_2_up_combinations = {tuple(p): combos(p,L1,L2,flav=2) for p in n_2_up}
    n_2_down_combinations = {tuple(p): combos(p,L1,L2,flav=2) for p in n_2_down}

    n_3_up_combinations = {tuple(p): combos(p,L1,L2,flav=3) for p in n_3_up}
    n_3_down_combinations = {tuple(p): combos(p,L1,L2,flav=3) for p in n_3_down}

    #partitions_final = product(n_1_up,n_1_down,n_2_up,n_2_down)

    combined_results = {}
    for x,y,z,w,f,g in product(n_1_up,n_1_down,n_2_up,n_2_down,n_3_up,n_3_down):
        key = (tuple(x), tuple(y),tuple(z),tuple(w),tuple(f),tuple(g))
        combined_results[key] = n_1_up_combinations[tuple(x)]*n_1_down_combinations[tuple(y)]*n_2_up_combinations[tuple(z)]*n_2_down_combinations[tuple(w)]*n_3_up_combinations[tuple(f)]*n_3_down_combinations[tuple(g)]
        # Extract values
    values = list(combined_results.values())
    max_value = max(values)
    bins = np.logspace(np.log10(min(values)), np.log10(max(values)), num=20)
    hist, edges, _ = plt.hist(values, bins=bins, edgecolor='black', log=True)  # Use log scale for frequency
    plt.xscale('log')  # Logarithmic scale for x-axis
    plt.title('Symmetry Sector Size')
    plt.xlabel('Value')
    plt.ylabel('Frequency (log scale)')
    ########
    # Frequency of the largest bin (last bin in histogram)
    largest_bin_index = -2  # Second last edge corresponds to the last bin
    largest_bin_freq = hist[largest_bin_index]  # Frequency of the last bin
    bin_left_edge = edges[largest_bin_index]  # Left edge of the last bin
    bin_right_edge = edges[largest_bin_index + 1]  # Right edge of the last bin

    # Annotate the frequency of the largest bin on the plot
    plt.annotate(
        f"Freq: {int(largest_bin_freq)}", 
        xy=((bin_left_edge + bin_right_edge) / 2, largest_bin_freq),  # Position in the center of the bin
        xytext=((bin_left_edge + bin_right_edge) / 2, largest_bin_freq * 1.5),  # Adjusted text position
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=10,
        color='red'
    )
    ##########
    # Enable gridlines for both major and minor ticks
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ##########
    plt.savefig('/mnt/users/kotssvasiliou/ED/figures/hist.png')
    ##########
    #print all largest sectors
    #values_in_largest_bin = [v for v in values if bin_left_edge <= v < bin_right_edge]
    #print(f"Values in the largest bin ({bin_left_edge:.2f}, {bin_right_edge:.2f}):")
    #print(values_in_largest_bin)
    return 
def configs_chains_max(N_1_up,N_1_down,N_2_up,N_2_down,N_3_up,N_3_down,L1,L2):
    '''
    Generates all partitions of particles in chains given the total number of particles of each spin of each valley
    Then, it returns the partition with the largest amount of configurations.
    ----------------
    Possible number of particles in each valley and spin is between 
    0 =< N_{\eta,s} =< L1*L2
    All these combos are possible
    ----------------
    N_TOT = N_1_up + N_1_down + N_2_up + N_2_down + N_3_up + N_3_down
    S_TOT = 0.5*(N_1_up - N_1_down + N_2_up -N_2_down + N_3_up - N_3_down)
    '''
    # setup
    #N_eta_i = number of chains of valley i
    #L_eta_i = number of sites in a chainof valley i
    N_eta_1 = L2
    L_eta_1 = L1

    N_eta_2 = L1
    L_eta_2 = L2
    
    N_eta_3 = L1 # these can be flipped. total hilbert space should be invariant,,,,,
    L_eta_3 = L2

    #generate partitions of each section and combine them in the end
    n_1_up = partitions(value = N_1_up,parts = N_eta_1,max = L_eta_1)
    n_1_down = partitions(value = N_1_down,parts = N_eta_1,max = L_eta_1)

    n_2_up = partitions(value = N_2_up,parts = N_eta_2,max = L_eta_2)
    n_2_down = partitions(value = N_2_down,parts = N_eta_2,max = L_eta_2)

    n_3_up = partitions(value = N_3_up,parts = N_eta_3,max = L_eta_3)
    n_3_down = partitions(value = N_3_down,parts = N_eta_3,max = L_eta_3)

    n_1_up_combinations = {tuple(p): combos(p,L1,L2,flav=1) for p in n_1_up}
    n_1_down_combinations = {tuple(p): combos(p,L1,L2,flav=1) for p in n_1_down}

    n_2_up_combinations = {tuple(p): combos(p,L1,L2,flav=2) for p in n_2_up}
    n_2_down_combinations = {tuple(p): combos(p,L1,L2,flav=2) for p in n_2_down}

    n_3_up_combinations = {tuple(p): combos(p,L1,L2,flav=3) for p in n_3_up}
    n_3_down_combinations = {tuple(p): combos(p,L1,L2,flav=3) for p in n_3_down}

    max = 0
    for x,y,z,w,f,g in product(n_1_up,n_1_down,n_2_up,n_2_down,n_3_up,n_3_down):
        value = n_1_up_combinations[tuple(x)]*n_1_down_combinations[tuple(y)]*n_2_up_combinations[tuple(z)]*n_2_down_combinations[tuple(w)]*n_3_up_combinations[tuple(f)]*n_3_down_combinations[tuple(g)]
        sector = (tuple(x),tuple(y),tuple(z),tuple(w),tuple(f),tuple(g))
        if value > max:
            max = value
            max_sec = sector
    return max,max_sec
def configs_chains_fixed_N(N_up,N_down,L1,L2):
    '''
    
    '''
    filling = np.round((N_up+N_down)/(1.*L1*L2),3)
    Sz = np.round((N_up-N_down)/(2.*L1*L2),3)
    N = int(3*L1*L2) # total number of degrees of freedom
    print(f'Generating all chain configurations compatible with total filling {filling} and spin {Sz}')
    ps_up = partitions(N_up,3,N_up)
    ps_down = partitions(N_down,3,N_down)
    ps = list(product(ps_up, ps_down))
    print(f'There are {len(ps)} *large* symmetry sectors')
    uniq_ps_up = {tuple(sorted(t)) for t in ps_up}
    uniq_ps_up = list(uniq_ps_up)
    uniq_ps = list(product(uniq_ps_up, ps_down))
    print(f'There are {len(uniq_ps)} unique *large* symmetry sectors')
    #print(uniq_ps)
    ps_N = []
    ps_S = []
    for p in ps:
        ps_N.append([a+b for a,b in zip(p[0],p[1])])
        ps_S.append([a-b for a,b in zip(p[0],p[1])])
    #need a smart way to get only those configs that are permutations of eachother!!!!!    
    #print(ps)
    return
#combinatorics for total symmetries.
def configs_global(L1,L2,n_up,n_down):
    '''
    just utilizing total 2xU(1)
    '''
    valleys = 3
    return comb(int(valleys*L1*L2),n_up)*comb(int(valleys*L1*L2),n_down)
def configs_valley(L1,L2,n_up,n_down):
    '''
    utilizing  2xU(1) per valley
    '''
    prod = 1
    valleys = 3
    for eta in range(valleys):
        prod *= comb(int(L1*L2),n_up[eta])*comb(int(L1*L2),n_down[eta])
    return prod
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
##############################################  STATMECH ##########################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
def partition_function(es,beta,emin=0):
    '''
    given a list of arrays for the energies in each sector, calculate the partition function at thempeature 1/beta
    '''
    Z = 0
    for sec in es:
        for state in range(sec.shape[0]):
            #Z += np.exp(-beta*sec[state])
            Z += np.exp(-beta * (sec[state] - emin))*np.exp(-beta * emin)
    return Z
def energy(es,beta,emin=0):
    '''
    given a list of arrays for the energies in each sector, calculate the <H>  and <H^2>at tempeature 1/beta
    '''
    H = 0
    H2 = 0
    for sec in es:
        for state in range(sec.shape[0]):
            #H += np.exp(-beta*sec[state])*sec[state]
            H += np.exp(-beta * (sec[state] - emin))*np.exp(-beta * emin)*sec[state]
            #H2 += np.exp(-beta*sec[state])*(sec[state])**2
            H2 += np.exp(-beta * (sec[state] - emin))*np.exp(-beta * emin)*(sec[state])**2
    return H,H2
    
def N_diagonal(vs,ns):
    '''
    for each eigenstate, calculate the diagonal elements of the number operator:
    
    <\\alpha|N|\\alpha> = \sum_{ij} <i|N|j>\\alpha^*_i \\alpha_j = \sum_{i} <i|N|i> |\\alpha|^2_i
    '''
    print('checking normalization')
    for i in range(len(vs)):
        norm = np.einsum('ij,ij->j',vs[i],vs[i].conj())
        if np.all(norm) !=1:
            print('uh? not normalized')
    print('normalization done')
    N_diag = []
    N_sq_diag = []
    for i in range(len(vs)):#go through each sector
        occupations = np.einsum('mn,m,mn->n',vs[i],ns[i],vs[i].conj())
        occupations_squared =  np.einsum('mn,m,m,mn->n',vs[i],ns[i],ns[i],vs[i].conj())
        N_diag.append(occupations.tolist())#m is basis index and n is order of vectors, i think***
        N_sq_diag.append(occupations_squared.tolist())
    #print(N_diag[3],N_diag[3][0].tolist()) #causes floating point errors....
    return N_diag,N_sq_diag
def Navg(es,N_diag,N_sq_diag,beta,emin=0):
    Obs = 0
    Obs_sq = 0
    for i in range(len(es)):#loop through sectors
        for state in range(es[i].shape[0]):
            #Obs += np.exp(-beta*es[i][state])*N_diag[i][state]
            Obs += np.exp(-beta * (es[i][state] - emin))*np.exp(-beta * emin)*N_diag[i][state]
            #Obs_sq += np.exp(-beta*es[i][state])*N_sq_diag[i][state]
            Obs_sq += np.exp(-beta * (es[i][state] - emin))*np.exp(-beta * emin)*N_sq_diag[i][state]
    return Obs,Obs_sq

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
##############################################  RUN FUNCS ##########################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
def feature1():
    '''
    does some ED for a single chain, partly full spectrum partly Lanczos and plots GS
    '''
    parameters = {}
    parameters['mu'] = 2
    parameters['int'] = 5
    parameters['hop'] = 1
    for L in range(3,7):
        parameters['L'] = L
        gs = getSpectrumFull(pars=parameters)[0]
        gs2 = getSpectrumLanczos(pars=parameters,k=1)[0]
        print('*'*10)
        print('Does lanczos match full spectrum for lowest energy?')
        print(gs-gs2)
        print('*'*10)
        plt.scatter(L,gs2/L)
    for L in range(7,13):
        parameters['L'] = L
        gs = getSpectrumLanczos(pars=parameters,k=1)[0]
        plt.scatter(L,gs/L)
    plt.savefig('/mnt/users/kotssvasiliou/ED/figures/testtest.png')
def feature2():
    '''
    combinatorics:
    Returns the sector with the largest basis size
    '''
    max,max_sec = configs_chains_max(N_1_up=2 ,N_1_down=2,N_2_up=2,N_2_down=2,N_3_up=2,N_3_down=2,L1=2,L2=2)
    print(max)
    print(max_sec)
def feature3():
    '''
    full combinatoric testing
    '''
    valleys = 3
    Max_secs = []
    for (L1,L2) in [(2,2),(2,3),(2,4)]:
        step_start_time = time.time()
        N = int(L1*L2)
        S1 = 0
        M1 = 0
        S2 = 0
        M2 = 0
        S3 = 0
        M3 = 0
        sectors = 0
        for n_up in range(0,int(valleys*N)+1):
            for n_down in range(0,int(valleys*N)+1):
                combo = configs_global(L1,L2,n_up,n_down)
                S1+= combo
                if combo > M1:
                    M1 = combo
        for n_up_1 in range(0,N+1):
            print('n_up_1',n_up_1)
            for n_down_1 in range(0,N+1):
                for n_up_2 in range(0,N+1):
                    for n_down_2 in range(0,N+1):
                        for n_up_3 in range(0,N+1):
                            for n_down_3 in range(0,N+1):
                                ######
                                n_up = [n_up_1,n_up_2,n_up_3]
                                n_down = [n_down_1,n_down_2,n_down_3]
                                combo = configs_valley(L1,L2,n_up,n_down)
                                S2 += combo
                                if combo > M2:
                                    M2 = combo
                                #######
                                s,l,m = configs_chains(n_up_1,n_down_1,n_up_2,n_down_2,n_up_3,n_down_3,L1=L1,L2=L2)
                                S3 += s
                                sectors += l
                                if m > M3:
                                    M3 = m
                                ######
                                if np.abs(combo-s)>1e-2:
                                    print('woooah vickyyyyyy')
        step_end_time = time.time()
        print('*'*100)
        print("Total elapsed time for (L1,L2)",str(L1),',',str(L2))
        print(f"is : {step_end_time - step_start_time:.2f} seconds")
        print('-'*50)
        print('hilbert space comps:')
        print('Ss / full H',S1/(64**(N)),S2/(64**(N)),S3/(64**(N)))
        print('max sectors',M1,M2,M3)
        print('gain',(64**(N))/M1,(64**(N))/M2,(64**(N))/M3)
        print('*'*100)
        Max_secs.append([M1,M2,M3])
    #now after loops, plot:
    p = 0 #scatter poin
    for (L1,L2) in [(2,2),(2,3),(2,4)]:
        N = L1*L2
        full_H = 64**(int(N))
        glob = Max_secs[p][0]
        vall = Max_secs[p][1]
        chain = Max_secs[p][2]
        p += 1
        plt.scatter(N,log10(glob/(N)),marker='x',c='black')
        plt.scatter(N,log10(vall/(N)),marker='v',c='blue')
        plt.scatter(N,log10(full_H),marker='o',c='red')
        plt.scatter(N,log10(chain/N),marker='^',c='green')
    plt.ylim([2,22])
    plt.xticks([4,6,8,9])
    plt.yticks([4,8,12,16,20],['$10^{4}$','$10^{8}$','$10^{12}$','$10^{16}$','$10^{20}$'])
    plt.scatter(N,1,marker='o',c='red',label='no symmetry')
    plt.scatter(N,1,marker='x',c='black',label='global$\\times T$')
    plt.scatter(N,1,marker='v',c='blue',label='flavor $\\times T$')
    plt.scatter(N,1,marker='^',c='green',label='full internal $\\times T$')
    plt.axhline(y=9,c='r',alpha=0.5,label='~ Lanczos cutoff')
    plt.title('Approx. Number of states in largest sector')
    plt.xlabel('System size $(L_1 \\times L_2)$')
    plt.legend()
    plt.savefig('/mnt/users/kotssvasiliou/ED/figures/test.png')
    return
def feature4():
    '''
    Concentrate at largest *large* sector, meaning largest (N_1_up,N_1_down,N_2_up,N_2_down,N_3_up,N_3_down) at eg half filling. This will be, for 2x2 the (2,2,2,2,2,2) section.
    Go through all chain configurations in there and plot their basis size
    '''
    #configs_chains_hist(N_1_up=2 ,N_1_down=2,N_2_up=2,N_2_down=2,N_3_up=2,N_3_down=2,L1=2,L2=2)
    #configs_chains_hist(N_1_up=3 ,N_1_down=3,N_2_up=3,N_2_down=3,N_3_up=3,N_3_down=3,L1=3,L2=3)
    configs_chains_hist(N_1_up=2 ,N_1_down=4,N_2_up=2,N_2_down=4,N_3_up=2,N_3_down=4,L1=3,L2=3)
    return
def feature5():
    L1 = 2
    L2 = 2
    N_up = 6
    N_down = 6
    configs_chains_fixed_N(N_up,N_down,L1,L2)
    return
def feature6():
    '''
    Comparing the Hamiltonian with https://stanford.edu/~xunger08/Exact%20Diagonalization%20Tutorial.pdf
    '''
    pars = {}
    L = 4
    n_up = 2
    n_down = 2
    pars['L'] = L
    pars['Electrons_up'] = n_up
    pars['Electrons_down'] = n_down
    pars['mu'] = 2
    pars['int'] = 5
    pars['hop'] = 1
    pars['sign'] = True
    H = ham_symm_full(pars)
    print(H.shape)
    plt.imshow(H)
    plt.colorbar()
    plt.savefig('sign_check_temp.png')
    quit()
    s1 = int('11100110010001000100',2)
    s2 = int('01100110010001000101',2)
    print(binp(s1,length=L),binp(s2,length=L))
    #print(isinstance(count_ones_between_flips(binp(s1,length=L), binp(s2,length=L)),int))
    print(count_ones_between_flips('11100110010001000100', '01100110010001000101'))
    print(count_ones_between_flips(binp(s1,length=20), binp(s2,length=20)))
    return
def feature7():
    '''
    runs ED for chain. Saves 1)Energies 2)eigenstates 3)N vector
    
    L = 6: T ~ 14s
    L = 7: T ~ 91s
    L = 8: T ~ 
    '''
    itime = time.time()
    pars = {}
    L = 6
    U = 20
    pars['L'] = L
    pars['mu'] = U/2
    pars['int'] = U
    pars['hop'] = 1
    pars['sign'] = True
    (lowestEnergy,GSSector,energies,eigenstates,eigennumber,eigenoccup) = EDFullSpectrum(pars)
    ftime = time.time()
    print(f"configuration reduction time: {ftime-itime:.2f} seconds")
    betas = np.array(list(np.linspace(0.1,1,num=100))+list(np.linspace(1,10,num=100))+list(np.linspace(10,20,num=100)))
    #betas = np.linspace(0.1,1,num=100)
    Es = np.zeros_like(betas)
    Es2 = np.zeros_like(betas)
    for i,beta in enumerate(betas):
        Z = partition_function(energies,beta)
        E1,E2 = energy(energies,beta)
        Es[i] = E1/Z
        Es2[i] = (beta**2)*((E2)/Z - Es[i]**2)
    ###########################################
    betas_inset = np.linspace(10,20,num=100)
    Es_inset = np.zeros_like(betas_inset)
    Es2_inset = np.zeros_like(betas_inset)
    for i,beta in enumerate(betas_inset):
        Z = partition_function(energies,beta)
        E1,E2 = energy(energies,beta)
        Es_inset[i] = E1/Z
        Es2_inset[i] = (beta**2)*((E2)/Z - Es_inset[i]**2)
    ########################
    fig, ax = plt.subplots()
    ax.plot(1/betas,Es, label="Main Plot")
    ax.set_xlabel("$T$")
    ax.set_ylabel("$E$")
    ax.set_title("$\\langle H \\rangle$ at U/t="+str(U))
    # Create the inset
    inset_ax = fig.add_axes([0.6, 0.2, 0.25, 0.25])  # [x, y, width, height] in figure coordinates
    inset_ax.plot(1/betas_inset,Es_inset, color="red", label="Inset Plot")
    inset_ax.set_title("low T", fontsize=10)
    inset_ax.tick_params(labelsize=8)
    plt.savefig('energy_observable_'+str(U)+'_.png',dpi=1000)
    plt.clf()
    #########
    fig, ax = plt.subplots()
    ax.plot(1/betas,Es2, label="Main Plot")
    ax.set_xlabel("$T$")
    ax.set_title("$C_{\chi}$ at U/t="+str(U))
    # Create the inset
    inset_ax = fig.add_axes([0.6, 0.6, 0.25, 0.25])  # [x, y, width, height] in figure coordinates
    inset_ax.plot(1/betas_inset,Es2_inset, color="red", label="Inset Plot")
    inset_ax.set_title("low T", fontsize=10)
    inset_ax.tick_params(labelsize=8)
    plt.savefig('heat_capacity_observable_'+str(U)+'_.png',dpi=1000)

    return
def feature8():
    '''
    runs ED for chain. Saves 1)Energies 2)eigenstates 3)N vector
    
    L = 6: T ~ 14s
    L = 7: T ~ 91s
    L = 8: T ~ 
    '''
    itime = time.time()
    pars = {}
    L = 6
    U = 6
    pars['L'] = L
    pars['mu'] = U/2
    pars['int'] = U
    pars['hop'] = 1
    pars['sign'] = True
    (lowestEnergy,GSSector,energies,eigenstates,eigenoccupation) = EDFullSpectrum(pars)
    ftime = time.time()
    print(f"configuration reduction time: {ftime-itime:.2f} seconds")
    #############################
    #calculate <\alpha|N|\alpha> for each basis state.
    N_diag,N_sq_diag = N_diagonal(eigenstates,eigenoccupation)
    betas = np.array(list(np.linspace(0.1,1,num=100))+list(np.linspace(1,10,num=100))+list(np.linspace(10,30,num=100)))
    #betas = np.linspace(0.01,1,num=200)
    Ns = np.zeros_like(betas)
    Ns_sq = np.zeros_like(betas)
    Zs = np.zeros_like(betas)
    for i,beta in enumerate(betas):
        Z = partition_function(energies,beta)
        ns,ns_sq = Navg(energies,N_diag,N_sq_diag,beta)
        #ns = Navg(energies,N_diag,beta)
        Zs[i] = Z
        Ns[i] = ns/Z
        Ns_sq[i] = ns_sq/Z
    ######################
    ###free energy plot###
    ######################
    fig, ax = plt.subplots()
    ax.plot(1/betas,-(1/betas)*np.log(Zs), label="Main Plot")
    ax.set_xlabel("$T$")
    ax.set_title("$F$ at U/t="+str(U)+'(h.f.)')
    plt.savefig('1d_chain_figures/free_energy.png')
    ######################
    #######<N> plot#######
    ######################
    #plt.plot(1/betas,(betas**2)*(Ns_sq - Ns**2),'-.',c='b')
    #plt.savefig('1d_chain_figures/test_N.png')
    return
def feature9():
    '''
    ED plotting <N> vs chemical potential
    '''
    itime = time.time()
    pars = {}
    L = 6
    U = 5
    pars['L'] = L
    #pars['int'] = U
    pars['hop'] = 1
    pars['sign'] = True
    ###########################################################
    delta_mus = np.linspace(0,1.5,200) #as multiples of U
    betas = np.array([5,10,15])
    Us = np.array([2.5,5,10])
    Ns = np.zeros((200,4,3),dtype=float)
    for i,mu in enumerate(delta_mus):
        for k,U in enumerate(Us):
            print('(i,k)',i,k)
            pars['mu'] = U*(1/2+mu)
            pars['int'] = U
            (lowestEnergy,GSSector,energies,eigenstates,eigenoccupation) = EDFullSpectrum(pars)
            N_diag,N_sq_diag = N_diagonal(eigenstates,eigenoccupation)
            for j,beta in enumerate(betas):
                Z = partition_function(energies,beta,lowestEnergy)
                ns,ns_sq = Navg(energies,N_diag,N_sq_diag,beta,emin=lowestEnergy)
                Ns[i,j,k] = ns/Z

    ######################
    ###plot###
    ######################
    fig, ax = plt.subplots()
    ax.plot(1/2+delta_mus,Ns[:,0,0],c='g',alpha=0.5)
    ax.plot(1/2+delta_mus,Ns[:,1,0],c='g',alpha=0.7)
    ax.plot(1/2+delta_mus,Ns[:,2,0],c='g',alpha=1.0,label='$U=2.5$')
    ax.plot(1/2+delta_mus,Ns[:,0,1],c='b',alpha=0.5)
    ax.plot(1/2+delta_mus,Ns[:,1,1],c='b',alpha=0.7)
    ax.plot(1/2+delta_mus,Ns[:,2,1],c='b',alpha=1.0,label='$U=5$')
    ax.plot(1/2+delta_mus,Ns[:,0,2],c='r',alpha=0.5)
    ax.plot(1/2+delta_mus,Ns[:,1,2],c='r',alpha=0.7)
    ax.plot(1/2+delta_mus,Ns[:,2,2],c='r',alpha=1.0,label='$U=10$')
    ax.set_xlabel("$\\mu/U$")
    ax.set_title("$\\langle N \\rangle $")
    plt.grid(visible=True,axis='y')
    plt.legend()
    plt.savefig('/mnt/users/kotssvasiliou/ED/utils/1d_chain_figures/N_vs_mu.png')
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
    parser.add_argument("feature", type=str, choices=["feature1", "feature2","feature3","feature4","feature5", "feature6","feature7","feature8","feature9"], help="Feature to run")

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
    elif args.feature == "feature8":
        feature8()
    elif args.feature == "feature9":
        feature9()


