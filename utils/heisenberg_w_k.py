import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from scipy.sparse import csr_matrix #optimizes H . v operations. to check if H already row sparse, do  isspmatrix_csr(H)
from scipy.sparse.linalg import eigsh
#############
def Tran(L,s):
    '''
    Translates state by one unit to the left, ie S'_{i} = S_{i-1}
    '''
    msb = (s >> (L - 1)) & 1
    s_shifted = ((s << 1) | msb) & ((1 << L) - 1)
    return s_shifted
def checkstate(L,k,s):
    '''
    Checks whether state is a valid representative.
    if output is R=-1 then:
        1)State has periodicity incompatible with the momentum k *or*
        2)State is ~ to another state s' that has smaller integer, s' < s. Hence, s is not a valid representative.(~ means same up to translation)
    If output is R != -1, R = Rs then:
        State is a valid representative and it has a periodicity Rs compatible with k
    '''
    R = -1 # initialize periodicity
    t = s #initialize state
    for i in range(1,L+1):
        #generate each translated state
        t = Tran(L,t)
        #if t<s, s is not valid represenative state
        if t < s :
            return R
        #if t==s that means by translation we've returned to original s. Is this periodicity comptible with k?
        elif t == s:
            if k%(L/i) != 0:
                return R
            #otherwise: change R and then return; we've found out all info we need about s
            R = i
            return R
    return R
def findrepstate(L,s):
    '''
    given |s>, returns n where
    T^{\ell}|s> = |s_0> or T^{-\ell}|s_0> = |s> 
    ie how does |s> relate to its representative
    '''
    #initialize translated state and rep state
    t = s; r = s; ell = 0
    for i in range(1,L):
        #generate translated state
        t = Tran(L,t)
        #if translated state is less than current rep state, it replaces the current rep state
        if t < r:
            r = t
            ell = i
    return ell,r
def findstate(Slist,s):
    '''
    Given a state (integer) s, find its basis index in the basis list Slist.
    Slist is ordered so, can do bisectional search (at each step halve the search space) ~\matchal{O}(logM)
    ---------------------------------------------------------------------------------------
    For faster searching algos, use Hash tables (PhysRevB.42.6561)
    The idea is it constructs a mapping between representatives {I} and position vectors h(I) by
    constructing a hashing function. ~\matchal{O}(1)
    ---------------------------------------------------------------------------------------
    The point is that the first method can impose memory constraints as len(Slist) ~< 2**L so storing Slist can be problematic. 
    Solution 
    1) A hash function h(s) so given a state H x s-->s2 we can immediately populate H[h(s),h(s2)]. Problem: Collisions. there will always in general be s2,s3 with same image under h...
    2)split Slist into two lists where the 1st list holds the first half of the lattice and the 2nd list holds the second half of the lattice. Then search individually in each list.
      This halves the memory required per list which is huge
    '''
    bmin=0
    bmax=len(Slist)
    b = bmin + int((bmax-bmin)/2)
    while s != Slist[b] and bmin < bmax: #latter is used as an exit condition if s is not in Slist!
        if s < Slist[b]:
            bmax = b - 1
        elif s > Slist[b]:
            bmin = b + 1
        b = bmin + int((bmax-bmin)/2)
    if bmin > bmax:
        #so b = -1 stands for the fact that s is not in the list!
        b = -1
    return b
def makeKBasis(L,k):
    '''
    Generates the basis at a given k.
    k are inputted as integers in {-L/2+1,....,0,....+L/2}. To get physical crystal momenta do 2*pi/N * k
    '''
    a = 0 #initialize state counter in magnetization block
    Slist = [] # list of states 
    Rlist = [] # list of periodicities of kstates
    for s in range(2**L):
        Rs = checkstate(L,k,s)
        if Rs >= 0:
            a += 1
            Slist.append(s)
            Rlist.append(Rs)
    return
def makeBasis(L,k,mz):
    '''
    create fixed mz,k basis
    k are inputted as integers in {-L/2+1,....,0,....+L/2}. To get physical crystal momenta do 2*pi/N * k
    '''
    a = 0 #initialize state counter in magnetization block
    Slist = [] # list of states ιν (κ,μζ) basis
    Rlist = [] # list of periodicities of states
    num_up = mz + L/2
    for s in range(2**L):
        if countBits(s) == num_up:
            Rs = checkstate(L,k,s)
            if Rs >= 0:
                a += 1
                Slist.append(s)
                Rlist.append(Rs)
    return a,Slist,Rlist

def constructH(L,k,mz,pars):
    '''
    creates basis then fills it up in a specific (k,mz) sector
    output: Hamiltonian
    '''
    #get basis
    M,S = makeBasis(L,k,mz)
    #real symmetric Hamiltonian allows for a speedup
    if k == 0 or k == L/2:
        H = np.zeros((M,M),dtype=float)
    else:
        H = np.zeros((M,M),dtype=complex)

############################
######## SOLVE H ###########
############################
def getSpectrumLanczos(L):
    '''Returns lowestEnergy, 
               (mz,k) sector of the GS, 
               and what else??'''
    '''
    mz goes from -L/2 to L/2
    k goes from -L/2+1 to L/2
    '''
    energies = []
    lowestEnergy = 1e10
    J = 1.0
    pars = {'J':J}

    for mz in np.arange(-L/2,L/2+1,+1):
        for k in np.arange(-L/2+1,L/2+1,+1):
            H = constructH(L,k,mz,pars)
            print('=============')
            print('diagonalizing (mz,k) sector (',mz,',',k,')')
            print('=============')
            lam,v = np.linalg.eigh(H)
            energies.append(lam)
            #keep track of GS
            if min(lam) < lowestEnergy:
                lowestEnergy  = min(lam)
                GSSector      = mz
                GSEigenvector = v[:,lam.argmin()]    
    print("Energies assembled!")
    print("Lowest energy:",lowestEnergy)
    print("The ground state occured in Sz=",GSSector)
    return (lowestEnergy,GSSector,GSEigenvector,energies)

############################
#####HELPER FUNCTIONS#######
############################
def findstate(Slist,s):
    '''
    Given a state (integer) s, find its basis index in the basis list Slist.
    Slist is ordered so, can do bisectional search (at each step halve the search space) ~\matchal{O}(logM)
    ---------------------------------------------------------------------------------------
    For faster searching algos, use Hash tables (PhysRevB.42.6561)
    The idea is it constructs a mapping between representatives {I} and position vectors h(I) by
    constructing a hashing function. ~\matchal{O}(1)
    ---------------------------------------------------------------------------------------
    The point is that the first method can impose memory constraints as len(Slist) ~< 2**L so storing Slist can be problematic. 
    Solution 
    1) A hash function h(s) so given a state H x s-->s2 we can immediately populate H[h(s),h(s2)]. Problem: Collisions. there will always in general be s2,s3 with same image under h...
    2)split Slist into two lists where the 1st list holds the first half of the lattice and the 2nd list holds the second half of the lattice. Then search individually in each list.
      This halves the memory required per list which is huge
    '''
    bmin=0
    bmax=len(Slist)
    b = bmin + int((bmax-bmin)/2)
    while s != Slist[b] and bmin < bmax: #latter is used as an exit condition if s is not in Slist!
        if s < Slist[b]:
            bmax = b - 1
        elif s > Slist[b]:
            bmin = b + 1
        b = bmin + int((bmax-bmin)/2)
    return b
def flip(s,i,j):
    '''
    given an integer s, flip its binary elements at indices i,j and return new integer s'
    '''
    mask = 2**(i)+2**(j)
    s = s^mask
    #bit_i = (s >> i) & 1  
    #bit_j = (s >> j) & 1
    #s ^= (1 << i)  
    #s ^= (1 << j)    
    return s
def basisVisualizer(L,psi):
    '''
    Given psi=(#)_10, outputs the state in arrows
    (psi is the decimal number not its binary rep)
    '''
    #ex: |↓|↑|↓|↑|↑|
    psi_2 = bin(psi)[2:]
    N  = len(psi_2)
    up = (L-N)*'0'+psi_2
    configStr = "|"
    uparrow   = '\u2191'
    downarrow = '\u2193'
    for i in range(L):
        blank = True
        if up[i] == '1':
            configStr+=uparrow
            blank = False
        if up[i] == '0':
            configStr+=downarrow
            blank = False
        if blank:
            configStr+="_"
        configStr +="|"
    print(configStr)
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
###########################
#### SANITY CHECKS ########
###########################
def findstate_check(S):
    '''
    GIven a list, makes sure the findstate function works correctly!
    '''
    for i,s in enumerate(S):
        print(s)
        if i != findstate(S,s):
            print('    ----')
            print('    ',i,findstate(S,s))
            print('    ----')
    return