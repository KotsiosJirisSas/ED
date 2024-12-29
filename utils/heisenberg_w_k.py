import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from scipy.sparse import csr_matrix,coo_matrix #optimizes H . v operations. to check if H already row sparse, do  isspmatrix_csr(H)
from scipy.sparse.linalg import eigsh
import time
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
    J = pars['J']
    #get basis
    M,S,Rlist = makeBasis(L,k,mz)
    #real symmetric Hamiltonian allows for a speedup
    if k == 0 or k == L/2:
        H = np.zeros((M,M),dtype=float)
    else:
        H = np.zeros((M,M),dtype=complex)
    for m in range(M):
        s = S[m]
        for i in range(L):
            j = (i+1)%L # mod takes care of boundary
            #1)build diagonal part   S^z_{i}S^z_{i+1}. If spins are equal add 1/4 if they are opposite add -1/4.
            #  if spins on i,i+1 the S^+ S^- will not act on it
            if (s>>i & 1) == (s>>j & 1):
                H[m,m] += (1./4)*J
            # if i and j have different spin, then hamiltonian maps s-->s' that is a state same as s but with spins in i,j flipped.
            else:
                H[m,m] -= (1./4)*J
                #now for the off-diagonal part we need to get s--->s' after flipping the spins and locate it in the basis. This is costly!
                #a) get new state by flipping spins
                s2 = flip(s,i,j)
                #b) get its representative, ie T^ell |r> = |s2>
                ell,r = findrepstate(L,s2)
                #c) look up its index. if it is not in the basis, output is -1
                n = findstate(S,r)
                #d) add matrix element; taking periodicities into account
                if n > -1:
                    if k == 0:
                        H[m,n] += (1./2)*J*np.sqrt(Rlist[m]/Rlist[n])
                    elif k == L/2:
                        H[m,n] += (1./2)*J*np.sqrt(Rlist[m]/Rlist[n])*(-1)**ell
                    else:
                        H[m,n] += (1./2)*J*np.sqrt(Rlist[m]/Rlist[n])*np.exp(1j*2.*np.pi**k*ell/L)
    print(np.all(H==H.T))
    return H 


def constructH_sparse_old(L,k,mz,pars):
    '''
    creates basis then fills it up in a specific (k,mz) sector
    output: Hamiltonian
    Note: Quite slow, this is the main time constraint. probably need to change method soon.
    SLower than dense matrix construction but dense memory construction runs out of memory by L=24
    this is too slow...
    -----------------------------------------------------------------------------------------------
    Note: COmparing this construction with the new sparse one, at eg L=22:
    x       | this| new
    ------------------
    memory  |  3MB | 6MB
    time   H| 800s | 12.55s
    time Lan| 0.1s | 0.24s

    so while this seems to produce a sparse matrix that makes lanczos a bit faster, since it takes way longer to construct the hamiltonian,
    and thats the most expensive part, it ends up being much, much slower...
    '''
    J = pars['J']
    #get basis
    M,S,Rlist = makeBasis(L,k,mz)
    #real symmetric Hamiltonian allows for a speedup
    if k == 0 or k == L/2:
        H = csr_matrix((M,M),dtype=float)
    else:
        H = csr_matrix((M,M),dtype=complex)
    for m in range(M):
        s = S[m]
        for i in range(L):
            j = (i+1)%L # mod takes care of boundary
            #1)build diagonal part   S^z_{i}S^z_{i+1}. If spins are equal add 1/4 if they are opposite add -1/4.
            #  if spins on i,i+1 the S^+ S^- will not act on it
            if (s>>i & 1) == (s>>j & 1):
                H[m,m] += (1./4)*J
            # if i and j have different spin, then hamiltonian maps s-->s' that is a state same as s but with spins in i,j flipped.
            else:
                H[m,m] -= (1./4)*J
                #now for the off-diagonal part we need to get s--->s' after flipping the spins and locate it in the basis. This is costly!
                #a) get new state by flipping spins
                s2 = flip(s,i,j)
                #b) get its representative, ie T^ell |r> = |s2>
                ell,r = findrepstate(L,s2)
                #c) look up its index. if it is not in the basis, output is -1
                n = findstate(S,r)
                #d) add matrix element; taking periodicities into account
                if n > -1:
                    if k == 0:
                        H[m,n] += (1./2)*J*np.sqrt(Rlist[m]/Rlist[n])
                    elif k == L/2:
                        H[m,n] += (1./2)*J*np.sqrt(Rlist[m]/Rlist[n])*(-1)**ell
                    else:
                        H[m,n] += (1./2)*J*np.sqrt(Rlist[m]/Rlist[n])*np.exp(1j*2.*np.pi**k*ell/L)
    return H 

def constructH_sparse(L,k,mz,pars):
    rows = []
    cols = []
    data = []
    J = pars['J']
    if k == 0 or k == L/2:
        dtype = complex
    else:
        dtype = float
    #get basis
    M,S,Rlist = makeBasis(L,k,mz)
    for m in range(M):
        s = S[m]
        for i in range(L):
            j = (i+1)%L # mod takes care of boundary
            #1)build diagonal part   S^z_{i}S^z_{i+1}. If spins are equal add 1/4 if they are opposite add -1/4.
            #  if spins on i,i+1 the S^+ S^- will not act on it
            if (s>>i & 1) == (s>>j & 1):
                rows.append(m)
                cols.append(m)
                data.append((1./4)*J)
            # if i and j have different spin, then hamiltonian maps s-->s' that is a state same as s but with spins in i,j flipped.
            else:
                rows.append(m)
                cols.append(m)
                data.append(-(1./4)*J)
                #now for the off-diagonal part we need to get s--->s' after flipping the spins and locate it in the basis. This is costly!
                #a) get new state by flipping spins
                s2 = flip(s,i,j)
                #b) get its representative, ie T^ell |r> = |s2>
                ell,r = findrepstate(L,s2)
                #c) look up its index. if it is not in the basis, output is -1
                n = findstate(S,r)
                #d) add matrix element; taking periodicities into account
                if n > -1:
                    rows.append(m)
                    cols.append(n)
                    data.append((1./2)*J*np.sqrt(Rlist[m]/Rlist[n])*np.exp(1j*2.*np.pi*k*ell/L))
    H_coo = coo_matrix((data, (rows, cols)), shape=(M,M), dtype=dtype)
    H_csr = H_coo.tocsr()
    return H_csr


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

###########################
if __name__ == "__main__":
    start_time = time.time()
    Gs = []
    Htime = []
    LanczosTime = []
    L = 30
    for l in range(10,L+2,+2):
        time_1 = time.time()
        #H = constructH(l,k=0,mz=0,pars={'J':1.})
        H_sparse = constructH_sparse(l,k=0,mz=0,pars={'J':1.})
        #sparsity_percentage = sparsity(H)
        time_2 = time.time()
        eigenvalues, eigenvectors = eigsh(csr_matrix(H_sparse), k=1, which='SA', tol=1e-10)
        #eigenvalues2, eigenvectors = eigsh(H, k=1, which='SA', tol=1e-10)
        #print('eigendifference',eigenvalues[0]/(l/2)-eigenvalues2[0]/(l/2) )
        data_memory = H_sparse.data.nbytes
        print(f"Memory used by data array: {data_memory / 1024**2:.2f} MB")
        time_3 = time.time()
        Dt1 = time_2 - time_1
        Dt2 = time_3 - time_2
        Htime.append(Dt1)
        LanczosTime.append(Dt2)
        print(f"H creation time: {Dt1:.2f} seconds")
        print(f"Lanczos time: {Dt2:.2f} seconds")
        Gs.append(eigenvalues[0]/(l/2))
        print('diagonalized system size L=',l)
        print('GS energy:',eigenvalues[0]/(l/2))
        print('-'*100)
    
    end_time = time.time()
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
    plt.plot(range(10,L+2,+2),np.array(Gs),'.--',c='k')
    plt.xlabel('$L$')
    plt.ylabel('$E/N_e$')
    plt.savefig('/mnt/users/kotssvasiliou/ED/figures/fig.png')
    plt.show()
    #---------------------
    plt.plot(range(10,L+2,+2),np.array(Htime),'--r',label='H')
    plt.plot(range(10,L+2,+2),np.array(LanczosTime),'--b',label='Lanczos')
    plt.ylabel('$t$(s)')
    plt.legend()
    plt.ylim([0,np.max(np.array(Htime))])
    plt.savefig('/mnt/users/kotssvasiliou/ED/figures/fig_time.png')