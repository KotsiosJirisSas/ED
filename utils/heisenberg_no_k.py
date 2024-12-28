import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from scipy.sparse import csr_matrix #optimizes H . v operations. to check if H already row sparse, do  isspmatrix_csr(H)
from scipy.sparse.linalg import eigsh
############################
##### Creates Sz basis #####
############################
def makeSzBasis(L):
    '''
    goes through hilbert space and assigns each i to a list depending on its Sz
    spin which is equal to #down - #up = #of 1's in the binary representation 
    of the state 
    '''
    basisSzList = [[] for i in range(0,2*L+1,2)] #S_z can range from -L to L, index that way as well
    #this is probably a bad way to do it
    # count bits is O(log(n)) and loop is O(2**L) :(
    for i in range(2**L):
        Szi = 2*countBits(i) - L
        basisSzList[(Szi+L)//2].append(i)
    print("L =",L,"basis size:",2**L)
    return basisSzList

def MzBasis(L,mz):
    '''
    Given a target magnetization mz loop through entire space and find 
    which states lie in this sector.
    mz = -L/2,-L/2+1,...,+L/2
    '''
    a = 0 #initialize state counter in magnetization block
    Slist = [] # list of states in increasing order
    num_up = mz + L/2
    for i in range(2**L):
        if countBits(i) == num_up:
            a += 1
            Slist.append(i)
    return a,Slist
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
def constructH(L):
    H = np.zeros((2**L,2**L),dtype=float)
    for s in range(2**L):
        for i in range(L):
            j = (i+1)%L # mod takes care of boundary
            #if i and i+1 have same spin, then S^+S^- terms vanish and only S^z S^z remains
            if (s>>i & 1) == (s>>j & 1):
                H[s,s] += 1./4
            # if i and j have different spin, then hamiltonian maps s-->s' that is a state same as s but with spins in i,j flipped.
            else:
                H[s,s] -= 1./4
                s2 = flip(s,i,j)
                H[s,s2] = 1./2
    print(np.all(H==H.T))
    return H
def constructH_fixedMz(L,mz):
    M,S = MzBasis(L,mz)
    H = np.zeros((M,M),dtype=float)
    for m in range(M):
        s = S[m]
        for i in range(L):
            j = (i+1)%L # mod takes care of boundary
            #if i and i+1 have same spin, then S^+S^- terms vanish and only S^z S^z remains
            if (s>>i & 1) == (s>>j & 1):
                H[m,m] += 1./4
            # if i and j have different spin, then hamiltonian maps s-->s' that is a state same as s but with spins in i,j flipped.
            else:
                H[m,m] -= 1./4
                s2 = flip(s,i,j)
                #basisVisualizer(L,s)
                #basisVisualizer(L,s2)
                #print('-')
                n = findstate(S,s2)
                #print(m,'--->',n)
                #print(basisVisualizer(L,s),'--->',basisVisualizer(L,s2))
                #print('-')
                H[m,n] = 1./2
                #if m == 0 and n == 1:
                #    print('---')
                #    print(i,j)
                #    basisVisualizer(L,s)
                #    basisVisualizer(L,s2)
    print(np.all(H==H.T))
    return H 
############################
##### Solve H full #########
############################
def getSpectrum(L):
    '''Returns lowestEnergy, 
               Sz sector of the GS, 
               GS eigenvector, 
               and all energies'''
    '''
    mz goes from -L/2 to L/2
    '''
    energies = []
    lowestEnergy = 1e10

    for mz in np.arange(-L/2,L/2+1,+1):
       H = constructH_fixedMz(L,mz)
       print('=============')
       print('diagonalizing mz sector',mz)
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
###########################
#### VISUALIZATION ########
###########################
def generatePlots(L):
    (lowestEnergy,GSSector,
     GSEigenvector,energies) = getSpectrum(L)
    total_energies = [en for szlist in energies for en in szlist]
    maxE           = np.max(total_energies)
    offset = 0
    for i in range(len(energies)):
        plt.plot(range(offset,len(energies[i])+offset),energies[i],'o')
        offset+=len(energies[i])
        if len(energies)-4>i>2:
            if i%2==0:
                plt.text(offset-200,maxE+1,"Sz="+str(-L/2+i))
            else:
                plt.text(offset-200,maxE+0.5,"Sz="+str(-L/2+i))

    plt.xlabel("Arbitrary Order",fontsize=15)
    plt.ylim([lowestEnergy-0.5,maxE+2])
    plt.ylabel("Energy",fontsize=15)
    #plt.title(r"XXZ model with $L="+str(L)+",\,\,\, J_z="+str(Jz)+",\,\,\, J_{xy}="+str(Jxy)+"$",fontsize=15)
    plt.plot([0,offset],[lowestEnergy,lowestEnergy],'--',label="Ground State")
    plt.legend(loc='lower right')
    plt.show()
    print('====')
    #now plot the distribution of the lowest energy eigenstate in its symmetry sector. ie |psi_GS> = sum_n c_n |psi_n>
    plt.plot(GSEigenvector,'o-')
    plt.xlabel("state order",fontsize=15)
    plt.ylabel(r"$|\psi_0\rangle$",fontsize=15)
    #plt.title("Ground State Eigenvector \nwith $L="+str(L)+",\,\,\, J_z="+str(Jz)+",\,\,\, J_{xy}="+str(Jxy)+"$",fontsize=15)
    plt.show()
    
    basisSzList   = MzBasis(L,GSSector)
    GSEigenvector = np.abs(np.round(GSEigenvector,10)) #rounding errors
    bigStatesID   = np.argwhere(np.abs(GSEigenvector) == np.max((GSEigenvector))).reshape((1,-1))
    
    #Get the states
    print("The biggest states are:")
    print(bigStatesID)
    #for state in bigStatesID[0]:
    #    bigStates = basisSzList[state]
    #    basisVisualizer(L,bigStates)

############################
#####HELPER FUNCTIONS#######
############################
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