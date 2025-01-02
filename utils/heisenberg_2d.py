#################
####IMPORTS######
#################
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from scipy.sparse import csr_matrix,coo_matrix #optimizes H . v operations. to check if H already row sparse, do  isspmatrix_csr(H)
from scipy.sparse.linalg import eigsh
import time
###########################
#######HELPER FUNCTIONS####
###########################
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
def checkstate(a):
    '''
    to be edited later. used in mzstategenerator to check if state a is a valid reprersentative re translation/ other discreet symms.
    '''
    return True
def mzstategenerator(L, a, vis = False):
    '''
    Starting from the *smallest* state (in terms of integer s), generate all other states
    with the same magnetization (i.e., number of 1's in binary representation).
    so eg 000111->001011->001101->001110->010110->...->111000
    ---------------------------
    Notes:
    0) Input state should be 2**(n_up)-1 which is 000....011...1  w/ n_up 1's.This state has mz = n_up - L/2
    1) cancel the visulaization part for speedup
    2) when i'm creating the bits i 'reverse the order' with the [::-1] 
       because the algo picks up the i-th *righmost* (smallest) bit which would correspond to 
       the L-i'th element of the list had it not been reversed. at the final step i reverse again.
    ---------------------------
    Time comparison vs looping thorugh all states and checking their bits: (only aplying mz convergence)
    -----|  L=26 | L=28 |
    Dumb |  26.8s|111.1s|
    -----|-------|------|
    Smart|  17.4s| 76.1s|
    => already a speed up of 50% which should go up maybe with L.
    Since building the basis and Hamiltonian is the thing taking up so much time, not Lanczos, this is great!
    Below L ~ 22 there's actually no difference and maybe the dump way is faster due to overhead.
    '''
    states = []  # List to store all generated states
    
    while True:
        # Visualize the current state
        if vis == True: basisVisualizer1D(L, a)
        
        # Add the current state to the list
        if checkstate(a):
            states.append(a)
        
        # Convert integer `a` to binary list, reverse for easier manipulation
        a_bits = list(bin(a)[2:].zfill(L))[::-1]
        c = 0  # Count consecutive 1's encountered so far
        for b in range(1, L):
            if a_bits[b-1] == '1':
                if a_bits[b] == '1':
                    c += 1
                else:
                    # Update state for next iteration
                    if c == 0:
                        for i in range(c, b):
                            a_bits[i] = '0'
                    else:
                        for i in range(0, c):
                            a_bits[i] = '1'
                        for i in range(c, b):
                            a_bits[i] = '0'
                    a_bits[b] = '1'
                    # Convert back to integer and update `a`
                    a = int(''.join(a_bits[::-1]), 2)
                    break
        else:
            # If we reach the end of the for loop without finding a valid transition
            return states
def mzstategenerator2(Lx,Ly, a,kx,ky, vis = False):
    '''
    Starting from the *smallest* state (in terms of integer s), generate all other states
    with the same magnetization (i.e., number of 1's in binary representation).
    so eg 000111->001011->001101->001110->010110->...->111000
    ---------------------------
    Notes:
    0) Input state should be 2**(n_up)-1 which is 000....011...1  w/ n_up 1's.This state has mz = n_up - L/2
    1) cancel the visulaization part for speedup
    2) when i'm creating the bits i 'reverse the order' with the [::-1] 
       because the algo picks up the i-th *righmost* (smallest) bit which would correspond to 
       the L-i'th element of the list had it not been reversed. at the final step i reverse again.
    ---------------------------
    Time comparison vs looping thorugh all states and checking their bits: (only aplying mz convergence)
    -----|  L=26 | L=28 |
    Dumb |  26.8s|111.1s|
    -----|-------|------|
    Smart|  17.4s| 76.1s|
    => already a speed up of 50% which should go up maybe with L.
    Since building the basis and Hamiltonian is the thing taking up so much time, not Lanczos, this is great!
    Below L ~ 22 there's actually no difference and maybe the dump way is faster due to overhead.
    '''
    states = []  # List to store all generated states
    norms = []
    L = Lx*Ly
    while True:
        # Visualize the current state
        #if vis == True: basisVisualizer1D(L, a)
        #####
        i2c,c2i = create_lattice_mapping(Lx,Ly)
        bonds = get_bonds(c2i,i2c,Lx,Ly)
        Tx,Ty = get_TxTy(Lx,Ly,i2c,c2i)
        translations = precompute_translations(Lx,Ly,Tx,Ty)
        #####
        # Add the current state to the list
        N = compatibility_k(translations=translations,Lx=Lx,Ly=Ly,kx=kx,ky=ky,s=a)
        if N != -1:
            states.append(a)
            norms.append(N)
            if vis == True: basisVisualizer2D(Lx,Ly, a)
        
        # Convert integer `a` to binary list, reverse for easier manipulation
        a_bits = list(bin(a)[2:].zfill(L))[::-1]
        c = 0  # Count consecutive 1's encountered so far
        for b in range(1, L):
            if a_bits[b-1] == '1':
                if a_bits[b] == '1':
                    c += 1
                else:
                    # Update state for next iteration
                    if c == 0:
                        for i in range(c, b):
                            a_bits[i] = '0'
                    else:
                        for i in range(0, c):
                            a_bits[i] = '1'
                        for i in range(c, b):
                            a_bits[i] = '0'
                    a_bits[b] = '1'
                    # Convert back to integer and update `a`
                    a = int(''.join(a_bits[::-1]), 2)
                    break
        else:
            # If we reach the end of the for loop without finding a valid transition
            return states
#####
#lattice
#####
def create_lattice_mapping(Lx, Ly):
    """
    Create mappings for a 2D lattice:
    - i to (x, y)
    - (x, y) to i
    Args:
        Lx (int): Number of columns in the lattice.
        Ly (int): Number of rows in the lattice.
    Returns:
        tuple: Two dictionaries:
            - index_to_coords: Maps i to (x, y).
            - coords_to_index: Maps (x, y) to i.
    """
    index_to_coords = {}
    coords_to_index = {}
    for y in range(Ly):
        for x in range(Lx):
            i = x + y * Lx
            index_to_coords[i] = (x, y)
            coords_to_index[(x, y)] = i
    return index_to_coords, coords_to_index
def get_TxTy(Lx, Ly, i2c, c2i):
    """
    Generate translation dictionaries Tx and Ty for x- and y-translations.
    Args:
        Lx (int): Number of columns in the lattice.
        Ly (int): Number of rows in the lattice.
        i2c (dict): Mapping from index (i) to coordinates (x, y).
        c2i (dict): Mapping from coordinates (x, y) to index (i)
    Returns:
        tuple: Two dictionaries:
            - Tx: Maps i to i' for translation in x-direction.
            - Ty: Maps i to i' for translation in y-direction.
    """
    Tx = {}
    Ty = {}
    for i in i2c.keys():
        x, y = i2c[i]  # Get the (x, y) coordinates for index i
        # Translation in x-direction
        xnew = (x - 1) % Lx  # Move left (wrap around)
        Tx[i] = c2i[(xnew, y)]
        # Translation in y-direction
        ynew = (y - 1) % Ly  # Move down (wrap around)
        Ty[i] = c2i[(x, ynew)]
    return Tx, Ty

def get_bonds(c2i,i2c,Lx,Ly):
    '''
    Generates a mapping a'th bond to pair of sites (i,j)~((x1,y1),(x2,y2))
    -----------
    Q: should i place an ordering on the sites? ie (i,j) describe a bond and must have i<j?
    '''
    bonds = []
    for i in i2c.keys():
        (x,y) = i2c[i]
        x1 = (x+1)%Lx
        y1 = (y+1)%Ly
        j1 = c2i[(x1,y)]
        j2 = c2i[(x,y1)]
        bonds.append((i,j1))
        bonds.append((i,j2))
    return bonds
def translate_spin_state(s,L, T):
    """
    Translate a spin state using a given translation dictionary.
    Args:
        s (str): Original spin state as an integer
        T (dict): Translation dictionary (e.g., Tx or Ty) mapping indices.
    Returns:
        str: Translated spin state.
    --------------------------------------
    Q: am i translating correctly or in the 'opposite' way? all is convention anyway

    """
    #L = Lx*Ly
    s_bin = bin(s)[2:].zfill(L)
    # Create a list for the translated spin state
    s_bin_translated = ['0'] * len(s_bin)
    # Rearrange the spins based on the translation map
    for i in range(len(s_bin)):
        s_bin_translated[T[i]] = s_bin[i]
    # Convert the list back to a string
    return int(''.join(s_bin_translated), 2)
#######
def compose_translation(T, L, times):
    """
    Compose a translation dictionary T multiple times.

    Args:
        T (dict): Translation dictionary (e.g., Tx or Ty).
        L (int): Total number of spins (Lx * Ly).
        times (int): Number of times to apply the translation.

    Returns:
        dict: Composed translation dictionary.
    """
    T_composed = {i: i for i in range(L)}  # Start with the identity mapping
    for _ in range(times):
        T_composed = {i: T[T_composed[i]] for i in range(L)}
    return T_composed
def build_Tnm(Tx, Ty, L, n, m):
    """
    Build the combined translation dictionary Tnm = Tx^n Ty^m.

    Args:
        Tx (dict): x-translation dictionary.
        Ty (dict): y-translation dictionary.
        L (int): Total number of spins (Lx * Ly).
        n (int): Number of x-translations.
        m (int): Number of y-translations.

    Returns:
        dict: Combined translation dictionary Tnm.
    """
    # Compose Ty m-times
    Ty_m = compose_translation(Ty, L, m)
    # Compose Tx n-times, starting from Ty^m
    Tnm = compose_translation(Tx, L, n)
    Tnm = {i: Tnm[Ty_m[i]] for i in range(L)}
    return Tnm
def precompute_translations(Lx, Ly, Tx, Ty):
    """
    Precompute Tnm dictionaries for a range of (n, m) translations.

    Args:
        Lx (int): Number of columns in the lattice.
        Ly (int): Number of rows in the lattice.
        Tx (dict): x-translation dictionary.
        Ty (dict): y-translation dictionary.
        max_n (int): Maximum x-translations to precompute.
        max_m (int): Maximum y-translations to precompute.

    Returns:
        dict: A dictionary where keys are (n, m) pairs, and values are Tnm dictionaries.
    """
    max_n = Lx
    max_m = Ly
    L = Lx * Ly
    translations = {}
    for n in range(max_n):
        for m in range(max_m):
            Tnm = build_Tnm(Tx, Ty, L, n, m)
            translations[(n, m)] = Tnm
    return translations

def compatibility_k(translations,Lx,Ly,kx,ky,s):
    '''
    Starting from a state, check if
        1) it is the representative
        2) if it is compatible with momentum k
    If both tests pass, output:
        1) The state's normalization
    '''
    D = 1 # counts number of unique states
    F = 0 # the complex phase
    N = -1 # normalization/pass
    #start generateing translations. as soon as you hit a state that has smaller integer representation, exit. This should speed up things.
    for n in range(Lx):
        for m in range(Ly):
            Tnm = translations[(n,m)]
            t = translate_spin_state(s,L=Lx*Ly,T = Tnm)
            if t < s:
                return N
            elif t > s:
                D += 1
            elif t == s:
                    #we came across og state
                    F += np.exp(-1j*(kx*n/Lx+ky*m/Ly)*2*np.pi)
    N = float(D * np.abs(F)**2)
    if N < 1e-5:
        N = -1         
    return N
######

#####
#####

def basisVisualizer1D(L,psi):
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
    return
def basisVisualizer2D(L1,L2,psi):
    '''
    turns an integer s representing the state into a picture of spins
    '''
    uparrow   = '\u2191'
    downarrow = '\u2193'
    #convert psi into binary and fill up states. eg 1->000001 or whatever
    total_sites = L1 * L2
    psi_2 = bin(psi)[2:].zfill(total_sites)
    #print each row
    for y in range(L2):
        row = psi_2[y*L1:(y+1)*L1][::-1]
        line = ''.join(uparrow if bit == '1' else downarrow for bit in row)
        print(line)
    print('*'*4)
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
