'''
I want to investigate how the 'classical' system

H = \sum_i UN_i^2 -\mu N_i

that is diagonal in the number basis behave with altering the chemical potential.
Ie i am interested in the thermal average of <n> as a function of \mu.

The idea is to see how roughly adding one particle to the system costs ~U and how in the cases

\\beta U >> 1 

we can project our hilbert space to that which only has \hat{n} near the thermal expectation value for that chemical potential
'''
import numpy as np
import matplotlib.pyplot as plt
def basis(Occ,Sites):
    '''
    store as binary strings
    '''
    return [f"{x:0{Occ*Sites}b}" for x in range(2 ** (Occ*Sites))]

def H(basis,mu,U,Occ,Sites):
    Es = np.zeros(len(basis),dtype=float)
    Occupancy_s = np.zeros(len(basis),dtype=float)
    for i,s in enumerate(basis):
        for site in range(Sites):
            Nhat = occupancy(s[Occ*site:Occ*site+Occ])
            Es[i] += 0.5*U*Nhat**2 - mu*Nhat
            Occupancy_s[i] += Nhat
    return Es,Occupancy_s
def occupancy(s):
    '''
    takes a binary string of length Occ and counts its ones
    '''
    state = int(s,2)
    #print(s,state,countBits(state))
    return countBits(state)
def PartFunc(basis,beta,Es):
    '''
    calculates partition function
    '''
    Z = 0
    for i,s in enumerate(basis):
        Z += np.exp(-beta*Es[i]) 
    return Z
def N_avg(basis,beta,Es,Occupancy_s):
    '''
    calculates avg occupation number
    '''
    navg = 0
    for i,s in enumerate(basis):
        navg += Occupancy_s[i]*np.exp(-beta*Es[i]) 
    return navg
def N_sq_avg(basis,beta,Es,Occupancy_s):
    '''
    calculates avg occupation number
    '''
    nsqavg = 0
    for i,s in enumerate(basis):
        nsqavg += (Occupancy_s[i]**2)*np.exp(-beta*Es[i]) 
    return nsqavg
def E_avg(basis,beta,Es):
    '''
    calculates avg energy Tr(Hexp(-\\betaH))
    '''
    eavg = 0
    for i,s in enumerate(basis):
        eavg += Es[i]*np.exp(-beta*Es[i]) 
    return eavg
def E_sq_avg(basis,beta,Es):
    '''
    calculates avg energy Tr(H**2exp(-\\betaH))
    '''
    esqavg = 0
    for i,s in enumerate(basis):
        esqavg += (Es[i]**2)*np.exp(-beta*Es[i]) 
    return esqavg
def countBits(x):
    '''Counts number of 1s in bin(n)'''
    #From Hacker's Delight, p. 66
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x0000003F
def basis_Proj(basis,Occ,Sites,minN,maxN):
    '''
    if state s has occupation outside of min,max it gets projected out
    '''
    basis_new = []
    for i,s in enumerate(basis):
        N = 0
        for site in range(Sites):
            N += occupancy(s[Occ*site:Occ*site+Occ])
        if N >= minN and N <= maxN:
            basis_new.append(s)
    return basis_new
def count_ones_between_flips(binary1, binary2):
    '''
    Used for ***SIGN*** due to fermion anticommutators
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

if __name__ == "__main__":
    '''
    doing static susceptibilities
    '''
    Occ = 7 #S=11/2
    Sites = 2 #number of sites
    U = 5
    mu = 3 
    betas = np.linspace(1,20,num=50)#leads t <N> =3
    #betas = #np.array(list(np.linspace(0.01,1,num=50))+list(np.linspace(1,20,num=50)))
    Energies = np.zeros(len(betas))
    Energies_sq = np.zeros(len(betas))
    Numbers = np.zeros(len(betas))
    Numbers_sq = np.zeros(len(betas))
    Basis = basis(Occ,Sites)
    for i,beta in enumerate(betas):
        Es,Occupancy_s = H(Basis,mu,U,Occ,Sites)
        Z = PartFunc(Basis,beta,Es)
        Eavgs = E_avg(Basis,beta,Es)
        E_sq_avgs = E_sq_avg(Basis,beta,Es)
        Navgs = N_avg(Basis,beta,Es,Occupancy_s)
        N_sq_avgs = N_sq_avg(Basis,beta,Es,Occupancy_s)
        Energies[i] = Eavgs/Z
        Energies_sq[i] = E_sq_avgs/Z
        Numbers[i] = Navgs/Z
        Numbers_sq[i] = N_sq_avgs/Z
    plt.plot(1/betas,(betas**2) * (Energies_sq - Energies**2) ,'.',c='b')
    #plt.plot(np.log(1/betas),(betas**2) * (Numbers_sq - Numbers**2) ,c='r')
    plt.xlabel('$\log{(1/\\beta)}$')
    #plt.xticks([1,3,5,7,9])
    #plt.ylabel('$\langle N \\rangle$')
    #plt.legend()
    plt.title('$\\mu=3$')
    plt.savefig('chi_charge.png',dpi=800)


    '''
    doing plot of energy vs temperature.
    For T--->0, nergy = ground state nergy as thermal fluctuations get killed off
    '''

    '''
    Occ = 12 #S=11/2
    Sites = 1 #number of sites
    U = 1
    mu = 3 #leads t <N> =3
    betas = np.linspace(5,20,num=100)
    Energies = np.zeros(len(betas))
    Numbers = np.zeros(len(betas))
    Basis = basis(Occ,Sites)
    for i,beta in enumerate(betas):
        Es,Occupancy_s = H(Basis,mu,U,Occ,Sites)
        Z = PartFunc(Basis,beta,Es)
        Eavgs = E_avg(Basis,beta,Es)
        Navgs = N_avg(Basis,beta,Es,Occupancy_s)
        Energies[i] = Eavgs/Z
        Numbers[i] = Navgs/Z
    #plt.plot(1/betas,Energies,c='b')
    plt.plot(1/betas,Numbers,c='r')
    plt.xlabel('$1/\\beta$')
    #plt.xticks([1,3,5,7,9])
    plt.ylabel('$\langle N \\rangle$')
    #plt.legend()
    plt.title('$\\mu=3$')
    plt.savefig('N_vs_T.png',dpi=800)
    '''
'''
    working out the boltzman weight stored in states away from filling dictated by \mu
'''

'''
    Occ = 12
    Sites = 1
    U = 1
    mu = 3 # ensures <N> = 3
    betas = np.linspace(2,8,num=50)
    Basis = basis(Occ,Sites)
    Basis_Proj1 = basis_Proj(Basis,Occ,Sites,2,4)
    Basis_Proj2 = basis_Proj(Basis,Occ,Sites,1,5)
    frac1 = np.zeros(len(betas),dtype = float)
    frac2 = np.zeros(len(betas),dtype = float)
    for i,beta in enumerate(betas):
        Es,Occupancy_s = H(Basis,mu,U,Occ,Sites)
        Z = PartFunc(Basis,beta,Es)

        Es_proj1,Occupancy_s = H(Basis_Proj1,mu,U,Occ,Sites)
        Es_proj2,Occupancy_s = H(Basis_Proj2,mu,U,Occ,Sites)
        
        Zproj1 = PartFunc(Basis_Proj1,beta,Es_proj1)
        Zproj2 = PartFunc(Basis_Proj2,beta,Es_proj2)
        
        frac1[i] = 1-Zproj1/Z
        frac2[i] = 1-Zproj2/Z
    plt.plot(betas,np.log(frac1),'r',label='$\delta N = 1$')
    plt.plot(betas,np.log(frac2),'b',label='$\delta N = 2$')
    plt.ylabel('$\log (Z_{\perp}/Z)$')
    plt.xlabel('$\\beta U$')
    plt.legend()
    plt.savefig('projected_boltzmann_weight.png',dpi=800)
'''


'''
    doing plot of <n> vs \mu
    '''
'''
    Occ = 12 #S=11/2
    Sites = 1 #number of sites
    U = 1
    beta = 7
    mus = np.linspace(0,10,num=100)
    Ns = np.zeros(len(mus))
    Basis = basis(Occ,Sites)
    for i,mu in enumerate(mus):
        Es,Occupancy_s = H(Basis,mu,U,Occ,Sites)
        Z = PartFunc(Basis,beta,Es)
        Navgs = N_avg(Basis,beta,Es,Occupancy_s)
        Ns[i]=Navgs/Z
    plt.plot(mus/U,Ns,c='g',label='$\\beta U$='+str(beta*U))
    beta = 10
    mus = np.linspace(0,10,num=100)
    Ns = np.zeros(len(mus))
    Basis = basis(Occ,Sites)
    for i,mu in enumerate(mus):
        Es,Occupancy_s = H(Basis,mu,U,Occ,Sites)
        Z = PartFunc(Basis,beta,Es)
        Navgs = N_avg(Basis,beta,Es,Occupancy_s)
        Ns[i]=Navgs/Z
    plt.plot(mus/U,Ns,c='b',label='$\\beta U$='+str(beta*U))
    beta = 15
    mus = np.linspace(0,10,num=100)
    Ns = np.zeros(len(mus))
    Basis = basis(Occ,Sites)
    for i,mu in enumerate(mus):
        Es,Occupancy_s = H(Basis,mu,U,Occ,Sites)
        Z = PartFunc(Basis,beta,Es)
        Navgs = N_avg(Basis,beta,Es,Occupancy_s)
        Ns[i]=Navgs/Z
    plt.plot(mus/U,Ns,c='r',alpha=0.7,label='$\\beta U$='+str(beta*U))
    beta = 18
    mus = np.linspace(0,10,num=100)
    Ns = np.zeros(len(mus))
    Basis = basis(Occ,Sites)
    for i,mu in enumerate(mus):
        Es,Occupancy_s = H(Basis,mu,U,Occ,Sites)
        Z = PartFunc(Basis,beta,Es)
        Navgs = N_avg(Basis,beta,Es,Occupancy_s)
        Ns[i]=Navgs/Z
    plt.plot(mus/U,Ns,c='k',alpha=0.7,label='$\\beta U$='+str(beta*U))
    plt.xlabel('$\\mu/U$')
    #plt.xticks([1,3,5,7,9])
    plt.ylabel('$\langle N \\rangle$')
    plt.legend()
    plt.savefig('test_toy.png',dpi=800)
    '''