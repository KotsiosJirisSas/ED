##########
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import gc
from scipy.special import logsumexp
import pickle
from collections import defaultdict
import h5py
sys.path.append('/mnt/users/kotssvasiliou/ED/core_scripts')
from hubbard_chain import hubbard_chain,thermodynamics,EDFullSpectrum
##########
class spinless_chain():
    '''
    A very easy class instance for spinless PBC bosons/fermions
    Creates basis
    Creates Hamiltonian
    Diagonalizes
    Does thermodynamics
    Does parity resolved Correlation fucntions both at T=0 and at finite T
    '''
    def __init__(self,params):
        '''
        Initializes params
        '''
        self.L = params['L']
        self.species = params['species']#'fermion' or 'boson'
        self.JWstring = params['JWstring']
        if self.species == 'fermion' and self.JWstring:
            print('for JW string we must have underlying bosonic dofs')
            raise ValueError
        self.t = params['t']
        self.V = params['V']
        self.mu = params['mu']
        if self.V != 0:
            print('nn repulsion not yet implemented')
            raise NotImplementedError
    def create_basis(self):
        '''
        Creates occupation basis using the decimal representation of binary states

        
        Args:
            L:                      Chain length
        Returns:
            basis(dict):          A dict of all states in the hilbert space of spin s
            index(dict):          The reverse of the basis_s... is it really necessary?
            len_basis(int):       The size of the Hilbert space for spin s
        '''
        L = self.L
        basis = {}
        index = {}
        count = 0
        for s in range(2**L):
            basis[count] = s
            index[s] = count
            count += 1
                
        self.basis = basis
        self.index = index
        self.dim = len(basis)
        self.Hamiltonian = np.zeros((self.dim,self.dim),dtype=float)
        return
    def hop_ij(self,i,j,m):
        '''
        Function:
        ---------
        Adds to the hamiltonian the elements due to hopping between site i and j of the *spinless* state with index m. 

        Input:
        -------
        i,j(int):       i,j, \in [0,L-1]; The sites on which hopping happens
        m(int):         The spinless state that is to be hopped from i to j

        '''
        s1 = self.basis[m]
        s2 = self.hop(i,j,s1)
        if s2 == -1:
            return
        else:
            try:
                n = self.index[s2]
            except ValueError:
                print('Index of new state not found during hopping...raising Value error')
                raise ValueError
            #get sign                 
            if self.species == 'fermion':
                sgn = self.fermion_sgn(self.binp(s1,length=self.L),self.binp(s2,length=self.L))
                if (sgn == -1) and (self.verbose>0):print('fermion negative sign',i,j,m)
            elif self.species == 'boson':
                sgn = 1
            else:
                raise ValueError
            self.Hamiltonian[m,n] += -self.t*sgn #NOTE: ANy issues with transpose etc?
            return
    def construct_H(self):
        '''
        constructs the hamiltonian on the chain with nearest neighbor hopping and a chemical potential term
        '''
        for i in range(self.L):
            j=(i+1)%self.L
            for m in self.basis:
                self.hop_ij(i,j,m)
        for m in self.basis:
            s = self.basis[m]
            for i in range(self.L):
                occ = self.occupancy(s,i)
                self.Hamiltonian[m,m] += -self.mu*occ
        return
    def solve(self,mode='full',shift = 'True'):
        '''
        solve hamiltonian
        shift: shifts GS to be at zero
        '''
        ####
        if not hasattr(self,'Hamiltonian'):
            self.create_basis()
            self.construct_H()
        ####
        if mode != 'full':
            print('Lanczos not yet Implemented')
            raise NotImplementedError
        lam,v = np.linalg.eigh(self.Hamiltonian)
        self.es = lam
        self.vs = v
        self.GS = np.min(self.es)
        self.GSindex = np.argmin(self.es)
        if shift:
           self.es -= self.GS
        return
    #####################
    ###### THERMO #######
    #####################
    def logZ(self,beta,parity):
        '''
        Function:
        ---------
        Calculates the partition function in a given parity sector \n
        
        TODO Add a generic function to sort parity depending on structure of symmetry sectors, making it compatible with more systems

        Input:
        ------
        beta(float):        The inverse temperature \n
        parity:             None,0 or 1. The parity of the sectors we are considering \n

        Output:
        -------
        logZ:               The log of the partiton function
        ''' 
        def parity_check(v, parity):
            '''
            Checks whether the eigenstate v has the desired parity (even=0 or odd=1)
            
            Inputs:
            -------
            v       : ndarray, eigenvector in physical basis
            parity  : int or None (0 for even, 1 for odd)
            basis   : dict mapping index -> basis state (int)
            L       : int, system size
            
            Returns:
            --------
            bool
            '''
            if parity is None:
                return True
            
            for idx, amplitude in enumerate(v):
                if np.abs(amplitude) > 1e-12:  # ignore numerical noise
                    state = self.basis[idx]
                    total_occ = sum((state >> i) & 1 for i in range(self.L))
                    return total_occ % 2 == parity
            return False  # shouldn't happen if v is normalized
        
        all_energies = []
        # Gather all energies
        for n,e in enumerate(self.es):
            v = self.vs[n]
            if parity_check(v,parity):
                all_energies.append(e)
        all_energies = np.array(all_energies)
        log_Z_P = logsumexp(- beta * all_energies)
        return log_Z_P
    def H_moments(self,beta,Cv = False):
        '''
        Function:
        ---------
        Calculates <H> and <H^2>, as well as the specific heat, if asked to.Uses logsumexp for numerical stability. \n
        Takes care to filter out E=0 (GS) as their log is ill defined and they shouldn't contribute to shifted expectation values \n
    
        Input:
        ------
        beta(float)         :The inverse temperature \n
        Cv(Boolean)         :Calculate the specific heat? C_v = β (<H^2> - <H>^2)/L \n

        Output:
        -------
        <H> \n
        <H_unshifted> \n
        <H^2> \n
        <H^2_unshifted> \n
        C_v \n
        '''
        log_Z = self.logZ(beta,parity = None)
        log_terms_H = []
        log_terms_H_sq = []
        for e in self.es:
            if e > 0:
                log_terms_H.append((-beta * e) + np.log(e))
                log_terms_H_sq.append((-beta * e) + np.log(e**2))
        log_H = logsumexp(np.array(log_terms_H)) - log_Z
        log_H_sq = logsumexp(np.array(log_terms_H_sq)) - log_Z

        ##################################
        #while not physically relevant, return also the values with shifted energy.
        H_avg = np.exp(log_H)
        H_avg_unshifted = H_avg + self.GS
        H_sq_avg = np.exp(log_H_sq)
        H_sq_avg_unshifted = H_sq_avg +2*self.GS*H_avg+self.GS**2

        if Cv == False:
            return H_avg,H_avg_unshifted,H_sq_avg,H_sq_avg_unshifted
        else:
           raise NotImplementedError
    def op_lookup_tau_zero(self):
        '''
        Stores info about
        <i|O^\dagger_r1 O_r2|j>
        which is the correlator matrix elements at tau = 0
        '''
        G = {}
        for r1 in range(self.L):
            for r2 in range(self.L):
                G[(r1,r2)] = {}
                for i in self.basis:
                    #act with c^dagger_r2
                    if (i>>r2) & 1:#means r2 is full. cant act with cdag
                        continue
                    i1 = i | (1<<r2)
                    #get sign.
                    if self.species == 'fermion':
                        sgn1 = self.fermion_sign_2(i,r2)
                    elif self.JWstring == True:
                        sgn1 = self.fermion_sign_2(i,r2)
                    else:
                        sgn1 = 1
                    #act with c_r1
                    if not ((i1 >> r1) & 1):#means r1 is empty. cant act with c
                        continue
                    j = i1 & ~(1 << r1)
                    #get sign.
                    if self.species == 'fermion':
                        sgn2 = self.fermion_sign_2(i1,r1)
                    elif self.JWstring == True:
                        sgn2 = self.fermion_sign_2(i1,r1)
                    else:
                        sgn2 = 1
                    G[(r1,r2)][j,i] = sgn1*sgn2
        return G
    def Green_function_tau_0(self,beta,parity = None):
        '''
        Use the greens function map on the occupation basis to calculate parity resolved greens functions
        '''
        def parity_check(v, parity):
            '''
            Checks whether the eigenstate v has the desired parity (even=0 or odd=1)
            
            Inputs:
            -------
            v       : ndarray, eigenvector in physical basis
            parity  : int or None (0 for even, 1 for odd)
            basis   : dict mapping index -> basis state (int)
            L       : int, system size
            
            Returns:
            --------
            bool
            '''
            if parity is None:
                return True
            
            for idx, amplitude in enumerate(v):
                if np.abs(amplitude) > 1e-12:  # ignore numerical noise
                    state = self.basis[idx]
                    total_occ = sum((state >> i) & 1 for i in range(self.L))
                    return total_occ % 2 == parity
            return False  # shouldn't happen if v is normalized
        ####
        log_Z = self.logZ(beta = beta,parity=parity)
        G_lookup = self.op_lookup_tau_zero()
        C = np.zeros((self.L,self.L),dtype = complex)
        for r1 in range(self.L):
            for r2 in range(self.L):
                for n in range(self.dim):
                    v = self.vs[:, n]
                    if parity_check(v,parity):
                        weight = 0
                        log_weight = -beta *self.es[n]
                        for (j,i),amp in G_lookup[(r1,r2)].items():
                            weight += np.conj(v[j]) * v[i] * amp
                        C[r1, r2] += weight * np.exp(log_weight - log_Z)
        if np.allclose(C,C.real):
            C = C.real
        else:
            print('not real')
        return C
    def correlator(self,op,beta,n_tau,parity=None,green_spin = 'up'):
        '''
        Function:
        ----------
        Using the operator eigen-matrrix elements <m|O_r1 exp(-tauH) O^dag_r2|m>, it calculates the dynamical correlator: \n
        C(r,r',τ) = <O_r(τ) O^\dagger_r'(0)> = (1/Z)x Σ_m {exp(-(β-τ)Em)<m|O_r1 exp(-tauH) O^dag_r2|m>}

        Input:
        ------
        op(str):            The operator name
        beta(float):        The inverse temperature
        n_tau(int):         The number of imaginary time slices
        parity:             None/0/1/GS. The parity resolution of the operator. If GS just means 


        Output:
        -------
        C(npcarray):        The L x L x Nτ correlator
        '''
        return
    
    #####################
    ##### HELPER ########
    #####################
    def fermion_sign_2(self,state, r):
        """
        Computes the fermionic sign (-1)^{sum_{j=0}^{r-1} n_j}
        when acting with c_r or c_r^\dagger on a spinless fermion state.

        Args:
            state (int): integer representing the occupation bitstring
            r (int): site where the operator acts
            L: total bin length

        Returns:
            int: +1 or -1
        """
        num_ones = self.binp(state & ((1 << r) - 1),length=self.L).count("1")
        return (-1) ** num_ones

    def occupancy(self,psi,i):
        '''
        Function:
        ---------
            Calculates occupancy of state psi at site i for a one species system on chain of length L \n
        Input:
        ------
            psi(int):       A one-species state psi = state \n
            i(int) :        i \in [0,L-1]; The site wwe are counting the occupation of \n
        Output:
        -------
            occ(int):       The occupation
        '''
        mask = 2**(i)
        occ = self.countBits(psi & mask)
        return occ
    @staticmethod
    def fermion_sgn(binary1,binary2):
        '''
        Function:
        ---------

        Takes two binary strings that are meant to be related by a flip, ie they only differ in two sites, s1=xxx0xxx1xxx and s2=xxx1xxx0xxx. \n
        It then counts the number of 1's that separate these flipped sites, and outputs (-1)**count \n
        This accounts for the anticommutative relations of the fermions \n

        NOTE Modified from 'count_ones_between_flips' function in hubbard_chains.py \n
        NOTE I think i could have just done 

        Input:
        ------
            binary1(str):       A binary string representing a spinless fermion state on a chain \n
            binaryw(str):       A binary string representing a spinless fermion state on a chain \n
        Return:
        ------
            sgn(int):           The sign relating these two states \n
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
        return (-1)**ones_count
    @staticmethod
    def countBits(x):
        '''Counts number of 1s in bin(n)'''
        #From Hacker's Delight, p. 66
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        x = x + (x >> 8)
        x = x + (x >> 16)
        return x & 0x0000003F
    @staticmethod
    def binp(num, length):
        '''
        print a binary number without python 0b and appropriate number of zeros
        regular bin(x) returns '0bbinp(x)' and the 0 and b can fuck up other stuff
        '''
        return format(num, '#0{}b'.format(length + 2))[2:]
    @staticmethod
    def hop(i,j,s):
        '''
        Function:
        ---------
        Checks if hopping is allowed between sites i and j for state s and if it is,it outputs the resulting state.
        This only treats a single spin as hopping preserves spin. Meaning, state_in and state_out are both single spin states.


        Input:
        -----
            s(bin):             A binary number with L digits(L=length of chain) signifying the state of the chain \n
            i(int),j(int):      0 =<i,j<L Integers representing sites on the chain \n

        Output:
        -----
            s2:                 Either -1 to signify no allowed hopping or a binary to denote the resulting state after the hopping \n
        '''
        mask = 2**(i)+2**(j)
        K = s & mask #bitwise AND.
        P = K ^ mask #bitwise XOR.
        if P == mask or P == 0:
            s2 = -1
        else:
            s2 = s - K + P
        return s2
#####################
def generate_benchmark_data_new(params,beta,mode):
    '''
    Calculate dynamical correlations for *mode* to be used for SSE benchmarking
    -----------------------------------------------------------
    params:     are the hamiltonian params to be fed into EDfullSpectrum
    mode:       Green_p/Green/Spin/Eta
    '''
    save_dir = '/mnt/users/kotssvasiliou/ED/benchmarks2/1d/dat_Dumitru'
    if params['species'] != 'boson':raise ValueError
    energies,eigenstates,bases,bases_inv,lowestEnergy = EDFullSpectrum(params)
    params['energies'] = energies
    params['eigenstates'] = eigenstates
    params['bases'] = bases
    params['bases_inv'] = bases_inv
    params['lowestEnergy'] = lowestEnergy
    params['JWstring'] = True
    thermo = thermodynamics(params)
    #####################
    #spin parity resolved green's function at tau = 0
    thermo = thermodynamics(params)
    n_tau = 11
    taus = np.linspace(0,beta, num=n_tau)
    if mode == 'Green_p':
        logZ_0 = thermo.logZ(beta=beta,parity = (0,0))
        logZ_1 = thermo.logZ(beta=beta,parity = (0,1))
        G_0_up = thermo.correlator(op='green',beta = beta,n_tau= 1,parity = (0,0),green_spin='up')
        G_1_up = thermo.correlator(op='green',beta = beta,n_tau= 1,parity = (0,1),green_spin='up')
        comp_0 = np.sum(np.abs(G_0_up.imag)) / np.sum(np.abs(G_0_up))
        comp_1 = np.sum(np.abs(G_1_up.imag)) / np.sum(np.abs(G_1_up))
        if (comp_0 <1e-8) and (comp_1<1e-8):
            G_0_up = G_0_up.real
            G_1_up = G_1_up.real
        else:
            raise ValueError
        Dat = {'G0':G_0_up,
               'G1':G_1_up,
               'logZ0':logZ_0,
               'logZ1':logZ_1}
    ########################
    #averaged  correlators for green's function, spin-spin and eta
    else:
        if mode == 'Green':
            Green = thermo.correlator(op='green',beta = beta,n_tau= n_tau,parity = None,green_spin='up')
            Dat = Green
        elif mode == 'Spin':
            Spin = thermo.correlator(op='spinspin',beta = beta,n_tau= n_tau,parity = None)
            Dat = Spin
        elif mode == 'Eta':
            Eta =  thermo.correlator(op='eta',beta = beta,n_tau= n_tau,parity = None)
            Dat = Eta
        else:
            raise ValueError
        #keep real part
        comp = np.sum(np.abs(Dat.imag)) / np.sum(np.abs(Dat))
        if (comp <1e-8):
            Dat = Dat.real
        else:
            raise ValueError
        print('tau = 0 data',Dat[:,:,0])
    Lsys = params['L']
    #save as pickle
    with open(save_dir+'/'+mode+f'_{Lsys}_beta_{beta}_data.pkl','wb') as f:
        pickle.dump(Dat,f)
    #save params
    params_save = params['H_params'].copy()
    params_save['beta'] = beta
    params_save['taus'] = taus
    with open(save_dir+f'/params_{Lsys}_beta_{beta}.pkl','wb') as f:
        pickle.dump(params_save,f)
    return  
def generate_benchmark_data(params,sector = None):
    '''
    Calculate dynamical correlations for *mode* to be used for SSE benchmarking
    -----------------------------------------------------------
    params:     are the hamiltonian params to be fed into EDfullSpectrum
    mode:       Green_p/Green/Spin/Eta
    '''
    energies,eigenstates,bases,bases_inv,lowestEnergy = EDFullSpectrum(params)
    params['energies'] = energies
    params['eigenstates'] = eigenstates
    params['bases'] = bases
    params['bases_inv'] = bases_inv
    params['lowestEnergy'] = lowestEnergy
    params['JWstring'] = True
    thermo = thermodynamics(params)
    #####################
    # E,N ns beta
    Beta_max = 10
    Betas = np.linspace(start=1,stop=Beta_max,num=10,endpoint=True)
    Es = []
    Ns = []
    print('GS:',lowestEnergy)
    for beta in Betas:
        Es.append(thermo.H_moments(beta=beta)[1])
        Ns.append(thermo.N_moments(beta=beta)[0])
    #print('Es',Es)
    #print('Ns',Ns)
    #####################
    #parity resolved green's function at tau = 0
    thermo = thermodynamics(params)
    n_tau = 30
    taus = np.linspace(0,Beta_max, num=n_tau)
    logZ_0 = thermo.logZ(beta=Beta_max,parity = 0)
    logZ_1 = thermo.logZ(beta=Beta_max,parity = 1)
    Z_0 = np.exp(logZ_0)
    Z_1 = np.exp(logZ_1)
    G_0 = thermo.correlator(op='green',beta = Beta_max,n_tau= n_tau,parity = 0,green_spin='up')
    G_1 = thermo.correlator(op='green',beta = Beta_max,n_tau= n_tau,parity = 1,green_spin='up')
    G_0_alt = thermo.correlator(op='green0',beta = Beta_max,n_tau= n_tau,parity = 0,green_spin='up')
    G_1_alt = thermo.correlator(op='green0',beta = Beta_max,n_tau= n_tau,parity = 1,green_spin='up')
    G_0_alt2 = thermo.correlator(op='green03',beta = Beta_max,n_tau= n_tau,parity = 0,green_spin='up')
    G_1_alt2 = thermo.correlator(op='green03',beta = Beta_max,n_tau= n_tau,parity = 1,green_spin='up')
    G = thermo.correlator(op='green',beta = Beta_max,n_tau= n_tau,parity = None,green_spin='up')
    Eta = thermo.correlator(op='eta',beta = Beta_max,n_tau= n_tau,parity = None)
    S = thermo.correlator(op='spinspin',beta = Beta_max,n_tau= n_tau,parity = None)
    G_0 = G_0.real
    G_1 = G_1.real
    G = G.real
    G_0_alt = G_0_alt.real
    G_1_alt = G_1_alt.real
    G_0_alt2 = G_0_alt2.real
    G_1_alt2 = G_1_alt2.real
    Eta = Eta.real
    S = S.real
    print('Z0',Z0)
    print('Z1',Z1)
    print('G0',G_0.shape)
    return  G_0,G_1,G_0_alt,G_1_alt,G_0_alt2,G_1_alt2,Eta,S

def investigate_data(params,beta):
    '''
    Investigates in small system sizes the stuff i want to investigate
    '''
    energies,eigenstates,bases,bases_inv,lowestEnergy = EDFullSpectrum(params)
    params['energies'] = energies
    params['eigenstates'] = eigenstates
    params['bases'] = bases
    params['bases_inv'] = bases_inv
    params['lowestEnergy'] = lowestEnergy
    thermo = thermodynamics(params)
    taus = np.linspace(0,beta, num=2,endpoint = True)
    logZ_0 = thermo.logZ(beta=beta,parity = 0)
    logZ_1 = thermo.logZ(beta=beta,parity = 1)
    Z_0 = np.exp(logZ_0)
    Z_1 = np.exp(logZ_1)
    G_0 = thermo.correlator(op='green',beta = beta,n_tau= 2,parity = (0,0),green_spin='up')
    G_1 = thermo.correlator(op='green',beta = beta,n_tau= 2,parity = (0,1),green_spin='up')
    G_0_dn = thermo.correlator(op='green',beta = beta,n_tau= 2,parity = (1,0),green_spin='dn')
    G_1_dn = thermo.correlator(op='green',beta = beta,n_tau= 2,parity = (1,1),green_spin='dn')
    print(np.allclose(G_0,G_0_dn))
    print(np.allclose(G_1,G_1_dn))
    return Z_0,Z_1,G_0.real,G_1.real
#####################
if __name__ == "__main__":

    '''
    params2 = {
        'L':5,
        'species':'boson',
        'verbose':1,
        'JWstring':True
    }
    params2['H_params'] = {
        't':1,
        'U':5,
        'V':0.7,
        'mu':-0
    }
    params2['diag_params'] = {'mode':'full'}
    beta = 1
    Z_0,Z_1,G_0,G_1 = investigate_data(params2,beta)
    print('Z0,Z1',Z_0,Z_1)
    print(np.round(G_0[:,:,0],5))
    print(np.round(G_1[:,:,0],5))
    quit()
    '''



    mode = str(sys.argv[1])
    beta = float(sys.argv[2])
    L = 6
    t = 1;U = 4;V = 1.5;mu = 1
    params = {'L':L,'verbose':1,'species':'boson','H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    generate_benchmark_data_new(params,beta,mode)
    
    quit()
    params = {
        'L':3,
        't':1,
        'mu':0,
        'V':0,
        'species':'boson',
        'JWstring':True
    }
    chain = spinless_chain(params=params)
    chain.solve(shift=True)
    beta = 10
    Z0 = np.exp(chain.logZ(beta=beta,parity=0))
    Z1 = np.exp(chain.logZ(beta=beta,parity=1))
    Z = np.exp(chain.logZ(beta=beta,parity=None))
    print(chain.H_moments(beta=beta))
    print(Z0,Z1,Z0+Z1,Z)
    print(chain.GS)
    C = chain.Green_function_tau_0(beta = beta,parity = 0)
    #########################################################
    params2 = {
        'L':5,
        'species':'fermion',
        'verbose':1,
        'JWstring':False
    }
    params2['H_params'] = {
        't':1,
        'U':0.5,
        'V':0.2,
        'mu':-2
    }
    params2['diag_params'] = {'mode':'full'}
    print(C)
    print('..... \n')
    G_0,G_1,G_0_alt,G_1_alt,G_0_alt2,G_1_alt2,Eta,S = generate_benchmark_data(params2)
    print('shapes')
    print(G_0.shape,G_1.shape,G_0_alt.shape,G_1_alt.shape,G_0_alt2.shape,G_1_alt2.shape)
    print(G_0[:,:,0])
    print(G_0_alt2[:,:,0])
    print(G_0_alt[:,:,0])
    print(Eta[:,:,0])
    print(S[:,:,0])
    #print(G[:,:,0])
    for i in range(params2['L']):
        for j in range(params2['L']):
            plt.plot(G_0[i,j,:],c='k',alpha=0.5)
            plt.plot(S[i,j,:],c='r',alpha=0.5)
            plt.plot(Eta[i,j,:],c='b',alpha=0.5)
    plt.savefig('fig.png',dpi=1000)
    
