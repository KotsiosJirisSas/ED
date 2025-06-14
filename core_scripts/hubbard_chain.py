'''
A barebones (ish) implementation of ED of a 1D Hubbard chain using only N,Z symmetries, splitting the hilbert spaces into ( N_up, N_down) sectors.
Key points:
1) Basis generation:
                    Index = Index_up + 2**L * Index_down
2) Hamiltonian generation and getting the spectrum:
                    Either full or Lanczos but in either case i store (sparesly) the entire Hamiltonian
3) Calculating thermal properties. Both static and dynamical
NOTE Latest update: 16 May 2025
--------------------------------------------------------------------------
TODO For systems above L = 8, can no longer perform full ED and Lanczos or Kernel Polynomial Methods should be used
TODO Allow for sparse creation of the matrix. Again,useful for larger system sizes beyong L=6-8
TODO More generic BCs?
--------------------------------------------------------------------------
NOTE 
BENCHMARKING:
1)  Phys. Rev. B 53, 6865:
        has plots of specific heat for small system sizes. Look at FIG2&3
2)  DQMC(can be made to have very small error for such small systems) 

'''
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import gc
from scipy.special import logsumexp
import pickle
from collections import defaultdict
import h5py

class hubbard_chain():
    '''
    Model:
    ------
    H = -t Σ (c^\dagger_i c_j) -μΣ(n_i) + (U/2) Σ (n_i - 1)^2 + V Σ (n_i - 1)(n_j - 1)

    DOFS & BCs:
    -----
    PBC fermions 
    PBC hardcore bosons
    NOTE these are controlled via the self.species property
    '''
    def __init__(self,params):
        '''
        Initializes the model and creates the Hilbert space basis of a given sector (N_up,N_dn)
        ----------------------------------------------------------------------------------------
        What params dict has to contain:
        1)system size L
        2)species
        3)(Nup,Ndn) symmetry sector
        4)Hamiltonian parameters t,U,V,μ
        5)diagonalization settings:
            mode = Full or Sparse (create dense or sparse hamiltonian matrix?) NOTE Sparse not Implemented
        6)verbosity
        '''
        self.L = params['L']

        self.species = params['species']# 'fermion' or 'boson' or 'mboson'
        self.Nup = params['Nup']
        self.Ndn = params['Ndn']

        self.H_params = params['H_params']
        self.t = self.H_params['t']
        self.U = self.H_params['U']
        self.V = self.H_params['V']
        self.mu = self.H_params['mu']

        self.diag_params = params['diag_params']
        
        self.verbose = params['verbose']
        if self.verbose>0:print('='*100,f'\n PERFORMING ED ON SYSTEM WITH l={self.L} and symmetry sector ({self.Nup},{self.Ndn}) \n','='*100)
        ###############################
        ########## sanity checks ######
        ###############################
        if self.L > 8:
            print(f'SYSTEM SIZE IS {self.L}. This might be too large... Proceed?')
        if (self.species != 'boson') and (self.species != 'mboson') and (self.species != 'fermion'):
            print(f'unrecognised particle species {self.species}')
            raise ValueError
        if (self.verbose > 0 ):
            if (self.species == 'mboson'):
                print('*'*100,f'\n TREATING SYSTEM AS mixed BC {self.species[1:]} \n ','*'*100)
            else:
                print('*'*100,f'\n TREATING SYSTEM AS PBC {self.species} \n ','*'*100)
        #############################
        ###### GENERATE BASIS #######
        #############################
        self.basis()#generates basis

    def basis(self):
        '''
        Generates Lin lookup tables for a given symmetry sector.
        Doesn't do full sum but only starts from min and max indices that are compatible with total number of electrons. 
        TBH gains are very small in the costly sectors w/ N=L/2. In easy sectors (N << L/2 or N ~ L) gain is abut x2
        Output:
        1)states and number of states and also nstates, ie the occupancy of each basis state. this final one is to be used later for ED
        2) lookup maps J_up,J_down,J
        
        Args:
            Nup:                    Number of down electrons
            Ndn:                    Number of up eleectrons
            L:                      Chain length
        Returns:
            basis_s(dict):          A dict of all states in the hilbert space of spin s
            index_s(dict):          The reverse of the basis_s... is it really necessary?
            len_basis_s(int):       The size of the Hilbert space for spin s
        '''
        Nup = self.Nup
        Ndn = self.Ndn
        L = self.L
        basis_dn = {}
        basis_up = {}
        index_dn = {}
        index_up = {}
        dn_imax = 0
        dn_imin = 0
        up_imax = 0
        up_imin = 0
        for i in range(0,Nup):
            up_imax += 2**(L-1-i)
            up_imin += 2**i
        for i in range(0,Ndn):
            dn_imax += 2**(L-1-i)
            dn_imin += 2**i
        count_dn = 0
        for I_dn in range(dn_imin,dn_imax+1):
            if self.countBits(I_dn) == Ndn:
                count_dn += 1
                basis_dn[count_dn] = I_dn
                index_dn[I_dn] = count_dn
        count_up = 0
        for I_up in range(up_imin,up_imax+1):
            if self.countBits(I_up) == Nup:
                count_up += 1
                basis_up[count_up] = I_up
                index_up[I_up] = count_up
                
        self.basis_up = basis_up
        self.basis_dn = basis_dn
        self.index_up = index_up
        self.index_dn = index_dn
        self.len_basis_up = count_up
        self.len_basis_dn = count_dn
        self.dim = count_up*count_dn
        if self.verbose>0:print('Hilbert Space size:',self.dim,'for symmetry sector (N_up,N_dn)=(',Nup,Ndn,')')
        #TODO add option for sparse construction

        #initializes hamiltonian to be filled in later one
        self.Hamiltonian = np.zeros((self.dim,self.dim),dtype=float)

        ####################
        ####SANITY CHECK####
        ####################
        for state_up in index_up:
            for state_dn in index_dn:
                N_el = self.countBits(state_up)+self.countBits(state_dn)
                if N_el != (Nup+Ndn):
                    print('number of electrons doesnt match number of N_up + N_dn')
                    raise ValueError
        return 
    
    def hop_ij_up(self,i,j,m):
        '''
        Function:
        ---------
        Adds to the hamiltonian the elements due to hopping between site i and j of the *spinless* state with index m. 

        Input:
        -------
        i,j(int):       i,j, \in [0,L-1]; The sites on which hopping happens
        m(int):         The spinless state that is to be hopped from i to j

        '''
        s1 = self.basis_up[m]
        s2 = self.hop(s1,i,j)
        if s2 == -1:
            return
        else:
            try:
                n = self.index_up[s2]
            except ValueError:
                print('Index not found...quitting!')
                quit()
            #get sign
            if self.species == 'fermion':
                sgn = self.fermion_sgn(self.binp(s1,length=self.L),self.binp(s2,length=self.L))
                if (sgn == -1) and (self.verbose>0):print('fermion negative sign',i,j,m)
            elif (self.species == 'mboson') and (i==self.L-1):
                #parity term
                sgn = (-1)**(bin(s1).count('1'))
                if (sgn == -1) and (self.verbose>0):print('boson negative sign',i,j,m)
            else:
                sgn = 1
            #generate all the basis states, since the other spin here is just playing spectator role
            for k in range(1,self.len_basis_dn+1):
                I1 = (k-1)*self.len_basis_up + (m-1)
                I2 = (k-1)*self.len_basis_up + (n-1)
                self.Hamiltonian[I1,I2] += -self.t*sgn
            return
    def hop_ij_dn(self,i,j,m):
        '''
        Function:
        ---------
        Adds to the hamiltonian the elements due to hopping between site i and j of the *spinless* state with index m. 

        Input:
        -------
        i,j(int):       i,j, \in [0,L-1]; The sites on which hopping happens
        m(int):         The spinless state that is to be hopped from i to j

        '''
        s1 = self.basis_dn[m]
        s2 = self.hop(s1,i,j)
        if s2 == -1:
            return
        else:
            try:
                n = self.index_dn[s2]
            except ValueError:
                print('Index not found...quitting!')
                quit()
            #get sign
            if self.species == 'fermion':
                sgn = self.fermion_sgn(self.binp(s1,length=self.L),self.binp(s2,length=self.L))
            elif (self.species == 'mboson') and (i==self.L-1):
                #parity term
                if self.verbose>0:print('boundary hop for hc bosons with mixed BC')
                sgn = (-1)**(bin(s1).count('1'))
            else:
                sgn = 1
            #generate all the basis states, since the other spin here is just playing spectator role
            for k in range(1,self.len_basis_up+1):
                I1 = (m-1)*self.len_basis_up + (k-1)
                I2 = (n-1)*self.len_basis_up + (k-1)
                self.Hamiltonian[I1,I2] += -self.t*sgn
        return
    def build_hopping_full(self):
        '''
        Function:
        ---------
        Creates hopping matrix as a dense object. \n
        It loops through all sites,states and spins and adds hopping terms \n
        '''
        if self.L == 2:
            L =1
        else:
            L = self.L
        for i in range(L):
            j=(i+1)%self.L
            #spin up
            for m_up in self.basis_up:
                self.hop_ij_up(i,j,m_up)
            # spin down
            for m_dn in self.basis_dn:
                self.hop_ij_dn(i,j,m_dn)
        return
    def build_hopping_sparse(self):
        '''
        Function:
        ---------

        Creates hopping matrix as a sparse object \n
        '''
        raise NotImplementedError

 
    def build_ham(self):
        '''
        Function:
        ---------
        Creates the Hamiltonian matrix \n

        TODO: Can calculate on-site terms by generating I_up and I_dn states individually. I don't do that here. This is covered in arXiv 1307.7542
        '''
        ################################################
        #hopping terms
        if self.diag_params['mode'] == 'full':
            self.build_hopping_full()
        elif self.diag_params['mode'] == 'sparse':
            self.build_hopping_sparse()
        else:
            print('diag params mode not recognised')
            raise ValueError
        #################################################
        # on site terms
        if self.diag_params['mode'] != 'full':
            raise NotImplementedError
        for m_up in self.basis_up:
            for m_dn in self.basis_dn:
                I_up = self.basis_up[m_up]
                I_dn = self.basis_dn[m_dn]
                m_tot = (m_dn-1)*self.len_basis_up + (m_up-1)
                I_physical = I_up+I_dn*(2**self.L)
                for i in range(self.L):
                    occ = self.occupancy(I_physical,i)
                    self.Hamiltonian[m_tot,m_tot] += -self.mu*occ + 0.5*self.U*(occ-1.)**2
                    
                    #n.n. repulsion
                    j = (i+1)%self.L
                    occ_j = self.occupancy(I_physical,j)#I_physical is the basis state and j is the site
                    self.Hamiltonian[m_tot,m_tot] += self.V*((occ-1.)*(occ_j-1))
        ########################
        if np.allclose(self.Hamiltonian,self.Hamiltonian.T) == False:
            print('Hamiltonian is not hermitian?')
            raise ValueError
        if self.verbose > 0:
            self.sparsity(self.Hamiltonian)
        return

    #############################
    ###### HELPER FUNCTIONS #####
    #############################    
    @staticmethod
    def countBits(x):
        '''
        Counts number of 1s in bin(n)
        '''
        #From Hacker's Delight, p. 66
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        x = x + (x >> 8)
        x = x + (x >> 16)
        return x & 0x0000003F 

    @staticmethod
    def hop(s,i,j):
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
        # L will have structure 0000000[i]0000[j]00000 and there's four cases:
        #1) L = mask means I1[i]=I1[j]=0 -> hopping is not allowed
        #2) L = 000..00 means I1[i]=I1[j]=1 -> hopping is not allowed
        #3&4) L = ...[1]...[0]... or L = ...[0]...[1]... means hopping is allowed, in which case new integer is 
        if P == mask or P == 0:
            s2 = -1
        else:
            s2 = s - K + P
        return s2

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
    def binp(num, length):
        '''
        print a binary number without python 0b and appropriate number of zeros
        regular bin(x) returns '0bbinp(x)' and the 0 and b can fuck up other stuff
        '''
        return format(num, '#0{}b'.format(length + 2))[2:]

    def occupancy(self,psi,i):
        '''
        Function:
        ---------
            Calculates occupancy of state psi at site i for a two species system on chain of length L \n
        Input:
        ------
            psi(int):       A two-species state psi = state_up + 2**L state_dn \n
            i(int) :        i \in [0,L-1]; The site wwe are counting the occupation of \n
        Output:
        -------
            occ(int):       The occupation
        '''
        mask = 2**(i)+2**(self.L+i)
        occ = self.countBits(psi & mask)
        return occ

    @staticmethod
    def sparsity(X):
        '''
        Function:
        ---------
        sparsity calculator. For eg L=22, sparsity is 99.98% and this should keep growing w/ L
        because basis grows exponentially while states connected via hopping  grow linearly(?) \n
        note: can't apply this function to sparse(csr) matrix. Have to convert to dense matrix first
        '''
        nnz = np.sum(np.abs(X) > 1e-10)  # Non-zero count
        total = X.size
        sparsity_percentage = 100 * (1 - nnz / total)
        print('Sparsity is:', np.round(sparsity_percentage, 2), '%')
        return sparsity_percentage

class thermodynamics():
    '''
    Class instance that given some basic parameters and a full set of eigenstuff, performs calculations for static and dynamical thermal quantities.
    
    TODO Make it so that it's reusable for generic systems not only the hubbard chain! Also for the 2xN flavored 2D Hubbard model we have in mind 
    '''
    def __init__(self,params):
        '''
        Function:
        ---------
        INitializes the thermodynamics class given the eigenstuff, the bases and some more parameters
        sign: treat dofs as PBC fermions
        JWstring: treat dofs as  weirdBC bosons (True) or as PBC bosons (False)
        '''
        self.L = params['L']
        ##############
        ##ED results##
        ##############
        self.energies = params['energies']
        self.eigenstates = params['eigenstates']
        self.eGS = params['lowestEnergy']
        self.bases = params['bases']
        self.bases_inv = params['bases_inv']
        if 'JWstring' in params.keys():
            self.JWstring = params['JWstring']
        else:
            self.JWstring = False
        if 'verbose' in params.keys():
            self.verbose = params['verbose']
        else:
            self.verbose = 0
        ################
        # SPECIES & BCS#
        ################
        self.species = params['species']
        if self.species == 'mboson':
            print('yo sure u want that?')
            raise ValueError
    def logZ(self,beta,parity):
        '''
        Function:
        ---------
        Calculates the partition function in a given parity sector \n
        
        NOTE 1 Jun 2025 added new option for a spin resolved parity check aswell!
        
        TODO Add a generic function to sort parity depending on structure of symmetry sectors, making it compatible with more systems

        Input:
        ------
        beta(float):        The inverse temperature \n
        parity:             None,0,1,(i,j). The parity of the sectors we are considering. Either total parity or parity of a given spin. If option is tuple, (i,j), i is the parity 0,1 and j is the spin (0=up,1=dn) \n

        Output:
        -------
        logZ:               The log of the partiton function
        ''' 
        def parity_check_basic(sec,parity):
            '''
            Checks the parity of a sector, if required
            works for sec = (sec0,sec1) = (N_up,N_dn)
            '''
            if parity == None:
                return True
            elif  not isinstance(parity,tuple):
                if (sec[0]+sec[1])%2 == parity:
                    return True
                else:
                    return False
            else:
                spin = parity[0]
                par = parity[1]
                if (sec[spin])%2 == par:
                    return True
                else:
                    return False
        all_energies = []
        # Gather all energies
        for sectors in self.energies:
            #print('parity check',sectors,parity,(sectors[0]+sectors[1])%2 == parity)
            if parity_check_basic(sectors,parity):
                all_energies.append(self.energies[sectors])
        # Convert lists to NumPy arrays for fast computation
        all_energies = np.concatenate(all_energies)
        # Compute log partition function in one vectorized step
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
        for sector in self.energies:
            energies = self.energies[sector]
            valid_mask = energies > 0
            valid_energies = energies[valid_mask]
            #
            if valid_energies.size > 0:
                log_terms_H.append((-beta * valid_energies) + np.log(valid_energies))  
                log_terms_H_sq.append((-beta * valid_energies) + np.log(valid_energies**2))
                log_H = logsumexp(np.concatenate(log_terms_H)) - log_Z
                log_H_sq = logsumexp(np.concatenate(log_terms_H_sq)) - log_Z

        ##################################
        #while not physically relevant, return also the values with shifted energy.
        H_avg = np.exp(log_H)
        H_avg_unshifted = H_avg + self.eGS
        H_sq_avg = np.exp(log_H_sq)
        H_sq_avg_unshifted = H_sq_avg +2*self.eGS*H_avg+self.eGS**2

        if Cv == False:
            return H_avg,H_avg_unshifted,H_sq_avg,H_sq_avg_unshifted
        else:
            specific_heat = (1./self.L)*(beta**2)*(H_sq_avg - H_avg**2)
            return H_avg,H_avg_unshifted,H_sq_avg,H_sq_avg_unshifted,specific_heat
    def N_moments(self, beta):
        """
        Function:
        --------
        Computes the thermal expectation value of \hat{N} and \hat{N}^2, using logsumexp for numerical stability. \n
        Takes care to filter out N=0 states as their log is ill defined and they shouldn't contribute to shifted expectation values \n
        
        Input:
        ------
        beta (float):                   Inverse temperature (1/kT).
        
        Output:
        -------
        (float,float):                  Thermal expectation value <N>,<N^2>.
        """
        log_Z = self.logZ(beta,parity = None)
        log_terms_N = []
        log_terms_N_sq = []
        for sector in self.energies:
            energies = self.energies[sector]
            N_sec = sector[0]+sector[1] 
            N_sec = np.array([N_sec]*len(energies))#electrons in sector
            # Filter out zero-occupation cases to prevent log(0) issues
            valid_mask = N_sec > 0
            valid_N_sec = N_sec[valid_mask]
            valid_energies = energies[valid_mask]

            if valid_N_sec.size > 0:
                log_terms_N.append(np.log(valid_N_sec) - beta * valid_energies)
                log_terms_N_sq.append(2*np.log(valid_N_sec) - beta * valid_energies)

        # Compute the final thermal average <N>
        log_N = logsumexp(np.concatenate(log_terms_N)) - log_Z
        log_N_sq = logsumexp(np.concatenate(log_terms_N_sq)) - log_Z

        return np.exp(log_N),np.exp(log_N_sq)
    ###########################
    #### OPERATOR MAPPINGS ####
    ###########################
    def create_operator_mapping(self,op):
        '''
        Function:
        ----------
        Creates a mapping between *basis states* under the action of an operator O:
        O: |I> ----> sgn,|J> \n
        This will connect two sectors (N↑,N↓) & (N'↑,N'↓) and specifically two basis elements J,I (or indices j,i) via a phase (sign) \n

        Input:
        ------
        op(str):        Current operator strings: 'green','spinspin','eta' \n

        Output:
        -------
        mapping_op(dict):      Has keys tuples of the form (r,sec,j) and values (sec_new,j,sign) \n 
        '''
        def count_ones_between(n, r1, r2, L):
            '''
            count 1s in a string between location r1 and location r2 = (r1 + Delta r)%L
            '''
            count = 0
            if r2 > r1:
                # Straightforward range: r1+1 to r2-1
                for i in range(r1 + 1, r2):
                    if (n >> i) & 1:
                        count += 1
            elif r2 < r1:
                # Wrap-around case: r1+1 to L-1, then 0 to r2-1
                for i in range(r1 + 1, L):
                    if (n >> i) & 1:
                        count += 1
                for i in range(0, r2):
                    if (n >> i) & 1:
                        count += 1
            # if r1 == r2, the range is empty, return 0
            return count
        def apply_sc_op(I_up,I_dn,r):
            '''
             Apply c_{↑r+1} c_{↑r} to a state (Iup + 2^L *Idn).
            '''
            r2 = (r+1)%self.L
            if ((I_up >> r2)&1 == 0) or (((I_up >> r)&1 == 0))==True: # operator cant act on this
                return None, None, None
            J_up = I_up ^ (1 << r)#replace 1 with 0 at r
            J_up = J_up ^ (1 << r2)#replace 1 with 0 at r2
            # Compute sign factor (count fermions between 0 and r-1)
            if (self.species == 'fermion') or (self.JWstring):#if we are dealing either with fermions or with Bosons+JW string, this erm might have a sign
                if r == self.L-1:#bond crosses boundary
                    sign = (-1) ** (bin(I_up).count('1') - 1)#exclude the 1 that is guaranteed to be at final site
                else:
                    sign = +1
            else:
                sign = +1
            return J_up,I_dn,sign
        def apply_dens_op(I_up,I_dn,r):
            if (I_up >> r)&1 == 0: # operator cant act on this
                return None, None, None
            sign = 1
            return I_up,I_dn,sign

        def apply_bond_op(I_up,I_dn,r):
            '''
             Apply c†_{↑r+1} c_{↑r} to a state (Iup + 2^L *Idn).
            '''
            DeltaR = 2
            r2 = (r+DeltaR)%self.L
            if ((I_up >> r2)&1 == 1) or (((I_up >> r)&1 == 0))==True: # operator cant act on this
                return None, None, None
            J_up = I_up ^ (1 << r)#replace 1 with 0 at r
            J_up = J_up | (1 << r2)#replace 0 with 1 at r2
            # Compute sign factor (count fermions between 0 and r-1)
            if (self.species == 'fermion'):#if we are dealing either with fermions or with Bosons+JW string, this erm might have a sign
                if r == self.L-1:#bond crosses boundary
                    sign = (-1) ** (bin(I_up).count('1') - 1)#exclude the 1 that is guaranteed to be at final site
                    if (self.JWstring) and (sign == -1):
                        print('negative sign',I_up,I_dn,r)
                else:
                    sign = +1
            elif  (self.JWstring):
                sign = (-1)**count_ones_between(I_up,r1=r,r2=r2,L=self.L)
                if (sign == -1) and (DeltaR==1):
                    print('????? should have negatives in this case for Delta r =1')
            else:
                sign = +1
            return J_up,I_dn,sign
        def apply_c_dagger_c(I_up,I_dn,r):
            """
            Apply c†_{↓r} c_{↑r} to a state (Iup + 2^L *Idn).
            Ordering:
            |full> = c^\dagger_{L,up}...c^\dagger_{1,up} c^\dagger_{L,down}...c^\dagger_{1,down} |0>
            ------------------------
            Input:
            ------
                I_up(int)           :The decimal representation of spin-up component of state
                I_dn(int)           :The decimal representation of spin-dn component of state
                r(int)              :The location of application of the operator
            Output:
            -------
                J_up(int)           :The decimal representation of spin-up component of new state
                J_dn(int)           :The decimal representation of spin-dn component of new state
                sgn(\pm 1)          :The associated sign
            -----------------------------
            If output is None then that means state cannot support this c^\dagger c term
            """
            if ((I_dn >> r)&1 == 1) or (((I_up >> r)&1 == 0))==True: # hopping not possible
                return None, None, None
            J_up = I_up ^ (1 << r)#replace 1 with 0 at r
            J_dn = I_dn | (1 << r)#replace 0 with 1 at r
            # Compute sign factor (count fermions between 0 and r-1)
            if (self.species == 'fermion') or (self.JWstring):#if we are dealing either with fermions or with Bosons+JW string, this erm might have a sign
                phase_up_left = (bin(I_up & ((1 << r) - 1)).count('1'))#from the c_{↑r}
                phase_dn_left = (bin(I_dn & ((1 << r) - 1)).count('1'))#from the c†_{↓r}
                phase_dn_total = (bin(I_up).count('1')-1)#from the c†_{↓r} due to total fock space ordering. The extra minus is because this is meant to be calculated on the intermediate state ie c_{↑r}|I_up>
                sign = (-1) ** (phase_up_left + phase_dn_left + phase_dn_total)
                #sign = (-1) ** ((bin(I_up & ((1 << r) - 1)).count('1')) + (bin(I_dn & ((1 << r) - 1)).count('1')))
                #BUG shouldnt this count all the rest of the electrons of the up species instead of only those to the left?
            else:
                sign = +1
            return J_up,J_dn,sign
        
        def apply_c_c(I_up,I_dn,r):     
            """
            Apply c_{↓r} c_{↑r} to a state (Iup+2^LIdn).
            Ordering:
            |full> = c^\dagger_{L,up}...c^\dagger_{1,up} c^\dagger_{L,down}...c^\dagger_{1,down} |0>
            ------------------------
            Input:
                I_up(int)           :The decimal representation of spin-up component of state
                I_dn(int)           :The decimal representation of spin-dn component of state
                r(int)              :The location of application of the operator
            Output:
                J_up(int)           :The decimal representation of spin-up component of new state
                J_dn(int)           :The decimal representation of spin-dn component of new state
                sgn(\pm 1)          :The associated sign
            -----------------------------
            If output is None then that means state cannot support this c^\dagger c term
            
            """
            if ((I_dn >> r)&1 == 0) or (((I_up >> r)&1 == 0))==True: # hopping not possible
                return None, None, None
            J_up = I_up ^ (1 << r)#replace 1 with 0 at r
            J_dn = I_dn ^ (1 << r)#replace 1 with 0 at r
            # Compute sign factor (count fermions between 0 and r-1)
            if (self.species == 'fermion') or (self.JWstring): #if we are dealing either with fermions or with Bosons+JW string, this erm might have a sign
                phase_up_left = (bin(I_up & ((1 << r) - 1)).count('1'))#from the c_{↑r}
                phase_dn_left = (bin(I_dn & ((1 << r) - 1)).count('1'))#from the c_{↓r}
                phase_dn_total = (bin(I_up).count('1')-1)#from the c_{↓r} due to total fock space ordering. The extra minus is because this is meant to be calculated on the intermediate state ie c_{↑r}|I_up>
                sign = (-1) ** (phase_up_left + phase_dn_left + phase_dn_total)
                #sign = (-1) ** ((bin(I_up & ((1 << r) - 1)).count('1')) + (bin(I_dn & ((1 << r) - 1)).count('1')))
            else:
                sign = +1
            return J_up,J_dn,sign
        def apply_c_c_old(I_up,I_dn,r):     
            """
            Apply c_{↓r} c_{↑r} to a state (Iup+2^LIdn).
            Ordering:
            |full> = c^\dagger_{L,up}...c^\dagger_{1,up} c^\dagger_{L,down}...c^\dagger_{1,down} |0>
            ------------------------
            Input:
                I_up(int)           :The decimal representation of spin-up component of state
                I_dn(int)           :The decimal representation of spin-dn component of state
                r(int)              :The location of application of the operator
            Output:
                J_up(int)           :The decimal representation of spin-up component of new state
                J_dn(int)           :The decimal representation of spin-dn component of new state
                sgn(\pm 1)          :The associated sign
            -----------------------------
            If output is None then that means state cannot support this c^\dagger c term
            
            """
            if ((I_dn >> r)&1 == 0) or (((I_up >> r)&1 == 0))==True: # hopping not possible
                return None, None, None
            J_up = I_up ^ (1 << r)#replace 1 with 0 at r
            J_dn = I_dn ^ (1 << r)#replace 1 with 0 at r
            # Compute sign factor (count fermions between 0 and r-1)
            if (self.species == 'fermion') or (self.JWstring): #if we are dealing either with fermions or with Bosons+JW string, this erm might have a sign
                sign = (-1) ** ((bin(I_up & ((1 << r) - 1)).count('1')) + (bin(I_dn & ((1 << r) - 1)).count('1')))
            else:
                sign = +1
            return J_up,J_dn,sign
        def apply_c(I_s, r,I_rest=None):
            """
            Given a state of spin s represented by its decimal I_s, find the state with decimal I_s2 that is connected to it via creation operator
            I_rest is the part of the wavefunction 'infront' of the spin component here.
            ordering:

            |full> = c^\dagger_{L,up}...c^\dagger_{1,up} c^\dagger_{L,down}...c^\dagger_{1,down} |0>

            so in this case when computing c^\dagger_up we don't care about the spin_down part but if computing c^\dagger_down we do care about the spin_up part.
            ------------------------
            Input:

            Output:
            
            """
            if (I_s >> r) & 1:  # If site j is already occupied, return None
                return None, None
            I_s2 = I_s | (1 << r)
            # Compute sign factor (count fermions to the left)
            if (self.species == 'fermion') or (self.JWstring):
                sign = (-1) ** ((bin(I_s & ((1 << r) - 1)).count('1')) + (bin(I_rest).count('1') if I_rest is not None else 0))
            else:
                sign = +1
            return I_s2, sign
        def apply_c_zero(I_s,r1,r2):
            '''
            Calculates <i|c_{r1,s} c^\dagger_{r2,s}|j>
            for the case of bosons with JW string. Works the same way with both spins, since the only JW string that's leftover is the one between r,s and r',s
            '''
            if (self.species != 'boson') or (self.JWstring == False):
                raise ValueError
            if r1 == r2:
                if (I_s >> r2) & 1:
                    return None,None
                else:
                    return I_s,+1
            else:
                if ((I_s >> r2) & 1) or not((I_s >> r1) & 1):  # If site j is already occupied, return None
                    return None, None
                I_s2 = I_s | (1 << r2)#replace 0 with 1 at r2
                I_s2 = I_s2 ^ (1 << r1)#replace 1 with 0 at r1
                # Compute sign factor (count fermions between the two sites)
                if r1>r2:
                    sign = count_ones_between(I_s,r2,r1,L=self.L)
                    sign = -1*sign#from the extra 1 at r2
                else:
                    sign =  count_ones_between(I_s,r1,r2,L=self.L)
                return I_s, sign
        def apply_c_zero_2(I_s,r1,r2):
            '''
            NOTE I THINK THE OTHER APPLY Czero HAS AN ISSUE
            Calculates <i|c_{r1,s} c^\dagger_{r2,s}|j>
            for the case of bosons with JW string. Works the same way with both spins, since the only JW string that's leftover is the one between r,s and r',s
            '''
            if (self.species != 'boson') or (self.JWstring == False):
                raise ValueError
            if r1 == r2:
                if (I_s >> r2) & 1:
                    return None,None
                else:
                    return I_s,+1
            else:
                if ((I_s >> r2) & 1):#r2 already full
                    return None,None
                I_s1 = I_s | (1 << r2)#replace 0 with 1 at r2
                sgn1 = 1#TODO
                if not ((I_s1 >> r1)& 1):#r1 already empty
                    return None,None
                I_s2 = I_s1 & ~(1 << r1)
                sgn2 = 1#TODO
                return I_s2,sgn1*sgn2
        def apply_c_zero_3(I_s,r1,r2):
            '''
            TODO:FIRST ONE HAS ANE RRROR!!!!
            Calculates <i|c_{r1,s} c^\dagger_{r2,s}|j>
            for the case of bosons with JW string. Works the same way with both spins, since the only JW string that's leftover is the one between r,s and r',s
            '''
            if (self.species != 'boson') or (self.JWstring == False):
                raise ValueError
            if r1 == r2:
                if (I_s >> r2) & 1:
                    return None,None
                else:
                    return I_s,+1
            else:
                if ((I_s >> r2) & 1) or not((I_s >> r1) & 1):  # If site j is already occupied, return None
                    return None, None
                I_s2 = I_s | (1 << r2)#replace 0 with 1 at r2
                I_s2 = I_s2 ^ (1 << r1)#replace 1 with 0 at r1
                # Compute sign factor (count fermions between the two sites)
                if r1>r2:
                    sign = count_ones_between(I_s,r2,r1,L=self.L)
                    sign = -1*sign#from the extra 1 at r2
                else:
                    sign =  count_ones_between(I_s,r1,r2,L=self.L)
                return I_s2, sign
                

        if op == 'green':
            apply_op = apply_c
        elif op == 'green0':
            apply_op = apply_c_zero
        elif op == 'green03':
            apply_op = apply_c_zero_3
        elif op == 'spinspin':
            apply_op = apply_c_dagger_c
        elif op == 'eta':
            apply_op = apply_c_c
        elif op == 'etaold':
            apply_op = apply_c_c_old
        elif op == 'bond':
            apply_op = apply_bond_op
        elif op == 'dens':
            apply_op = apply_dens_op
        elif op == 'sc':
            apply_op = apply_sc_op
        else:
            raise ValueError
        if (op != 'green') and (op != 'green0') and (op != 'green03'):
            mapping_op = {}
            for r in range(self.L):#location of operator
                for sector in self.bases:
                    sec_basis_up,sec_basis_dn = self.bases[sector]
                    for state_up in sec_basis_up:#sec_basis_up is dict with keys the indices and values the decimal representation of states
                        for state_dn in sec_basis_dn:
                            I_up = sec_basis_up[state_up]
                            I_dn = sec_basis_dn[state_dn]
                            J_up,J_dn,sign = apply_op(I_up,I_dn,r)
                            if J_up == None:continue#skip this (I_up,I_dn) state
                            if (op == 'spinspin') or (op == 'spinspinold'):
                                sector_new = (sector[0]-1, sector[1]+1)
                            elif (op == 'eta') or (op == 'etaold'):
                                sector_new = (sector[0]-1, sector[1]-1)
                            elif (op == 'bond') or (op == 'dens'):
                                sector_new = sector
                            elif op == 'sc':
                                sector_new = (sector[0]-2, sector[1])
                            #now turn decimal representation of states to indices
                            i = self.State2Ind(sector,I_up,I_dn)
                            j = self.State2Ind(sector_new,J_up,J_dn)
                            mapping_op[(r,sector,i)] = (sector_new,j,sign)

        elif op == 'green':
            mapping_op = {}
            mapping_up = {}
            mapping_dn = {}
            for sector in self.bases:
                sec_basis_up,sec_basis_dn = self.bases[sector]
                for state_up in sec_basis_up:
                    I_up = sec_basis_up[state_up]
                    for state_dn in sec_basis_dn:
                        I_dn = sec_basis_dn[state_dn]
                        for j in range(self.L):
                            #apply creation operators
                            I_up2,sign_up = apply_op(I_up,j,I_rest=None)
                            if I_up2 != None:
                                #I_new_up = I_up2+I_dn*(2**self.L)
                                sector_new_up = (sector[0]+1, sector[1])
                            I_dn2,sign_dn = apply_op(I_dn,j,I_rest = I_up)
                            if I_dn2 != None:
                                #I_new_dn = I_up+I_dn2*(2**self.L)
                                sector_new_dn = (sector[0], sector[1]+1)
                            # turn states (I's) into indices
                            #print(I_up,I_dn,I_up2,I_dn2)
                            ind_in = self.State2Ind(sector,I_up,I_dn)
                            ind_out_up = self.State2Ind(sector_new_up,I_up2,I_dn)
                            ind_out_dn = self.State2Ind(sector_new_dn,I_up,I_dn2)
                            if ind_out_up is not None:
                                mapping_up[(j,sector,ind_in)] = (sector_new_up,ind_out_up,sign_up)
                            if ind_out_dn is not None:
                                mapping_dn[(j,sector,ind_in)] = (sector_new_dn,ind_out_dn,sign_dn)  
            mapping_op['up'] = mapping_up
            mapping_op['dn'] = mapping_dn


        elif (op == 'green0') or (op == 'green03'):
            #only does up_spins for now
            #mapping_op = {}
            mapping_up = {}
            mapping_dn = {}
            for sector in self.bases:
                sec_basis_up,sec_basis_dn = self.bases[sector]
                for state_up in sec_basis_up:
                    I_up = sec_basis_up[state_up]
                    for state_dn in sec_basis_dn:
                        I_dn = sec_basis_dn[state_dn]
                        for r1 in range(self.L):
                            for r2 in range(self.L):
                                I_up2,sign_up = apply_op(I_up,r1,r2)
                                if I_up2 != None:
                                    sector_new_up = (sector[0], sector[1])
                                #I_dn2,sign_dn = apply_op(I_dn,r1,r2)
                                #if I_dn2 != None:
                                    #I_new_dn = I_up+I_dn2*(2**self.L)
                                    #sector_new_dn = (sector[0], sector[1])
                                # turn states (I's) into indices
                                #print(I_up,I_dn,I_up2,I_dn2)
                                ind_in = self.State2Ind(sector,I_up,I_dn)
                                ind_out_up = self.State2Ind(sector_new_up,I_up2,I_dn)
                                #ind_out_dn = self.State2Ind(sector_new_dn,I_up,I_dn2)
                                if ind_out_up is not None:
                                    mapping_up[(r1,r2,sector,ind_in)] = (sector_new_up,ind_out_up,sign_up)
                                #if ind_out_dn is not None:
                                #    mapping_dn[(r1,r2,sector,ind_in)] = (sector_new_dn,ind_out_dn,sign_dn)  
            #mapping_op['up'] = mapping_up
            #mapping_op['dn'] = mapping_dn
            mapping_op = mapping_up
        
        return mapping_op

    def operator_matrix_elements(self,map,flag_green0=False):
        '''
        Function:
        ----------
        Turns the operator mapping (which is the matrix elements in state basis) to  a matrix in eigenbasis 
        to be used for correlator calculations \n
        Ie computes <n|O|m> in the eigenbasis given that you have the operator in the occupation basis <j|O|i> \n

        Input:
        ------
        map(dict):                  A dictionary of the form dict[(r,sector,index_in)] = (sector_new,index_out,sign) 
                                    w/ r: the location of the operator, sector,sector_new the symmetry sectors related by O,
                                    index_in & index_out the states related by O and sign the associated phase \n
        flag_green0(Bool):          A flag treating system differently if operator is green0
        Output:
        -------
        mat(dict):                  A dictionary holding info for the matrix <n|O|m>. It has structure
                                    mat[(r, sec, sec_new)] is an |ℋ|_secnew x |ℋ|_sec matrix                      
        '''
        if flag_green0 == False:
            matrix_elements = {}
            for (r, sector, ind_in), (sector_new, ind_out, sign) in map.items():
                if sector not in self.eigenstates or sector_new not in self.eigenstates:
                    print('sector not found?',sector,sector_new)
                    continue
                eigvecs_sec = self.eigenstates[sector]
                eigvecs_sec_new = self.eigenstates[sector_new]
                num_states_sec = eigvecs_sec.shape[0]
                num_states_sec_new = eigvecs_sec_new.shape[0]
                
                matrix_elements.setdefault((r,sector, sector_new), np.zeros((num_states_sec_new, num_states_sec), dtype=np.complex128))
                for n in range(num_states_sec_new):
                    for m in range(num_states_sec):
                        matrix_elements[(r,sector, sector_new)][n, m] += (
                            np.conj(eigvecs_sec_new[ind_out,n]) * sign * eigvecs_sec[ind_in,m]
                        )
        elif flag_green0 == True:
            matrix_elements = {}
            for (r1,r2, sector, ind_in), (sector_new, ind_out, sign) in map.items():
                if sector not in self.eigenstates or sector_new not in self.eigenstates:
                    print('sector not found?',sector,sector_new)
                    continue
                eigvecs_sec = self.eigenstates[sector]
                eigvecs_sec_new = self.eigenstates[sector_new]
                num_states_sec = eigvecs_sec.shape[0]
                num_states_sec_new = eigvecs_sec_new.shape[0]
                
                matrix_elements.setdefault((r1,r2,sector, sector_new), np.zeros((num_states_sec_new, num_states_sec), dtype=np.complex128))
                for n in range(num_states_sec_new):
                    for m in range(num_states_sec):
                        matrix_elements[(r1,r2,sector, sector_new)][n, m] += (
                            np.conj(eigvecs_sec_new[ind_out,n]) * sign * eigvecs_sec[ind_in,m]
                        )

        return matrix_elements
    def correlator(self,op,beta,n_tau,parity=None,green_spin = 'up'):
        '''
        Function:
        ----------
        Using the operator mapping and the operator eigen-matrix elements <m|O|n>, it calculates the dynamical correlator: \n
        C(r,r',τ) = <O^\dagger_r(τ) O_r'(0)> = (-1/Z)x Σ_n,m {exp(-(β-τ)Em)exp(-τEn) x [O_r]nm x [O_r']*nm }

        Input:
        ------
        op(str):            The operator name
        beta(float):        The inverse temperature
        n_tau(int):         The number of imaginary time slices
        parity:             None/0/1. The parity resolution of the operator
        green_spin(str)     If op=='green',choose to calculate G_up or G_dn, since G_σσ' ~ δ_σσ'
        Output:
        -------
        C(npcarray):        The L x L x Nτ correlator
        '''
        def parity_check_basic(sec,parity):
            '''
            NOTE Same as in logZ... just make it a class property....
            Checks the parity of a sector, if required
            works for sec = (sec0,sec1) = (N_up,N_dn)
            '''
            if parity == None:
                return True
            elif  not isinstance(parity,tuple):
                if (sec[0]+sec[1])%2 == parity:
                    return True
                else:
                    return False
            else:
                spin = parity[0]
                par = parity[1]
                if (sec[spin])%2 == par:
                    return True
                else:
                    return False
        mapping = self.create_operator_mapping(op)
        if op == 'green':
            mapping = mapping[green_spin]
        if (op == 'green0') or (op == 'green03'):
            flag = True
        else:
            flag = False
        mat = self.operator_matrix_elements(map=mapping,flag_green0=flag)
        #sec_pairs holds all pairs of sectors related by the operator
        if (op != 'green0') and (op != 'green03'):
            sec_loc = 1#location of sector info in dictionary
        else:
            sec_loc = 2#location of sector info in dictionary
        sec_pairs = set()
        for key in mapping:
            sec = key[sec_loc]
            sec_new = mapping[key][0]
            sec_pairs.add((sec,sec_new))
        sec_pairs = list(sec_pairs)
        if self.verbose>0:print('sec pairs \n',sec_pairs)

        taus = np.linspace(0, beta, num=n_tau)
        C = np.zeros((self.L, self.L, n_tau), dtype=np.complex128)
        log_Z = self.logZ(beta,parity)

        #start calculation for the correlator
        if (op != 'green0') and (op != 'green03'):
            for (sec, sec_new) in sec_pairs:
                if not parity_check_basic(sec,parity): continue # filters out certain sectors
                #if (parity!=None) and (sum(sec) % 2 != parity):continue# NOTE filter out certain sectors. more basic version of check. Now replaced with the function
                if self.verbose>0:print('sectors',sec,sec_new)
                for r1 in range(self.L):
                    for r2 in range(self.L):
                            for m in range(len(self.eigenstates[sec])):
                                for n in range(len(self.eigenstates[sec_new])):
                                    Em = self.energies[sec][m]
                                    En = self.energies[sec_new][n]
                                    amp_r2 = mat[(r2, sec, sec_new)][n, m]
                                    amp_r1 = mat[(r1, sec, sec_new)][n, m].conj()
                                    log_terms = -beta * Em - taus * (En - Em)
                                    C[r1, r2, :] += amp_r1*amp_r2 * np.exp(log_terms - log_Z)
        if (op == 'green0') or (op == 'green03'):
            for (sec, sec_new) in sec_pairs:
                if not parity_check_basic(sec,parity): continue # filters out certain sectors
                #if (parity!=None) and (sum(sec) % 2 != parity):continue## NOTE filter out certain sectors. more basic version of check. Now replaced with the function
                if self.verbose>0:print('sectors',sec,sec_new)
                for r1 in range(self.L):
                    for r2 in range(self.L):
                            if (r1,r2, sec, sec_new) in mat.keys():
                                for m in range(len(self.eigenstates[sec])):
                                    #sec = sec_new here.
                                    Em = self.energies[sec][m]
                                    amp = mat[(r1,r2, sec, sec_new)][m, m]
                                    log_terms = -beta * Em
                                    C[r1, r2, :] += amp* np.exp(log_terms - log_Z)
        return C

    ##################
    ###HELPER FUNCS###
    ##################
    def State2Ind(self,sector,I_up,I_dn):
        '''
        Goes from decimal representation of the state I to its index i
        '''
        if (I_up is None) or (I_dn is None):
            return None
        ind_up = self.bases_inv[sector][0]
        m_up = ind_up[I_up]
        ind_dn = self.bases_inv[sector][1]
        m_dn = ind_dn[I_dn]
        ind = (m_dn-1)*len(ind_up)+(m_up-1) # maybe doesn't need the -1's
        return ind
    @staticmethod
    def binp(num, length):
        '''
        print a binary number without python 0b and appropriate number of zeros
        regular bin(x) returns '0bbinp(x)' and the 0 and b can fuck up other stuff
        '''
        return format(num, '#0{}b'.format(length + 2))[2:]
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

class thermodynamics_old():
    def __init__(self,params):
        '''
        sign: treat dofs as PBC fermions
        JWstring: treat dofs as  weirdBC bosons (True) or as PBC bosons (False)
        '''
        self.params = params
        self.L = params['L']
        self.beta = params['beta']
        self.EDFullSpectrum()
        if 'verbose' in params.keys():
            self.verbose = params['verbose']
        else:
            self.verbose = 0
        self.sign = params['sign']
        if 'JW string' in params.keys():
            self.JWstring = params['JW string']
        else:
            self.JWstring = False
        return
    
    
    def apply_creation_operator_spinless(self,I_s, j,I_rest=None):
        """
        Given a state of spin s represented by its decimal I_s, find the state with decimal I_s2 that is connected to it via creation operator
        I_rest is the part of the wavefunction 'infront' of the spin component here.
        ordering:

        |full> = c^\dagger_{L,up}...c^\dagger_{1,up} c^\dagger_{L,down}...c^\dagger_{1,down} |0>

        so in this case when computing c^\dagger_up we don't care about the spin_down part but if computing c^\dagger_down we do care about the spin_up part.
        ------------------------
        Input:

        Output:
        
        """
        if (I_s >> j) & 1:  # If site j is already occupied, return None
            return None, None
        I_s2 = I_s | (1 << j)
        # Compute sign factor (count fermions to the left)
        if (self.sign == True) or (self.JWstring == True):
            sign = (-1) ** ((bin(I_s & ((1 << j) - 1)).count('1')) + (bin(I_rest).count('1') if I_rest is not None else 0))
        else:
            sign = +1
        return I_s2, sign
    def create_mapping(self):
        """
        Construct creation operator mapping using separate spin-up and spin-down bases.
        """
        mapping_up = {}
        mapping_dn = {}
        for sector in self.bases:
            sec_basis_up,sec_basis_dn = self.bases[sector]
            #print('sector',sector)
            for state_up in sec_basis_up:
                I_up = sec_basis_up[state_up]
                for state_dn in sec_basis_dn:
                    I_dn = sec_basis_dn[state_dn]
                    for j in range(self.L):
                        #apply creation operators
                        I_up2,sign_up = self.apply_creation_operator_spinless(I_up,j)
                        if I_up2 != None:
                            #I_new_up = I_up2+I_dn*(2**self.L)
                            sector_new_up = (sector[0]+1, sector[1])
                        I_dn2,sign_dn = self.apply_creation_operator_spinless(I_dn,j,I_up)
                        if I_dn2 != None:
                            #I_new_dn = I_up+I_dn2*(2**self.L)
                            sector_new_dn = (sector[0], sector[1]+1)
                        # turn states (I's) into indices
                        #print(I_up,I_dn,I_up2,I_dn2)
                        ind_in = self.State2Ind(sector,I_up,I_dn)
                        ind_out_up = self.State2Ind(sector_new_up,I_up2,I_dn)
                        ind_out_dn = self.State2Ind(sector_new_dn,I_up,I_dn2)
                        #print(ind_in,ind_out_dn,ind_out_up)
                        #print('*')
                        #save....
                        if ind_out_up is not None:
                            mapping_up[(j,sector,ind_in)] = (sector_new_up,ind_out_up,sign_up)
                        if ind_out_dn is not None:
                            mapping_dn[(j,sector,ind_in)] = (sector_new_dn,ind_out_dn,sign_dn)  
        self.mapping_up = mapping_up
        self.mapping_dn = mapping_dn
        return
    
    def create_spin_spin_mapping(self):
        '''
        Create a mapping for the *basis states* under the action of
        
        c†_{↓r} c_{↑r}

        This will connect two sectors (N↑,N↓) & (N↑-1,N↓ +1) and specifically two basis elements J,I (or indices j,i) via a phase (sign)

        Output:
            map(dict)           :Has keys tuples of the form (r,sec,j) and values (sec_new,j,sign)
        '''
        def apply_c_dagger_c(I_up,I_dn,r):
                    
            """
            Apply c^\dagger_dn c_up to a state (Iup+2^LIdn).
            Ordering:
            |full> = c^\dagger_{L,up}...c^\dagger_{1,up} c^\dagger_{L,down}...c^\dagger_{1,down} |0>
            ------------------------
            Input:
                I_up(int)           :The decimal representation of spin-up component of state
                I_dn(int)           :The decimal representation of spin-dn component of state
                r(int)              :The location of application of the operator
            Output:
                J_up(int)           :The decimal representation of spin-up component of new state
                J_dn(int)           :The decimal representation of spin-dn component of new state
                sgn(\pm 1)          :The associated sign
            -----------------------------
            If output is None then that means state cannot support this c^\dagger c term
            
            """
            if ((I_dn >> r)&1 == 1) or (((I_up >> r)&1 == 0))==True: # hopping not possible
                return None, None, None
            J_up = I_up ^ (1 << r)#replace 1 with 0 at r
            J_dn = I_dn | (1 << r)#replace 0 with 1 at r
            # Compute sign factor (count fermions between 0 and r-1)
            if (self.sign == True) or (self.JWstring == True): #if we are dealing either with fermions or with Bosons+JW string, this erm might have a sign
                sign = (-1) ** ((bin(I_up & ((1 << r) - 1)).count('1')) + (bin(I_dn & ((1 << r) - 1)).count('1')))
            else:
                sign = +1
            return J_up,J_dn,sign

        mapping_cdagc = {}
        for r in range(self.L):#location of creation/annihilation operator
            for sector in self.bases:
                sec_basis_up,sec_basis_dn = self.bases[sector]
                for state_up in sec_basis_up:#sec_basis_up is dict with keys the indices and values the decimal representation of states
                    for state_dn in sec_basis_dn:
                        I_up = sec_basis_up[state_up]
                        I_dn = sec_basis_dn[state_dn]
                        J_up,J_dn,sign = apply_c_dagger_c(I_up,I_dn,r)
                        if J_up == None:continue#skip this (I_up,I_dn) state
                        sector_new = (sector[0]-1, sector[1]+1)
                        #now turn dec reps of states to indices
                        i = self.State2Ind(sector,I_up,I_dn)
                        j = self.State2Ind(sector_new,J_up,J_dn)
                        mapping_cdagc[(r,sector,i)] = (sector_new,j,sign)
        self.mapping_cdagc = mapping_cdagc
        return
    def create_eta_mapping(self):
        '''
        Create a mapping for the *basis states* under the action of
        
        c_{↓r} c_{↑r}

        which is the generator of the hidden SU(2) symmetry of the Hubbard model.

        This will connect two sectors (N↑,N↓) & (N↑-1,N↓ -1) and specifically two basis elements J,I (or indices j,i) via a phase (sign)

        Output:
            map(dict)           :Has keys tuples of the form (r,sec,j) and values (sec_new,j,sign)
        '''
        def apply_c_c(I_up,I_dn,r):
                    
            """
            Apply c_dn c_up to a state (Iup+2^LIdn).
            Ordering:
            |full> = c^\dagger_{L,up}...c^\dagger_{1,up} c^\dagger_{L,down}...c^\dagger_{1,down} |0>
            ------------------------
            Input:
                I_up(int)           :The decimal representation of spin-up component of state
                I_dn(int)           :The decimal representation of spin-dn component of state
                r(int)              :The location of application of the operator
            Output:
                J_up(int)           :The decimal representation of spin-up component of new state
                J_dn(int)           :The decimal representation of spin-dn component of new state
                sgn(\pm 1)          :The associated sign
            -----------------------------
            If output is None then that means state cannot support this c^\dagger c term
            
            """
            if ((I_dn >> r)&1 == 0) or (((I_up >> r)&1 == 0))==True: # hopping not possible
                return None, None, None
            J_up = I_up ^ (1 << r)#replace 1 with 0 at r
            J_dn = I_dn ^ (1 << r)#replace 1 with 0 at r
            # Compute sign factor (count fermions between 0 and r-1)
            if (self.sign == True) or (self.JWstring == True): #if we are dealing either with fermions or with Bosons+JW string, this erm might have a sign
                sign = (-1) ** ((bin(I_up & ((1 << r) - 1)).count('1')) + (bin(I_dn & ((1 << r) - 1)).count('1')))
            else:
                sign = +1
            return J_up,J_dn,sign
        
        mapping_cdagc = {}
        for r in range(self.L):#location of annihilation operators
            for sector in self.bases:
                sec_basis_up,sec_basis_dn = self.bases[sector]
                for state_up in sec_basis_up:#sec_basis_up is dict with keys the indices and values the decimal representation of states
                    for state_dn in sec_basis_dn:
                        I_up = sec_basis_up[state_up]
                        I_dn = sec_basis_dn[state_dn]
                        J_up,J_dn,sign = apply_c_c(I_up,I_dn,r)
                        if J_up == None:continue#skip this (I_up,I_dn) state
                        sector_new = (sector[0]-1, sector[1]-1)
                        #now turn dec reps of states to indices
                        i = self.State2Ind(sector,I_up,I_dn)
                        j = self.State2Ind(sector_new,J_up,J_dn)
                        mapping_cdagc[(r,sector,i)] = (sector_new,j,sign)
        self.mapping_cdagc = mapping_cdagc
        return
    
    def test_eta_mapping(self,sector_test=(3,3)):
        '''
        meant to test the ss_mapping function
        '''
        print('\n \n \n','LOOKING AT C^\DAGGER_DOWN C_UP AT SECTOR',sector_test,'\n \n \n')
        if not hasattr(self,'mapping_cdagc'):
            timei = time.time()
            self.create_spin_spin_mapping()
            timef = time.time()
            print('time for spin spi mapping',timef-timei)
        sector_new_test = (sector_test[0]-1,sector_test[1]+1)
        basis_up = self.bases[sector_test][0]
        dim_up = len(basis_up)
        basis_dn = self.bases[sector_test][1]
        #print(basis_up,basis_dn)
        #quit()
        basis_up_new = self.bases[sector_new_test][0]
        dim_up_new = len(basis_up_new)
        basis_dn_new = self.bases[sector_new_test][1]
        for key in self.mapping_cdagc.keys():
            #print('key',key)
            #continue
            r = key[0]
            sector = key[1]
            i = key[2]
            #state_
            if sector == sector_test:
                ##############################
                i_dn = i//dim_up +1
                i_up = i%dim_up +1
                #print(i,i_up,i_dn)
                Iup = basis_up[i_up]
                Idn = basis_dn[i_dn]
                ##############################
                vals = self.mapping_cdagc[key]
                j = vals[1]
                j_dn = j//dim_up_new +1
                j_up = j%dim_up_new +1
                Jup = basis_up_new[j_up]
                Jdn = basis_dn_new[j_dn]
                sgn = vals[2]
                print('hopping at site ',r,': \n','from state',(self.binp(Iup,length=self.L),self.binp(Idn,length=self.L)),' to  state ',(self.binp(Jup,length=self.L),self.binp(Jdn,length=self.L)),' with sign',sgn,'\n','*'*100)
        return
    
    def test_ss_mapping(self,sector_test=(3,2)):
        '''
        meant to test the ss_mapping function
        '''
        print('\n \n \n','LOOKING AT C^\DAGGER_DOWN C_UP AT SECTOR',sector_test,'\n \n \n')
        if not hasattr(self,'mapping_cdagc'):
            timei = time.time()
            self.create_spin_spin_mapping()
            timef = time.time()
            print('time for spin spi mapping',timef-timei)
        sector_new_test = (sector_test[0]-1,sector_test[1]+1)
        basis_up = self.bases[sector_test][0]
        dim_up = len(basis_up)
        basis_dn = self.bases[sector_test][1]
        #print(basis_up,basis_dn)
        #quit()
        basis_up_new = self.bases[sector_new_test][0]
        dim_up_new = len(basis_up_new)
        basis_dn_new = self.bases[sector_new_test][1]
        for key in self.mapping_cdagc.keys():
            #print('key',key)
            #continue
            r = key[0]
            sector = key[1]
            i = key[2]
            #state_
            if sector == sector_test:
                ##############################
                i_dn = i//dim_up +1
                i_up = i%dim_up +1
                #print(i,i_up,i_dn)
                Iup = basis_up[i_up]
                Idn = basis_dn[i_dn]
                ##############################
                vals = self.mapping_cdagc[key]
                j = vals[1]
                j_dn = j//dim_up_new +1
                j_up = j%dim_up_new +1
                Jup = basis_up_new[j_up]
                Jdn = basis_dn_new[j_dn]
                sgn = vals[2]
                print('hopping at site ',r,': \n','from state',(self.binp(Iup,length=self.L),self.binp(Idn,length=self.L)),' to  state ',(self.binp(Jup,length=self.L),self.binp(Jdn,length=self.L)),' with sign',sgn,'\n','*'*100)
        return
    
    def testing_mapping(self):
        '''
        testing the (1,2) sector for L=2... problem has to be here or on matrix_element calculation...
        '''
        self.create_mapping()
        map = self.mapping_up
        if self.L != 3:
            raise ValueError
        sector = (1,2)
        index = 0
        for key in map.keys():
            if key[1] == sector:
                index +=1
                state_in = key[-1]
                state_out = map[key][1]
                sign = map[key][-1]
                print('(',index,')  state in',state_in,'---->state out',state_out,'sign',sign)
        return
    def State2Ind(self,sector,I_up,I_dn):
        if (I_up is None) or (I_dn is None):
            return None
        ind_up = self.bases_inv[sector][0]
        m_up = ind_up[I_up]
        ind_dn = self.bases_inv[sector][1]
        m_dn = ind_dn[I_dn]
        ind = (m_dn-1)*len(ind_up)+(m_up-1) # maybe doesn't need the -1's
        return ind

    def allowed_transitions(self, i, s):
        '''
        Determine which symmetry sectors {|sec_new>} and |{sec}> are connected via c^\dagger_{i,s}.
        '''
        mapping = self.mapping_up if s == 'up' else self.mapping_dn
        pairs = set()
        for key in mapping:
            if i == key[0]:  # Check if the creation operator acts on site i
                sec = key[1]
                sec_new = mapping[key][0]
                pairs.add((sec, sec_new))
        return list(pairs)
    #######
    def partition_function(self, beta):
        """
        Computes the partition function using logsumexp for numerical stability,
        leveraging NumPy vectorization to speed up calculations.
        returns *log_Z*
        """
        all_energies = []
        # Gather all energies
        for sectors in self.energies:
            all_energies.append(self.energies[sectors])
        # Convert lists to NumPy arrays for fast computation
        all_energies = np.concatenate(all_energies)
        # Compute log partition function in one vectorized step
        log_Z = logsumexp(- beta * all_energies)
        return log_Z
    def partition_function_partial(self, beta,parity=0):
        """
        Computes the partition function in a given parity sector using logsumexp for numerical stability,
        leveraging NumPy vectorization to speed up calculations.
        returns *log_Z*

        Parity = 0: even number of particles
        Parity = 1: odd number of particles
        """
        all_energies = []
        # Gather all energies
        for sectors in self.energies:
            #print('parity check',sectors,parity,(sectors[0]+sectors[1])%2 == parity)
            if (sectors[0]+sectors[1])%2 == parity:
                all_energies.append(self.energies[sectors])
        # Convert lists to NumPy arrays for fast computation
        all_energies = np.concatenate(all_energies)
        # Compute log partition function in one vectorized step
        log_Z_P = logsumexp(- beta * all_energies)
        return log_Z_P
        
    def compute_matrix_elements(self, mapping):
        """
        Compute <n|O|m> in the eigenbasis given that you have the operator in the occupation basis <j|O|i>
        Used for:
                1)The single particle Green's function
                2)The Spin-Spin Green's function

        Input:
            mapping(dict)           :Dictionary of the form dict[(j,sector,index_in)] = (sector_new,index_out,sign)
                                    Where j is the location of the operator O_r, sector and sector_new are the two sectors connected via O ,
                                    index_in and index_new are indices of states related by O and sign is due to fermionic ordering or JW string.
        """
        matrix_elements = {}
        
        for (j, sector, ind_in), (sector_new, ind_out, sign) in mapping.items():
            if sector not in self.eigenstates or sector_new not in self.eigenstates:
                print('sector not found?',sector,sector_new)
                continue
            eigvecs_sec = self.eigenstates[sector]
            eigvecs_sec_new = self.eigenstates[sector_new]
            num_states_sec = eigvecs_sec.shape[0]
            num_states_sec_new = eigvecs_sec_new.shape[0]
            
            matrix_elements.setdefault((j,sector, sector_new), np.zeros((num_states_sec_new, num_states_sec), dtype=np.complex128))
            for n in range(num_states_sec_new):
                for m in range(num_states_sec):
                    matrix_elements[(j,sector, sector_new)][n, m] += (
                        np.conj(eigvecs_sec_new[ind_out,n]) * sign * eigvecs_sec[ind_in,m]
                    )
        return matrix_elements

    def Energy(self,beta):
        log_Z =  self.partition_function(beta)#changed pervious code so that it returns logsumexp() rather than its exponential
        log_terms_H = []
        for sector in self.energies:
            energies = self.energies[sector]
            valid_mask = energies > 0
            valid_energies = energies[valid_mask]
            #
            if valid_energies.size > 0:
                log_terms_H.append((-beta * valid_energies) + np.log(valid_energies))  
                log_H = logsumexp(np.concatenate(log_terms_H)) - log_Z

        ##################################
        #while not physically relevant, return also the values with shifted energy.
        H_avg = np.exp(log_H)
        H_avg_unshifted = H_avg + self.lowestEnergy
        return H_avg_unshifted
    def OccNum(self,beta):
        '''
        <N> = \sum_sectors \sum_a <a|Nexp(-betaH)|a> = \sum_sectors N_sector \sum_a exp(-beta E_a)
        '''

        log_Z =  self.partition_function(beta)#changed pervious code so that it returns logsumexp() rather than its exponential
        log_terms_N = []
        for sector in self.energies:
            energies = self.energies[sector]
            occupation_num = sector[0]+sector[1]
            #print(sector,occupation_num)  
            #
            if occupation_num > 0:
                log_terms_N.append((-beta * energies) + np.log(occupation_num))  
                log_N = logsumexp(np.concatenate(log_terms_N)) - log_Z
        ##################################
        #while not physically relevant, return also the values with shifted energy.
        N_avg = np.exp(log_N)
        return N_avg
    def matrix_elements_debug(self,beta):
        '''
        Zooming in: L=3 sector (1,2)--->(2,2) does not give equal contribution at different sites...
        Which sectors have this issue in general????
        Comparing G_11 and G_00 here only!
        '''
        tau = beta/4
        self.matrix_elements_up = self.compute_matrix_elements(mapping=self.mapping_up)
        if self.L != 3:
            print('this debugging is made for L=3 for now')
            #raise ValueError
        log_Z = self.partition_function(beta)
        sec_0_contributions = {}
        sec_1_contributions = {}
        sectors_0 = set(self.allowed_transitions(0, 'up'))
        sectors_1 = set(self.allowed_transitions(1, 'up'))
        if sectors_0 != sectors_1:
            print('???')
            quit()
        for (sec, sec_new) in sectors_0:
            #if sec != (1,0):continue
            sec_0_contributions[sec] =0
            sec_1_contributions[sec] =0
            for m in range(len(self.eigenstates[sec])):
                for n in range(len(self.eigenstates[sec_new])):
                    spin = 'up'
                    Em = self.energies[sec][m]
                    En = self.energies[sec_new][n]
                    matrix_elements = self.matrix_elements_up if spin == 'up' else self.matrix_elements_dn
                    amp_0 = matrix_elements[(0, sec, sec_new)][n, m]
                    amp_1 = matrix_elements[(1, sec, sec_new)][n, m]
                    log_terms = -beta * Em - tau * (En - Em)
                    sec_0_contributions[sec]+= (np.abs(amp_0)**2) * np.exp(log_terms - log_Z)
                    sec_1_contributions[sec]+= (np.abs(amp_1)**2) * np.exp(log_terms - log_Z)
                    if sec == (1,2):
                        print('contribution 0',(np.abs(amp_0)**2)) #* np.exp(log_terms - log_Z))
                        print('contribution 1',(np.abs(amp_1)**2)) #* np.exp(log_terms - log_Z))
        #print('contributions difference',np.abs(sec_0_contributions[(1,0)]-sec_1_contributions[(1,0)]))
        for key in sec_0_contributions.keys():
            if np.abs(sec_0_contributions[key]-sec_1_contributions[key])>1e-10:
                print('secs contributions not equal',key,np.abs(sec_0_contributions[key]-sec_1_contributions[key]))
        #print('comparing matrix elements')
        #amp_0 = matrix_elements[(0,(1,0), (2,0))]
        #amp_1 = matrix_elements[(1, (1,0), (2,0))]
        #print(amp_0)
        print('----')
        #print(amp_1)
        return
    def GreenFuncDebug(self,beta,n_tau,sector,parity):
        '''
        do it in a single sector....
        same as original Green's function but to debug..
        I want to check if G_{ij} = G(|r_i-r_j|)
        step(1) G_00 vs G_11 vs G_22 etc
        step(2) G_r,0 vs G_(r+1),1 ...

        THOUGHTS SO FAR:
        SGN=TRUE WORKS FINE, ITS ONLY SGN = FALSE THAT FAILS. ALSO EG SGN = FALSE STILL WORKS FOR L=4,DeltaR=2
        '''
        if not hasattr(self,"matrix_elements_up"):
            print('calclulating matrix elements up')
            self.matrix_elements_up = self.compute_matrix_elements(mapping=self.mapping_up)
        #####
        taus = np.linspace(0, beta, num=n_tau)
        if parity != None:
            log_Z = self.partition_function_partial(beta,parity=parity)
        elif sector != None:
            log_Z = 1
        else:
            log_Z = self.partition_function(beta)
        G = np.zeros((self.L,n_tau), dtype=np.complex128)
        for i in range(self.L):
            j = (i)%self.L
            sectors_i = set(self.allowed_transitions(i, 'up'))
            sectors_j = set(self.allowed_transitions(j, 'up'))
            sectors = sectors_i.intersection(sectors_j)
            if self.verbose>1:print('*'*100,'\n ALLOWED SECTORS FOR SITE',i,': \n',sectors,'\n ','*'*100)
            for (sec, sec_new) in sectors:
                if (sector == None) and (parity != None):
                    sectors_i = [elem for elem in sectors_i if sum(elem[0]) % 2 == parity]
                    sectors_j = [elem for elem in sectors_j if sum(elem[0]) % 2 == parity]
                    sectors_i = set(sectors_i)
                    sectors_j = set(sectors_j)
                    sectors = sectors_i.intersection(sectors_j)
                elif (sector != None) and (parity == None):
                    if (sec not in  sector): continue
                elif (sector != None) and (parity != None):
                    raise ValueError
                #if self.verbose>1:print('now doing sectors',sec,sec_new)
                for m in range(len(self.eigenstates[sec])):
                    for n in range(len(self.eigenstates[sec_new])):
                        spin = 'up'
                        Em = self.energies[sec][m]
                        En = self.energies[sec_new][n]
                        matrix_elements = self.matrix_elements_up if spin == 'up' else self.matrix_elements_dn
                        amp_j = matrix_elements[(j, sec, sec_new)][n, m]
                        amp_i = matrix_elements[(i, sec, sec_new)][n, m].conj()
                        log_terms = -beta * Em - taus * (En - Em)
                        G[i, :] += amp_i*amp_j * np.exp(log_terms - log_Z)
                #if sec == (1,2):print('SITE',i,'CONTRIBUTION',sec_contribution)
           # if self.verbose>1:print('*'*100,'\n GREENS FUNCTION AT SITE ',i,' has elements \n',G[i,:],'\n','*'*100)
        #if self.verbose>0:print(G)
        return G
    def GreenFunc(self, beta,n_tau):
        """
        Returns the Green's function for the system. Has size L x L x s x s x Ntau --> L x L x s x Ntau
        
        NOTE 03 APRIL UPDATES: ADDED AN OVERALL MINUS SIGN. G(0,0,TAU=0+)= -1 + <N> = -1/2 EG, NOT 1/2
        """
        if not hasattr(self,"matrix_elements_up"):
            print('calclulating matrix elements up')
            self.matrix_elements_up = self.compute_matrix_elements(mapping=self.mapping_up)
        if not hasattr(self,"matrix_elements_dn"):
            print('calclulating matrix elements dn')
            self.matrix_elements_dn = self.compute_matrix_elements(mapping=self.mapping_dn)
        taus = np.linspace(0, beta, num=n_tau)
        G = np.zeros((self.L, self.L, 2, n_tau), dtype=np.complex128)
        
        log_Z = self.partition_function(beta)
        
        for i in range(self.L):
            for j in range(self.L):
                for spin_idx, spin in enumerate(['up', 'down']):
                    sectors_i = set(self.allowed_transitions(i, spin))
                    sectors_j = set(self.allowed_transitions(j, spin))
                    sectors = sectors_i.intersection(sectors_j)
                    for (sec, sec_new) in sectors:
                        for m in range(len(self.eigenstates[sec])):
                            for n in range(len(self.eigenstates[sec_new])):
                                Em = self.energies[sec][m]
                                En = self.energies[sec_new][n]
                                matrix_elements = self.matrix_elements_up if spin == 'up' else self.matrix_elements_dn
                                amp_j = matrix_elements[(j, sec, sec_new)][n, m]
                                amp_i = matrix_elements[(i, sec, sec_new)][n, m].conj()
                                log_terms = -beta * Em - taus * (En - Em)
                                G[i, j, spin_idx, :] += amp_i*amp_j * np.exp(log_terms - log_Z)
                
        return G

    def GreenFunc_partial(self, beta,n_tau,parity=0):
        """
        A modification of the GreenFunc function, calculating it only considering 'even'/'odd' symmetry sectors.
        Used to benchmark with Dumitru's SSE code
        Adding 'even' and 'odd' terms should match the full green's function
        ----------------------------------
        Parity=0(even) or 1 (odd)
        ----------------------------------
        Has size L x L x s x s x Ntau --> L x L x s x Ntau
        ----------------------------------
        NOTE 03 APRIL UPDATES:  NOW USING THE CORRECT PARTITIONA FUNCTION WITH SAME PARITY AS GREENS FUNCTION
                                ALSO, ADDED AN OVERALL MINUS SIGN. G(0,0,TAU=0+)= -1 + <N> = -1/2 EG, NOT 1/2
        """
        if not hasattr(self,"matrix_elements_up"):
            print('calclulating matrix elements up')
            self.matrix_elements_up = self.compute_matrix_elements(mapping=self.mapping_up)
        if not hasattr(self,"matrix_elements_dn"):
            print('calclulating matrix elements dn')
            self.matrix_elements_dn = self.compute_matrix_elements(mapping=self.mapping_dn)
        taus = np.linspace(0, beta, num=n_tau)
        G = np.zeros((self.L, self.L, 2, n_tau), dtype=np.complex128)
        log_Z_p = self.partition_function_partial(beta,parity=parity)#NOTE Changed it in 03 April to use the partition function of that sector
        for i in range(self.L):
            for j in range(self.L):
                for spin_idx, spin in enumerate(['up', 'down']):
                    sectors_i = self.allowed_transitions(i, spin)
                    sectors_j = self.allowed_transitions(j,spin)
                    #project out some sections
                    sectors_i = [elem for elem in sectors_i if sum(elem[0]) % 2 == parity]
                    sectors_j = [elem for elem in sectors_j if sum(elem[0]) % 2 == parity]
                    sectors_i = set(sectors_i)
                    sectors_j = set(sectors_j)
                    sectors = sectors_i.intersection(sectors_j)
                    #print('allowed sectors w/ parity',parity,':',sectors)
                    for (sec, sec_new) in sectors:
                        for m in range(len(self.eigenstates[sec])):
                            for n in range(len(self.eigenstates[sec_new])):
                                Em = self.energies[sec][m]
                                En = self.energies[sec_new][n]
                                matrix_elements = self.matrix_elements_up if spin == 'up' else self.matrix_elements_dn
                                amp_j = matrix_elements[(j, sec, sec_new)][n, m]
                                amp_i = matrix_elements[(i, sec, sec_new)][n, m].conj()
                                log_terms = -beta * Em - taus * (En - Em)
                                G[i, j, spin_idx, :] -= amp_i*amp_j * np.exp(log_terms - log_Z_p) #NOTE: THE UPDATED MINUS SIGN (-= instead of +=)
        return G
    #################
    def Spin_correlation_debug(self,beta,n_tau,sector):
        log_Z = self.partition_function(beta)
    
        if not hasattr(self,'mapping_cdagc'):
            self.create_spin_spin_mapping()
        ###then create matrix elements
        if not hasattr(self,"matrix_elements_spin_spin"):
            print('calclulating matrix elements for spin spin correlations')
            timei = time.time()
            self.matrix_elements_spin_spin = self.compute_matrix_elements(mapping=self.mapping_cdagc)
            timef = time.time()
            print('TOOK ',timef-timei,' seconds to calculate spin-spin matrix elmements')
        #sec_pairs holds all pairs of sectors related by c^dagger c
        sec_pairs = set()
        for key in self.mapping_cdagc:
                sec = key[1]
                sec_new = self.mapping_cdagc[key][0]
                sec_pairs.add((sec,sec_new))
        sec_pairs = list(sec_pairs)

        taus = np.linspace(0, 1, num=n_tau)*beta
        G = np.zeros((self.L, n_tau), dtype=np.complex128)
        #
        for (sec, sec_new) in sec_pairs:
            if (sec not in sector):continue#filter out certain sectors
            print('sectors',sec,sec_new)
            for x in range(self.L):
                    y = (x)%self.L
                    for m in range(len(self.eigenstates[sec])):
                        for n in range(len(self.eigenstates[sec_new])):
                            Em = self.energies[sec][m]
                            En = self.energies[sec_new][n]
                            amp_y = self.matrix_elements_spin_spin[(y, sec, sec_new)][n, m]
                            amp_x = self.matrix_elements_spin_spin[(x, sec, sec_new)][n, m].conj()
                            log_terms = -beta * Em - taus * (En - Em)
                            G[x, :] += amp_x*amp_y * np.exp(log_terms - log_Z)
        return G
    def Spin_correlation_function(self,beta,n_tau,parity=None):
        '''
        Calculates the S^+(\\tau) S^-(0) correlation function
        -----------------------------------------------------
        Input:
        beta(float)             :The INverse temperature of the system
        n_tau(int)              :At how many imaginary times to calculate the dynamic correlation function
        parity(0/1 or None)     :Calculate the full or partial Green's function
        -----------------------------------------------------
        TODO:       1) Implement parity
                    2) Check translation invariance 
        '''
        #####
        #internal function to calculate the matrix elements <n|c^\dagger c|m>
        #####
        if parity != None:
            log_Z = self.partition_function_partial(beta,parity)
        else:
            log_Z = self.partition_function(beta)
    
        if not hasattr(self,'mapping_cdagc'):
            self.create_spin_spin_mapping()
        ###then create matrix elements
        if not hasattr(self,"matrix_elements_spin_spin"):
            print('calclulating matrix elements for spin spin correlations')
            timei = time.time()
            self.matrix_elements_spin_spin = self.compute_matrix_elements(mapping=self.mapping_cdagc)
            timef = time.time()
            print('TOOK ',timef-timei,' seconds to calculate spin-spin matrix elmements')
        #sec_pairs holds all pairs of sectors related by c^dagger c
        sec_pairs = set()
        for key in self.mapping_cdagc:
                sec = key[1]
                sec_new = self.mapping_cdagc[key][0]
                sec_pairs.add((sec,sec_new))
        sec_pairs = list(sec_pairs)

        taus = np.linspace(0, 1, num=n_tau)*beta
        G = np.zeros((self.L, self.L, n_tau), dtype=np.complex128)
        #
        for (sec, sec_new) in sec_pairs:
            if (parity!=None) and (sum(sec) % 2 != parity):continue#filter out certain sectors
            print('sectors',sec,sec_new)
            for x in range(self.L):
                for y in range(self.L):
                        for m in range(len(self.eigenstates[sec])):
                            for n in range(len(self.eigenstates[sec_new])):
                                Em = self.energies[sec][m]
                                En = self.energies[sec_new][n]
                                amp_y = self.matrix_elements_spin_spin[(y, sec, sec_new)][n, m]
                                amp_x = self.matrix_elements_spin_spin[(x, sec, sec_new)][n, m].conj()
                                log_terms = -beta * Em - taus * (En - Em)
                                G[x, y, :] += amp_x*amp_y * np.exp(log_terms - log_Z)
        return G

    def spectral_function(self,beta,omega,broadening):
        '''
        Calculates the spectral function in real space :
        A(r1,r2,ω) = Z^{-1} \sum_{m,n} [exp(-βE_n)+exp(-βE_m)]<m|c_1|n><n|c^\dagger_2|m> delta(ω - (E_n - E_m)
        ---------------------------------------------------------
        Input:
            beta(float)         :The inverse temperature of the system
            broadening(float)   :How much to broaden the delta function
            sum_test_rule(Bool) :Test that the sum rule is satisfied
            omega(list)         :A list of the form [ω_min,ω_max,ω_steps]
            kspace(Bool)        :If true, fourier transform
        Output:
            A(npc array)        :An array of size L x L x ω_steps
        '''
        #0) define broadening function
        def f_eta(x,eta=broadening):
            return (1/np.pi)*(eta)/(eta**2 + x**2)
        #1)get matrix elements and intiialize
        omegas = np.linspace(omega[0], omega[1], num= omega[2])
        A = np.zeros((self.L, self.L, omega[2]), dtype=np.complex128) #spectral func
        if not hasattr(self,"matrix_elements_up"):
            print('calclulating matrix elements up')
            self.matrix_elements_up = self.compute_matrix_elements(mapping=self.mapping_up)
        if not hasattr(self,"matrix_elements_dn"):
            print('calclulating matrix elements dn')
            self.matrix_elements_dn = self.compute_matrix_elements(mapping=self.mapping_dn)

        log_Z = self.partition_function(beta)
        
        for i in range(self.L):
            for j in range(self.L):
                for spin_idx, spin in enumerate(['up', 'down']):
                    sectors_i = set(self.allowed_transitions(i, spin))
                    sectors_j = set(self.allowed_transitions(j, spin))
                    sectors = sectors_i.intersection(sectors_j)
                    for (sec, sec_new) in sectors:
                        for m in range(len(self.eigenstates[sec])):
                            for n in range(len(self.eigenstates[sec_new])):
                                Em = self.energies[sec][m]
                                En = self.energies[sec_new][n]
                                weight = np.exp(-beta*Em - log_Z) + np.exp(-beta*En - log_Z)
                                matrix_elements = self.matrix_elements_up if spin == 'up' else self.matrix_elements_dn
                                amp_j = matrix_elements[(j, sec, sec_new)][n, m]
                                amp_i = matrix_elements[(i, sec, sec_new)][n, m].conj()
                                A[i, j,:] += amp_i*amp_j * weight * f_eta(omegas)
                
        return A
    #################
    @staticmethod
    def binp(num, length):
        '''
        print a binary number without python 0b and appropriate number of zeros
        regular bin(x) returns '0bbinp(x)' and the 0 and b can fuck up other stuff
        '''
        return format(num, '#0{}b'.format(length + 2))[2:]
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

#######################################
def EDFullSpectrum(params):
        '''
        Function:
        ---------
        This is a bridging function to feed as input to a thermodynamics class instance \n

        Input:
        ------
        params(dict):       Contains all info needed by self.hubbard_chain (look at init) except Nup,Ndn. \n

        Output:
        -------
        Does ED on all possible sectors and returns
        0) Info on lowest sector and energy
        1) Energies as a list of 1D arrays, one array per sector
        2) Eigenstates as a list of 2D arrays, one per sector
        '''
        energies = {}
        eigenstates = {}
        bases = {}
        bases_inv = {}
        lowestEnergy = 1e10
        L = params['L']
        timei = time.time()
        for n_up in range(0,L+1):
            for n_down in range(0,L+1):
                if params['verbose']>0:print('='*20,f' \n Diagonalizing sector{n_up,n_down} \n','='*20)
                params['Nup'] = n_up
                params['Ndn'] = n_down
                chain = hubbard_chain(params)
                chain.build_ham()
                lam,v = np.linalg.eigh(chain.Hamiltonian)
                energies[(n_up,n_down)] = lam
                eigenstates[(n_up,n_down)] = v
                bases[(n_up,n_down)] = [chain.basis_up,chain.basis_dn]
                bases_inv[(n_up,n_down)] = [chain.index_up,chain.index_dn]
                if min(lam) < lowestEnergy:
                    lowestEnergy  = min(lam)
                    GSSector      = (n_up,n_down)
        timef = time.time()
        if params['verbose']>0:
            print('time needed to diagonalize all sectors',timef-timei)
            print("The ground state occured in (n_up,n_down)=",GSSector,':',lowestEnergy)
            print('-'*100)
            print('\n \n ')
        ####################################
        #shift minmum#
        for sector in energies:
           energies[sector] -= lowestEnergy
        return energies,eigenstates,bases,bases_inv,lowestEnergy

#######################################
def mu_vs_filling(params,num= 50,beta = 10):
    '''
    gives a plot of mu vs filling so we know what mu to target for a specific filling
    '''
    mus = np.linspace(-1.5,1.5,num=num)*params['H_params']['U']
    filling = np.zeros_like(mus)
    for i,mu in enumerate(mus):
        params['H_params']['mu'] = mu
        energies,eigenstates,bases,bases_inv,lowestEnergy = EDFullSpectrum(params)
        params['energies'] = energies
        params['eigenstates'] = eigenstates
        params['bases'] = bases
        params['bases_inv'] = bases_inv
        params['lowestEnergy'] = lowestEnergy
        thermo = thermodynamics(params)
        n,nsq = thermo.N_moments(beta)
        filling[i] = n/params['L']
    print(filling)
    plt.plot(mus/params['H_params']['U'],filling,'.-',alpha = 0.5)
    plt.axhline(y=0.8,xmin=-10,xmax = 10,c='k')
    #plt.axhline(y=0.8,xmin=mus[0]/params['H_params']['U'],xmax = mus[-1]/params['H_params']['U'],c='k')
    plt.xlabel('$\\mu/U$')
    plt.ylabel('$\\nu$')
    plt.savefig('/mnt/users/kotssvasiliou/ED/core_scripts/figs/n_vs_mu.png',dpi=500)
def cv_vs_temp(params,num = 200,beta_min = 0.5,beta_max = 30,save_fig = False):
    '''
    plots cv vs T to benchmark with Phys. Rev. B 53, 6865
    '''
    energies,eigenstates,bases,bases_inv,lowestEnergy = EDFullSpectrum(params)
    params['energies'] = energies
    params['eigenstates'] = eigenstates
    params['bases'] = bases
    params['bases_inv'] = bases_inv
    params['lowestEnergy'] = lowestEnergy
    thermo = thermodynamics(params)
    betas = np.linspace(beta_min,beta_max,num = num)
    Cs = np.zeros_like(betas)
    for i,beta in enumerate(betas):
        Cs[i] = thermo.H_moments(beta,Cv=True)[-1]
        if i == (num-1):
            print('filling at this chem pot at beta = beta_max is ',thermo.N_moments(beta)[0]/params['L'])
    if save_fig != False:
        plt.plot(1/betas,Cs,'.-',alpha = 0.5)
        plt.xlabel('$T/t$')
        plt.ylabel('$C_V$')
        plt.savefig('/mnt/users/kotssvasiliou/ED/core_scripts/figs/specific_heat.png',dpi=500)
    return betas,Cs
def reproduce_fig_3_paper():
    '''
    reproduces figure 3 in the paper 
    '''
    ###
    L = 6
    t = 1;U = 2;V = 0;mu = -1.13
    params = {'L':L,'verbose':0,'species':'fermion','H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    betas,C1 = cv_vs_temp(params,num = 100)
    ####
    L = 4
    t = 1;U = 2;V = 0;mu = -0.587
    params_2 = {'L':L,'verbose':0,'species':'fermion','H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    betas,C2 = cv_vs_temp(params_2,num = 100)
    ######
    plt.plot(1/betas,C1,'o-',alpha = 0.5,c='b',label='L=6')
    plt.plot(1/betas,C2,'o-',alpha = 0.5,c='r',label='L=4')
    plt.xlabel('$T/t$')
    plt.ylabel('$C_V$')
    plt.ylim([0.0,0.6])
    plt.title('U=2,n=0.8')
    plt.legend()
    plt.savefig('/mnt/users/kotssvasiliou/ED/core_scripts/figs/specific_heat_reproduced_fig_3.png',dpi=500)
def reproduce_fig_2_paper():
    '''
    reproduces figure 2 in the paper 
    '''
    ###
    L = 6
    t = 1;U = 16;V = 0;mu = -6.802
    params = {'L':L,'verbose':0,'species':'fermion','H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    betas,C1 = cv_vs_temp(params,num = 100)
    ###
    plt.plot(1/betas,C1,'o-',alpha = 0.5,c='b',label='L=6')
    plt.xlabel('$T/t$')
    plt.ylabel('$C_V$')
    plt.ylim([0.0,0.4])
    plt.legend()
    plt.title('U=16,L=6,n=0.8')
    plt.savefig('/mnt/users/kotssvasiliou/ED/core_scripts/figs/specific_heat_reproduced_fig_2.png',dpi=500)
def check_TI(C,savefig = True):
    '''
    checks translational invariance for a correlator of the form C = L x L x tau
    '''
    L, _, T = C.shape
    r_data = {}
    cols = ['red','purple','blue','green','pink','black','gray','orange']
    styles = ['-', '--', '-.', ':','-','-.']

    for r1 in range(L):
        for r2 in range(L):
            r = (r1 - r2) % L  # use mod L if PBC
            if r not in r_data:
                r_data[r] = []
            r_data[r].append(C[r1, r2, :])

    plt.figure()
    for r in sorted(r_data):
        for c_tau in r_data[r]:
            plt.plot(range(T), c_tau, c=cols[r],linestyle = styles[r],alpha=0.25)
    plt.xlabel("$\\tau$")
    plt.ylabel("$C(r1, r2, \\tau)$")
    #plt.legend()
    plt.tight_layout()
    if savefig == True:
        plt.savefig('/mnt/users/kotssvasiliou/ED/core_scripts/figs/TI_corr_.png',dpi = 500)
    return
def observable_collapse(C,beta,mode = 'average',save_fig = None,title = None):
    '''
    Given C(r_i,r_j,tau) returns C with keys the dissplacements r and values arrays of length Ntau
    mode = average:
        average over arrays of each key
    mode = check_ti:
    checks if each array is the same 
    '''
    L = C.shape[0]
    Ntau = C.shape[-1]
    C_r = {}
    for r1 in range(L):
        for r2 in range(L):
            r = (r1 - r2) % L  
            if r not in C_r:
                C_r[r] = []
            C_r[r].append(C[r1, r2, :])
    if mode == 'check':
        print('checking TI')
        for r in C_r:
            print('len C_r',len(C_r[r]),C_r[r][0].shape)
            if len(C_r[r])>1:
                all_close = all(np.allclose(C_r[r][0], dat) for dat in C_r[r][1:])
                print(f'Is r={r} Translationally invariant?',all_close)
                if all_close == True:
                    del C_r[r][1:]
            print('len C_r',len(C_r[r]),C_r[r][0].shape)
    if mode == 'average':
        for r in C_r:
            avg = np.zeros_like(C_r[r][0])
            for dat in C_r[r]:
                avg += dat
            avg *= 1./L
            C_r[r][0] = avg
            del C_r[r][1:]
    if save_fig is not None:
        for r in sorted(C_r):
            if r != 1:
                continue
            for c_tau in C_r[r]:
                plt.plot((beta*1./Ntau)*np.arange(Ntau), c_tau,alpha=0.5,label=f'$r={r}$')
        plt.xlabel("$\\tau$")
        #plt.title("$B(r=r_1-r_2;\\tau) = B(r_1,r_2;\\tau)$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_fig + title,dpi = 500)
    return C_r
def partial_Zs(params,nbetas=100):
    '''
    Investigates the behaviour of the partial partition functions.
    '''
    thermo = thermodynamics(params)
    Z_1 = []
    Z_0 = []
    betas = np.linspace(0.5,20,num=nbetas)
    for beta in betas:
        Z_0.append(np.exp(thermo.logZ(beta,parity = 0)))
        Z_1.append(np.exp(thermo.logZ(beta,parity = 1)))
        if np.abs((Z_0[-1] + Z_1[-1]) - np.exp(thermo.logZ(beta,parity = None)))>1e-5:
            print(np.abs((Z_0[-1] + Z_1[-1]) - np.exp(thermo.logZ(beta,parity = None))))
            raise ValueError
    plt.plot(1/betas,np.array(Z_1)/np.array(Z_0))
    plt.xlabel('$T$')
    plt.ylabel('Z_1/Z_0')
    plt.savefig('core_scripts/figs/partial_Z.png')
    return
def TI_greens_func():
    '''
    '''
    return
def generate_benchmark_data(params,out = 'results.h5'):
    '''
    All info needed for Benchmarking with SSE in an h5 file
    -----------------------------------------------------------
    params: are the hamiltonian params to be fed into EDfullSpectrum
    out:    Output name
    '''
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
    # E,N ns beta
    Beta_max = 10
    Betas = np.linspace(start=1,stop=Beta_max,num=10,endpoint=True)
    Es = []
    Ns = []
    print('GS:',lowestEnergy)
    for beta in Betas:
        Es.append(thermo.H_moments(beta=beta)[1])
        Ns.append(thermo.N_moments(beta=beta)[0])
    #####################
    #parity resolved green's function at tau = 0
    thermo = thermodynamics(params)
    logZ_0 = thermo.logZ(beta=Beta_max,parity = 0)
    logZ_1 = thermo.logZ(beta=Beta_max,parity = 1)
    G_0_up = thermo.correlator(op='green',beta = Beta_max,n_tau= 10,parity = 0,green_spin='up')
    G_0_dn = thermo.correlator(op='green',beta = Beta_max,n_tau= 10,parity = 0,green_spin='dn')
    G_1_up = thermo.correlator(op='green',beta = Beta_max,n_tau= 10,parity = 1,green_spin='up')
    G_1_dn = thermo.correlator(op='green',beta = Beta_max,n_tau= 10,parity = 1,green_spin='dn')
    if np.allclose(G_0_dn,G_0_up) and np.allclose(G_1_dn,G_1_up):
        comp_0 = np.sum(np.abs(G_0_up.imag)) / np.sum(np.abs(G_0_up))
        comp_1 = np.sum(np.abs(G_1_up.imag)) / np.sum(np.abs(G_1_up))
        if (comp_0 <1e-8) and (comp_1<1e-8):
            G_0_up = G_0_up.real
            G_1_up = G_1_up.real
        else:
            raise ValueError
        G_0 = {}
        G_1 = {}
        #group by distance
        for i in range(params['L']):
            for j in range(params['L']):
                r = (i-j)%params['L']
                if r not in G_0.keys():
                    G_0[r] = []
                if r not in G_1.keys():
                    G_1[r] = []
                G_0[r].append(G_0_up[i,j,0])
                G_1[r].append(G_1_up[i,j,0])
        #reduce to translationally invariant....
        #we dont actually have translational invariance soooooo
        '''
        for r in G_0.keys():
            if len(G_0[r])>1:
                all_close = all(np.allclose(G_0[r][0], dat,rtol=1e-6) for dat in G_0[r][1:])
                if all_close == True:
                    del G_0[r][1:]
                else:
                    print(G_0[r])
                    #raise ValueError
            if len(G_1[r])>1:
                all_close = all(np.allclose(G_1[r][0], dat) for dat in G_1[r][1:])
                if all_close == True:
                    del G_1[r][1:]
                else:
                    continue
                    #raise ValueError
        '''
    else:
        print('GREEN FUNCTIONS NOT SPIN INDPT?')
        raise ValueError
    ########################
    #averaged  correlators for green's function, spin-spin and eta
    n_tau = 11
    taus = np.linspace(0,Beta_max, num=n_tau)
    Green = thermo.correlator(op='green',beta = Beta_max,n_tau= n_tau,parity = None,green_spin='up')
    Spin = thermo.correlator(op='spinspin',beta = Beta_max,n_tau= n_tau,parity = None)
    Eta =  thermo.correlator(op='eta',beta = Beta_max,n_tau= n_tau,parity = None)
    #keep real part
    comp_1 = np.sum(np.abs(Green.imag)) / np.sum(np.abs(Green))
    comp_2 = np.sum(np.abs(Spin.imag)) / np.sum(np.abs(Spin))
    comp_3 = np.sum(np.abs(Eta.imag)) / np.sum(np.abs(Eta))
    if (comp_1 <1e-8) and (comp_2<1e-8) and (comp_3<1e-8):
        Green = Green.real
        Spin = Spin.real
        Eta = Eta.real
    else:
        raise ValueError
    #
    Green_dat = observable_collapse(C=Green,beta = Beta_max,mode = 'average')
    Spin_dat = observable_collapse(C=Spin,beta = Beta_max,mode = 'average')
    Eta_dat = observable_collapse(C=Eta,beta = Beta_max,mode = 'average')
    ##########################
    ##########################
    ##########################
    #save in hdf5 file
    ##########################
    ##########################
    ##########################
    params_save = params['H_params'].copy()
    params_save['beta'] = Beta_max
    params_save['taus'] = taus
    with h5py.File(out, 'w') as f:
    # Save parameters as attributes
        param_grp = f.create_group('params')
        for k, v in params_save.items():
            if isinstance(v, np.ndarray):
                param_grp.create_dataset(k, data=v)
            else:
                param_grp.attrs[k] = v
        
        # G_0 and G_1 as groups
        for name, dictionary in {'G_0': G_0, 'G_1': G_1,'Greens':Green_dat,'SpinSpin':Spin_dat,'Pairing':Eta_dat}.items():
            grp = f.create_group(name)
            for k, arr in dictionary.items():
                grp.create_dataset(str(k), data=arr)
        f.create_dataset('logZ_0', data=logZ_0)
        f.create_dataset('logZ_1', data=logZ_1)
    return    
def compare_spectra(params):
    '''
    Q: is it true that PBC fermions and mBC hc bosons have same spectra?
    '''
    for L in [2,3]:
        print(params['H_params']['mu'])
        params['L'] = L
        params['species'] = 'fermion'
        energies_f = EDFullSpectrum(params)[0]
        params['species'] = 'mboson'
        energies_b = EDFullSpectrum(params)[0]
        for (n_up,n_dn) in energies_f.keys():
            if (n_up)%2 == 1 and (n_dn)%2 == 1:
                print(np.allclose(energies_f[(n_up,n_dn)],energies_b[(n_up,n_dn)]))
                if not np.allclose(energies_f[(n_up,n_dn)],energies_b[(n_up,n_dn)]):
                    print(L,(n_up,n_dn))
                    print(energies_f[(n_up,n_dn)])
                    print(energies_b[(n_up,n_dn)])
                    print('.')

        #E_b = EDFullSpectrum(params)[-1]
        #print(f'L={L} and GS energies are {E_f} vs {E_b}')
########
#temp functions
def test_G(C0,C1,beta,save_fig = None,title = None):
    '''
    checks paritties of greens functions
    '''
    L = C0.shape[0]
    Ntau = C0.shape[-1]
    C0_r = {}
    C1_r = {}
    for r1 in range(L):
        for r2 in range(L):
            r = (r1 - r2) % L
            if r == 0:
                print('r1,r2',r1,r2,C1[r1,r2,0])  
                if r not in C0_r:
                    C0_r[r] = []
                if r not in C1_r:
                    C1_r[r] = []
                C0_r[r].append(C0[r1, r2, :])
                C1_r[r].append(C1[r1, r2, :])
    if save_fig is not None:
        for r in sorted(C0_r):
            for c_tau in C0_r[r]:
                plt.plot((beta*1./Ntau)*np.arange(Ntau), c_tau,alpha=0.5,label=f'$r={r}$')
            for c_tau in C1_r[r]:
                plt.plot((beta*1./Ntau)*np.arange(Ntau), c_tau,alpha=0.5,label=f'$r={r}$',linestyle='--')
        plt.xlabel("$\\tau$")
        #plt.title("$B(r=r_1-r_2;\\tau) = B(r_1,r_2;\\tau)$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_fig,dpi = 500)
    return C0_r,C1_r
#######################################
def generate_benchmark_data_new(params,mode):
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
    # E,N ns beta
    Beta_max = 10
    Betas = np.linspace(start=1,stop=Beta_max,num=10,endpoint=True)
    Es = []
    Ns = []
    print('GS:',lowestEnergy)
    for beta in Betas:
        Es.append(thermo.H_moments(beta=beta)[1])
        Ns.append(thermo.N_moments(beta=beta)[0])
    print('Es',Es)
    print('Ns',Ns)
    #####################
    #parity resolved green's function at tau = 0
    thermo = thermodynamics(params)
    n_tau = 11
    taus = np.linspace(0,Beta_max, num=n_tau)
    if mode == 'Green_p':
        logZ_0 = thermo.logZ(beta=Beta_max,parity = 0)
        logZ_1 = thermo.logZ(beta=Beta_max,parity = 1)
        G_0_up = thermo.correlator(op='green',beta = Beta_max,n_tau= 10,parity = 0,green_spin='up')
        G_0_dn = thermo.correlator(op='green',beta = Beta_max,n_tau= 10,parity = 0,green_spin='dn')
        G_1_up = thermo.correlator(op='green',beta = Beta_max,n_tau= 10,parity = 1,green_spin='up')
        G_1_dn = thermo.correlator(op='green',beta = Beta_max,n_tau= 10,parity = 1,green_spin='dn')
        if np.allclose(G_0_dn,G_0_up) and np.allclose(G_1_dn,G_1_up):
            comp_0 = np.sum(np.abs(G_0_up.imag)) / np.sum(np.abs(G_0_up))
            comp_1 = np.sum(np.abs(G_1_up.imag)) / np.sum(np.abs(G_1_up))
            if (comp_0 <1e-8) and (comp_1<1e-8):
                G_0_up = G_0_up.real
                G_1_up = G_1_up.real
            else:
                raise ValueError
            G_0 = {}
            G_1 = {}
            #group by distance
            for i in range(params['L']):
                for j in range(params['L']):
                    r = (i-j)%params['L']
                    if r not in G_0.keys():
                        G_0[r] = []
                    if r not in G_1.keys():
                        G_1[r] = []
                    G_0[r].append(G_0_up[i,j,0])
                    G_1[r].append(G_1_up[i,j,0])
            #reduce to translationally invariant....
            #we dont actually have translational invariance soooooo
        else:
            print('GREEN FUNCTIONS NOT SPIN INDPT?')
            raise ValueError
        Dat = {'G0':G_0,
               'G1':G_1,
               'logZ0':logZ_0,
               'logZ1':logZ_1}
    ########################
    #averaged  correlators for green's function, spin-spin and eta
    else:
        if mode == 'Green':
            Green = thermo.correlator(op='green',beta = Beta_max,n_tau= n_tau,parity = None,green_spin='up')
            Dat = Green
        elif mode == 'Spin':
            Spin = thermo.correlator(op='spinspin',beta = Beta_max,n_tau= n_tau,parity = None)
            Dat = Spin
        elif mode == 'Eta':
            Eta =  thermo.correlator(op='eta',beta = Beta_max,n_tau= n_tau,parity = None)
            Dat = Eta
        else:
            raise ValueError
        #keep real part
        comp = np.sum(np.abs(Dat.imag)) / np.sum(np.abs(Dat))
        if (comp <1e-8):
            Dat = Dat.real
        else:
            raise ValueError
        #average
        Dat = observable_collapse(C=Dat,beta = Beta_max,mode = 'average')
    #save as pickle
    with open(save_dir+'/'+mode+'data.pkl','wb') as f:
        pickle.dump(Dat,f)
    #save params
    params_save = params['H_params'].copy()
    params_save['beta'] = Beta_max
    params_save['taus'] = taus
    with open(save_dir+'/params.pkl','wb') as f:
        pickle.dump(params_save,f)
    return  

#######################################
if __name__ == "__main__":
    mode = str(sys.argv[1])
    L = 6
    t = 1;U = 4;V = 1.5;mu = 1
    params = {'L':L,'verbose':1,'species':'boson','H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    generate_benchmark_data_new(params=params,mode = mode)
    #generate_benchmark_data(params=params,out = 'results_new.h5')
    quit()
    #quit()
    #generate_benchmark_data()
    L = 6
    t = 1;U = 5*1;V = 1.1*1;mu = -0.61
    params = {'L':L,'verbose':0,'species':'boson','H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    #compare_spectra(params)
    energies,eigenstates,bases,bases_inv,lowestEnergy = EDFullSpectrum(params)
    params['energies'] = energies
    params['eigenstates'] = eigenstates
    params['bases'] = bases
    params['bases_inv'] = bases_inv
    params['lowestEnergy'] = lowestEnergy
    params['JWstring'] = True
    thermo = thermodynamics(params)
    beta = 3
    #C = thermo.correlator(op='green0',beta = beta,n_tau= 1,parity = None)
    #C2 = thermo.correlator(op='green',beta = beta,n_tau= 10,parity = None,green_spin='up')
    #print(C.real)
    #print(C2[:,:,0].real)
    #quit()
    #Corr = thermo.correlator(op='spinspin',beta = beta,n_tau= 50,parity = None,green_spin='up')
    Corr0 = thermo.correlator(op='eta',beta = beta,n_tau= 50,parity = None,green_spin='up')
    Corr1 = thermo.correlator(op='spinspin',beta = beta,n_tau= 50,parity = None,green_spin='up')
    #Corr0 = Corr0.real
    #Corr1 =Corr1.real
    #comp = np.sum(np.abs(Corr.imag)) / np.sum(np.abs(Corr))
    #print(f'how complex is the operator? {comp}%')
    #if comp<1e-8:
    #    Corr = Corr.real
        #Corr = np.abs(Corr)
    #observable_collapse(C = Corr.real,beta = beta,mode = 'check',save_fig='/mnt/users/kotssvasiliou/ED/core_scripts/figs/',title='green.svg')
    #check_TI(C=Corr,savefig=True)
    #quit()
    #print()
    C0,C1 = test_G(Corr0,Corr1,beta = 5,save_fig='figs/green_parity.png')
    #print(C0[0][0][-1:])
    #print(C0[0][1][-1:])
    #print(C0[0][2][-1:])
    #print(C0[0][3][-1:])
    #print(C1[0].shape)

