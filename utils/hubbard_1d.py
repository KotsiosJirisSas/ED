'''
some skeleton code for running 1d Hubbard chain ED using only ( N_up, N_down) symmetry
maybe also reflection? Or maybe translation as sectors.
Key points:
1) Basis generation:
                    Index = Index_up + 2**L * Index_down
2) Hamiltonian generation and getting the spectrum:
                    Either full or Lanczos but in either case i store (sparesly) the entire Hamiltonian
3) Calculating thermal averages. Mainly intereated in Green's function
'''
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import gc
from scipy.special import logsumexp
import pickle

class hubbard_chain():
    '''
    '''
    def __init__(self,params):
        self.Nup = params['Nup']
        self.Ndn = params['Ndn']
        self.H_params = params['H_params']
        self.t = self.H_params['t']
        self.U = self.H_params['U']
        self.V = self.H_params['V']
        self.mu = self.H_params['mu']
        self.L = params['L']
        #self.loc = params['loc']
        self.sign = params['sign']# boolean T/F
        print('keys',params.keys())
        if 'JW string' in params.keys():
            self.JWstring  = params['JW string']
        else:
            self.JWstring = False
        print('OVERWRITTING... FORCING JWSTRING TO BE FALSE FOR HAMILTONIAN, BUT INCLUDE IN GREENS FUNC')
        self.JWstring = False
        if self.sign == True and self.JWstring == True:
            raise ValueError
        if self.JWstring == True:
            print('*'*100,'\n TREATING SYSTEM AS HARDCORE BOSONS W/ JW STRING \n ','*'*100)
        elif self.sign == True:
            print('*'*100,'\n TREATING SYSTEM AS PBC FERMIONS \n ','*'*100)
        else:
            print('*'*100,'\n TREATING SYSTEM AS PBC HARDCORE BOSONS \n ','*'*100)
        self.basis()#generates basis
        self.diag_params = params['diag_params']

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
        print('Hilbert Space size:',self.dim,'for (N_up,N_dn)=(',Nup,Ndn,')')
        # add option for sparse construction
        self.Hamiltonian = np.zeros((self.dim,self.dim),dtype=float)
        for state_up in index_up:
            for state_dn in index_dn:
                N_el = self.countBits(state_up)+self.countBits(state_dn)
                if N_el != (Nup+Ndn):
                    print('gucvfhkbdj')
        return 
    
    def hop_ij_up(self,i,j,m):
        '''
        adds to the hamiltonian the elements due to hopping between site i and j of the state with index m (between 1 and whatever; not 0!)
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
            if self.sign == True:
                sgn = self.fermion_sgn(self.binp(s1,length=self.L),self.binp(s2,length=self.L))
                if sgn == -1:
                    print('.')
            elif (self.JWstring == True) and (i==self.L-1):
                #parity term
                print('boundary hop for hc bosons')
                sgn = (-1)**(bin(s1).count('1'))
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
        adds to the hamiltonian the elements due to hopping between site i and j of the state with index m
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
            if self.sign == True:
                sgn = self.fermion_sgn(self.binp(s1,length=self.L),self.binp(s2,length=self.L))
            elif (self.JWstring == True) and (i==self.L-1):
                #parity term
                print('boundary hop for hc bosons')
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
        loop through all sites and all states and all spins and add hopping terms
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
        raise NotImplementedError
    
    def build_ham(self):
        ################################################
        #hopping terms
        if self.diag_params['mode'] == 'full':
            self.build_hopping_full()
        elif self.diag_params['mode'] == 'sparse':
            self.build_hopping_sparse()
        else:
            print('?')
            quit()
        #################################################
        # on site terms
        # can do this only by generating I_up and I_dn states individually, not both; this speeeds it up considerably. but idc.... if u do look at arXiv 1307.7542

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

                    #second way of adding interaction: equivalent up to chem pot + shift in energy.
                    # chem pot: mu = 0 vs mu = U/2 (half filling)
                    # shift:         0 vs L*U/2 
                    # this is used in hubbard_chain.py code
                    # if occ == 2:
                    #    self.Hamiltonian[m_tot,m_tot] += U
                    #self.Hamiltonian[m_tot,m_tot] -= self.mu*occ
        ########################
        if np.allclose(self.Hamiltonian,self.Hamiltonian.T) == False:
            print('not hermitian?????')
            quit()
        self.sparsity(self.Hamiltonian)
        return

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
    def hop(s,i,j):
        '''
        CHecks if hopping is allowed between sites i and j for state s and if it is,
        it outputs the resulting state

        Args:
            s(bin):         A binary number with L digits(L=length of chain) signifying the state of the chain
            i(int),j(int):  0 =<i,j<L Integers representing sites on the chain

        Returns:
            s2:             Either -1 to signify no allowed hopping or a binary to denote the resulting state after the hopping
        '''
        mask = 2**(i)+2**(j)
        K = s & mask #bitwise AND.
        P = K ^ mask #bitwise XOR.
        # L will have structure 0000000[i]0000[j]00000 and there's four cases:
        #1) L = mask means I1[i]=I1[j]=0 -> hopping is not allowed
        #2) L = 000..00 means I1[i]=I1[j]=1 -> hopping is not allowed
        #3&4) L = ...[1]...[0]... or L = ...[0]...[1]... means hopping is allowed, in which case new integer is 
        if P == mask or P == 0:
            s2 = -1#flag to signify no hopping
        else:
            s2 = s - K + P
        return s2

    @staticmethod
    def fermion_sgn(binary1,binary2):
        '''
        Modified from 'count_ones_between_flips' function in hubbard_chains.py
        --------------------
        Takes two binary strings that are meant to be related by a flip, ie they only differ in two sites, s1=xxx0xxx1xxx and s2=xxx1xxx0xxx.
        It then counts the number of 1's that separate these flipped sites, and outputs (-1)**count
        This accounts for the anticommutaative relations of the fermions
        
        Input:
            binary1(str):       A binary string representing a spinless fermion state on a chain
            binaryw(str):       A binary string representing a spinless fermion state on a chain
        Return:
            sgn(int):           The sign relating these two states
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
        Calculates occupancy of state psi at site i for a two species system on chain of length L
        '''
        mask = 2**(i)+2**(self.L+i)
        occ = self.countBits(psi & mask)
        return occ

    @staticmethod
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

class thermodynamics():
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
    def EDFullSpectrum(self):
        '''
        Does ED on all sectors and returns
        0) Info on lowest sector and energy
        1) Energies as a list of 1D arrays, one array per sector
        2) Eigenstates as a list of 2D arrays, one per sector
        3) An 'occupation' vector for each sector (that tells us the occupation number of each basis element in a sector)
        N_up goes from 0 to L
        N_down goes from 0 to L
        '''
        energies = {}
        eigenstates = {}
        bases = {}
        bases_inv = {}
        lowestEnergy = 1e10
        L = self.L
        timei = time.time()
        for n_up in range(0,L+1):
            for n_down in range(0,L+1):
                #print('=='*20)
                #print('Diagonalizing sector',n_up,n_down)
                #print('=='*20)
                self.params['Nup'] = n_up
                self.params['Ndn'] = n_down
                chain = hubbard_chain(self.params)
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
        print('time needed to diagonalize all sectors',timef-timei)
        print("The ground state occured in (n_up,n_down)=",GSSector,':',lowestEnergy)
        self.energies = energies
        self.eigenstates = eigenstates
        self.bases = bases
        self.bases_inv = bases_inv
        self.lowestEnergy = lowestEnergy
        ####################################
        #shift minmum#
        for sector in self.energies:
            self.energies[sector] -= lowestEnergy
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
if __name__ == "__main__":
    L = 6
    t = 1;U = 4;V = 0;mu = 0*U/2
    params = {'L':L,'sign':True,'loc':0,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    params['beta'] = 4.0
    thermo = thermodynamics(params)
    #print(thermo.Energy(beta=4))
    timei = time.time()
    thermo.create_mapping()
    timef = time.time()
    DATA = {}
    G = thermo.GreenFunc(beta=4,n_tau=100)
    timeff = time.time()
    print('mapping',timef-timei)
    print('green func',timeff-timef)
    DATA['greens'] = G
    Es = []
    timei = time.time()
    for beta in [1,2,3,4,5,6]:
        Es.append(thermo.Energy(beta=beta))
    DATA['es'] = Es
    timef = time.time()
    print('energy time',timef-timei)
    with open('data.pkl', 'wb') as handle:
        pickle.dump(DATA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    quit()



    #
    #map = thermo.mapping_dn
    #for key in map.keys():
    #    print('G_i,up',key[0],':   ',key[1:],'--->',map[key][:-1],'|   sign:',map[key][-1])
    total_size = 0
    for obj in gc.get_objects():
        try:
            size = sys.getsizeof(obj)
            total_size += size
            #print(obj,size)
        except TypeError:
            continue

    print(f"Total memory used by all objects: {total_size/(1024**2)} Mb")
    print('dictionaries(Mb):',sys.getsizeof(thermo.mapping_dn)/(1024**2)+sys.getsizeof(thermo.mapping_up)/(1024**2))
    print('eigenstuff(Mb)',sys.getsizeof(thermo.eigenstates)/(1024**2))
    print('matrix elements(Mb)',sys.getsizeof(thermo.matrix_elements_dn)/(1024**2)+sys.getsizeof(thermo.matrix_elements_up)/(1024**2))
    print('-'*100)
    #print('allowed transitions for i,spin=(0,up)')
    #print(thermo.allowed_transitions(0,'up'))
    #print('-'*100)
    #print('checking')
    #for sec in thermo.bases.keys():
    #    print('sector',sec,'states',thermo.bases[sec])
    #    print('~')
    #    print('energies',thermo.energies[sec])
    #    print('eigenvectors',thermo.eigenstates[sec])
    #    print('~')
    #    print('*'*50)
    #print('-'*100)
    #print('allowed transitions')
    #for i in range(L):
    #    for s in ['up','dn']:
    #        print('i,s=',i,s,':   ',thermo.allowed_transitions(i,s))
    #print('~'*100)
    #print('mappigs')
    #for key in thermo.mapping_up:
    #    print('mapping:',key,'--->',thermo.mapping_up[key])
    #for key in thermo.matrix_elements_up:
    #    for key2 in thermo.matrix_elements_up:
    #        if key[1:] == key2[1:] and key[0] != key2[0]:
    #            m1 = np.abs(thermo.matrix_elements_up[key])
    #            m2 = np.abs(thermo.matrix_elements_up[key2])
    #            if not np.allclose(m1,m2):
    #                print('not equal elements for',key,key2)
                    #print(m1)
                    #print(m2)
                    #quit()
    #for key in thermo.matrix_elements_up:
    #    sec_in = key[1]
    #    sec_out = key[2]
    #    sec_in = (sec_in[1],sec_in[0])
    #    sec_out = (sec_out[1],sec_out[0])
    #    m1 = np.abs(thermo.matrix_elements_up[key]) 
    #   key_flip = (key[0],sec_in,sec_out)
    #   m2 = np.abs(thermo.matrix_elements_dn[key_flip])
    #    if not np.allclose(m1,m2):
    #        print('not equal elements for',key,key_flip)


        #sec_in_flip  = 
    #print(thermo.matrix_elements_up[0,(0,1),(1,1)])
    #print(thermo.matrix_elements_up[1,(0,1),(1,1)])
    #print(thermo.matrix_elements_dn[0,(1,0),(1,1)])
    #print(thermo.matrix_elements_dn[1,(1,0),(1,1)])
    #print(thermo.partition_function(beta=10))
    #check translation invariance
    G_diag = np.abs(np.einsum('iiab->iab',np.real(G)))
    print(np.allclose(G_diag[0,0,:],G_diag[1,0,:],G_diag[2,0,:]))
    print(np.allclose(G_diag[0,1,:],G_diag[1,1,:],G_diag[2,1,:]))
    #print(G_diag[:,0,1])
    #print(G_diag[:,1,1])
    plt.plot(np.log10(G_diag[0,0,:]))
    plt.plot(np.log10(G_diag[1,0,:]))
    plt.plot(np.log10(G_diag[0,1,:]))
    #plt.plot(np.log10(G_diag[3,0,:]))
    #plt.plot(np.log10(G_diag[0,1,:]))
    plt.savefig('/mnt/users/kotssvasiliou/ED/figures/Gs.png')


    #plt.imshow(H,cmap='coolwarm')
    #plt.colorbar()
    #plt.savefig('hamiltonian.png')
