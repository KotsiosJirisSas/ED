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
        self.mu = self.H_params['mu']
        self.L = params['L']
        self.loc = params['loc']
        self.sign = params['sign']# boolean T/F
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
        print('Hilbert Space size:',self.dim)
        # add option for sparse construction
        self.Hamiltonian = np.zeros((self.dim,self.dim),dtype=float)
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
                    self.Hamiltonian[m_tot,m_tot] += -self.mu*occ + 0.5*U*(occ-1.)**2
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
        Calculates occupancy of site i for a two species system on chain of length L
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
        self.params = params
        self.L = params['L']
        self.beta = params['beta']
        self.EDFullSpectrum()
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
        """
        if (I_s >> j) & 1:  # If site j is already occupied, return None
            return None, None
        I_s2 = I_s | (1 << j)
        # Compute sign factor (count fermions to the left)
        #forget about sign for now
        sign = (-1) ** ((bin(I_s & ((1 << j) - 1)).count('1')) + (bin(I_rest).count('1') if I_rest is not None else 0))
        return I_s2, sign
    def create_mapping(self):
        """
        Construct creation operator mapping using separate spin-up and spin-down bases.
        """
        mapping_up = {}
        mapping_dn = {}
        for sector in self.bases:
            sec_basis_up,sec_basis_dn = self.bases[sector]
            print('sector',sector)
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
    
    def compute_matrix_elements(self, mapping):
        """
        Compute \\langle n | c^\dagger | m \\rangle in the eigenbasis.
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
    def GreenFunc(self, beta,n_tau):
        """
        Returns the Green's function for the system. Has size L x L x s x s x Ntau --> L x L x s x Ntau
        """
        self.matrix_elements_up = self.compute_matrix_elements(mapping=self.mapping_up)
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
    @staticmethod
    def binp(num, length):
        '''
        print a binary number without python 0b and appropriate number of zeros
        regular bin(x) returns '0bbinp(x)' and the 0 and b can fuck up other stuff
        '''
        return format(num, '#0{}b'.format(length + 2))[2:]
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
