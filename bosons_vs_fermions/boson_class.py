'''
Code to treat a simple chain system
Options:
1)PBC hardcore bosons
2)PBC/APBC fermions
Goal is to match appropriate spectra
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import time

class chain_system:
    def __init__(self, params):
        self.L = params['L']
        self.t = params['t']
        self.species = params['species']   # 'bosons' or 'fermions'
        self.bcs = params['bcs']          # 'pbc' or 'apbc' or 'mixed'
        self.verbose = params['verbose']
        
        if self.species not in ['bosons','fermions']:
            raise ValueError("species must be 'bosons' or 'fermions'")
        if self.bcs not in ['pbc','apbc','mixed']:
            raise ValueError("bcs must be 'pbc' or 'apbc'")
        if self.species == 'bosons' and self.bcs == 'apbc':
            raise ValueError
        if self.L ==2:
            raise ValueError
        if self.L > 10 and self.verbose > 0:
            print('system size probably too large....')
        
        # For now, we still only split by parity of particle number,
        # exactly as before.
        self.even_states, self.even_index, self.odd_states, self.odd_index = self.create_basis()

    def create_basis(self):
        """
        Enumerate all 2^L basis states (each state is an integer 0..2^L-1).
        Count how many bits are set in 's'. If even, goes to even_states; else odd_states.
        """
        all_states = range(2**self.L)
        even_states, odd_states = [], []
        for s in all_states:
            if bin(s).count('1') % 2 == 0:
                even_states.append(s)
            else:
                odd_states.append(s)

        # Indices within each parity block
        even_index = {s: i for i, s in enumerate(even_states)}
        odd_index  = {s: i for i, s in enumerate(odd_states)}
        return even_states, even_index, odd_states, odd_index

    #
    def hop_operator_bosons(self,s, r):
        """
        Attempt to move one boson from site r+1 to site r (mod L).
        Return (new_state, sign_factor).
        If invalid hop, return (None, 0).
        For HCB with PBC, sign_factor = 1 always.
        """
        rplus = (r + 1) % self.L
        occ_rplus = (s >> rplus) & 1
        occ_r     = (s >> r) & 1
        if occ_rplus == 1 and occ_r == 0:
            new_s = s ^ (1 << r) ^ (1 << rplus)
            return (new_s, 1.0)  # no extra sign
        else:
            return (None, 0.0)
    def hop_operator_fermions(self,s,r,parity=0):
        """
        Attempt to move one fermion from site r+1 to site r (mod L).
        Return (new_state, sign_factor).
        If invalid hop, return (None, 0).
        For spinless fermions, we incorporate any boundary phase for APBC.
        """
        rplus = (r + 1) % self.L
        occ_rplus = (s >> rplus) & 1
        occ_r     = (s >> r) & 1
        if occ_rplus == 1 and occ_r == 0:
            new_s = s ^ (1 << r) ^ (1 << rplus)
            # boundary hop?
            if r == self.L - 1:
                # boundary link from site L-1 to site 0
                anticommutation_sign = (-1)**self.count_fermions_in_open_interval(s,start=rplus,end=r)
                #if anticommutation_sign==-1:print('ruh roh')
                if self.bcs == 'apbc':
                    sign = -1.0*anticommutation_sign
                    return (new_s,sign)
                elif (self.bcs == 'mixed') and (parity==0):
                    sign = -1.0*anticommutation_sign
                    return (new_s,sign)
                else:
                    sign = anticommutation_sign
                    return (new_s, sign)
            else:
                sign = 1.0
                return (new_s, 1.0)
        else:
            return (None, 0.0)

    def create_Ham(self):
        """
        Build the Hamiltonian blocks (even and odd) for either
        - hardcore bosons (PBC)
        - spinless fermions (PBC/APBC)

        Returns eigenvalues/eigenvectors for the even/odd blocks.
        """
        # --- Initialize blocks ---
        Heven = np.zeros((len(self.even_states), len(self.even_states)), dtype=np.complex128)
        Hodd  = np.zeros((len(self.odd_states),  len(self.odd_states)),  dtype=np.complex128)

        # --- Define the appropriate "hop_operator" for each species ---
        # For bosons: no sign factors for boundary (since we only coded 'pbc' for bosons).
        # Decide on the correct hopping function:
        def hop_op_wrapper(state, site, parity):
            """
            Decide which function to call based on species,
            and pass 'parity' if it's fermions.
            """
            if self.species == 'bosons':
                # For bosons, we ignore parity
                return self.hop_operator_bosons(state, site)
            else:
                # Fermions: pass the parity argument for boundary conditions, etc.
                return self.hop_operator_fermions(state, site, parity=parity)
        ########
        for (block_states, block_index, H, p) in [
                (self.even_states, self.even_index, Heven, 0),
                (self.odd_states,  self.odd_index,  Hodd, 1),
            ]:
                for s in block_states:
                    i = block_index[s]
                    for r in range(self.L):
                        s_new, sign_factor = hop_op_wrapper(s, r, p)
                        if s_new is not None:
                            # Check if s_new is in the same block
                            if s_new in block_index:
                                j = block_index[s_new]
                                # matrix element = -t * sign_factor
                                melem = -self.t * sign_factor

                                if (self.verbose > 1) and (abs(sign_factor - (-1)) < 1e-9):
                                    print('negative sign', self.species, self.bcs, i, j)

                                # fill in H[j, i] and H[i, j] for the Hermitian part
                                H[j, i] += melem
                                H[i, j] += melem.conjugate()
        #########
        # --- Fill the parity-block Hamiltonians ---
        # We'll do the same pattern for even and odd subspaces:
        '''
        for block_states, block_index, H in [
            (self.even_states, self.even_index, Heven),
            (self.odd_states,  self.odd_index,  Hodd),
        ]:
            for s in block_states:
                i = block_index[s]
                for r in range(self.L):
                    s_new, sign_factor = hop_op(s, r) #!!!!!!!!!!!
                    if s_new is not None:
                        # Check if new_s is in the same block
                        if s_new in block_index:
                            j = block_index[s_new]
                            # matrix element = -t * sign_factor
                            melem = -self.t * sign_factor
                            if (self.verbose>1) and (sign_factor==-1):
                                print('negative sign',self.species,self.bcs,i,j)
                            # fill in H[j, i] and the h.c.
                            H[j, i] += melem
                            H[i, j] += melem.conjugate()
        '''
        # --- Store / diagonalize ---
        self.H = {'0': Heven, '1': Hodd} 
        evals_even, evecs_even = np.linalg.eigh(Heven)
        evals_odd,  evecs_odd  = np.linalg.eigh(Hodd)
        #############
        #shift the spectra so that the GS energy is 0. The unshifted GS is saved.
        #note that depending on the system, we dont necessarily care about both sectors
        #so eg for PBC fermions, we only care about odd sector.
        if (self.species == 'bosons'):
            self.e_gs = np.min(np.array([np.min(evals_even),np.min(evals_odd)]))
        elif (self.species == 'fermions') and (self.bcs=='pbc'):
            self.e_gs = np.min(evals_odd)
        elif (self.species == 'fermions') and (self.bcs=='apbc'):
            self.e_gs = np.min(evals_even)
        elif (self.species == 'fermions') and (self.bcs=='mixed'):
            self.e_gs = np.min(np.array([np.min(evals_even),np.min(evals_odd)]))
        if self.verbose>0:print('\n GROUND STATE ENERGY: \n',self.e_gs)
        self.evals_even = evals_even
        self.evals_odd = evals_odd
        self.evecs_even = evecs_even
        self.evecs_odd = evecs_odd
        return 

    def partition_function(self,β,parity,true_Z=False):
        '''
        Function that calculates the parition function in a given parity sector
        It always first shifts the energies in that sector so they are all >=0.
        Z_p = \sum_{|n> \in P} exp{-β E_n}

        Note: If one needs the 'true' Z, maybe when comparing between even-odd sectors,
              then one needs to add the shift back in, via the true_Z term
              however, the Zs will be very large in this case, so tread carefully
        --------------------------------------
        Input:
            β(float)        :The inverse temperature
            parity(0,1)     :The parity sector we are considering
            true_Z(Bool)    :Involves an additional shift to give 'true' Z
        Output:
            Z(int)       :The total weight of this sector at temp 1/β 
        '''
        if not hasattr(self,'H'): raise ValueError
        #first shift energies 
        e_shift = np.min(np.array([np.min(self.evals_even),np.min(self.evals_odd)]))
        if parity == 0:
            Es = self.evals_even.copy() - e_shift
        elif parity == 1:
            #e_shift = np.min(self.evals_odd)
            Es = self.evals_odd.copy() - e_shift
        elif parity == None:
            Es = np.concatenate([self.evals_odd.copy() - e_shift,self.evals_even.copy() - e_shift])
        log_Z_P = logsumexp(-β* Es)
        if true_Z == True:
            raise NotImplementedError
            return np.exp(log_Z_P)*np.exp(-β*e_shift)
        return log_Z_P

    def operator_mapping(self,op=None):
        '''
        given an operator from selected modes 'op', calculates its matrix elements
        in the occupation number basis, <j|O_r|i> as a dictionary. Since for each |i> there is asingle |j> ,
        the data is stored as tuples of basis indices instead of a full matrix
        ------------------------------------
        Note: mask = (1<<loc)-1 isolates first loc bits to count the 1's
        to count the 'last' (most significant bits) bits, do  ((1<<L)-1)^ ((1<<loc)-1)
        ------------------------------------
        Input:
            op(str)             :Available operators:
                                    1)'b'   :creation operator for bosons
                                    2)'f'   :creation operator for fermions (includes the phase due to anticommutation in definition of FOck states)
                                    3)'JWb' :creation operator for bosons + the JW string
                                    4)'JWf' :creation operator for fermions + the (inverse) JW string
        Output:
            mapping(dict)       :A dictionary with keys 'r', the location of the operator on the chain, and values tuples of the form (i,j,phase),
                                 for elements of the operator <j|O_r|i> = phase
        '''
        # 1) Define all different operators
        def b_dagger(state_in,loc):
            if (state_in >> loc) & 1:
                return None,None
            state_out = state_in | (1<<loc)
            phase = 1
            return state_out,phase
        def f_dagger(state_in,loc):
            if (state_in >> loc) & 1:
                return None,None
            state_out = state_in | (1<<loc)
            mask = (1<<loc)-1
            phase = (-1)**(bin(state_out&mask).count('1'))#count 1's to the left of site r
            return state_out,phase
        # 2) assign operator depending on mode
        if op == 'b':
            apply_op = b_dagger
        elif op == 'f':
            apply_op = f_dagger
        else:
            raise NotImplementedError(f"Operator '{op}' not recognized yet.")
        # 3) Initialize the dictionaries
        dim_odd = len(self.odd_states)
        dim_even = len(self.even_states)
        mapping = {
                'e->o': {r: np.zeros((dim_odd,dim_even),dtype=complex) for r in range(self.L)},
                'o->e': {r: np.zeros((dim_even,dim_odd),dtype=complex) for r in range(self.L)},
                    }
        # 4) Create mapping
        for r in range(self.L):
            for i,I in enumerate(self.even_states):
                J,phase = apply_op(state_in=I,loc=r)
                if J != None: 
                    j = self.odd_index[J]
                    mapping['e->o'][r][j,i] += phase
            for i,I in enumerate(self.odd_states):
                J,phase = apply_op(state_in=I,loc=r)
                if J != None: 
                    j = self.even_index[J]
                    mapping['o->e'][r][j,i] += phase
        return mapping
    def matrix_elements_from_mapping(self,mapping):
        '''
        Given a mapping \hat{O}:|i> ---->(|j>,phase)
        for an operator, calculate the matrix elements of the operator
        in the eigenbasis:
        <m|O|n>
        --------------------------------------------------
        Input:
            mapping(dict)           :A mapping with keys 'r', the location the operator is acting in, and values the pairs of states it relates (and the added phase)
        Output:
            matrix_el(dict)         :A mapping with keys 'r' for the location of the operator, and values the matrix in the eigenbasis
        '''
        # 1)Initialize structure of matrix elements 
        dim_odd = len(self.odd_states)
        dim_even = len(self.even_states)
        matrix_el = {
                'e->o': {r: np.zeros((dim_odd,dim_even),dtype=complex) for r in range(self.L)},
                'o->e': {r: np.zeros((dim_even,dim_odd),dtype=complex) for r in range(self.L)},
                    }
        # 2) Change basis
        v_o = self.evecs_odd
        v_e = self.evecs_even
        for r in range(self.L):
            mapping_e2o = mapping['e->o'][r]
            mapping_o2e = mapping['o->e'][r]
            matrix_el['e->o'][r] = np.einsum('jm,ji,in',v_o.conj(),mapping_e2o,v_e)
            matrix_el['o->e'][r] = np.einsum('jm,ji,in',v_e.conj(),mapping_o2e,v_o)
        return matrix_el
    
    def corr_func(self,β,n_taus,op,parity):
        '''
        Calculates C_p(r,r',τ) =-1/Z_p x Tr_p{exp(-(β-τ)H_{1-p}) x O_r x exp(-τH_p) x O†_r'}
        for an operator O that maps between two parity sectors
        -----------------------------------------------------
        Inputs:
            β(float)            :The inverse temperature
            n_tau(int)          :The imaginary times at which to evaluate the Green's function
            op(str)             :The operator O we are evaluating the correlations of
            parity(0/1)         :The parity sector we are evaluating the correlations in. 
        '''
        #1)get matrix elements and intiialize
        taus = np.linspace(0, β, num=n_taus)
        C = np.zeros((self.L, self.L, n_taus), dtype=np.complex128) #correlator
        timei = time.time()
        op_map = self.operator_mapping(op=op)
        Mat = self.matrix_elements_from_mapping(mapping=op_map)
        timef = time.time()
        if self.verbose>0:print(' \n TIME TO CALCULATE THE MATRIX: \n',timef-timei,' seconds')
        #2) Get partition function
        log_Z_p = self.partition_function(β=β,parity=parity,true_Z = False)
        #3) isolate appropriate quantities
        if parity == 0:
            Ei = self.evals_even.copy()
            Vi = self.evecs_even.copy()
            Ef = self.evals_odd.copy()
            Vf = self.evecs_odd.copy()
            M = Mat['e->o']
        elif parity == 1:
            Ei = self.evals_odd.copy()
            Vi = self.evecs_odd.copy()
            Ef = self.evals_even.copy()
            Vf = self.evecs_even.copy()
            M = Mat['o->e']
        #4) shift energies
        e_shift = np.min(np.array([np.min(Ei),np.min(Ef)]))
        Ei -= e_shift
        Ef -= e_shift
        #5) calculate correlation
        for r in range(self.L): # the location of operator O^\dagger
            for k in range(self.L): # the location of operator O
                for m,vm in enumerate(Vi):
                    for n,vn in enumerate(Vf):
                        Em = Ei[m]
                        En = Ef[n]
                        log_terms = -β * Em - taus * (En - Em)
                        O_O_dagger_term = M[r][n,m] * M[k][n,m].conj()
                        C[k,r,:] += O_O_dagger_term * np.exp(log_terms-log_Z_p)
        return  -C # account for sign in definition

    def spectral_function(self,beta,omega,broadening=1e-2,sum_rule_test=False,kspace=False):
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
        if self.species == 'fermions':op = 'f'
        else:op = 'b'
        op_map = self.operator_mapping(op=op)
        Mat = self.matrix_elements_from_mapping(mapping=op_map)
        #2) Get partition function
        log_Z = self.partition_function(β=beta,parity=None,true_Z = False)
        #4) shift energies
        e_shift = np.min(np.array([np.min(self.evals_even.copy()),np.min(self.evals_odd.copy())]))
        E_0 = self.evals_even.copy() - e_shift
        E_1 = self.evals_odd.copy() - e_shift
        V_0 = self.evecs_even.copy()
        V_1 = self.evecs_odd.copy()
        #5) calculate spectral function
        for i,freq in enumerate(omegas):
            for r in range(self.L): # the location of operator O^\dagger
                for k in range(self.L): # the location of operator O
                    #parity sector 0
                    M = Mat['e->o']
                    for m,vm in enumerate(V_0):
                        for n,vn in enumerate(V_1):
                            Em = E_0[m]
                            En = E_1[n]
                            weight = (np.exp(-beta*Em-log_Z)+np.exp(-beta*En-log_Z))
                            delta_func_term = f_eta(x = freq - (En - Em))
                            O_O_dagger_term = M[r][n,m] * M[k][n,m].conj()
                            A[k,r,i] += O_O_dagger_term * weight * delta_func_term
                    #parity sector 1
                    M = Mat['o->e']
                    for m,vm in enumerate(V_1):
                        for n,vn in enumerate(V_0):
                            Em = E_1[m]
                            En = E_0[n]
                            weight = (np.exp(-beta*Em-log_Z)+np.exp(-beta*En-log_Z))
                            delta_func_term = f_eta(x = freq - (En - Em))
                            O_O_dagger_term = M[r][n,m] * M[k][n,m].conj()
                            A[k,r,i] += O_O_dagger_term * weight * delta_func_term
        if kspace == True:
            ks = np.linspace(start=0,stop=self.L,endpoint=False)*(2*np.pi/self.L)
            rs = np.linspace(start=0,stop=self.L,endpoint=False)
            A_k = np.zeros_like(A)
            for x in rs:
                for y in rs:
                    for i in range(self.L):
                        for j in range(self.L):
                            A_k[i,j,:] += np.exp(1j*(ks[i]*x-ks[j]*y))*A[i,j,:]    
            return A_k                                 
            #raise NotImplementedError
        if sum_rule_test:
            print('TESTING SUM RULE')
            temp = np.einsum('ijk->ij',A)
            print('inetgrated spectral function: SHould be ~identity \n:',temp)
        return A
    def MB_T(self,block_diag=False):
        '''
        Creates the many-body translation operator's basis.
        --------------------------------
        Input:
            block_diag(Bool)        :If true, returns H as a block off-diagonal matrix in the parity sectors
        '''
        if block_diag == False:
            T = np.zeros((2**self.L,2**self.L),dtype=int)
            for I in range(2**self.L):
                bit_positions = [pos for pos in range(self.L) if ((I >> pos) & 1) == 1]
                #shift all bits to the left
                new_positions = [((pos - 1) % self.L) for pos in bit_positions]
                J = 0
                for pos in new_positions:
                    J |= (1 << pos) 
                #RULE FOR SIGN: IF PARITY=EVEN AND IT CROSSES BOUNDARY, ADD A MINUS
                sign = 1
                if self.bcs == 'mixed':
                    # even number of occupied sites?
                    if len(bit_positions) % 2 == 0:
                        # does any occupant cross from L-1 to 0?
                        if (self.L-1 in bit_positions):
                            sign = -1
                if self.L-1 in bit_positions:
                    fock_space_order_factor = (-1)**(len(bit_positions)-1)
                else:
                    fock_space_order_factor = 1
                sign *= fock_space_order_factor
                T[J,I] = sign
        else:
            T = {'o':np.zeros((len(self.odd_states),len(self.odd_states)),dtype=int),
                 'e':np.zeros((len(self.even_states),len(self.even_states)),dtype=int)
                 }
            for I in self.odd_states:
                bit_positions = [pos for pos in range(self.L) if ((I >> pos) & 1) == 1]
                new_positions = [((pos - 1) % self.L) for pos in bit_positions]
                J = 0
                for pos in new_positions:
                    J |= (1 << pos) 
                sign = 1
                if self.bcs == 'mixed':
                    if len(bit_positions) % 2 == 0:
                        if (self.L-1 in bit_positions):
                            sign = -1
                if self.L-1 in bit_positions:
                    fock_space_order_factor = (-1)**(len(bit_positions)-1)
                else:
                    fock_space_order_factor = 1
                sign *= fock_space_order_factor
                T['o'][self.odd_index[J],self.odd_index[I]] = sign
            for I in self.even_states:
                bit_positions = [pos for pos in range(self.L) if ((I >> pos) & 1) == 1]
                new_positions = [((pos - 1) % self.L) for pos in bit_positions]
                J = 0
                for pos in new_positions:
                    J |= (1 << pos) 
                sign = 1
                if self.bcs == 'mixed':
                    if len(bit_positions) % 2 == 0:
                        if (self.L-1 in bit_positions):
                            sign = -1
                if self.L-1 in bit_positions:
                    fock_space_order_factor = (-1)**(len(bit_positions)-1)
                else:
                    fock_space_order_factor = 1
                sign *= fock_space_order_factor
                T['e'][self.even_index[J],self.even_index[I]] = sign
        return T
    def MB_Tb(self,block_diag=False):
        '''
        Creates the many-body bosonic translation operator's basis.
        --------------------------------
        Input:
            block_diag(Bool)        :If true, returns H as a block off-diagonal matrix in the parity sectors
        '''
        if self.species == 'bosons':
            raise ValueError
        raise NotImplementedError
    ##################
    ## HELPER FUNCS ##
    ##################
    def count_fermions_in_open_interval(self, s, start, end):
        """
        Count how many sites are occupied in the *open interval* (start, end)
        on a ring of length L, moving forward in ascending order.

        Example: if L=4, start=3, end=1, then the open interval is sites [0] 
                 (since you wrap around: after 3 comes 0, then 1 is the end).
        We'll define the ring in ascending order: 0,1,2,...,L-1, then wrap to 0.

        Returns an integer number of occupied sites in that open arc.
        """
        L = self.L
        count = 0
        # We'll step from 'start+1' mod L until we hit 'end' mod L (excluded).
        i = (start + 1) % L
        while i != end:
            # Check if site i is occupied in state s
            if (s >> i) & 1:
                count += 1
            i = (i + 1) % L
        return count
    def binp(self,num):
        """
        Print a binary number with exactly L bits (no leading '0b').
        """
        return format(num, '0{}b'.format(self.L))
#----------------TESTING FUNCTIONS--------------
def compare_Hs():
    L=6
    chain_bos = chain_system({'L':L, 't':1., 'species':'bosons', 'bcs':'pbc', 'verbose':1})
    chain_ferm_PBC = chain_system({'L':L, 't':1., 'species':'fermions', 'bcs':'pbc', 'verbose':1})
    chain_ferm_APBC = chain_system({'L':L, 't':1., 'species':'fermions', 'bcs':'apbc', 'verbose':1})
        
    Es_Bosons_even, evecsE_b, Es_Bosons_odd, evecsO_b = chain_bos.create_Ham()
    temp, evecsE_b, Es_fermions_odd, evecsO_b = chain_ferm_PBC.create_Ham()
    Es_fermions_even, evecsE_b, evalsO_b, evecsO_b = chain_ferm_APBC.create_Ham()
    Hbos_even = chain_bos.H['0']
    Hbos_odd = chain_bos.H['1']
    Hferm_even = chain_ferm_APBC.H['0']
    Hferm_odd = chain_ferm_PBC.H['1']
    plt.imshow(np.real(Hferm_odd))
    plt.colorbar()
    plt.savefig('bos.png')
    plt.clf()
    plt.imshow(np.real(chain_ferm_PBC.H['0']))
    plt.colorbar()
    plt.savefig('ferm.png')
    return

def compare_spectra(Ls):
    '''
    Runs ED on small systems and compares:
    1)PBC Bosons (odd sectors) Vs PBC Fermions (odd sectors)
    2)PBC Bosons (even sectors) Vs APBC Fermions (even sectors)
    The fact that these have equal spectra is due to JW string
    '''
    for L in Ls:
        chain_bos = chain_system({'L':L, 't':1., 'species':'bosons', 'bcs':'pbc', 'verbose':1})
        chain_ferm_PBC = chain_system({'L':L, 't':1., 'species':'fermions', 'bcs':'pbc', 'verbose':1})
        chain_ferm_APBC = chain_system({'L':L, 't':1., 'species':'fermions', 'bcs':'apbc', 'verbose':1})
        chain_ferm_mixed = chain_system({'L':L, 't':1., 'species':'fermions', 'bcs':'mixed', 'verbose':1})

        chain_bos.create_Ham()
        chain_ferm_PBC.create_Ham()
        chain_ferm_APBC.create_Ham()
        chain_ferm_mixed.create_Ham()
        Hbos_even = chain_bos.H['0']
        Hbos_odd = chain_bos.H['1']
        Hferm_even = chain_ferm_APBC.H['0']
        Hferm_odd = chain_ferm_PBC.H['1']
        #
        Hferm_even = chain_ferm_mixed.H['0']
        Hferm_odd = chain_ferm_mixed.H['1']
        #
        Es_Bosons_odd = chain_bos.evals_odd
        Es_Bosons_even = chain_bos.evals_even
        Es_fermions_even = chain_ferm_APBC.evals_even
        Es_fermions_odd = chain_ferm_PBC.evals_odd
        #
        Es_fermions_even = chain_ferm_mixed.evals_even
        Es_fermions_odd = chain_ferm_mixed.evals_odd
        #
        print('*'*100)
        print('L=',L)
        print('COMPARING ODD SECTORS: \n PBC BOSONS  VS PBC FERMIONS')
        print('IS HAMILTONIAN THE SAME?',np.allclose(Hbos_odd,Hferm_odd))
        print('IS SPECTRUM THE SAME?',np.allclose(Es_Bosons_odd,Es_fermions_odd))
        print('\n')
        print('COMPARING ODD SECTORS: \n PBC BOSONS  VS PBC FERMIONS')
        print('IS HAMILTONIAN THE SAME?',np.allclose(Hbos_even,Hferm_even))
        print('IS SPECTRUM THE SAME?',np.allclose(Es_Bosons_even,Es_fermions_even))
        print('\n \n')
    return
def fermion_greens_func(L):
    chain_ferm = chain_system({'L':L, 't':1., 'species':'fermions', 'bcs':'mixed', 'verbose':1})
    chain_ferm.create_Ham()
    C = chain_ferm.corr_func(β=10,n_taus=11,op='f',parity=1)
    #print(C[:,:,5])
    T = chain_ferm.MB_T(block_diag=False)
    print(T)
    T2 = chain_ferm.MB_T(block_diag=True)
    print('To \n',T2['o'])
    print('Te \n',T2['e'])
    return
def boson_greens_func(L):
    chain_bos = chain_system({'L':L, 't':1., 'species':'bosons', 'bcs':'pbc', 'verbose':1})
    #chain_bos = chain_system({'L':L, 't':1., 'species':'fermions', 'bcs':'pbc', 'verbose':1})
    chain_bos.create_Ham()
    C = chain_bos.corr_func(β=10,n_taus=11,op='f',parity=1)
    plt.figure()
    for r in range(L):
        r_p = (r+0)%L
        plt.plot(np.linspace(0,1,num=11), C[r, r_p, :], '-', label=f"bos G({r},{r_p})")
    plt.legend()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$G_{rr}(\tau)$')
    plt.tight_layout()
    plt.savefig('temp.png')
    print(C[:,:,5])
    return
def spectral_function(L):
    chain_ferm = chain_system({'L':L, 't':1., 'species':'fermions', 'bcs':'pbc', 'verbose':1})
    chain_ferm.create_Ham()
    A = chain_ferm.spectral_function(beta=5,omega=[-5,5,100],broadening=1e-2,sum_rule_test=True,kspace=True)
    A_diag = np.einsum('iik->ik',A).real
    plt.figure(figsize=(4,4))
    plt.imshow(A_diag.T,origin='lower',cmap='inferno')
    plt.xlabel('r')
    plt.ylabel('omega')
    plt.colorbar()
    plt.gca().set_aspect('auto')
    plt.savefig('temp.png')
# --------------- Usage example ----------------
if __name__ == "__main__":
    A = spectral_function(L=4)
    quit()
    #compare_Hs()
    #compare_spectra(Ls =[3,4,5,6,7,8,9,10])
    #quit()
    #chain_bos = chain_system({'L':6, 't':1., 'species':'bosons', 'bcs':'pbc', 'verbose':1})
    #chain_bos.create_Ham()
    #print(chain_bos.partition_function(β=10,parity=0,true_Z=False))
    fermion_greens_func(L=4)
    #boson_greens_func(L=4)
