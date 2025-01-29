import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from scipy.sparse import csr_matrix,coo_matrix,kron,identity #optimizes H . v operations. to check if H already row sparse, do  isspmatrix_csr(H)
from scipy.sparse.linalg import eigsh
import time
import pickle
import argparse
from itertools import product,combinations
from math import comb,log10
from collections import Counter
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import random
from scipy.special import logsumexp
from scipy.sparse import SparseEfficiencyWarning
import warnings

warnings.simplefilter("ignore", SparseEfficiencyWarning) #supress warning???

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CODE THAT PERFORMS ED FOR AN LXL THREE-VALLEY HUBBARD SYSTEM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HOW A CALCULATION WORKS:

    THE FIRST PART OF THE CODE IS TO USE THE U(1) SYMMETRIES OF THE CHAINS TO GENERATE ALL NON-EQUIVALENT CHAIN CONFIGURATIONS.
        NON-EQUIVALENT MEANING YOU CANNOT REACH ONE FROM THE OTHER VIA A C3,TRANSLATION,TR TRANSFORMATION
    THEN, FOR A GIVEN CONFIGURATION, WHICH WILL HAVE A HILBERT SPACE OF AT MOST (L CHOOSE L/2)**(6L) WE CREATE THE HAMILTONIAN BY CREATING THE TENSOR PRODUCT OF THE HAMILTONIANS FOR THE INDIVIDUAL CHAINS
        EACH CHAIN (SPINLESS) HAMILTONIAN WILL HAVE A SMALL SIZE, AT MOST, (L CHOOSE L/2) SO 2X2 FOR L=2 AND 3X3 FOR L=3, SO ESSENTIALLY OUR FULL HAMILTONIAN IS A BLOCK HAMILTONIAN (ADD STATS ABOUT SPARSITY).
        THIS IS SORTED BY THE 'CHAINS' CLASS THAT IS INITIALIZED BY A SINGLE CONFIGURATION.
    THEN WHAT ONE NEEDS TO SAVE IS THE EIGENSTATES/EIGENENERGIES FOR EACH CONFIGURATION. FOR A THERMAL AVERAGE CALCULATION THEN ONE COMBINES THE DATA FROM ALL SECTORS
        THIS IS ORGANIZED, ALONG WITH SOME OBSERVABLES IN THE LAST PART OF THE CODE
----------------------------------------------------------------------------------------------------------------
Code that can generate all symmetry-inequivalent configurations of chain occupations.
These unique configurations can then be fed to a different function to create their Hamiltonian and diagonalize.
For 2x2 system, there are ~ 5*1e5 different configurations, but only ~ 2*1e4 unique ones.
For 3x3 system, there are ~ 7*1e10 different configurations, but ~ 1*1e9 unique ones.

The optimal reduction is by the order of the symmetry group which i think is 6L**2 (3 from C3, 2 from spin-inversion (time-reversal), L**2 from translations)


|   |   |   |               |   |   |   |               |   |   |   |       
|   |   |   |               |   |   |   |               |   |   |   | 
|   |   |   |               |   |   |   |               |   |   |   | 
|   |   |   |               |   |   |   |               |   |   |   |  x ....
|   |   |   |       x       |   |   |   |       x       |   |   |   | 
|   |   |   |               |   |   |   |               |   |   |   | 
|   |   |   |               |   |   |   |               |   |   |   | 
|   |   |   |               |   |   |   |               |   |   |   | 

   flavor 1                    flavor 2                      flavor 3

These are permutted by C3 and each one has its own action under T1,T2

For L1 != L2 things are tricky because how does the C3 action make sense? chains along one direction differ in amount and in how many electrons they fit depending on the direction (flavor)


A given sector is labelled by {n_i,s} with i the chain number and s the spin. we have 0=<n_{i,s} <= L (length of each chain)

This system has
1) T1,T2
2) C_3
3) Spin Inversion

So what i want to do is enumerate all possible combos and count the inequivalent configurations.
ie if g \in G does g( {n_i,s} ) =  {m_i,s} then in a sense  {n_i,s} ~ {m_i,s}.
Further, a state that you get multiple times after the action of g 's has to be weighed differently

We are not using translational symmetry the usual way because of the way we are build the basis.

The total number of sectors here is (1+L)**(6*L)
and i want to see the effective number of sectors, ie those sectors not related by any g \in G.
These should scale like ~ (1+L)**(6*L)/(6L**2)
------------------------------------------------------------------------------------------------
17 Jan Log:
   Everything is working great.
   Tested the action of the symmetries on arbitrary L x L systems.
   To do:
            1) Figure out how to consistencly define the chains for a L1 x L2 system with L1 != L2
            2) Figure out how to dynamically generate the configurations for systems larger than 2x2.
23 Jan Log:
   Added a sign function to take care of fermion sign, although that only applied to chains with more than two electrons and that allow hopping. So the smallest such case is for L=3 chain with 2 electrons.
   To do:   
            0) Add n.n. repulsion V
            1) Not important for now<------Figure out how to consistencly define the chains for a L1 x L2 system with L1 != L2
            2) For small temperatures, projecting Hilbert space to N_electrons = \\nu *(L**2) \pm 1 should be enough<--------Figure out how to dynamically generate the configurations for systems larger than 2x2.
'''
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
###########################################  Chain config class ###############################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
class chain_configs():
    '''
    Class that holds information about all possible chain configurations.
    An LxL lattice will have:
        a) 3L chains per spin for triangular lattice
        b) 2L chains per spin for rectangular lattice

    The class is initialized by 
        1) The size L
        2) If the configurations should be fully or partially generated; since for large L even listing out the configs takes up TBs of data...
        3) The lattice geometry. This also defines which symmetries to use

    The derived properties of this class are mainly:
        1) The non-equivalent configurations and their associated weights:
            ....
        2).... 

        
    A given configuration will have the form of a nested tuple

    c = ((c_up),(c_down))

    with c_spin = (c_1,c_2,.....,c_{{2,3}L})

    
    -----------------------------------------------------------------------------------------------------------------------------
    Some stats:
    #configs = (L+1)**(2L*{2,3}) depending on square or lattice

    #configs|L = 2|L = 3|L = 4|
    --------|-----|-----|-----|
    Triangle| 5e5 | 7e10| 6e16|
    Rectangl| 7e3 | 2e7 | 1e11|

    and w/ symmetries(T1,T2,C3/C2,TR), approximately, |G| = 2*{2,3}*L**2:

    #configs|L = 2|L = 3|L = 4|
    --------|-----|-----|-----|
    Triangle| 2e4 | 1e9 | 6e14|
    Rectangl| 4e2 | 5e5 | 2e9 |

    Also largest sectors:
    
    #Hilbert|L = 2|L = 3|L = 4|
    --------|-----|-----|-----|
    Triangle| 4e3 | 4e8 | 5e18|
    Rectangl| 3e2 | 5e5 | 5e11 |

    So conclusion:
    Can do full spectrum analysis for L=2 triangular and L=2,3 square.

    Can do Lanczos for L=3 triangular and L=4 square, but only on part of the HIlbert space, eg around N_el = L**2 \pm 1, so nearby filling 1.
    '''
    def __init__(self,params):
        self.geometry = params['geometry'] #'triangular' or 'square'
        self.L = params['L'] #L lattice size
        self.partial = params['partial'] #boolean
        self.projection = params['projection'] #boolean
        if 'n_el_max' in params: #if there's no projection, no need to define min and max number of electrons
            self.n_el_max = params['n_el_max']
        else:
            self.n_el_max = None
        if 'n_el_min' in params:
            self.n_el_min = params['n_el_min']
        else:
            self.n_el_min = None


        if self.geometry == 'triangular':
            self.Nchains = 3*self.L
            self.Gs = ['TR','T1','T2','C3'] #symmetries
            self.rotate = self.C3
        elif self.geometry == 'square':
            self.Nchains = 2*self.L
            self.Gs = ['TR','T1','T2','C2']
            self.rotate = self.C2
        self.Nconfigs = (self.L+1)**(2*self.Nchains)

        if self.partial == True and 'N_c' in params:
            self.N_c = params['N_c'] # how many configs to generate
        else:
            self.N_c = self.Nconfigs
        self.section_generator(n=self.N_c) # generates the configurations
        return
    
    #####
    def section_generator(self,n=0):
        """
        **PREVIOUSLY CALLED SECTION_GENERATION_PARTIAL**

        Generates the first n configurations for a given L, saves them temporarily to a file,
        and returns them as a list of tuples. The temporary file is deleted after use.
        This may be useful for systems larger than L=2 when number of configurations is larger than that allowed by RAM to hold it at the same time

        Args:
            L (int):        The system size (number of chains per valley and sites per chain).
            n (int):        The number of configurations to generate. If self.partial is off, n is set to the total combinatorial value of configurations

        Returns:
            list:           A list of configurations (tuples).

        --------------------------------------------------------------------------------------------------------------------------------------------------------
        TO DO:
            0)Check that this generates correctly all the configs for L=2 triangular.
            1)ALLOW ALL SYMMETRY RELATED CONFIGURATIONS TO BE GENERATED AND COUNTED AT THIS STAGE. THIS SHOULD SPEED UP THE CODE AND ALSO SIMPLIFY THE STORING PROCESS....
                This should involve the c2i function. Start w/ config c and enumerate all symmetry related ones. Then somehow add a 'mask' so that these are not generated later on.
            2)ALLOW A SMARTER WAY TO GENERATE STATES
        """
        # Temporary file to store configurations
        temp_file = "configurations_temp.txt"

        # Step 1: Generate configurations and save to file
        electron_count = range(self.L + 1)
        count = 0
        with open(temp_file, 'w') as f:
            for c_up in product(electron_count, repeat=self.Nchains):
                for c_down in product(electron_count, repeat=self.Nchains):
                    config = (c_up, c_down)
                    if self.projection == True:
                        if self.n_flag(config):  # Only save if it passes the test
                            f.write(str(config) + '\n')
                            count += 1
                    else:
                        f.write(str(config) + '\n')
                        count += 1
                    if count >= n:
                        break
                if count >= n:
                    break

        # Step 2: Read configurations back into a list of tuples
        configurations = []
        with open(temp_file, 'r') as f:
            for line in f:
                configurations.append(eval(line.strip()))  # Convert string back to tuple

        # Step 3: Delete the temporary file
        os.remove(temp_file)

        self.configurations = configurations
        return 

    def T(self,s,n,m):
        '''
        Takes a configuration s and acts on it with the translation operator
        T^m_2   T^n_1 
        Translates it by n sites along a_1 and m sites along a_2.
        Works the same for both geometries

        Args:
            s(tuple):               The configuration of chains
            n(int \in [0,L)):       Translation about a_1
            m(int \in [0,L)):       Translation about a_2
        Returns:
            s'(tuple):              The transformed configuration
        '''
        if self.geometry == 'triangular':
            s_up,s_down = s[0],s[1]
            s_up_1 = s_up[:self.L]
            s_up_2 = s_up[self.L:2*self.L]
            s_up_3 = s_up[2*self.L:3*self.L]
            s_down_1 = s_down[:self.L]
            s_down_2 = s_down[self.L:2*self.L]
            s_down_3 = s_down[2*self.L:3*self.L]
            #check len(s_up_1) == L

            s_up_1 = [s_up_1[i-m] for i in range(self.L)]
            s_up_2 = [s_up_2[i-n] for i in range(self.L)]
            s_up_3 = [s_up_3[i-(n+m)] for i in range(self.L)]
            s_down_1 = [s_down_1[i-m] for i in range(self.L)]
            s_down_2 = [s_down_2[i-n] for i in range(self.L)]
            s_down_3 = [s_down_3[i-(n+m)] for i in range(self.L)]
            s_up_t = tuple(s_up_1+s_up_2+s_up_3)
            s_down_t = tuple(s_down_1+s_down_2+s_down_3)
        if self.geometry == 'square':
            s_up,s_down = s[0],s[1]
            s_up_1 = s_up[:self.L]
            s_up_2 = s_up[self.L:2*self.L]
            s_down_1 = s_down[:self.L]
            s_down_2 = s_down[self.L:2*self.L]

            s_up_1 = [s_up_1[i-m] for i in range(self.L)]
            s_up_2 = [s_up_2[i-n] for i in range(self.L)]
            s_down_1 = [s_down_1[i-m] for i in range(self.L)]
            s_down_2 = [s_down_2[i-n] for i in range(self.L)]
            s_up_t = tuple(s_up_1+s_up_2)
            s_down_t = tuple(s_down_1+s_down_2)
        return (s_up_t,s_down_t)

    def C3(self,s):
        '''
        Implements C3 rotation which cycles through the chain directions while also permuting the order withing a chain type.
        Only works for self.geometry = Triangular

        (1,x)----->(2,x)
        (2,x)----->(3,L-x)
        (3,x)----->(1,L-x)

        Args:
            s(tuple):               The configuration of chains
            L(int):                 System size
        Returns:
            s'(tuple):              The transformed configuration
        '''
        L = self.L
        s_up,s_down = s[0],s[1]
        s_up_1 = s_up[:L]
        s_up_2 = s_up[L:2*L]
        s_up_3 = s_up[2*L:3*L]
        s_down_1 = s_down[:L]
        s_down_2 = s_down[L:2*L]
        s_down_3 = s_down[2*L:3*L]

        # Rotate the chains:
        # a_1 -> a_2 (unchanged order)
        # a_2 -> a_3 (reversed order)
        # a_3 -> a_1 (reversed order)
        #s_up_1_rot = s_up_3[::-1]

        s_up_1_rot = tuple([s_up_3[L-i-1] for i in range(L)]) #could do tuple(s_up[::-1]) instead... but for such small lists difference is marginal...
        s_up_2_rot = s_up_1
        s_up_3_rot = tuple([s_up_2[L-i-1] for i in range(L)])
        s_down_1_rot = tuple([s_down_3[L-i-1] for i in range(L)])
        s_down_2_rot = s_down_1
        s_down_3_rot = tuple([s_down_2[L-i-1] for i in range(L)])

        s_up_rot = tuple(s_up_1_rot+s_up_2_rot+s_up_3_rot)
        s_down_rot = tuple(s_down_1_rot+s_down_2_rot+s_down_3_rot)

        return (s_up_rot,s_down_rot)
    def C2(self,s):
        '''
        Implements C4 rotation which cycles through the chain directions while also permuting the order withing a chain type.
        Only works for self.geometry = Square

        Note this is a 90 degree rotation but doing it twice returns identity since chains are headless (no direction)

        (1,x)----->(2,x)
        (2,x)----->(1,L-x)
        
        The 'opposite' rule is also fine and is essentially doing a -90 degree rotation instead.

        Args:
            s(tuple):               The configuration of chains
            L(int):                 System size
        Returns:
            s'(tuple):              The transformed configuration
        '''
        L = self.L
        s_up,s_down = s[0],s[1]
        s_up_1 = s_up[:L]
        s_up_2 = s_up[L:2*L]
        s_down_1 = s_down[:L]
        s_down_2 = s_down[L:2*L]

        # Rotate the chains:
        # a_1 -> a_2 (unchanged order)
        # a_2 -> a_1 (reversed order)

        s_up_1_rot = tuple([s_up_2[L-i-1] for i in range(L)]) #could do tuple(s_up[::-1]) instead... but for such small lists difference is marginal...
        s_up_2_rot = s_up_1
    
        s_down_1_rot = tuple([s_down_2[L-i-1] for i in range(L)])
        s_down_2_rot = s_down_1

        s_up_rot = tuple(s_up_1_rot+s_up_2_rot)
        s_down_rot = tuple(s_down_1_rot+s_down_2_rot)

        return (s_up_rot,s_down_rot)

    def TR(self,s):
        '''
        **previoulsy called S_inv**
        exchanges the two tuples
        Equivalent to S_z -> - S_z
        '''
        return (s[1],s[0])
    def GroupAction(self,s,n,m,m_rot,m_spin):
        '''
        Acts on a configuration with a generic element from the symmetry group

        Args:
            s(tuple):                   Original configuration
            L(int):                     System size
            n(int \in [0,L)):           Translation about a_1
            m(int \in [0,L)):           Translation about a_2
            m_rot(int \in [0,1,(2)]):   C2 or C3 rotation
            m_spin(int \in [0,1]):      Spin inversion


        Returns:
            s'(tuple):              The transformed configuration
        '''
        L = self.L

        s_temp = s
        s_temp = self.T(s_temp,n,m)
        for x in range(m_rot):
            s_temp = self.rotate(s_temp)
        for y in range(m_spin):
            s_temp = self.TR(s_temp)
        return s_temp
    
    def c2i(self,config):
        """
        Finds the index of a given configuration.
        *add explanation for mapping*

        Args:
            config(tuple):          A configuration in the form [(up_spin), (down_spin)].
            L(int):                 Linear system size.
            Nchains(int):           The number of chains in the system

        Returns:
            total_index(int):       The index of the configuration.
        """
        L = self.L
        # Unpack the red and blue configurations
        up_config,down_config = config
        # Convert the tuple to a unique index
        base = L + 1
        up_index= sum(r * (base ** i) for i, r in enumerate(reversed(up_config)))
        down_index = sum(b * (base ** i) for i, b in enumerate(reversed(down_config)))
        # Combine the two indices
        total_index = up_index * (base ** (self.Nchains)) + down_index
        return total_index

    def configuration_reduction(self):
        '''
        Generates all the states, then goes throught them, applies all the group elements and bunches the configs together in equivalence classes: if \exists g \in G: g(c)=c' => c ~c'
        Then it calls the reweighing function that turns equivalence classes into two numbers; Their representative and their order(weight)
        Args:
            dict(dictionary):       Dictionary of the form (key,value):(x_1,x_1,x_2,x_3,.....) with number equal to the order of the group

        Returns:
            symm_configs(dict):     Dictionaty of the equivalence classes. Keys are the representative configurations and values their weights.
            Reduced_number(int):    Number of configurations after this reduction
        '''
        #only works now with full configuration creation
        if not hasattr(self,'configurations'):
            print('GENERATING CONFIGURATIONS')
            self.section_generator()
        L = self.L
        Equivalence_classes = {}
        mask = [True]*self.Nconfigs # checks if this state has been 'met' before
        for i,c in enumerate(self.configurations):
            if mask[i] == False:
                continue
            #Equivalence_classes[i] = []                                                                                                #changes 29 Jan; BUG? What should keys of new dict be....
            Equivalence_classes[c] = []
            #######
            if self.geometry == 'triangular':
                M_rot = 3
            else:
                M_rot = 2
            for n in range(2):
                for m in range(2):
                    for m_rot in range(M_rot):
                            for m_spin in range(2):
                                c_new = self.GroupAction(c,n,m,m_rot,m_spin)
                                i_new = self.c2i(c_new)
                                #Equivalence_classes[i].append(i_new)                                                                       #changes 29 Jan; BUG?
                                Equivalence_classes[c].append(i_new)
                                mask[i_new] = False
        self.symm_configs = self.reweight(Equivalence_classes)
        Reduced_number = len(self.symm_configs)
        print('Total Number of configurations:',self.Nconfigs)
        print('Reduced Number of configurations:',Reduced_number)
        print('Gain:',self.Nconfigs/Reduced_number)
        return Reduced_number

    @staticmethod
    def reweight(dict):
        '''
        Takes the symmetry-reduced configurations and 'reweights' them. Ie for eg [x_1]:(x_1,x_2,x_3) in the same equivalence class, it turns it into [x_1]:(3) 
        Meaning for calculations, x_1 cnofiguration should count 3 times as it really represents three states.

        Args:
            dict(dictionary):       Dictionary of the form (key,value):(x_1,x_1,x_2,x_3,.....) with number equal to the ordger of the group

        Returns:
            dict_out(dictionary):   Dictionary of the form (key,value):(x_1,w_1) with w_1 the weight corresponding to configuration x_1
        
        '''
        dict_out = {}
        for k in dict.keys():
            lst = dict[k]
            element_counts = Counter(lst)
            unique_elements = list(element_counts.keys())
            multiplicities = list(element_counts.values())
            if not multiplicities.count(multiplicities[0]) == len(multiplicities):
                #checks that all items in an equivalence class appear an equal amount of times: 1,2,3,4,6,8,12,24
                print('uhhhhhhhhhhhhhhhhh')
                quit()
            weight = len(unique_elements)
            #dict_out[k] = (unique_elements,weight) # this returns something that holds info of all equivalent states
            dict_out[k] = weight # this only holds info of the *representative* configuration through the dict key
        return dict_out

    def hilbertspacedim(self,c):
        '''
        ** previously calld fcombinatoric**
        Calculates the Hilbert space dimension of a configuration.
        '''
        c_up = c[0]
        c_down = c[1]
        dim = 1
        L = self.L
        for i in range(L):
            for flav in range(int(self.Nchains/self.L)):
                dim *= comb(L,c_up[i+int(flav*L)])*comb(L,c_down[i+int(flav*L)])
        return dim

    def n_flag(self,config):
        '''
        Tests if the configuration has a given number of electrons.
        If it does, the configuration passes through
        If not, it doesn't

        Input:
            config(nested tuple):           The configuration of electrons on chains
            (n_el_min,n_el_max)(int tuple): The allowed range of electrons in the system 
        Returns:
            flag(bool):                     Does config pass the test?
        '''
        n_el = sum(sum(inner) for inner in config)
        if n_el <= self.n_el_max and n_el >= self.n_el_min:
            flag = True
        else:
            flag = False
        return flag
 
    def Qnumber_printout(self,c):
        '''
        ** needs checking and potential debugging***
        ---------------
        Outputs the quantum numbers associated with a configuration

        Args:
            c(nested tuple):        The configuration

        Returns:
            Qs(tuple):              Nested Tuple of quantum numbers:(\\nu,Sz_tot,N_i,Sz_i) for i= 1....number of flavors
        
        '''
        up_c,down_c = c
        L = self.L
        Nfl = int(len(up_c)/L)
        N_ups = [sum(up_c[fl*L:L+fl*L]) for fl in range(Nfl)]
        N_downs = [sum(down_c[fl*L:L+fl*L]) for fl in range(Nfl)]
        
        N_up = sum(up_c)
        N_down = sum(down_c)

        nu = (N_up+N_down)/L**2
        Sz_tot = (N_up-N_down)/2
        N_fl = np.array(N_ups) + np.array(N_downs)
        Sz_fl = 0.5*(np.array(N_ups) - np.array(N_downs))
        return (nu,Sz_tot,N_fl,Sz_fl)

 #####################################   
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
##########################################  Single chain Hamiltonian ###############################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
class chains():
    '''
    An instance of this class represents a Hamiltonian associated with a particular configuration of chains.
    It contains the Hilbert space and Hamiltonian of individual chains, as well as the total hilbert space and Hamiltonian, plus its eigenstuff

    The underlying lattice is only inputed through:
        1) the self.loc property, mapping the chain numbers to sites on the lattice
        2) the self.geometry property. May depreciate it in the future.
        3) Implicitly throught the number of chans for a n LxL system.
    '''
    def __init__(self,params):
        self.config = params['config']
        self.H_params = params['H_params']
        self.L = params['L']
        self.loc = params['loc']
        self.sign = params['sign']# boolean T/F
        self.basis()#generates basis
        self.diag_params = params['diag_params']
        self.sites = {} #uhhhhh forgot what this does exactly....
        for i,v in enumerate(self.loc):
            if v not in self.sites.keys():
                self.sites[v] = 2**((6*self.L**2)-i-1)
            else:
                self.sites[v] += 2**((6*self.L**2)-i-1)
    def basis_old(self):
        '''
        Generates the basis in terms of binaries. An LxL system will have a basis with 3L*2*L=6L**2 sites. Each site is 0 or 1 so full hilbert space is ofcourse 2**(6L**2)=64**L**2
        The ordering of the basis is: chain_1_up,chain_1_down,chain_2_up,chain_2_down,......,chain_3L_up,chain_3L_down
        Args:
            c(nested tuple):        The chain configuration
        Returns:
            basis(list):            A list of all states in the hilbert space spanned by the configuration
            length_basis(int):      The size of the HIlbert space
            Hamiltonians(list):     List of 6L**2 npcarray Hamiltonians to be combined into a full hamiltonian
        '''
        c = self.config
        L = self.L
        basis = []
        Hamiltonians = []
        for chain in range(int(3*L)):
            chain_up = c[0][chain]
            chain_down = c[1][chain]
            basis_up = generate_partitions(L,chain_up)
            Hamiltonians.append(self.chain_Hamiltonian(basis_up))
            basis_down = self.generate_partitions(L,chain_down)
            Hamiltonians.append(self.chain_Hamiltonian(basis_down))
            if len(basis) == 0: 
                basis = basis_up
                basis = [old + new for old in basis for new in basis_down]
            else:
                basis = [old + new for old in basis for new in basis_up]
                basis = [old + new for old in basis for new in basis_down]
        self.basis = basis
        self.dim = len(basis)
        self.chain_hamiltonians = Hamiltonians
        return 
    def basis(self):
        '''
        Generates the basis in terms of binaries. An LxL system will have a basis with 3L*2*L=6L**2 sites. Each site is 0 or 1 so full hilbert space is ofcourse 2**(6L**2)=64**L**2
        
        Args:
            c(nested tuple):        The chain configuration
        Returns:
            basis(list):            A list of all states in the hilbert space spanned by the configuration
            length_basis(int):      The size of the HIlbert space
            Hamiltonians(list):     List of 6L**2 npcarray Hamiltonians to be combined into a full hamiltonian
        '''
        c = self.config
        L = self.L
        basis = []
        Hamiltonians = []
        for chain in range(int(3*L)):
            chain_up = c[0][chain]
            chain_down = c[1][chain]
            basis_up = self.generate_partitions(L,chain_up)
            Hamiltonians.append(self.chain_Hamiltonian(basis_up))
            basis_down = self.generate_partitions(L,chain_down)
            Hamiltonians.append(self.chain_Hamiltonian(basis_down))
            if len(basis) == 0: 
                basis = basis_up
                basis = [old + new for old in basis for new in basis_down]
            else:
                basis = [old + new for old in basis for new in basis_up]
                basis = [old + new for old in basis for new in basis_down]
        self.basis = basis
        self.dim = len(basis)
        self.chain_hamiltonians = Hamiltonians
        return 
    def chain_Hamiltonian(self,chain_basis):
        '''
        Generates the small non-interacting Hamiltonian for a single chain and spin. The dimension is read from the basis.
        The full Hilbert space for a single chain and spin is 2**L but in our case the Hilbert space will be L Choose N_spin. For L=2, dim =1 or 2 while for L=3, dim = 1 or 3 

        Parameters:
            Basis(list):        A list containing all states (represented by binaries) in the chain's Hilbert space
            t(float):           Hopping strength
        Returns:
            H(npc array):       A (densely constructed) Hamiltonian
        
        '''
        #step1) Build lookup table for all states(very small table). That can just be the basis_up/down list
        #step2) go through basis size, associate index with state and check hopping, mapping it back to a new state.
        #step3) done#
        dim = len(chain_basis)
        basis_dec = [int(el,2) for el in chain_basis]
        L_chain = self.L
        t = self.H_params['t']
        H = np.zeros((dim,dim),dtype=float)
        for m in range(dim):
            s = basis_dec[m]
            if L_chain == 2: L_max = 1
            else:
                L_max = L_chain
            #in L=2 case there is overcounting of processes since 0<-->1 and 1<--->0 . compare to eg L=3: 0<-->1,1<-->2,2<-->0 
            for i in range(L_max):
                j=(i+1)%L_chain
                s2 = self.hop(s,i,j)
                if s2 != -1:
                    try:
                        n = basis_dec.index(s2)
                    except ValueError:
                        print('Index not found...quitting!')
                        quit()
                    sgn = 1
                    if self.sign == True:
                        sgn = self.fermion_sgn(self.binp(s,length=L_chain),self.binp(s2,length=L_chain))
                        if sgn == -1 and L_chain == 2:
                            print('negative sign? Shouldnt happen for L=2')
                    H[n,m] -= t*sgn
        return H
    #@staticmethod
    def count_occupancies(self,s):
        '''
        s is the configuration as a binary string. it has size 6*L**2
        v has same length as s and holds the location of s 
        '''
        #s = int(s,2) #convert to integer from string
        #for each site (0 to L**2-1) create a mask based on v
        masks = self.sites
        state = int(s,2)
        occupancy = {}
        occupancy_tot = 0
        for i in masks.keys():
            print('--')
            print(self.binp(masks[i],length=6*self.L**2))
            print(self.binp(state,length=6*self.L**2))
            print(self.binp(masks[i]&state,length=6*self.L**2))
            print(self.countBits(masks[i] & state))
            print('--')
            occupancy[i] = self.countBits(masks[i] & state)
            occupancy_tot += occupancy[i]
        return occupancy,occupancy_tot
    def configuration_Hamiltonian(self):
        '''
        Returns the *full* Hamiltonian of the configuration
        Steps:
        1)Creates tensor product for hopping Hamiltonians sparsely
        2)Adds interactions and chemical potential (all are diagonal terms)
        '''
        U = self.H_params['U']
        V = self.H_params['V']
        mu = self.H_params['mu']
        num_chains = len(self.chain_hamiltonians)
        total_dim = np.prod([h.shape[0] for h in self.chain_hamiltonians])
        H = csr_matrix((total_dim, total_dim), dtype=np.float64)
        # Loop through each local Hamiltonian and embed it in the tensor product space
        for i, h_local in enumerate(self.chain_hamiltonians):
            h_local_sparse = csr_matrix(h_local)
            # Identity operators for spaces before and after the current subspace
            identity_before = identity(np.prod([self.chain_hamiltonians[j].shape[0] for j in range(i)]), format="csr") if i > 0 else 1
            identity_after = identity(np.prod([self.chain_hamiltonians[j].shape[0] for j in range(i + 1, num_chains)]), format="csr") if i < num_chains - 1 else 1
            # Embed the local Hamiltonian in the full tensor product space
            term = kron(kron(identity_before, h_local_sparse), identity_after, format="csr")
            H += term
        ######################################
        ##### interactions#######
        ##########################
        basis_dec = [int(el,2) for el in self.basis]
        for m in range(total_dim):
            s = basis_dec[m]
            occupations = []
            for site in range(1,self.L**2+1): # keyyyyyyyy need to go all the way from 1 to L**2 not L**2 -1!!!!! BUG!!! 
                occ_site = self.countBits(self.sites[site] & s)
                occupations.append(occ_site)
                H[m,m] += -mu*occ_site + U*occ_site**2 
            #n.n. electron repulsion
            H[m,m] += V*self.nn_repulsion(self.L,occupations)
        ###################
        diff = H - H.getH()
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        if max_diff != 0:
            print('Hamiltonian is not hermitian!!!!',max_diff)
        return H
    
    def diagonalization(self):
        '''
        Sorts out the diagonalization of this configuration. The output contains the full information necessary to calculate thermodynamic properties.

        Returns:
            diag_states(dict):      A dictionary with keys: 'params':           Holds minimal system information such as configuration and weight of the configuration
                                                            'eig_energies':     A 1xM array of dtype=float containing the eigenstates
                                                            'occupations':      A 1xM array of dtype=int containing the occupation number of the eigenstates
                                                            'eig_states':       A MxM array of dtype=complex containing the eigenstates. Its the main memory bottleneck by far
        '''
        #lanczos or full?
        H = self.configuration_Hamiltonian()

        if self.diag_params['mode'] == 'full':
            H_dense = H.toarray()
            e,v = np.linalg.eigh(H_dense)
            
        elif self.diag_params['mode'] == 'Lanczos':
            k = self.diag_params['k']
            dim = self.dim
            if 1<dim<10:
                k = min(dim-1,k)
            e,v = eigsh(H,k=k, which='SA', tol=1e-10)
        else:
            print('no mode added')
            quit()
        diag_states = {}
        #diag_states['configuration'] = self.config
        diag_states['energies'] = e
        diag_states['states'] = v
        return diag_states
    @staticmethod
    def generate_partitions(L, N):
        """
        Generate all possible partitions of N electrons in a chain with L sites as binary strings.

        Parameters:
            L (int): Number of sites.
            N (int): Number of electrons of certain spin.

        Returns:
            list: List of binary *strings* representing the partitions.
        """

        if N > L:
            raise ValueError("Number of electrons (N) cannot exceed number of sites (L).")

        # Generate all combinations of N positions from L sites
        partitions = []
        for positions in combinations(range(L), N):
            # Create a binary representation of the partition
            binary = ['0'] * L
            for pos in positions:
                binary[pos] = '1'
            partitions.append(''.join(binary))

        return partitions
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
    def nn_repulsion(length,occupations):
        '''
        calculates n.n. repulsion for L=2 where all sites are nn with all others
        '''
        if length != 2:
            print('n.n. interaction not implemented for L >2')
            return 0
            #quit()
        count = 0
        for i,occ1 in enumerate(occupations):
            for j,occ2 in enumerate(occupations):
                if i < j:
                    count += occ1*occ2
        return count
    @staticmethod
    def binp(num, length=4):
        '''
        print a binary number without python 0b and appropriate number of zeros
        regular bin(x) returns '0bbinp(x)' and the 0 and b can fuck up other stuff
        '''
        return format(num, '#0{}b'.format(length + 2))[2:]
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
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################### THERMODYNAMICS #############################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
class correlations():
    '''
    A class that takes as input the configurations and their weights from the chain_configs class instance and also the eigenbasis of a chains class instance.
    An instance of this class then calculates information about 
        1) Partition function
        2) <E>,<E^2>
        3) other non-diagonal operators like <N>,<N^2>
        4) To calculate other quantities, need to add some additional structure: 
            Other take more info in from the chains instance (like how does the eigenbasis act on an operator O)
            -or- devise a way to calculate that from within. this is more flexible.

    The instance is initialized with the configurations, the eigenbasis, and another occupation vector.
    For each of Nc configurations, there will be one M_c x 1 energy array, an M_c x 1 vector describing \hat{N}|n_c> for the occupation basis of config c, and an M_c x M_c array for the eigenbasis.
    '''
    def __init__(self,input_dict):
        '''
        input dict will have structure {'config':{'multiplicity:int,'energies,:1xM array,'basis':MxM array,'ns':1xM array}}

        This also shifts the spectrum to allow non-problematic handling of logs of energies

        This also 'changes the basis' for the occupation numbers. for each sector, ns is a 1xM array giving the expectation of the occupation operator in the original basis
        <j|N|i> =\delta_{ij} N_i.
        Here,  i change the basis to the eigenbasis {|a>}. The exp. values only care about the diagonal elements:
        <a|N|a> = \sum_{i}ns[i]|a_i|**2
        '''
        #calculate max and min energies
        Es_flat =  np.concatenate([sector_data['es'] for sector_data in input_dict.values()])
        self.Emin = np.min(Es_flat)
        self.Emax = np.max(Es_flat)
        print('Extremal energies:',self.Emin,self.Emax)
        self.data_dict = input_dict
        #shifts energies
        for sector_data in self.data_dict.values():
            sector_data['es'] -= self.Emin
        Es_flat = np.concatenate([sector_data['es'] for sector_data in self.data_dict.values()])
        print('after rescaling,Extremal energies:',np.min(Es_flat),np.max(Es_flat))
        #################
        # change basis on occupation operator expectations
        for sector_data in self.data_dict.values():
            ns = sector_data['ns']
            vs = sector_data['vs']
            sector_data['ns'] = np.sum(ns * np.abs(vs) ** 2, axis=0)  # Sum over basis states
            #sector_data['ns'] = np.einsum('ij,i,ij->j',vs,ns,vs.conj()) #are two methods equivalent?Should be. Check
        return
    def partition_function(self,beta):
        '''
        Computes the partition function using logsumexp for numerical stability
        Parameters:
        data_dict (dict):           Dictionary where keys label fundamental sectors, and each sector contains:
                                    - 'weight': Number of symmetry-equivalent sectors
                                    - 'es': Mx1 array of eigenvalues (spectrum)
                                    - 'vs': MxM array of eigenvectors

        beta (float):               Inverse temperature (1/kT).
        
        Returns:
        float:                      The partition function Z.
        '''
        log_terms = []
        for sector_data in self.data_dict.values():
            weight = sector_data['weight']
            energies = sector_data['es']
            log_terms.append(np.log(weight) + (-beta * energies))
        log_terms = np.concatenate(log_terms)
        log_Z = logsumexp(log_terms)
        return log_Z
    def H_moments(self,beta):
        '''
        Computes the <H> and <H^2> observables using logsumexp for numerical stability.
        Takes care to filter out E=0 (GS) as their log is ill defined and they shouldn't contribute to shifted expectation values
        ------------------------------------------------------------------------------
        Notes:
            1) Filters out the ground state which has energy zero by definition (after shifting)
            2) Just taking log(energies) wouldn't work for negative energies so 
        ------------------------------------------------------------------------------
        Parameters:
        data_dict (dict):           Dictionary where keys label fundamental sectors, and each sector contains:
                                    - 'weight': Number of symmetry-equivalent sectors
                                    - 'es': Mx1 array of eigenvalues (spectrum)
                                    - 'vs': MxM array of eigenvectors

        beta (float):               Inverse temperature (1/kT).
        
        Returns:
        (float, float, float, float): 
                                    - H_avg (shifted)
                                    - H_avg_unshifted (original energy scale)
                                    - H_sq_avg (shifted)
                                    - H_sq_avg_unshifted (original energy scale
        '''
        log_Z =  self.partition_function(beta)#changed pervious code so that it returns logsumexp() rather than its exponential
        log_terms_H = []
        log_terms_H_sq = []
        for sector_data in self.data_dict.values():
            weight = sector_data['weight']
            energies = sector_data['es']
            valid_mask = energies > 0
            valid_energies = energies[valid_mask]
            #
            if valid_energies.size > 0:
                log_terms_H.append(np.log(weight) + (-beta * valid_energies) + np.log(valid_energies))  
                log_terms_H_sq.append(np.log(weight) + (-beta * valid_energies) + np.log(valid_energies**2))
                log_H = logsumexp(np.concatenate(log_terms_H)) - log_Z
                log_H_sq = logsumexp(np.concatenate(log_terms_H_sq)) - log_Z
                #log_terms.append(np.log(weight) + np.log(valid_energies)+ (-beta * valid_energies)-log_Z)
                #log_terms_sq.append(np.log(weight) + np.log(valid_energies**2)  + (-beta * valid_energies)-log_Z)

        ##################################
        #while not physically relevant, return also the values with shifted energy.
        H_avg = np.exp(log_H)
        H_avg_unshifted = H_avg + self.Emin
        H_sq_avg = np.exp(log_H_sq)
        H_sq_avg_unshifted = H_sq_avg +2*self.Emin*H_avg+self.Emin**2
        return H_avg,H_avg_unshifted,H_sq_avg,H_sq_avg_unshifted 
    def N_moments(self, beta):
        """
        Computes the thermal expectation value <N> using logsumexp for numerical stability.
        Takes care to filter out N=0 states as their log is ill defined and they shouldn't contribute to shifted expectation values
        Parameters:
        data_dict (dict):               Dictionary where keys label fundamental sectors, and each sector contains:
                                        - 'weight': Number of symmetry-equivalent sectors
                                        - 'es': Mx1 array of eigenvalues (spectrum)
                                        - 'vs': MxM array of eigenvectors
                                        - 'ns': Mx1 array of the <a|N|a> expectation value of the occupation in each eigenstate

        beta (float):                   Inverse temperature (1/kT).
        
        Returns:
        (float,float):                  Thermal expectation value <N>,<N^2>.
        """
        log_Z = self.partition_function(beta)  # log of partition function
        log_terms_N = []
        log_terms_N_sq = []

        for sector_data in self.data_dict.values():
            weight = sector_data['weight']
            energies = sector_data['es']
            eigenvectors = sector_data['vs']  # MxM matrix of eigenstates
            N_a = sector_data['ns']  # 1xM (Mx1?) array of occupation numbers

            # Filter out zero-occupation cases to prevent log(0) issues
            valid_mask = N_a > 0
            valid_N_a = N_a[valid_mask]
            valid_energies = energies[valid_mask]

            if valid_N_a.size > 0:
                log_terms_N.append(np.log(weight) + np.log(valid_N_a) - beta * valid_energies)
                log_terms_N_sq.append(np.log(weight) + 2*np.log(valid_N_a) - beta * valid_energies)

        # Compute the final thermal average <N>
        log_N = logsumexp(np.concatenate(log_terms_N)) - log_Z
        log_N_sq = logsumexp(np.concatenate(log_terms_N_sq)) - log_Z

        return np.exp(log_N),np.exp(log_N_sq)

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#################################################### TERMINAL FUNCTIONS ############################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
def test_mapping():
    ''''
    Tests the config to index function
    '''
    print('=='*50)
    print('=='*15,'Testing Config 2 index function after group action','=='*15)
    print('=='*50)
    itime = time.time()
    L = 2
    params = {'geometry':'triangular','L':L,'partial':False,'projection':False}
    CCs = chain_configs(params)
    configs = CCs.configurations
    for i,c in enumerate(configs):
        for n in range(2):
            for m in range(2):
                for m_c3 in range(3):
                        for m_spin in range(2):
                            c_new = CCs.GroupAction(c,n,m,m_c3,m_spin)
                            i_new = CCs.c2i(c_new)
                            #checking:
                            if c_new != configs[i_new]:
                                print('error mapping')
                                print(c_new,configs[i_new])
                                quit()
    ftime = time.time()
    print('=='*50)
    print('=='*40,'Success','=='*40)
    print(f"mapping check elapsed time: {ftime-itime:.2f} seconds")
    print('=='*50)
    return
def test_commutators():
    '''
    tests commutator relations for triangular lattice
    '''
    L = 2
    params = {'geometry':'triangular','L':2,'partial':False,'projection':False}
    itime = time.time()
    CCs = chain_configs(params)
    '''
    checks C3T1C3^-1 == T2^-1
    '''
    ftime = time.time()
    print('*'*100)
    print(f"configuration creation time: {ftime-itime:.2f} seconds")
    print('='*50)
    print('='*20,'Testing','='*21)
    print('='*50)
    for i,config in enumerate(CCs.configurations):
        config_temp = config
        config_temp = CCs.GroupAction(config_temp,n=0,m=0,m_rot=2,m_spin=0)
        config_temp = CCs.GroupAction(config_temp,n=1,m=0,m_rot=1,m_spin=0)
        config_temp_alt = CCs.GroupAction(config,n=0,m=L-1,m_rot=0,m_spin=0)
        if config_temp != config_temp_alt:
            print('test1:Fail')
            print('config',i)
            print(config_temp)
            print(config_temp_alt)
            quit()
    print('Test 1 Success')
    '''
    checks C3T2C3^-1 == T1 x T2^-1
    '''
    for i,config in enumerate(CCs.configurations):
        config_temp = config
        config_temp = CCs.GroupAction(config_temp,n=0,m=0,m_rot=2,m_spin=0)
        config_temp = CCs.GroupAction(config_temp,n=0,m=1,m_rot=1,m_spin=0)
        config_temp_alt = CCs.GroupAction(config,n=1,m=L-1,m_rot=0,m_spin=0)
        if config_temp != config_temp_alt:
            print('test2:Fail')
            print('config',i)
            print(config_temp)
            print(config_temp_alt)
            quit()
    print('Test 2 Success')
    '''
    checks C^3_3 == id
    '''
    for i,config in enumerate(CCs.configurations):
        config_temp = config
        config_temp = CCs.GroupAction(config_temp,n=0,m=0,m_rot=3,m_spin=0)
        if config_temp != config:
            print('test3:Fail')
            print('config',i)
            print(config_temp)
            print(config)
            quit()
    print('Test 3 Success')
    ftime = time.time()
    print('='*50)
    print('='*20,'Done Test','='*19)
    print('='*50)
    print(f"commutator check time: {ftime-itime:.2f} seconds")
    print('*'*100)
    return
def test_equivalent_sectors():
    '''
    Tests the Equivalent sector generator
    Specifically tests that the *sum_total* of the reduced configs equals the number of total configs
    '''
    print('*'*100)
    L = 2
    params = {'geometry':'triangular','L':2,'partial':False,'projection':False}
    CCs = chain_configs(params)
    itime = time.time()
    reduced_number = CCs.configuration_reduction()
    symmetric_configs = CCs.symm_configs
    ftime = time.time()
    print(f"configuration reduction time: {ftime-itime:.2f} seconds")
    sum_reduced = 0
    sum_total = 0
    for k in symmetric_configs.keys():
        sum_reduced += 1
        sum_total += symmetric_configs[k]
    print('do reduced sectors agree?',sum_reduced,reduced_number)
    print('do total sectors agree?',sum_total,CCs.Nconfigs)
    print('*'*100)
    return
def test_symmetry_related_spectra():
    '''
    checks that configurations related by symmetry have the same spectrum
    '''
    L = 2
    t = 3
    U = 10
    V = 2
    mu = 1
    loc = [1,3,1,3,2,4,2,4,1,2,1,2,3,4,3,4,1,4,1,4,2,3,2,3]
    params = {'geometry':'triangular','L':2,'partial':False,'projection':False}
    CCs = chain_configs(params)
    c = CCs.configurations[random.randint(0, len(CCs.configurations)-1)]
    print('randomly chosen configuration:',c)
    Es = []
    for n in range(2):
        for m in range(2):
            for m_rot in range(3):
                for m_spin in range(2):
                    c_new = CCs.GroupAction(c,n,m,m_rot,m_spin)
                    H_params = {'L':L,'loc':loc,'config':c,'sign':True,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
                    chain_instance = chains(H_params)
                    diag_states = chain_instance.diagonalization()
                    Es.append(diag_states['energies'])
    diff = 0
    for i in range(len(Es)):
        for j in range(len(Es)):
            diff += np.sum(np.abs(Es[i] - Es[j]))
    if diff >1e-10:
        print('uhhhhhhh')
    print('Do configurations related by symmetry have identical spectr?Delta E = ',str(diff))
    return
def testclass():
    '''
    testing the class functions:
    1) Tests all commutators are working correctly
    2) Tests symmetry-related configurations are generated correctly
    3) Tests symmetry-related configurations have same spectrum
    '''
    print('executing....')
    print('1) Testing commutator relations:')
    test_commutators()
    print('2)Testing the state to index mapping')
    test_mapping()
    print('3)Testing the section reduction')
    test_equivalent_sectors()
    print('4)Tesing that symmetry-related configurations have same spectrum')
    test_symmetry_related_spectra()
    print('end of tests')   
def exe1():
    '''
    Diagonalize full system and store everything in dictionary. check memory of dictionary...
    '''
    L = 2;t = 1;U = 10;V = 2;mu = 1;loc = [1,3,1,3,2,4,2,4,1,2,1,2,3,4,3,4,1,4,1,4,2,3,2,3]
    print('executing....')
    print('Step1:Generate Configurations')
    section_params = {'geometry':'triangular','L':2,'partial':False,'projection':False}
    H_params = {'L':L,'loc':loc,'sign':True,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    CCs = chain_configs(params=section_params)
    CCs.configuration_reduction()
    symmetric_configs = CCs.symm_configs
    count = 0
    states = 0
    dimH = 64**(L**2) # for triangular case
    data_dict = {}
    itime = time.time()
    print('Step2:Diagonalizing')
    for k in symmetric_configs.keys():
        H_params['config'] = k
        chain_instance = chains(H_params)
        diag_states = chain_instance.diagonalization()
        data_dict[k] = diag_states.copy()
        data_dict[k]['weights'] = symmetric_configs[k]
        count +=1
        states += symmetric_configs[k]*chain_instance.dim
        if count%1000 == 0:
            #print('keys',data_dict[k].keys())
            #print('M size',data_dict[k]['weights'],data_dict[k]['energies'].shape)
            print(f"Progress: {100*(states/dimH):.2f} %")   
            print(f"Time elapsed:{(time.time()-itime)/60:.2f} minutes")
            print('*'*100)
    print('ED ENDED')
    print('-'*100)
    print('MEMORY OF VARS')
    for name, obj in locals().items():
        size_mb = sys.getsizeof(obj) / (1024 * 1024)
        print(f"{name}: {size_mb:.6f} MBs")
    print('-'*100)
    Es_flat =  np.concatenate([sector_data['energies'] for sector_data in data_dict.values()])
    Emin = np.min(Es_flat)
    Emax = np.max(Es_flat)
    print('Extremal energies:',Emin,Emax)
    return
#########################################################
########incorporated into chain_configs class###############
#########decommissioned. delete after double checking########
############################################################
'''
def section_generator(L):
    #Generates all configurations for a system of size L
    #Args:
    #    L(int):         The number of chains in the system (per flavor)
    #    L(int):         The sites in each chain
    #    (Equal in this case. Need to generalize)
    #Returns:
    #    All possible configurations
    #    A given configuration(section) has the form (n_1,n_2,.....,n_3L)
    
    
    ###########################################################################################################################################################
    #Convention: First L numbers are for the first valley parallel to a1, [L->2L] are second valley parallel to a2, [2L->3L] are third valley parallel to a3.
    
    #L=1: #~64      time = 0s
    #L=2: #~5*1e5   time = 0.06s
    #L=3: #~7*1e10  time ~ 1.5 hours process is killed, maybe too much memory. would require about 24TB of memory woOoOah

    #memory= # x 4 bytes per integer x 2*3*L integers for a configuration. + overhead for tuples (overhead is 3x memory to store a tuple so its bad but its order 1 bad)
    #L=1: ~1e-6 GB
    #L=2: ~1e-2 GB
    #L=3: ~1e+4 GB
    #L=4: ~1e+10 GB

    #Clearly for L=3 to stand a chance we have to dynamically create the configurations. Start from lowest, map all the ones related by symmetry, then generate second lowest (not related by symmetry etc etc)
    #So at each process, we are only storing integers. Also after we are done at each step, don't sort them but save them in a dictionary or something, because just storing 4**18 numbers needs ~100GB.
    
    #The result we actually care about will be about the 4**18/ 10 ~ 10**9 unique configurations and their weights.
    #If we do this smartly we can get final memory of ~<10 GBs
    
    electron_count = range(L + 1)
    configs_spinup = list(product(electron_count, repeat=3*L))
    configs_spindown = list(product(electron_count, repeat=3*L))
    configs =[(c_up,c_down) for c_up in configs_spinup for c_down in configs_spindown]
    return configs
def section_generator_partial(L, n):
    
    #Generates the first n configurations for a given L, saves them temporarily to a file,
    #and returns them as a list of tuples. The temporary file is deleted after use.
    #This may be useful for systems larger than L=2 when number of configurations is larger than that allowed by RAM to hold it at the same time

    #Args:
    #    L (int):        The system size (number of chains per valley and sites per chain).
    #    n (int):        The number of configurations to generate.

    #Returns:
    #    list:           A list of configurations (tuples) equivalent to section_generator(L).
    
    # Temporary file to store configurations
    temp_file = "configurations_temp.txt"

    # Step 1: Generate configurations and save to file
    electron_count = range(L + 1)
    count = 0
    with open(temp_file, 'w') as f:
        for c_up in product(electron_count, repeat=3 * L):
            for c_down in product(electron_count, repeat=3 * L):
                config = (c_up, c_down)
                f.write(str(config) + '\n')
                count += 1
                if count >= n:
                    break
            if count >= n:
                break

    # Step 2: Read configurations back into a list of tuples
    configurations = []
    with open(temp_file, 'r') as f:
        for line in f:
            configurations.append(eval(line.strip()))  # Convert string back to tuple

    # Step 3: Delete the temporary file
    os.remove(temp_file)

    return configurations
def T(s,n,m,L):
    #Takes a configuration s and acts on it with the translation operator
    #T^m_2   T^n_1 
    #Translates it by n sites along a_1 and m sites along a_2.

    #Args:
    #    s(tuple):               The configuration of chains
    #    n(int \in [0,L)):       Translation about a_1
    #    m(int \in [0,L)):       Translation about a_2
    #Returns:
    #    s'(tuple):              The transformed configuration
    
    s_up,s_down = s[0],s[1]
    s_up_1 = s_up[:L]
    s_up_2 = s_up[L:2*L]
    s_up_3 = s_up[2*L:3*L]
    s_down_1 = s_down[:L]
    s_down_2 = s_down[L:2*L]
    s_down_3 = s_down[2*L:3*L]
    #check len(s_up_1) == L

    s_up_1 = [s_up_1[i-m] for i in range(L)]
    s_up_2 = [s_up_2[i-n] for i in range(L)]
    s_up_3 = [s_up_3[i-(n+m)] for i in range(L)]
    s_down_1 = [s_down_1[i-m] for i in range(L)]
    s_down_2 = [s_down_2[i-n] for i in range(L)]
    s_down_3 = [s_down_3[i-(n+m)] for i in range(L)]
    s_up_t = tuple(s_up_1+s_up_2+s_up_3)
    s_down_t = tuple(s_down_1+s_down_2+s_down_3)
    return (s_up_t,s_down_t)
def T_old(s, n, m, L):
    
    #supposedly faster but at least for 10**5 elements, my code is slightly faster.
    #############
    #Applies the translation operator to a configuration.
    
    #Args:
    #    s (tuple): configuration, containing two tuples of integers (s_up, s_down).
    #    n (int): number of sites to translate in the x-direction.
    #    m (int): number of sites to translate in the y-direction.
    #    L (int): number of sites in one chain direction.

    #Returns:
    #    tuple: Translated configuration (s_up_t, s_down_t).
    
    s_up, s_down = s
    
    # Extract slices for s_up and s_down
    s_up_1, s_up_2, s_up_3 = s_up[:L], s_up[L:2*L], s_up[2*L:3*L]
    s_down_1, s_down_2, s_down_3 = s_down[:L], s_down[L:2*L], s_down[2*L:3*L]
    
    # Translate each segment using modular arithmetic
    s_up_1 = [s_up_1[(i - m) % L] for i in range(L)]
    s_up_2 = [s_up_2[(i - n) % L] for i in range(L)]
    s_up_3 = [s_up_3[(i - (n + m)) % L] for i in range(L)]
    s_down_1 = [s_down_1[(i - m) % L] for i in range(L)]
    s_down_2 = [s_down_2[(i - n) % L] for i in range(L)]
    s_down_3 = [s_down_3[(i - (n + m)) % L] for i in range(L)]

    # Combine translated segments and return as tuples
    s_up_t = tuple(s_up_1 + s_up_2 + s_up_3)
    s_down_t = tuple(s_down_1 + s_down_2 + s_down_3)

    return (s_up_t, s_down_t)
def C3(s,L):
    
    #Implements C3 rotation which cycles through the chain directions while also permuting the order withing a chain type

    #(1,x)----->(2,x)
    #(2,x)----->(3,L-x)
    #(3,x)----->(1,L-x)

    #Args:
    #    s(tuple):               The configuration of chains
    #    L(int):                 System size
    #Returns:
    #    s'(tuple):              The transformed configuration
    
    s_up,s_down = s[0],s[1]
    s_up_1 = s_up[:L]
    s_up_2 = s_up[L:2*L]
    s_up_3 = s_up[2*L:3*L]
    s_down_1 = s_down[:L]
    s_down_2 = s_down[L:2*L]
    s_down_3 = s_down[2*L:3*L]

    # Rotate the chains:
    # a_1 -> a_2 (unchanged order)
    # a_2 -> a_3 (reversed order)
    # a_3 -> a_1 (reversed order)
    #s_up_1_rot = s_up_3[::-1]

    s_up_1_rot = tuple([s_up_3[L-i-1] for i in range(L)]) #could do tuple(s_up[::-1]) instead... but for such small lists difference is marginal...
    s_up_2_rot = s_up_1
    s_up_3_rot = tuple([s_up_2[L-i-1] for i in range(L)])
    s_down_1_rot = tuple([s_down_3[L-i-1] for i in range(L)])
    s_down_2_rot = s_down_1
    s_down_3_rot = tuple([s_down_2[L-i-1] for i in range(L)])

    s_up_rot = tuple(s_up_1_rot+s_up_2_rot+s_up_3_rot)
    s_down_rot = tuple(s_down_1_rot+s_down_2_rot+s_down_3_rot)

    return (s_up_rot,s_down_rot)
def S_inv(s):
    
    #exchanges the two tuples
    #quivalent to S_z -> S_(-z)
    
    return (s[1],s[0])
def GroupAction(s,n,m,m_c3,m_spin,L):
    
    #Acts on a configuration with a generic element from the symmetry group

    #Args:
    #    s(tuple):               Original configuration
    #    L(int):                 System size
    #   n(int \in [0,L)):       Translation about a_1
    #    m(int \in [0,L)):       Translation about a_2
    #    m_c3(int \in [0,1,2]):  C3 rotation
    #    m_spin(int \in [0,1]):  Spin inversion


    #Returns:
    #    s'(tuple):              The transformed configuration
    
    s_temp = s
    s_temp = T(s_temp,n,m,L)
    for x in range(m_c3):
        s_temp = C3(s_temp,L)
    for y in range(m_spin):
        s_temp = S_inv(s_temp)
    return s_temp
def c2i(config,L):
    
    #Finds the index of a given configuration.
    #add explanation for mapping*

    #Args:
    #    config(tuple):          A configuration in the form [(up_spin), (down_spin)].
    #    L(int):                 Linear system size.

    #Returns:
    #    total_index(int):       The index of the configuration.
    
    # Unpack the red and blue configurations
    up_config,down_config = config

    # Convert the tuple to a unique index
    base = L + 1
    up_index= sum(r * (base ** i) for i, r in enumerate(reversed(up_config)))
    down_index = sum(b * (base ** i) for i, b in enumerate(reversed(down_config)))
    
    # Combine the two indices
    total_index = up_index * (base ** (3*L)) + down_index
    
    return total_index
def section_reduction(L):
    
    #Generates all the states, then goes throught them, applies all the group elements and bunches the configs together in equivalence classes: if \exists g \in G: g(c)=c' => c ~c'
    #Then it calls the reweighing function that turns equivalence classes into two numbers; Their representative and their order(weight)
    #Args:
    #    dict(dictionary):       Dictionary of the form (key,value):(x_1,x_1,x_2,x_3,.....) with number equal to the order of the group

    #Returns:
    #    Equivalence_c...(dict): Dictionaty of the equivalence classes. Keys are the representative configurations and values their weights.
    #    Tot_number(int):        Total number of configurations
    #    Reduced_number(int):    Number of configurations after this reduction
    
    configs = section_generator(L)
    Tot_number = (1+L)**(6*L)
    Equivalence_classes = {}
    mask = [True]*Tot_number # checks if this state has been 'met' before
    for i,c in enumerate(configs):
        if mask[i] == False:
            continue
        Equivalence_classes[i] = []
        #######
        for n in range(2):
            for m in range(2):
                for m_c3 in range(3):
                        for m_spin in range(2):
                            c_new = GroupAction(c,n,m,m_c3,m_spin,L)
                            i_new = c2i(c_new,L)
                            Equivalence_classes[i].append(i_new)
                            mask[i_new] = False
    Equivalence_classes = reweight(Equivalence_classes)
    Reduced_number = len(Equivalence_classes)
    print('Total Number of configurations',Tot_number)
    print('Reduced Number of configurations',Reduced_number)
    return Equivalence_classes,Reduced_number,Tot_number
def reweight(dict):
    
    #Takes the symmetry-reduced configurations and 'reweights' them. Ie for eg [x_1]:(x_1,x_2,x_3) in the same equivalence class, it turns it into [x_1]:(3) 
    #Meaning for calculations, x_1 cnofiguration should count 3 times as it really represents three states.

    #Args:
     #   dict(dictionary):       Dictionary of the form (key,value):(x_1,x_1,x_2,x_3,.....) with number equal to the ordger of the group

    #Returns:
    #    dict_out(dictionary):   Dictionary of the form (key,value):(x_1,w_1) with w_1 the weight corresponding to configuration x_1
    
    
    dict_out = {}
    for k in dict.keys():
        lst = dict[k]
        element_counts = Counter(lst)
        unique_elements = list(element_counts.keys())
        multiplicities = list(element_counts.values())
        if not multiplicities.count(multiplicities[0]) == len(multiplicities):
            #checks that all items in an equivalence class appear an equal amount of times: 1,2,3,4,6,8,12,24
            print('uhhhhhhhhhhhhhhhhh')
            quit()
        weight = len(unique_elements)
        #dict_out[k] = (unique_elements,weight) # this returns something that holds info of all equivalent states
        dict_out[k] = weight # this only holds info of the *representative* configuration through the dict key
    return dict_out
def Qnumber_printout(c):
    
    #Outputs the quantum numbers of each configuration

    #Args:
    #    c(nested tuple):        The configuration

    #Returns:
    #    Qs(tuple):              Nested Tuple of quantum numbers:(\\nu,Sz_tot,N_1,N_2,N_3,Sz_1,Sz_2,Sz_3)
    
    
    up_c,down_c = c
    L = int(len(up_c)/3)
    N_1_up,N_2_up,N_3_up = sum(up_c[:L]),sum(up_c[L:2*L]),sum(up_c[2*L:3*L])
    N_1_down,N_2_down,N_3_down = sum(down_c[:L]),sum(down_c[L:2*L]),sum(down_c[2*L:3*L])
    N_up = sum(up_c)
    N_down = sum(down_c)
    nu = (N_up+N_down)/L**2
    Sz_tot = (N_up-N_down)/2
    N_1 = N_1_up + N_1_down
    N_2 = N_2_up + N_2_down
    N_3 = N_3_up + N_3_down
    Sz_1 = (N_1_up - N_1_down)/2
    Sz_2 = (N_2_up - N_2_down)/2
    Sz_3 = (N_3_up - N_3_down)/2
    return (nu,Sz_tot,N_1,N_2,N_3,Sz_1,Sz_2,Sz_3)
def fcombinatoric(c):
    
    #Calculates the Hilbert space dimension of a configuration.
    
    c_up = c[0]
    c_down = c[1]
    dim = 1
    L = int(len(c_up)/3)
    for i in range(L):
        dim *= comb(L,c_up[i])*comb(L,c_down[i])*comb(L,c_up[i+L])*comb(L,c_down[i+L])*comb(L,c_up[i+2*L])*comb(L,c_down[i+2*L])
    return dim

'''
#####################################
#########decommissioned functions####
##incorporated into chains class####
#############delete after checking###
#####################################
'''
def generate_basis(c):
    
    #Generates the basis in terms of binaries. An LxL system will have a basis with 3L*2*L=6L**2 sites. Each site is 0 or 1 so full hilbert space is ofcourse 2**(6L**2)=64**L**2
   # 
    #Args:
    #    c(nested tuple):        The chain configuration
    #Returns:
    #    basis(list):            A list of all states in the hilbert space spanned by the configuration
    #    length_basis(int):      The size of the HIlbert space
    #    Hamiltonians(list):     List of 6L**2 npcarray Hamiltonians to be combined into a full hamiltonian
    
    t = 1
    basis = []
    Hamiltonians = []
    L = int(len(c[0])/3)
    for chain in range(int(3*L)):
        chain_up = c[0][chain]
        chain_down = c[1][chain]
        basis_up = generate_partitions(L,chain_up)
        Hamiltonians.append(generate_chain_Hamiltonian(basis_up))
        basis_down = generate_partitions(L,chain_down)
        Hamiltonians.append(generate_chain_Hamiltonian(basis_down))
        if len(basis) == 0: 
            basis = basis_up
            basis = [old + new for old in basis for new in basis_down]
        else:
            basis = [old + new for old in basis for new in basis_up]
            basis = [old + new for old in basis for new in basis_down]
    return basis,len(basis),Hamiltonians
def generate_chain_Hamiltonian(basis,t=1,L=2):
    
    #Generates the small non-interacting Hamiltonian for a single chain and spin. The dimension is read from the basis.
    #The full Hilbert space for a single chain and spin is 2**L but in our case the Hilbert space will be L Choose N_spin. For L=2, dim =1 or 2 while for L=3, dim = 1 or 3 

    #Parameters:
    #    Basis(list):        A list containing all states (represented by binaries) in the chain's Hilbert space
    #    t(float):           Hopping strength
    #Returns:
    #    H(npc array):       A (densely constructed) Hamiltonian
    
    
    #step1) Build lookup table for all states(very small table). That can just be the basis_up/down list
    #step2) go through basis size, associate index with state and check hopping, mapping it back to a new state.
    #step3) done#
    dim = len(basis)
    basis_dec = [int(el,2) for el in basis]
    L_chain = L#no better way???
    H = np.zeros((dim,dim),dtype=float)
    for m in range(dim):
        s = basis_dec[m]
        for i in range(L_chain):
            j=(i+1)%L_chain
            s2 = hop(s,i,j)
            if s2 != -1:
                try:
                    n = basis_dec.index(s2)
                except ValueError:
                    print('Index not found...quitting!')
                    quit()
                H[n,m] -= t
    return H
def hop(s,i,j):
    
   # CHecks if hopping is allowed between sites i and j for state s and if it is,
    #it outputs the resulting state

    #Args:
    #    s(bin):         A binary number with L digits(L=length of chain) signifying the state of the chain
    #    i(int),j(int):  0 =<i,j<L Integers representing sites on the chain

    #Returns:
    #    s2:             Either -1 to signify no allowed hopping or a binary to denote the resulting state after the hopping
    
    mask = 2**(i)+2**(j)
    K = s & mask #bitwise AND.
    L = K ^ mask #bitwise XOR.
    # L will have structure 0000000[i]0000[j]00000 and there's four cases:
    #1) L = mask means I1[i]=I1[j]=0 -> hopping is not allowed
    #2) L = 000..00 means I1[i]=I1[j]=1 -> hopping is not allowed
    #3&4) L = ...[1]...[0]... or L = ...[0]...[1]... means hopping is allowed, in which case new integer is 
    if L == mask or L == 0:
        s2 = -1#flag to signify no hopping
    else:
        s2 = s - K + L
    return s2

def generate_partitions(L, N):
    
    #Generate all possible partitions of N electrons in a chain with L sites as binary strings.

    #Parameters:
    #    L (int): Number of sites.
    #    N (int): Number of electrons.

    #Returns:
    #    list: List of binary strings representing the partitions.
    

    if N > L:
        raise ValueError("Number of electrons (N) cannot exceed number of sites (L).")

    # Generate all combinations of N positions from L sites
    partitions = []
    for positions in combinations(range(L), N):
        # Create a binary representation of the partition
        binary = ['0'] * L
        for pos in positions:
            binary[pos] = '1'
        partitions.append(''.join(binary))

    return partitions
    def test10():
    
    #Plots histogram of size of hilbert spaces
    L = 2
    configs = section_generator(L)
    Equivalence_clases,Reduced_number,Tot_number = section_reduction(L = L)
    dimensions = []
    for k in Equivalence_clases.keys():
        c = configs[k]
        basis,lenbasis = generate_basis(c)
        dimensions.append(lenbasis)
    bins = np.logspace(np.log10(min(dimensions)), np.log10(max(dimensions)), num=10)
    hist, edges, _ = plt.hist(dimensions, bins=bins, edgecolor='black', log=True)  # Use log scale for frequency
    plt.savefig('uhhhhhhhh.png')
    return
'''

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
##########################################  COMMAND LINE RUN #######################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multiple features.")
    parser.add_argument("test", type=str, choices=["no_test","testclass"], help="test to run")
    parser.add_argument("execute", type=str, choices=["no_exe","exe1"], help="execute ting")

    args = parser.parse_args()

    if args.test == "testclass":
        testclass()
    if args.execute == "exe1":
        exe1()