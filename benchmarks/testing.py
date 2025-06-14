import sys
import numpy as np
import time
import random
import h5py
from collections import defaultdict
sys.path.append('/mnt/users/kotssvasiliou/ED/utils')
import ED_chains_final as mod
def generate_data(geometry):
    L = 2
    params = {'L':L,'geometry':geometry,'projection':False}
    if params['geometry'] == 'square':
        config_number = (L+1)**(4*L)
        group_order = 2*2*L**2
    elif params['geometry'] == 'triangular':
        config_number = (L+1)**(6*L)
        group_order = 2*3*L**2
    if params['projection'] == True:
        params['Nel_min'] = 2*(L**2) - 1
        params['Nel_max'] = 2*(L**2) + 1
    ###################################
    #generate all sectors#
    ###################################
    timei = time.time()
    configs = mod.chain_configs(params)
    timef = time.time()
    print('time to generate sectors:',timef-timei,' secs')
    ####################################
    #initialize hamiltonian params for system#
    ####################################
    t = 1
    U = 6
    V = 1
    mu = 0
    H_params = {'L':L,'sign':True,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    H_params['geometry'] = params['geometry']
    DATA = configs.compressed_data
    return DATA,H_params

def testing():
    '''
    diagonalizes *all* sectors and checks that those sectors that are in symmetry equivalent configurations have eigenstates related by a permutation.
    ***********
    projection has a bug i need to fix
    '''
    L = 2
    params = {'L':L,'geometry':'square','projection':False}
    if params['geometry'] == 'square':
        config_number = (L+1)**(4*L)
        group_order = 2*2*L**2
    elif params['geometry'] == 'triangular':
        config_number = (L+1)**(6*L)
        group_order = 2*3*L**2
    if params['projection'] == True:
        params['Nel_min'] = 2*(L**2) - 1
        params['Nel_max'] = 2*(L**2) + 1
    ###################################
    #generate all sectors#
    ###################################
    timei = time.time()
    configs = mod.chain_configs(params)
    timef = time.time()
    print('time to generate sectors:',timef-timei,' secs')
    ####################################
    #initialize hamiltonian params for system#
    ####################################
    DATA = configs.compressed_data
    t = 1
    U = 6
    V = 1
    mu = 0
    H_params = {'L':L,'sign':True,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    H_params['geometry'] = params['geometry']
    ####################################
    #for a random config generate basis#
    ####################################
    symmetry_sector = random.choice(list(DATA.keys()))
    #symmetry_sector = ((1,1,1,1),(1,1,1,1))
    equiv_sectors, equiv_perms = DATA[symmetry_sector]

    H_params['config'] = symmetry_sector
    print('symmetry sector',symmetry_sector)
    print('equivalent sectors',equiv_sectors)
    sector_sys = mod.chains(H_params)
    sector_basis = [int(el,2) for el in sector_sys.basis]
    for i,state in enumerate(sector_basis):
        print('basis:',i,state,binp(state,length=group_order))
    ################################################
    #    ACT ON BASIS STATES WITH CREATION OPERATOR#
    #    AND THEN FIND SECTOR NEW STATE IS IN      #
    #    AND THEN FIND REPRESENTATIVE SECTOR       #
    #    AND SEE THAT USING BOTH WAYS BASES MATCH  #
    ################################################
    timei = time.time()
    for i,I in enumerate(sector_basis):
        J,sgn = cdagger(loc=14,state=I)
        if J == None:
            print(I,binp(I,length=group_order),'cdagger 10---->None')
            print('sector:',symmetry_sector,'--->None')
        else:
            sec_new = state2sector(J,L=2)
            sec_principle = find_principle_sec(sec_new,DATA)
            #####################
            #create new basis   #
            # and find new index#
            #####################
            H_params['config'] = sec_new
            new_sector_sys = mod.chains(H_params)
            new_sector_basis = [int(el,2) for el in new_sector_sys.basis]
            j = new_sector_basis.index(J)

            print(I,binp(I,length=group_order),'cdagger 10---->',J,binp(J,length=group_order))
            print('sector:',symmetry_sector,'--->',sec_new,sec_principle)
            print('index',i,j)
        print('-'*100)
    timef = time.time()
    print(timef-timei)    
#
def location_mapping(geometry,L):
    """
    Maps each tuple (j, eta, s) to the corresponding x'th binary place in self.loc.
    self.loc is a list/array of length 6*L**2 
    """
    #
    if geometry == 'triangular':
        Nflav = 3
    elif geometry == 'square':
        Nflav = 2
    if L == 2 and geometry == 'triangular':
        loc = [1,3,1,3,2,4,2,4,1,2,1,2,3,4,3,4,1,4,1,4,2,3,2,3]
    elif L == 2 and geometry == 'square':
        loc = [1,3,1,3,2,4,2,4,1,2,1,2,3,4,3,4]
    elif L == 3 and geometry == 'square':
        loc = [1,4,7,1,4,7,2,5,8,2,5,8,3,6,9,3,6,9,1,2,3,1,2,3,4,5,6,4,5,6,7,8,9,7,8,9]
    else:
        print('not implemented geometry or size')
        raise NotImplementedError
    
    mapping = {}
    for eta in range(Nflav):
        for s in range(2):
            for j in range(1,L**2+1):
                # Only consider indices in the current eta block.
                start = eta * 2 * L**2
                end = (eta + 1) * 2 * L**2
                # Find the first index in the block with the correct parity and j value.
                for i in range(start, end):
                    if ((i // L) % 2 == s) and (loc[i] == j):
                        mapping[(j, eta, s)] = i
                        break
                else:
                    raise ValueError(f"No index found for (j={j}, eta={eta}, s={s})")
    return mapping
#
def build_reverse_lookup(data):
    lookup = {}
    for key, value in data.items():
        for sec in value[0]:
            lookup[sec] = key
    return lookup

#
def gen_creation_mapping(DATA,H_params,L,geometry='square', filename='greens_mapping.h5'):
    '''
    generate <i|c^\dagger|j> data as HD5 with structure:
    greens_mapping.h5
    └── c^{\dagger}_{r0,eta0,s0}
        ├── sector0_to_sector_new0
        |   ├── sectors  : [rep_sector,sector_new,rep_sector_new]
        │   ├── ij_pairs : (i, j) pairs for basis states in sec0 and sec_new0
        │   ├── signs    : Fermionic signs for each pair
        ├── sector1_to_sector_new1
        │   ├── ...
    └── cdagger_r1_eta2_s1
        ├── ...
    
    ------------------------------------------------
    TODO        0) Implement mapping (eta,s,r)--->location on string
                1) Implement sign
                2) Implement projection
    '''
    if geometry == 'square':
        group_order = 2*2*L**2
        flav = 2
    elif geometry == 'triangular':
        group_order = 2*3*L**2
        flav = 3
    Sites = range(1,L**2+1)
    Flavors = range(flav)
    Spin = range(2)
    mapping = location_mapping(geometry=geometry,L=2)
    #print('mapping',mapping)
    #####
    #generate lookup table for sections
    timei = time.time()
    sector_lookup = build_reverse_lookup(DATA)
    timef = time.time()
    print('lookup table time',timef-timei)
    #####
    with h5py.File(filename, 'w') as f:
        for x, eta, s in ((x, eta, s) for x in Sites for eta in Flavors for s in Spin):
        #for operator in range(group_order):#these replace the (r,eta,s) tuples as there  is an ordering between them
            operator_loc = mapping[(x,eta,s)]
            #print(x,eta,s,operator_loc)
            op_group = f.create_group(f"cdagger_op_{x}_{eta}_{s}")
            for rep_sector in DATA.keys():
                new_sector = None#initialized value
                new_rep_sector = None
                H_params['config'] = rep_sector
                rep_sector_system = mod.chains(H_params)
                rep_sector_basis = [int(el,2) for el in rep_sector_system.basis]
                ij_pairs = [] #where ill store the mappings
                signs = [] #where ill store the signs
                for i,I in enumerate(rep_sector_basis):
                    J,sgn = cdagger(loc=operator_loc,state=I) # apply creation operator
                    if J == None:
                        continue
                    else:
                        new_sector_temp = state2sector(J,L=2) #within a sector, all i should map to a j that lies in the same new_sector
                        if new_sector == None:
                            new_sector = new_sector_temp
                        elif new_sector != new_sector_temp:
                            print("Sector mismatch!",new_sector,new_sector_temp)
                            print('....')
                            print('ABORTING')
                            quit()
                        #find representative secotr related to new sector
                        #one method is regular binary search and the other is with a lookup table.
                        #at L=2 square, you get about 10% faster performance with lookup table.
                        #as sanity check one can make sure the mappings agree
                        new_rep_sector_temp = sector_lookup.get(new_sector)
                        #new_rep_sector_temp_2 = find_principle_sec(new_sector,DATA)
                        #if new_rep_sector_temp != new_rep_sector_temp_2:
                        #    print('lookup methods disagrreee')
                        #    quit() 
                        if new_rep_sector == None:
                            new_rep_sector = new_rep_sector_temp
                        elif new_rep_sector != new_rep_sector_temp:
                            print("Sector mismatch!",new_sector,new_sector_temp)
                            print('....')
                            print('ABORTING')
                            quit()
                        H_params['config'] = new_sector
                        new_sector_sys = mod.chains(H_params)
                        new_sector_basis = [int(el,2) for el in new_sector_sys.basis]
                        try:
                            j = new_sector_basis.index(J)
                        except ValueError:
                            print(f"J={J} not found in new sector basis!")
                            print('....')
                            print('ABORTING')
                            #continue
                            quit()
                        ij_pairs.append((i, j))
                        signs.append(sgn)
                if ij_pairs:  # only save non-empty transitions
                    group_name = f"{rep_sector}_to_{new_sector}"
                    trans_group = op_group.create_group(group_name)
                    trans_group.create_dataset('sectors',data=[rep_sector,new_sector,new_rep_sector])
                    trans_group.create_dataset("ij_pairs", data=np.array(ij_pairs, dtype=np.int32), compression="gzip")
                    trans_group.create_dataset("signs", data=np.array(signs, dtype=np.int8), compression="gzip")

def state2sector(state,L,geometry='square'):
    ''''
    given a state, find the sector its in. brings out the configuration (symm. sector) it lies in
    '''
    if geometry == 'square':
        group_order =  4*L**2
    else:
        raise NotImplementedError
    string = binp(state,length= group_order)
    chain_length = L
    num_chains = len(string) // chain_length
    config = [string[i*chain_length:(i+1)*chain_length].count('1') for i in range(num_chains)]
    config_up,config_dn = tuple(config[::2]),tuple(config[1::2])
    return (config_up,config_dn)

def find_principle_sec(sec,data):
    '''
    FIND REPRESENTATIVE SECTOR GIVEN THE GENERIC SECTOR
    --------------------------------------------------
    LOOKUP IS O(N) WHICH MIGHT GET SLOW FOR LARGER SYSTEMS. 
    TODO: MAYBE IMPLEMENT A LOOKUP TABLE
    '''
    if sec == None:
        return None
    for key,values in data.items():
        #if key == sec:
            #print('principle!')
        if sec in values[0]:
            return key

def cdagger(loc, state):
    #need to add sign as well
    if (state >> loc) & 1:  # check if bit at loc is 1
        return None,None
            # Compute sign factor (count fermions to the left)
    #forget about sign for now
    sign = (-1) **(binp(state & ((1 << loc) - 1)).count('1'))
    #sign_temp = (-1) **(binp(state & ((1 << loc) - 1),length=16).count('1'))#im pretty sure the dumb bin function that doesn't retain the most-important-digit stuff (ie 000110->110) also returns correct results
    #if sign != sign_temp:
    #    print('ihfnvfn')
    #    quit()
    return state ^ (1 << loc),sign
def binp(num, length=4):
    if num == None:
        return None
    return format(num, '#0{}b'.format(length + 2))[2:]
###################
if __name__ == "__main__":
    timei = time.time()
    DATA,H_params = generate_data(geometry='triangular')
    timef = time.time()
    print('TIME TO GENERATE DATA',timef-timei)
    timei = time.time()
    gen_creation_mapping(DATA,H_params,L=2,geometry='triangular',filename='greens_mapping.h5')
    timef = time.time()
    print('TIME TO GENERATE GREENS FUNCTION MAPPING',timef-timei)
    