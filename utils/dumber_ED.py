'''
L=3 PBC bosons for t=1,U=0,mu=0,V=0 between sectors
(1,0)----c^\dagger_up---->(2,0)
to see G_00 vs G_11
'''
L = 3
t = 1
JWstring = True
import numpy as np

# sector (1,0) and (2,0) basis states as strings
base_i = {0: '001', 1: '010', 2: '100'}     # (1,0) sector
base_f = {0: '011', 1: '101', 2: '110'}     # (2,0) sector

# trivial H = -t hopping matrix in both sectors
H = np.array([[0, -t, -t],
              [-t, 0, -t],
              [-t, -t, 0]])

e, v = np.linalg.eigh(H)
print("Energies:\n", e)
print("Eigenvectors:\n", v)

# build c^\dagger_r matrix: maps (1,0) → (2,0)
cdag_r = {}

for r in range(L):
    cdag_r[r] = np.zeros((3, 3))  # rows: final (2,0), cols: initial (1,0)

    for i, state_i in base_i.items():
        int_i = int(state_i, 2)
        if ((int_i >> r) & 1) == 0:
            int_f = int_i | (1 << r)  # apply c^\dagger at site r
            str_f = format(int_f, f'0{L}b')
            #print(r,state_i,str_f)
            for j, state_f in base_f.items():
                if state_f == str_f:
                    # calculate JW sign (optional here — can be off if you want pure bosonic)
                    sign = (-1) ** bin(int_i & ((1 << r) - 1)).count("1") if JWstring else 1
                    cdag_r[r][j, i] = sign
                    print(f"c†_{r}: |{state_i}⟩ → |{state_f}⟩,     {i} → {j},      sign={sign}")
##############
#now create eigenstate mapping
#check translational invariance

