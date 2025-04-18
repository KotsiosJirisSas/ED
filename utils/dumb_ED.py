'''
Some dumb ED code for small (L=3 or L=4) chains
This is for PBC hc fermions and i want to check the JW green's function, make sure its translationally invariant

L=3: 64x64 matrix
L=4: 256x256 matrix
'''
import numpy as np
from itertools import product
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from collections import defaultdict


L = 3
t = 1.0
U = 0.0
mu = 0

# Build Hilbert space: tuple of bits, length 2L
# even indices: spin-up, odd indices: spin-down
hilbert = [tuple(s) for s in product([0, 1], repeat=2 * L)]
state_to_index = {s: i for i, s in enumerate(hilbert)}
dim = len(hilbert)

def idx(site, spin):
    return 2 * site + spin  # spin=0: up, spin=1: down

# Initialize Hamiltonian
H = np.zeros((dim, dim), dtype=np.float64)

for state in hilbert:
    s = np.array(state)
    i = state_to_index[state]

    # On-site interaction + chemical potential
    for site in range(L):
        n_up = s[idx(site, 0)]
        n_dn = s[idx(site, 1)]
        n_tot = n_up + n_dn
        onsite_energy = (U / 2) * (n_tot - 1) ** 2 - mu * n_tot
        H[i, i] += onsite_energy
    #hop
    for site in range(L):
        next_site = (site + 1) % L  # PBC
        for spin in [0, 1]:
            if s[idx(site, spin)] == 1 and s[idx(next_site, spin)] == 0:
                # hop from site → next_site
                s_new = s.copy()
                s_new[idx(site, spin)] = 0
                s_new[idx(next_site, spin)] = 1
                s_new_tup = tuple(s_new)
                j = state_to_index.get(s_new_tup)
                if j is not None:
                    H[i, j] -= t
            if s[idx(next_site, spin)] == 1 and s[idx(site, spin)] == 0:
                # hop from next_site → site
                s_new = s.copy()
                s_new[idx(next_site, spin)] = 0
                s_new[idx(site, spin)] = 1
                s_new_tup = tuple(s_new)
                j = state_to_index.get(s_new_tup)
                if j is not None:
                    H[i, j] -= t



#plt.imshow(np.abs(H))
#plt.colorbar()
#plt.savefig('temp.png')
print(np.sum(np.abs(H-H.T)))
for i in range(dim):
    for j in range(i, dim):
        if abs(H[i, j]) > 1e-10:
            print(f"H[{i},{j}] = {H[i,j]:.2f}")

# Diagonalize
eigvals, eigvecs = eigh(H)
print('spectrum: \n',eigvals)
print("Ground state energy:", eigvals[0])
###################################################
##### greens function############
#################################
def compute_fermionic_Gtau_fixed_sector(Nup, Ndown, beta, tau_vals):
    # Get basis states for each sector
    source_basis = [i for i in range(dim)
                    if sum(hilbert[i][::2]) == Nup and sum(hilbert[i][1::2]) == Ndown]
    target_basis = [i for i in range(dim)
                    if sum(hilbert[i][::2]) == Nup + 1 and sum(hilbert[i][1::2]) == Ndown]

    # Build C†_r for spin-up
    Cdag_r = {}
    for r in range(L):
        r_idx = idx(r, 0)
        op = np.zeros((dim, dim))

        for j in range(dim):  # ket
            if hilbert[j][r_idx] == 0:
                new_state = list(hilbert[j])
                new_state[r_idx] = 1
                i = state_to_index.get(tuple(new_state))
                if i is not None:
                    sign = (-1) ** sum(hilbert[j][:r_idx])
                    op[i, j] = sign
        Cdag_r[r] = op

    # Project to eigenbasis
    G = np.zeros((L, L, len(tau_vals)))
    psi = eigvecs  # shape [dim, dim]
    Em = eigvals

    for r in range(L):
        for rp in range(L):
            C = Cdag_r[r].conj().T
            Cdag = Cdag_r[rp]

            for m in range(dim):  # |m⟩ in source sector
                if m not in source_basis:
                    continue
                Em_m = Em[m]
                psi_m = psi[:, m]

                for n in range(dim):  # |n⟩ in target sector
                    if n not in target_basis:
                        continue
                    Em_n = Em[n]
                    psi_n = psi[:, n]

                    mat_elem = (psi_m.conj().T @ C @ psi_n) * (psi_n.conj().T @ Cdag @ psi_m)
                    weight = np.exp(-(beta - tau_vals) * Em_m) * np.exp(-tau_vals * Em_n)
                    G[r, rp, :] += -mat_elem.real * weight

    Z = np.sum(np.exp(-beta * Em[source_basis]))
    return G / Z

def compute_bosonic_Gtau(Nup, Ndown, beta, tau_vals):
    # Filter indices in fixed sector
    sector_indices = [i for i, state in enumerate(hilbert)
                      if sum(state[::2]) == Nup and sum(state[1::2]) == Ndown]

    Em = eigvals
    psi = eigvecs  # shape: [dim, dim]
    G = np.zeros((L, L, len(tau_vals)))

    # Build full operator matrices b†_r and b_r
    Bdag_r = {}
    for r in range(L):
        r_idx = idx(r, 0)
        op = np.zeros((dim, dim))

        for i, state in enumerate(hilbert):
            if state[r_idx] == 0:
                new_state = list(state)
                new_state[r_idx] = 1
                new_state = tuple(new_state)
                j = state_to_index.get(new_state, None)
                if j is not None:
                    op[j, i] = 1.0  # b† turns |i⟩ → |j⟩
        Bdag_r[r] = op

    # Compute G_rr'(tau)
    for r in range(L):
        for rp in range(L):
            Bdag = Bdag_r[rp]
            B = Bdag_r[r].T

            for m in sector_indices:
                psi_m = psi[:, m]
                Em_m = Em[m]

                for n in sector_indices:
                    psi_n = psi[:, n]
                    Em_n = Em[n]

                    amp = (psi_m.conj().T @ B @ psi_n) * (psi_n.conj().T @ Bdag @ psi_m)
                    weight = np.exp(-(beta - tau_vals) * Em_m) * np.exp(-tau_vals * Em_n)
                    G[r, rp, :] += -amp.real * weight

    Z = np.sum(np.exp(-beta * Em[sector_indices]))
    return G / Z



###############
beta=1
tau_vals = np.linspace(0, beta, 100)
Nup, Ndown = 1, 1

Gf = compute_fermionic_Gtau_fixed_sector(Nup, Ndown, beta, tau_vals)
#Gb = compute_bosonic_Gtau(Nup, Ndown, beta=beta, tau_vals = tau_vals)

# Plot comparison (diagonal)

plt.figure()
for r in range(L):
    #plt.plot(tau_vals, Gf[r, r, :], '--', label=f"ferm G({r},{r})")
    plt.plot(tau_vals, Gf[r, r, :], '-', label=f"bos G({r},{r})")
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$G_{rr}(\tau)$')
plt.tight_layout()
plt.savefig('temp.png')