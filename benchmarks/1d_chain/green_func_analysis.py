import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
'''
Testing the 1d chain Green's function

'''
file='Greens_func_benchmark_data_L_6_new.pkl'
with open(file, 'rb') as handle:
    G = pickle.load(handle)
file_old='Greens_func_benchmark_data_L_6.pkl'
with open(file_old, 'rb') as handle:
    Gold = pickle.load(handle)

#####
#sanity test 1: do spins agree?
diff = 0
anisotropy = 0
for g in G:
    for key in g.keys():
        print('key',key,'Z length',len(g[key][0]),'G length',len(g[key][1]),'G shape',g[key][1][0].shape)
        Zs = g[key][0]
        Gs = g[key][1]
        for n,greenfunc in enumerate(Gs):#go over even, odd,total greens functions
            for i in range(6):#L
                for j in range(6):
                    for tau in range(5):
                        diff += np.abs(greenfunc[i,j,0,tau]-greenfunc[i,j,1,tau]) #spin up - spin down
                        #check between odd and even Green's functions?
                        #for s in range(2):
                            #if n == 2:#check between odd and even?
                            #    anisotropy -= greenfunc[i,j,s,tau]
                            #else:
                            #     anisotropy += greenfunc[i,j,s,tau]
print('diff',diff)
#print('anisotropy',anisotropy)
#quit()
####
#save as hdf5
'''
# Create HDF5 file
new_G = []
params = []
for idx,entry in enumerate(G):
    #mu = entry.keys
    new_entry = {}
    for old_key in entry.keys():
        new_key = old_key[-1]
        mu = old_key[0] 
        V = old_key[1]
        #print('V',V,'mu',mu)
        new_entry[new_key] = entry[old_key]
    new_G.append(new_entry)
    params.append((mu,V))
#print(len(G),len(new_G))
#quit()
'''
'''
seen_par = set()
with h5py.File("Gfunc_data.h5", "w") as h5f:
    for idx, entry in enumerate(new_G):
        # Use group name based on (x, y), or just index if x,y not stored
        (mu,V) = params[idx]
        ##
        par_key = (mu, V)

        if par_key in seen_par:
            print(f"Skipping duplicate (x, y) = ({mu}, {V})")
            continue
        seen_par.add(par_key)
        ##
        group_name = f"mu_{mu}_V_{V}" if mu is not None and V is not None else f"entry_{idx}"
        muV_group = h5f.create_group(group_name)
        print(f"Group: {group_name}")

        for beta_key, mat_list in entry.items():
            if beta_key in ['x', 'y']:
                continue  # skip scalar values used in group name
            print('beta key',beta_key)
            beta_group = muV_group.create_group(f"beta_{beta_key}")
            beta_group.create_dataset("Geven", data=np.array(mat_list[0]))
            beta_group.create_dataset("Godd",  data=np.array(mat_list[1]))
            beta_group.create_dataset("Gtot",  data=np.array(mat_list[2]))
            print(f"  Subgroup: beta_{beta_key}")
            print(f"    Datasets: even, odd, tot")
'''
#####
def test_G_1(G):
    '''
    relation that should hold:
    G_p(0)/Z_p + G_p(β)/Z_{1-p}=-1/Z_p
    '''
    diff = 0
    for g in G:#different (mu,V) parameters
        for key in g.keys():#different betas
            Z0 = g[key][0][1]
            Z1 = g[key][0][0]
            G0 = g[key][1][1][0,5,0,:]
            G1 = g[key][1][0][0,5,0,:]
            #term = np.abs((G0[0]/Z0+G0[-1]/Z1)-(-1/Z0))
            #term2 = np.abs((G1[0]/Z1+G1[-1]/Z0)-(-1/Z1))
            #diff += (G0[0]/Z0+G0[-1]/Z1)-(-1/Z0)
            #diff += (G1[0]/Z1+G1[-1]/Z0)-(-1/Z1)
            term = np.abs((G0[0]/Z1+G1[-1]/Z0)-0*(-1/Z1))
            term2 = np.abs((G1[0]/Z0+G0[-1]/Z1)-0*(-1/Z0))
            diff += np.abs((G0[0]/Z1+G1[-1]/Z0)-0*(-1/Z1))
            diff += np.abs((G1[0]/Z0+G0[-1]/Z1)-0*(-1/Z0))
            print(term,term2)
            if term >1e-5 or term2>1e-5:
                #continue
                print('key........',key,term,term2)
            else:
                continue
                print('key',key)
    print('test diff',diff)
def test_G_2(G):
    '''
    relation that should hold with particle-hole symmetry
    '''
    diff = 0
    for g in G:#different (mu,V) parameters
        for key in g.keys():#different betas
            Z0 = g[key][0][1]
            Z1 = g[key][0][0]
            G0 = g[key][1][1]
            G1 = g[key][1][0]
            G0 = (Z0/(Z0+Z1))*G0
            G1 = (Z1/(Z0+Z1))*G1
            for tau in range(G0.shape[-1]):
                #print(tau,G0.shape[-1]-tau-1)
                term = np.abs(G0[:,:,:,tau]-np.transpose(G1[:,:,:,G1.shape[-1]-tau-1],(1,0,2)))
                term = np.einsum('ijk->',term)
                #print(term)
                if term <1e-3:
                    print(key,term)
def test_G_3(G):
    '''
    relation that should hold:
    G_p(R,0)(τ) = (-1)**p G_p(L-R)(τ)
    '''
    diff = 0
    for g in G:#different (mu,V) parameters
        for key in g.keys():#different betas
            if key == (0.0,0.0,1): 
                Z0 = g[key][0][1]
                Z1 = g[key][0][0]
                GR0 = g[key][1][1][:,:,0,2]
                GR1 = g[key][1][0][:,:,0,2]
                Gtot = g[key][1][2][:,:,0,2]
                G0_r = []
                G1_r = []
                rs = []
                for i in range(6):
                    for j in range(6):
                        dist = i - j
                        if dist == 0:
                            print('dist=0',i,j,Gtot[i,j])
                        rs.append(dist)
                        G0_r.append(GR0[i,j])
                        G1_r.append(GR1[i,j])
                plt.plot(rs,G0_r,'.b')
                #plt.plot(rs,G1_r,'.r')
                plt.axhline(0,alpha=0.5)
                plt.savefig('G_dist.png')
                print(len(rs))
                quit()
    print('test diff',diff)
def test_G_4(G):
    '''
    relation that should hold:
    G_p(R,0)(τ) = (-1)**p G_p(L-R)(τ)
    '''
    diff = 0
    for g in G:#different (mu,V) parameters
        for key in g.keys():#different betas
            Z0 = g[key][0][1]
            Z1 = g[key][0][0]
            GR0 = g[key][1][1][:,0,0,:]
            GR1 = g[key][1][0][:,0,0,:]
            Gtot = g[key][1][2][:,0,0,:]
            for i in range(6):
                print(GR0[i,-1],GR1[i,-1])
            quit()
    print('test diff',diff)
def test_G_5(G):
    diff = 0
    for g in G:#different (mu,V) parameters
        for key in g.keys():#different betas
            Godd = g[key][0]
            Geven = g[key][1]
            G = g[key][-1]
            for i in range(6):
                print('i',i,'G',G[i,i,0,:])
            quit()

######
#test_G_1(G)
#quit()
test_G_5(Gold)
quit()
plot = True
if plot == True:
    g = G[3] #V=0,mu=0
    betas = []
    G_even = []
    G_odd = []
    G_tot = []
    Z_even = []
    Z_odd = []
    for key in g.keys():
        if key[-1] == 10:#beta
            print('KEY',key)
            Zeven = g[key][0][1]
            Zodd = g[key][0][0]
            Geven = g[key][1][1]
            Godd = g[key][1][0]
            Gtot = g[key][1][2]
            #plt.plot(Geven[0,0,0,:],'--',c='b',alpha=0.5,label='$G_0$')
            #plt.plot(Godd[0,0,0,:],'--',c='r',alpha=0.5,label='$G_1$')
            #plt.plot(Geven[0,0,0,:]*(Zeven/(Zeven+Zodd)),c='b',alpha=0.5,label='$G_0 \\times Z/Z_0$')
            #plt.plot(Godd[0,0,0,:]*(Zodd/(Zeven+Zodd)),c='r',alpha=0.5,label='$G_1 \\times Z/Z_1$')
            plt.plot(-Gtot[0,0,0,:],'--',c='k',alpha=0.25,label='Gtot')
            #plt.plot(-Gtot[1,1,0,:]+Gtot[2,2,0,:],'.--',c='r',alpha=0.25,label='Gtot')
            #plt.plot(-Gtot[2,2,0,:],'.--',c='r',alpha=0.25,label='Gtot')
            #plt.plot(-Gtot[4,4,0,:],'-',c='r',alpha=0.5,label='Gtot')
            #plt.plot(-Gtot[2,2,0,:],'-',c='r',alpha=0.5,label='Gtot')
            #plt.plot(-Gtot[3,3,0,:],'-',c='r',alpha=0.5,label='Gtot')
            plt.ylim([-1.5,0.1])
            plt.title('$\\mu,\\beta=$'+str(key[0])+','+str(key[-1]))
            plt.legend()
            plt.savefig('Gtest.png')
            plt.clf()
    Betas = []
    Ratios = []
    for key in g.keys():
        beta = key[-1]
        Betas.append(beta)
        #print('beta',beta,'Zodd',g[key][0][0],'Zeven',g[key][0][1])
        #print(g[key][0][0]/g[key][0][1])
        Ratios.append(g[key][0][0]/g[key][0][1])
        print('beta',beta,'Z odd',g[key][0][0],'Zeven',g[key][0][1])
    plt.plot(Betas,Ratios)
    plt.title('$\\mu=$'+str(key[0]))
    plt.ylabel('$Z_1/Z_0$')
    plt.xlabel('$\\beta$')
    plt.savefig('Ztest.png')
