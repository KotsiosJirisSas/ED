#takes pkl data and puts it into h5 to send
#code to see wtf the h5 data contains
import h5py
import numpy as np
import pickle
def avg_f(dat,L = 6):
    C_r = {}
    for r1 in range(L):
        for r2 in range(L):
            r = (r1 - r2) % L  
            if r not in C_r:
                C_r[r] = []
            C_r[r].append(dat[r1, r2, :])

    for r in C_r:
            avg = np.zeros_like(C_r[r][0])
            for dat in C_r[r]:
                avg += dat
            avg *= 1./L
            C_r[r][0] = avg
            del C_r[r][1:]
    return C_r
with open('params.pkl','rb') as f:
    params = pickle.load(f)
with open('Green_p_6_data.pkl','rb') as f:
    Green_p = pickle.load(f)
with open('Green_6_data.pkl','rb') as f:
    Green = pickle.load(f)
with open('Spin_6_data.pkl','rb') as f:
    Spin = pickle.load(f)
with open('Eta_6_data.pkl','rb') as f:
    Eta = pickle.load(f)
G0 = Green_p['G0']
G1 = Green_p['G1']
logZ0 = Green_p['logZ0']
logZ1 = Green_p['logZ1']
print('')
print('G0 \n',np.round(G0[:,:,0],3))
print('G1 \n',np.round(G1[:,:,0],3))
print('G \n',np.round(Green[:,:,0],3))
print('Eta \n',np.round(Eta[:,:,0],3))
print('Spin \n',np.round(Spin[:,:,0],3))
Eta = avg_f(Eta,L = 6)
Spin = avg_f(Spin,L = 6)
Green = avg_f(Green,L = 6)

#print('G0 keys',G0.keys())
#Green_P_0 = np.stack([G0[k] for k in range(6)], axis=0)
#Green_P_1 = np.stack([G1[k] for k in range(6)], axis=0)
Green = np.stack([Green[k] for k in range(6)], axis=0)
Spin = np.stack([Spin[k] for k in range(6)], axis=0)
Eta = np.stack([Eta[k] for k in range(6)], axis=0)
#print('green 0 shape',Green_P_0.shape)
#print('green shape',Green_P_1.shape)
print('green shape',Green.shape)
print('eta shape',Eta.shape)
print('Spin spahe',Spin.shape)
params = {'beta':10,
          'L':6,
          't':1.,
          'mu':1.,
          'U':4,
          'V':1.5,
        }
print('params',params)
arrays = {
    "logZ_0":logZ0,
    "logZ_1":logZ1,
    "Green_even":G0[:,:,0],
    "Green_odd":G1[:,:,0],
    "Green":Green[:,0,:],
    "Eta":Eta[:,0,:],
    "Spin":Spin[:,0,:],
}


with h5py.File("ED_SSE_CHAIN_BENCHMARK_GREENSFUNCS.h5", "w") as f:
    param_grp = f.create_group("params")
    for k, v in params.items():
        param_grp.attrs[k] = v
    # Save arrays as datasets
    array_grp = f.create_group("arrays")
    for name, arr in arrays.items():
        array_grp.create_dataset(name, data=arr)

#####
#print('green',Green)
#print('spin',Spin)
#print('eta',Eta)
