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
'''
filename = 'NonInt.h5'
def print_h5_contents(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"\n[Group: {name}]")
        if obj.attrs:
            print(" Attributes:")
            for k, v in obj.attrs.items():
                print(f"  {k}: {v}")
    elif isinstance(obj, h5py.Dataset):
        print(f"\n[Dataset: {name}]")
        print(f" shape: {obj.shape}, dtype: {obj.dtype}")
        G0 = obj[()][0]
        G1 = obj[()][1]
        print('G0 \n',np.round(G0,5))
        print('G1 \n',np.round(G1,5))
        #print(obj[()][0].shape)
        #print(f" data:\n{obj[()]}")

with h5py.File(filename, "r") as f:
    f.visititems(print_h5_contents)
'''
#quit()
with open('Green_p_6_beta_10.0_data.pkl','rb') as f:
    Green_p_10= pickle.load(f)
G0_10 = Green_p_10['G0']
G1_10 = Green_p_10['G1']    
logZ0_10 = Green_p_10['logZ0']
logZ1_10 = Green_p_10['logZ1']
with open('Green_p_6_beta_1.0_data.pkl','rb') as f:
    Green_p_1= pickle.load(f)  
G0_1 = Green_p_1['G0']
G1_1 = Green_p_1['G1']    
logZ0_1 = Green_p_1['logZ0']
logZ1_1 = Green_p_1['logZ1']
#green
with open('Green_6_beta_10.0_data.pkl','rb') as f:
    Green_10= pickle.load(f)
Green_10 = avg_f(Green_10)
with open('Green_6_beta_1.0_data.pkl','rb') as f:
    Green_1= pickle.load(f)  
Green_1 = avg_f(Green_1)
#eta
with open('Eta_6_beta_10.0_data.pkl','rb') as f:
    Eta_10= pickle.load(f)
Eta_10 = avg_f(Eta_10)
with open('Eta_6_beta_1.0_data.pkl','rb') as f:
    Eta_1= pickle.load(f)  
Eta_1 = avg_f(Eta_1)
#spin
with open('Spin_6_beta_10.0_data.pkl','rb') as f:
    Spin_10= pickle.load(f)
Spin_10 = avg_f(Spin_10)
with open('Spin_6_beta_1.0_data.pkl','rb') as f:
    Spin_1= pickle.load(f)  
Spin_1 = avg_f(Spin_1)
##################
Green_10 = np.stack([Green_10[k] for k in range(6)], axis=0)
Spin_10 = np.stack([Spin_10[k] for k in range(6)], axis=0)
Eta_10 = np.stack([Eta_10[k] for k in range(6)], axis=0)
Green_1 = np.stack([Green_1[k] for k in range(6)], axis=0)
Spin_1 = np.stack([Spin_1[k] for k in range(6)], axis=0)
Eta_1 = np.stack([Eta_1[k] for k in range(6)], axis=0)
##################
print(np.round(Green_p_1['G0'][:,:,0],4))
#print(Green_1.shape,Green_10.shape,Eta_1.shape,Eta_10.shape)
###############
#############
params_10 = {'beta':10,
          'L':6,
          't':1.,
          'mu':1.,
          'U':4,
          'V':1.5,
        }
arrays_10 = {
    "logZ_even_spinup":logZ0_10,
    "logZ_odd_spinup":logZ1_10,
    "Green_even_spinup":G0_10[:,:,0],
    "Green_odd_spinup":G1_10[:,:,0],
    "Green":Green_10[:,0,:],
    "Eta":Eta_10[:,0,:],
    "Spin":Spin_10[:,0,:],
}
params_1 = {'beta':1,
          'L':6,
          't':1.,
          'mu':1.,
          'U':4,
          'V':1.5,
        }
arrays_1 = {
    "logZ_even_spinup":logZ0_1,
    "logZ_odd_spinup":logZ1_1,
    "Green_even_spinup":G0_1[:,:,0],
    "Green_odd_spinup":G1_1[:,:,0],
    "Green":Green_1[:,0,:],
    "Eta":Eta_1[:,0,:],
    "Spin":Spin_1[:,0,:],
}
################
with h5py.File("ED_SSE_CHAIN_BENCHMARK_final.h5", "w") as f:
    # For beta=10
    grp_10 = f.create_group("beta_10")
    param_grp_10 = grp_10.create_group("params")
    for k, v in params_10.items():
        param_grp_10.attrs[k] = v
    array_grp_10 = grp_10.create_group("arrays")
    for name, arr in arrays_10.items():
        array_grp_10.create_dataset(name, data=arr)
    #For beta = 1
    grp_1 = f.create_group("beta_1")
    param_grp_1 = grp_1.create_group("params")
    for k, v in params_1.items():
        param_grp_1.attrs[k] = v
    array_grp_1 = grp_1.create_group("arrays")
    for name, arr in arrays_1.items():
        array_grp_1.create_dataset(name, data=arr)