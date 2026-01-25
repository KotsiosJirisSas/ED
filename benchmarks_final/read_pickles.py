import pickle
import numpy as np
from pathlib import Path
import h5py
#filename = '1stbatch/U_5.0_V_1.0_mu_-10.0_sgn_False.pkl'
#U = filename.split('_')[0][:-3]
#with open(filename,mode='rb') as f:
#    data = pickle.load(f)
#print(filename)
#print('beta|     E       |   E_fluct    |   N  |   N_sq')
#for i in range(10):
#    print(np.round(data['betas'][i],2),'|',np.round(data['Es'][i],10),'|',np.round(data['Es_sq'][i]-data['Es'][i]**2,10),'|',np.round(data['Ns'][i],10),'|',np.round(data['Ns_sq'][i]-data['Ns'][i]**2,10))
#print('')
###############
#save as h5####
pkl_dir = Path('2ndbatch')
with h5py.File('dat_new.h5', "w") as h5:
    for p in sorted(pkl_dir.glob("*.pkl")):
        #if str(p).split('_')[1] == '5.0':
        #    print('skipping pickle',p)
        #    continue
        #print('keeping pickle',str(p))
        with open(p, "rb") as f:
            d = pickle.load(f)          # expects keys 'x' and 'y'
        betas = np.asarray(d["betas"])
        Es = np.asarray(d["Es"])
        Es_sq = np.asarray(d["Es_sq"])
        Ns = np.asarray(d["Ns"])
        Ns_sq = np.asarray(d["Ns_sq"])
        Es_fluct = Es_sq - Es**2
        Ns_fluct = Ns_sq - Ns**2

        g = h5.create_group(p.stem)    # one group per pickle
        g.create_dataset("Betas", data=betas, compression="gzip", compression_opts=4, chunks=True)
        g.create_dataset("Es", data=Es, compression="gzip", compression_opts=4, chunks=True)
        g.create_dataset("Ns", data=Ns, compression="gzip", compression_opts=4, chunks=True)
        g.create_dataset("E fluctuations", data=Es_fluct, compression="gzip", compression_opts=4, chunks=True)
        g.create_dataset("N fluctuations", data=Ns_fluct, compression="gzip", compression_opts=4, chunks=True)
with h5py.File("dat_new.h5", "r") as h5:
    for name in h5:
        print('anme',name)
        x = h5[name]["Betas"][...]  # numpy arrays
        y = h5[name]["E fluctuations"][...]
        print('name',name)
        print('beta',x,'flucts',y)