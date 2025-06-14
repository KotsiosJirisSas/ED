import h5py
import numpy as np
threshold = 0.0 # ie if a quantity is 0.1 * t_max, ignore.
spin_chain_threshold = 10
spin_chain_regime = []
with h5py.File('interaction.h5', 'r') as f:
    # List all groups
    for key in f.keys():
        print('key',key)
        dat = f[key][()]
        #lattice_vcecotrs = 
        print('NUMBER OF MONOLAYERS: \n',len(dat),'\n \n')
        for monolayer in dat.dtype.names:
            dat_mono = dat[monolayer]
            #max hopping:
            t_max = np.max(np.abs(dat_mono[2]))
            #max interaction
            V_max = 0
            for i,mat in enumerate(dat_mono[4]):
                mat = mat.reshape((-1,))
                mat_avg = np.sum(mat)/mat.shape[0]
                if mat_avg>=V_max:
                    V_max = mat_avg
            ########
            if V_max/t_max>spin_chain_threshold:
                spin_chain_regime.append([monolayer,V_max/t_max])
                continue
            compound, stacking, twist, eps, screen = monolayer.split('_')
            print('*'*40,f'\n COMPOUND:{compound} & STACKING:{stacking} \n TWIST:{twist} PERMITTIVITY:{eps} SCREENING LENGTH:{screen}nm \n','*'*40,'\n')
            print(f'Umax/tmax ={V_max/t_max}')
            print('LATTICE VECTORS:',dat_mono[0],'\n')
            print('HOPPING VECTORS AND STRENGTH:')
            for i,DeltaR in enumerate(dat_mono[1]):
                if np.abs((dat_mono[2][i]/t_max))>threshold:
                    print(f'vector:{DeltaR} and relative strength:{(dat_mono[2][i]/t_max):.2f}')
            print('\n INTERACTION VECTORS AND STRENGTH:')
            for i,DeltaR in enumerate(dat_mono[3]):
                mat = dat_mono[4][i]
                mat = mat.reshape((-1,))
                mat_avg = np.sum(mat)/mat.shape[0]
                mat_error = np.std(mat)/mat_avg
                if np.abs(mat_avg/t_max)>threshold:
                    print(f'vector:{DeltaR} and relative strength:{mat_avg/t_max:.2f} and SU(6) breaking:{100*mat_error:.2g}%')
            print('='*100,'\n \n')
        #print(dat.dtype.names)
    # Access a specific dataset
    #data = f['your_dataset_name'][:]
    #print(data)
    print(f'COMPOUNDS IN THE SPIN-CHAIN REGIME({len(spin_chain_regime)} of 72):\n \n') 
    for (monolayer,ut) in spin_chain_regime:
         compound, stacking, twist, eps, screen = monolayer.split('_')
         print(f'COMPOUND:{compound} & STACKING:{stacking} \n TWIST:{twist} PERMITTIVITY:{eps} SCREENING LENGTH:{screen}nm \n  U/t ~ {ut:.2f} \n')