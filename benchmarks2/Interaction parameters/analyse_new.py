import h5py
import numpy as np
threshold = 0.05 # ie if a quantity is 0.1 * t_max, ignore.
spin_chain_threshold = 20
spin_chain_regime = []
##########################################
with open("output.txt","w") as outfile:
    with h5py.File('interaction.h5', 'r') as f:
        for key in f.keys():
            dat = f[key][()]
            for compound in dat.dtype.names:
                data_compound = dat[compound]
                # Sort t-related entries
                ts = np.abs(data_compound[2])
                t_max = np.max(ts)#sets the scale
                args_t = np.argsort(ts)[::-1]  # descending
                data_compound[1] = data_compound[1][args_t]
                data_compound[2] = data_compound[2][args_t]

                # Sort V-related entries
                Vs = [np.mean(np.abs(V)) for V in data_compound[4]]
                V_max = np.max(Vs)
                args_V = np.argsort(Vs)[::-1]
                data_compound[3] = data_compound[3][args_V]
                data_compound[4] = data_compound[4][args_V]
                ###########################################
                ########
                if V_max/t_max>spin_chain_threshold:
                    spin_chain_regime.append([compound,V_max/t_max])
                    continue
                compound_name, stacking, twist, eps, screen = compound.split('_')
                print('*'*40,f'\n COMPOUND:{compound_name} & STACKING:{stacking} \n TWIST:{twist} PERMITTIVITY:{eps} SCREENING LENGTH:{screen}nm \n','*'*40,'\n',file=outfile)
                print(f'Umax/tmax ={V_max/t_max}',file=outfile)
                #print('LATTICE VECTORS:',data_compound[0],'\n') #always the same
                print('HOPPING VECTORS AND STRENGTH:',file=outfile)
                for i,DeltaR in enumerate(data_compound[1]):
                    if np.abs((data_compound[2][i]/t_max))>threshold:
                        print(f'vector:{DeltaR} and relative strength:{(data_compound[2][i]/t_max):.2f} and absolute strength:{data_compound[2][i]:.2f}',file=outfile)
                print('\n INTERACTION VECTORS AND STRENGTH:',file=outfile)
                for i,DeltaR in enumerate(data_compound[3]):
                    mat = data_compound[4][i]
                    mat_avg = np.mean(mat)
                    mat_error = np.std(mat)/mat_avg
                    if np.abs(mat_avg/t_max)>threshold:
                        print(f'vector:{DeltaR} and relative strength:{mat_avg/t_max:.2f} and SU(6) breaking:{100*mat_error:.2g}% and absolute strength:{mat_avg:.2f}',file=outfile)
                print('='*100,'\n \n',file=outfile)
                #quit()

        print(f'COMPOUNDS IN THE SPIN-CHAIN REGIME({len(spin_chain_regime)} of 72):\n \n',file=outfile) 
        for (monolayer,ut) in spin_chain_regime:
            compound, stacking, twist, eps, screen = monolayer.split('_')
            print(f'COMPOUND:{compound} & STACKING:{stacking} \n TWIST:{twist} PERMITTIVITY:{eps} SCREENING LENGTH:{screen}nm \n  U/t ~ {ut:.2f} \n',file=outfile)
