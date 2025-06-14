import h5py
import numpy as np
##########################
def all_close(lst, tol=1e-3):
    return all(abs(x - lst[0]) < tol for x in lst)
######
nn_bonds = [[0,1],[1,1],[1,0],[0,-1],[-1,-1],[-1,0]]
nnn_bonds = [[1,2],[2,1],[1,-1],[-1,-2],[-2,-1],[-1,1]]
nnnn_bonds = [[0,2],[2,2],[2,0],[0,-2],[-2,-2],[-2,0]]

t_bonds = [[0,1],[0,-1]]
t_perp_bonds = [[-2,-1],[2,1]]
t_nnn_bonds = [[0,2],[0,-2]]
t_else_bonds = [[-1,-1]]

hop_threshold = 0.05
int_threshold = 0.05
ratio_threshold = 2.5
strong_U_threshold = 20
##########################
data_out = {}#analyse data and save in here the relevant tidbits
with h5py.File('interaction.h5', 'r') as f:
    for key in f.keys():
        dat = f[key][()]
        for compound in dat.dtype.names:
            data_compound = dat[compound]
            #####################
            ####HOPPINGS#########
            #####################
            bonds = data_compound[1].tolist()
            bonds = [x.tolist() if isinstance(x, np.ndarray) else x for x in bonds]
            hops = data_compound[2].tolist()
            bond_max = bonds[np.argmax(np.abs(hops))]
            t_max = np.max(np.abs(hops))
            #if not (bond_max in t_bonds):
            #if not any(np.array_equal(bond_max, row) for row in np.array(t_bonds)):
                #if not any(np.array_equal(bond_max, row) for row in t_perp_bonds):
            #    print(f'skipping {compound}.. does not have expected hopping')
            #    continue
            try:
                t = hops[bonds.index(t_bonds[0])]
            except ValueError:
                t_perp = 0
            try:
                t_perp = hops[bonds.index(t_perp_bonds[0])]
            except ValueError:
                t_perp = 0
            try:
                t_nnn = hops[bonds.index(t_nnn_bonds[0])]
            except ValueError:
                t_nnn = 0
            try:
                t_else = hops[bonds.index(t_else_bonds[0])]
            except ValueError:
                t_else = 0
            #####################
            ####INTERATIONS######
            #####################
            bonds = data_compound[3].tolist()
            bonds = [x.tolist() if isinstance(x, np.ndarray) else x for x in bonds]
            Vs = [np.mean(np.abs(V)) for V in data_compound[4]]
            Vs_err = [np.std(V)/np.mean(V) for V in data_compound[4]]
            V_max = np.max(Vs)
            if not (bonds[np.argmax(Vs)] in [[0,0]]):
                raise ValueError
            if (V_max/t_max)>strong_U_threshold:
                print(f'Skipping compound{compound}... U/t is {V_max/t_max:.2f} \n')
                continue
            V_0 = []
            V_0_err = []
            V_1 = []
            V_1_err = []
            V_2 = []
            V_2_err = []
            V_3 = []
            V_3_err = []
            for i,DeltaR in enumerate(bonds):
                if DeltaR in [[0,0]]:
                    V_0.append(Vs[i])
                    V_0_err.append(Vs_err[i])
                elif DeltaR in nn_bonds:
                    V_1.append(Vs[i])
                    V_1_err.append(Vs_err[i])
                elif DeltaR in nnn_bonds:
                    V_2.append(Vs[i])
                    V_2_err.append(Vs_err[i])
                elif DeltaR in nnnn_bonds:
                    V_3.append(Vs[i])
                    V_3_err.append(Vs_err[i])
                else:
                    raise ValueError
            ##################
            # check C6 rotation symmetry
            if all_close(V_0) and all_close(V_1) and all_close(V_2) and all_close(V_3):
                Vs = np.array([V_0[0],V_1[0],V_2[0],V_3[0]])
                Vs_err = np.array([V_0_err[0],V_1_err[0],V_2_err[0],V_3_err[0]])
            else:
                print('no rotational symmetry?')
                raise ValueError
            ####################
            ####check sign problem free region
            ####################
            if Vs[0]/Vs[1] < ratio_threshold:
                print(f'Skipping compound {compound}... U/V is {Vs[0]/Vs[1]:.2f} \n')
                continue
            #######################################
            ######saving#######
            ###################
            data_out[compound] = {}
            data_out[compound]['ts'] = np.array([t,t_perp,t_nnn,t_else])
            data_out[compound]['Vs'] = Vs
            data_out[compound]['Vs err'] = Vs_err
            data_out[compound]['ts relative'] = data_out[compound]['ts']/data_out[compound]['ts'][0]#scaling by t no matter if its the max
            data_out[compound]['Vs relative'] = data_out[compound]['Vs']/data_out[compound]['ts'][0]#scaling by t no matter if its the max


##########################
#showing results
for key,dat in data_out.items():
    print('compound',key,' \n ts/t:',dat['ts relative'],'\n Vs/t:',dat['Vs relative'],' \n SU(6) deviation(%)',100*dat['Vs err'],'\n','*'*100)

##########################
'''
#counts differen dominant hoppings
with h5py.File('interaction.h5', 'r') as f:
    for key in f.keys():
        dat = f[key][()]
        index = 0
        count_0 = 0
        count_1 = 0
        count_2 = 0
        for compound in dat.dtype.names:
            if index >100:continue
            index += 1
            data_compound = dat[compound]
            if index == 1:
                vecs_init = data_compound[0]
            elif np.allclose(vecs_init,data_compound[0])== False:
                print('???')
                quit()
            #print('vectors \n',data_compound[0])
            hops = data_compound[2]
            bond_max = data_compound[1][np.argmax(np.abs(hops))]
            #print('bond max \n',bond_max)
            if any(np.array_equal(bond_max, row) for row in t_bonds):
                count_0 += 1
                print('0:',compound)
            elif any(np.array_equal(bond_max, row) for row in t_perp_bonds):
                print('1:',compound)
                #print(hops)
                count_1 += 1
            else:
                print('2:',compound)
                #print('bond max',bond_max)
                #print(compound)
                count_2 += 1
                print(data_compound[1])
                print('?',hops)
            print('-'*100)
print(count_0,count_1,count_2)
'''