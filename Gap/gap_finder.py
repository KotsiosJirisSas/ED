'''
Sets up runs that only calculate <H>,<H^2>,<N>,<N^2>; not greens functions
'''
import sys
import numpy as np
sys.path.append('/mnt/users/kotssvasiliou/ED/core_scripts')
sys.path.append('/mnt/users/kotssvasiliou/ED/utils')
import ED_chains_final_final_anis
import time
import pickle
import os
import matplotlib.pyplot as plt
def ED_gap(parameters,verbose=1):
    '''
    Docstring for ED_gap
    
    :param parameters: Description
    '''
    L = parameters['L'];t = parameters['t'];U = parameters['U'];V = parameters['V'];mu = parameters['mu'];anis = parameters['anis'];geometry = parameters['geometry']
    partial = parameters['partial'];projection = parameters['projection'];sign = parameters['sign'];mode = parameters['mode']
    #mu = 0 #half-filling; for 1electron per site, mu=-3U and for two, mu=-1.5U
    system_params = {'geometry':geometry,'L':L,'partial':partial,'projection':projection}
    sector_params = {'L':L,'geometry':geometry,'sign':sign,'H_params':{'t':t,'mu':mu,'U':U,'V':V,'anis':anis},'diag_params':{'mode':mode}}
    if verbose>0:print('='*100+'\n GENERATING ALL SYMMETRY SECTORS \n'+'='*100)
    timei = time.time()
    CCs = ED_chains_final_final_anis.chain_configs(params=system_params)
    CCs.sector_check()
    config_data = CCs.compressed_data
    timef = time.time()
    if verbose>0:
        print('time to generate sectors:',timef-timei,' secs')
        if geometry=='square': secs_tot = (L+1)**(4*L); order = 2*4*L**2
        elif geometry=='triangular': secs_tot = (L+1)**(6*L); order = 2*3*L**2
        rep_secs = len(config_data)
        print('information on symmetry sectors:\n Total sectors: '+str(secs_tot)+'\n Number of representative sectors: '+str(rep_secs)+'\n Optimal number of representative sectors: '+str(int(secs_tot/order)))
    ##
    #ED
    ##
    if verbose>0:print('='*100+'\n PERFORMING EXACT DIAGONALIZATION ON ALL REPRESENTATIVE SYMMETRY SECTORS \n'+'='*100)
    itime = time.time()
    combined_data = {}
    for k in config_data.keys():
        sector_params['config'] = k
        chain_instance = ED_chains_final_final_anis.chains(sector_params)
        diag_states = chain_instance.diagonalization()
        combined_data[k] = {}
        combined_data[k]['es'] = diag_states['es']
        combined_data[k]['vs'] = diag_states['vs']
        combined_data[k]['ns'] = diag_states['ns']
        combined_data[k]['equivalent sectors'] = config_data[k]
        combined_data[k]['weight'] = len(config_data[k][0])
    ftime = time.time()
    if verbose>0:print('time to perform ED:',ftime-itime,' seconds \n'+'='*100)
    ###############################
    #   Initialize thermodynamic  #
    #   calculation instance      #
    ###############################
    parameters['loc'] = chain_instance.loc
    parameters['verbose'] = 1
    parameters['greens function'] = True
    thermo = ED_chains_final_final_anis.thermodynamics(parameters=parameters,combined_data=combined_data)
    gap = thermo.charge_gap()
    fftime = time.time()
    if verbose>0:print('time to init thermodynamics:',fftime-ftime,' seconds \n'+'='*100)
    return gap
#######
if __name__ == "__main__":
    V = 0.0
    mu = 0.0
    Us = np.linspace(start=0.1,stop=20,num=30)
    Gaps =[]
    for U in Us:
        parameters = {'L':2,
                    'geometry':'triangular',
                    't':1.,
                    'mu':mu,
                    'U':U,
                    'V':V,
                    'anis':1,
                    'partial':False,
                    'projection':False,
                    'sign':False,
                    'mode':'full'}
        #print('params',parameters)
        results = ED_gap(parameters=parameters,verbose=0)
        Gaps.append(results)
        print('U',U,'gap',results)
Gaps = np.array(Gaps)
plt.plot(Us,Gaps,'.')
plt.savefig('gaps_triangular_iso.png',dpi=200)
