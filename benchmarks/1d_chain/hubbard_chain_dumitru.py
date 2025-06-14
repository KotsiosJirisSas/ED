'''
here i want to benchmark the hubbard chain Energy and Green's function code with Dumitru's SSE code

Complications: 
    The SSE approach uses essentially hardcore boson formulation so to go back to fermionic results one has to add some fermion strings when going accross the boundary
'''
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import gc
from scipy.special import logsumexp
import pickle
import os
sys.path.append('/mnt/users/kotssvasiliou/ED/utils')
from hubbard_1d import hubbard_chain,thermodynamics
import fcntl
########################################################
######## PARITY PARTITION FUNCTION COMPARISON ##########
########################################################
def parity_Z_plot(params):
    '''
    plots Z_odd/Z_even as a function of beta, at a given filling.
    -------------------------------------------------------------
    params example:
    L=6,t=1,U=4,V=1.5,mu=0,SGN=False
    params = {'L':L,'sign':SGN,'beta':beta,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    '''
    L = params['L']
    t = params['H_params']['t']
    U = params['H_params']['U']
    V = params['H_params']['V']
    mu = params['H_params']['mu']
    SGN = params['sign']
    ####################
    #open text file#
    ####################
    '''
    with open(text_file, "a") as file:
        file.write(f"#\n#\n#\n")  # Empty lines for spacing
        file.write(f"# Parameters:\n")
        file.write(f"# L = {L}, t = {t}, U = {U}, V = {V}, mu = {mu}, sign = {SGN}\n")
        file.write(f"#\n#\n#\n")  # Empty lines for spacing'
    '''
    ###################
    # run#
    ###################
    DATA = {}
    thermo = thermodynamics(params)
    #quit()
    Es = []
    Ns = []
    Betas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]+[1.5,2.0,2.5,3,3.5,4,4.5,5]+[6,7,8,9,10]
    '''
    with open(text_file, "a") as file:
        file.write("beta\t E \t <n> \n")  # Header
        for beta in Betas:
            E = thermo.Energy(beta=beta)
            N = thermo.OccNum(beta=beta)
            file.write(f"{beta}\t{E} \t{N} \n")  # Tab-separated values
        file.write("*"*100)'
    '''
    ##############################
    ######## Z1/Z0 #####
    ##############################
    ratio = np.zeros_like(np.array(Betas))
    for i,beta in enumerate(Betas):
        logZodd = thermo.partition_function_partial(beta=beta,parity=1)
        logZeven = thermo.partition_function_partial(beta=beta,parity=0)
        logZ = thermo.partition_function(beta=beta)
        #print(np.exp(logZodd)+np.exp(logZeven)-np.exp(logZ))
        print('ratio for beta',beta,':',np.exp(logZodd),np.exp(logZeven),np.exp(logZodd)/np.exp(logZeven))
        ratio[i] = np.exp(logZodd)/np.exp(logZeven)
    plt.plot(np.array(Betas)*U,ratio,'.')
    #plt.plot(np.array(Betas)*U,np.exp(-0.15*np.array(Betas)*U),c='k')
    #plt.plot(np.array(Betas)*U,np.exp(-0.18*np.array(Betas)*U),c='r')
    plt.xlabel('$\\beta U$')
    plt.title('$Z_1/Z_0$')
    plt.ylim([-0.1,1.1])
    plt.savefig('Z_comparison.png',dpi=500)
    return

########################################################
############# TEST TRANSLATION INVARIANCE ##############
########################################################
def test_spinspin_correlator(params,parity=0):
    '''
    Tests the <S^+S^-> correlator on the chain
    '''
    thermo = thermodynamics(params)
    thermo.test_ss_mapping(sector_test=(3,2))
    G = thermo.Spin_correlation_function(beta=5,n_tau=10,parity=parity)
    for i in range(10):
        print('\n \n ')
        print(G[:,:,i])
    #quit()
    ####################
    #check translational invariance
    G_r = {}
    for x in range(params['L']):
        for y in range(params['L']):
            r = np.abs(x-y)
            if r>=3:
                r= params['L']-r
            if r not in G_r.keys():
                G_r[r] = []
                G_r[r].append(G[x,y,:])
            else:
                if not np.allclose(G[x,y,:],G_r[r][0]):
                    G_r[r].append(G[x,y,:]) 
    for r in G_r.keys():
        print('location',r,'SS correlator',G_r[r])
    ####################
    #plot#
    taus = np.linspace(0,1,10)
    for r,f in G_r.items():
        plt.plot(taus,f[0][:],'.-',label='$r=$'+str(r),alpha=0.5)
    plt.ylabel('$\\langle S^+(\mathbf{r},\\tau)S^-(0,0)\\rangle$') 
    plt.xlabel('$\\tau/\\beta$')
    plt.legend()
    #plt.title('L='+params['L']+'$\\beta$='+str(5)+'($U/t,V/t,\mu/t$)='+str(params['H_params']['U'])+str(params['H_params']['V'])+str(params['H_params']['mu']))
    plt.savefig('/mnt/users/kotssvasiliou/ED/benchmarks/1d_chain/spin_correlator.png',dpi=500)
    return
def test_translation_inv(params,sector,parity):
    '''
    prints results for the Green's function at different distances on the chain, to check if translation invariance is satisfied
    '''
    L = params['L']
    t = params['H_params']['t']
    U = params['H_params']['U']
    V = params['H_params']['V']
    mu = params['H_params']['mu']
    SGN = params['sign']

    #####################
    thermo = thermodynamics(params)
    Es = []
    for key,value in  thermo.energies.items():
        for e in value:
            Es.append(e+thermo.lowestEnergy)
    #print(np.sort(np.array(Es)))
    #quit()
    #print('energies',thermo.energies[(3,2)]+thermo.lowestEnergy)
    #print('states',thermo.eigenstates[(3,2)])
    #quit()
    ##############################
    thermo.create_mapping()
    G = thermo.GreenFuncDebug(beta=params['beta'],n_tau=100,sector=sector,parity=parity)
    #G = thermo.Spin_correlation_debug(beta=params['beta'],n_tau=100,sector=sector)
    plt.figure()
    tau_vals = np.linspace(0,1,100)
    for r in range(L):
            plt.plot(tau_vals, G[r,:], '-', label=f"fer G({r},{r})",alpha=0.5)
    plt.legend()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$G_{rr}(\tau)$')
    plt.tight_layout()
    plt.savefig('temp.png')
    print('?')
    return
########################################################
##################### PARAMETER RUN ####################
########################################################
def parameter_run(params,text_file,file_path = '/mnt/users/kotssvasiliou/ED/benchmarks/Greens_func_benchmark_data_L_6_new.pkl'):
    '''
    Does Green's function calculation for one set of parameters. Run by doing smth like:
    ####
    V = float(sys.argv[1])
    mu = float(sys.argv[2])
    params = {'L':L,'sign':SGN,'beta':4.0,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    parameter_run(params,text_file='benchmark_data_new.txt',file_path = '/mnt/users/kotssvasiliou/ED/benchmarks/Greens_func_benchmark_data_L_6_new.pkl')
    ####

    text_file:drops results fro static observables in a text file
    file_path:greens function pickle output location
    ----------------------------------
    example of params input:
    L=6,t=1,U=4,V=1.5,mu=0,SGN=False
    params = {'L':6,'sign':False,'beta':4,'H_params':{'t':1,'mu':0,'U':4,'V':1.5},'diag_params':{'mode':'full'}}
    '''
    def safe_append_pickle(file_path, new_dict):
        '''
        pickles data and is safe in case of access at the same time
        '''
        # Ensure file exists
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                pickle.dump([], f)

        with open(file_path, 'r+b') as f:
            # Lock file exclusively
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                data = pickle.load(f)
                data.append(new_dict)
                f.seek(0)
                pickle.dump(data, f)
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    L = params['L']
    t = params['H_params']['t']
    U = params['H_params']['U']
    V = params['H_params']['V']
    mu = params['H_params']['mu']
    SGN = params['sign']
    ####################
    #open text file#
    ####################
    with open(text_file, "a") as file:
        file.write(f"#\n#\n#\n")  # Empty lines for spacing
        file.write(f"# Parameters:\n")
        file.write(f"# L = {L}, t = {t}, U = {U}, V = {V}, mu = {mu}, sign = {SGN}\n")
        file.write(f"#\n#\n#\n")  # Empty lines for spacing
    ###################
    # run#
    ###################
    DATA = {}
    thermo = thermodynamics(params)
    #quit()
    Es = []
    Ns = []
    Betas = [1,2,3,4,5,6,7,8,9,10]
    for beta in Betas:
        Es.append(thermo.Energy(beta=beta))
        Ns.append(thermo.OccNum(beta=beta))
    DATA['params'] = params
    DATA['betas'] = Betas
    DATA['H_avg'] = Es
    DATA['n_avg'] = Ns
    with open(text_file, "a") as file:
        file.write("beta\t E \t <n> \n")  # Header
        for beta in Betas:
            E = thermo.Energy(beta=beta)
            N = thermo.OccNum(beta=beta)
            file.write(f"{beta}\t{E} \t{N} \n")  # Tab-separated values
        file.write("*"*100)
    ##############################
    ######## GREENS FUNCTION #####
    ##############################
    green_func_calc = True
    if green_func_calc == True:
        G_data = {}
        timei = time.time()
        thermo.create_mapping()
        for beta in Betas:
            timei = time.time()
            Zodd = np.exp(thermo.partition_function_partial(beta=beta,parity=1))
            Zeven = np.exp(thermo.partition_function_partial(beta=beta,parity=0))
            Ztot = np.exp(thermo.partition_function(beta=beta))
            Godd = thermo.GreenFunc_partial(beta=beta,n_tau=5,parity=1)
            Geven = thermo.GreenFunc_partial(beta=beta,n_tau=5,parity=0)
            Gtot = thermo.GreenFunc(beta=beta,n_tau=5)
            
            G_data[(params['H_params']['mu'],params['H_params']['V'],beta)] = [[Zodd,Zeven,Ztot],[Godd,Geven,Gtot]]
            timef = time.time()
            print('Gfunc time','beta',beta,'time',timef-timei)
        #########################
        #### save data ######
        #########################
        print('size of mapping',sys.getsizeof(thermo.matrix_elements_up))
        # Load if exists, else start with empty list
        print('SAVING IN PICKLEEEE')
        safe_append_pickle(file_path = file_path,new_dict = G_data)
        return
#######################################################

########################################################
#testing the V nn repulsion
def nn_repulsion_test(params):
    '''
    tests nn repulsion function
    ---------------------------
    example input:
    params = {'L':L,'sign':SGN,'H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
    '''
    L = params['L']
    t = params['H_params']['t']
    U = params['H_params']['U']
    V = params['H_params']['V']
    mu = params['H_params']['mu']
    SGN = params['sign']
    for n_up in range(0,L+1):
     for n_down in range(0,L+1):
            if (n_up,n_down) == (3,0):
                #print('=='*20)
                #print('Diagonalizing sector',n_up,n_down)
                #print('=='*20)
                params['Nup'] = n_up
                params['Ndn'] = n_down
                chain = hubbard_chain(params)
                chain.build_ham()
                print(chain.Hamiltonian)
                #
                for m_up in chain.basis_up:
                    for m_dn in chain.basis_dn:
                        I_up = chain.basis_up[m_up]
                        I_dn = chain.basis_dn[m_dn]
                        I_physical = I_up+I_dn*(2**L)
                        print(I_physical,chain.binp(I_physical,int(2*L)))
####################
def collect_data(params,filename=''):
    return

def test_spectral_func(params):
    L = params['L']
    t = params['H_params']['t']
    U = params['H_params']['U']
    V = params['H_params']['V']
    mu = params['H_params']['mu']
    SGN = params['sign']

    #####################
    thermo = thermodynamics(params)
    Es = []
    for key,value in  thermo.energies.items():
        for e in value:
            Es.append(e+thermo.lowestEnergy)
    #####################
    thermo.create_mapping()
    A = thermo.spectral_function()
    #G = thermo.Spin_correlation_debug(beta=params['beta'],n_tau=100,sector=sector)
    plt.figure()
    tau_vals = np.linspace(0,1,100)
    for r in range(L):
            plt.plot(tau_vals, G[r,:], '-', label=f"fer G({r},{r})",alpha=0.5)
    plt.legend()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$G_{rr}(\tau)$')
    plt.tight_layout()
    plt.savefig('temp.png')
    print('?')

def charge_gap(params):
    '''
    prints the single particle charge gap, to be compared to DQMC
    '''
    L = params['L']
    t = params['H_params']['t']
    U = params['H_params']['U']
    V = params['H_params']['V']
    mu = params['H_params']['mu']
    SGN = params['sign']
    ###################
    ########RUN########
    ###################
    DATA = {}
    thermo = thermodynamics(params)

#####################
if __name__ == "__main__":
    params = {'L':3,'loc':0,'H_params':{'t':1,'mu':0,'U':0,'V':0},'diag_params':{'mode':'full'}}
    params['verbose'] = 1
    params['beta'] = 1
    params['sign'] = True
    params['JW string'] = False
    #params['sign'] = False
    #params['JW string'] = True
    #test_spinspin_correlator(params,parity=None)
    test_translation_inv(params,sector=[(1,0)],parity=None)
    