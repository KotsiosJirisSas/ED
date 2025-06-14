import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import gc
from scipy.special import logsumexp
import pickle
from collections import defaultdict
import hubbard_chain
L = 3
t = 1;U = 10;V = 1.1*0;mu = -1.14*0
params = {'L':L,'verbose':0,'species':'mboson','H_params':{'t':t,'mu':mu,'U':U,'V':V},'diag_params':{'mode':'full'}}
params['Nup'] = 1
params['Ndn'] = 0
chain = hubbard_chain.hubbard_chain(params)
chain.build_ham()
print(chain.Hamiltonian)
lam,v = np.linalg.eigh(chain.Hamiltonian)
print('es',lam)