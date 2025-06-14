import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from scipy.sparse import csr_matrix,coo_matrix,kron,identity #optimizes H . v operations. to check if H already row sparse, do  isspmatrix_csr(H)
from scipy.sparse.linalg import eigsh
import time
import pickle
import argparse
from itertools import product,combinations
from math import comb,log10
from collections import Counter
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import random
from scipy.special import logsumexp
from scipy.sparse import SparseEfficiencyWarning
import warnings
from numba import njit
import random
from functools import lru_cache
import h5py
from collections import defaultdict
import ast
sys.path.append('/mnt/users/kotssvasiliou/ED/utils')
from ED_chains_final import chain_configs
parameters = {'L':2,
                  'geometry':'triangular',
                  't':1,
                  'mu':0,
                  'U':4,
                  'V':1,
                  'partial':False,
                  'projection':True,
                  'sign':True,
                  'JW string':True,
                  'mode':'full'}
L = parameters['L']
t = parameters['t']
U = parameters['U']
V = parameters['V']
mu = parameters['mu']
geometry = parameters['geometry']
projection = parameters['projection']
sign = parameters['sign']
JWstring = parameters['JW string']
mode = parameters['mode']
system_params = {'geometry':geometry,'L':L,'partial': parameters['partial'],'projection':projection}
system_params = {'geometry':geometry,'L':L,'partial': parameters['partial'],'projection':projection,'Nel_min':2,'Nel_max':6}
print('='*100+'\n GENERATING ALL SYMMETRY SECTORS FOR L='+str(L)+'in '+geometry+'geometry'+' \n'+'='*100)
timei = time.time()
CCs = chain_configs(params=system_params)
CCs.sector_check()
config_data = CCs.compressed_data
timef = time.time()
print('time to generate sectors:',timef-timei,' secs')
if geometry=='square': secs_tot = (L+1)**(4*L); order = 2*4*L**2
elif geometry=='triangular': secs_tot = (L+1)**(6*L); order = 2*3*L**2
rep_secs = len(config_data)
print('information on symmetry sectors:\n Total sectors: '+str(secs_tot)+'\n Number of representative sectors: '+str(rep_secs)+'\n Optimal number of representative sectors: '+str(int(secs_tot/order)))