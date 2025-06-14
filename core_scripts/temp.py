import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import gc
from scipy.special import logsumexp
import pickle
from collections import defaultdict
import h5py
with h5py.File('results_L_6.h5', 'r') as f:
    # Load params
    param_grp = f['params']
    print("Params (attributes):")
    for k in param_grp.attrs:
        print(f"  {k}: {param_grp.attrs[k]}")

    print(f"  taus shape: {param_grp['taus'].shape}")  # assuming 'taus' was stored as dataset

    # Loop over groups like G_0, G_1, etc.
    for group_name in ['G_0', 'G_1', 'Greens', 'SpinSpin', 'Pairing']:
        grp = f[group_name]
        print(f"\nGroup '{group_name}':")
        for k in grp:
            data = grp[k]
            print(f"  {k}: shape {data.shape}")

    # Load standalone datasets
    for name in ['logZ_0', 'logZ_1']:
        print(f"\n{name}: shape {f[name].shape}")