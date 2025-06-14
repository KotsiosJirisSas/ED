#code to see wtf the h5 data contains
import h5py
import numpy as np

filename = "/mnt/users/kotssvasiliou/ED/core_scripts/results_L_6.h5"  # change this
filename = "/mnt/users/kotssvasiliou/ED/benchmarks2/1d/dat_Dumitru/ED_SSH_CHAIN_BENCHMARK.h5"
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
        #print(f" data:\n{obj[()]}")

with h5py.File(filename, "r") as f:
    f.visititems(print_h5_contents)

