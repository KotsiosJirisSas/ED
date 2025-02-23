import numpy as np
# Define parameter range (modify as needed)
tot = 400
param_values = np.linspace(1, tot, tot,dtype=int) 
# Path to the output file
params_file = "/mnt/users/kotssvasiliou/ED/new_dir_66ec702c/params.txt"
# Generate the command list
with open(params_file, "w") as f:
    for param in param_values:
        command = f"python /mnt/users/kotssvasiliou/ED/utils/chain_sections.py exe_multirun_ED {int(param)} {tot}\n"
        f.write(command)
print(f"Generated {params_file} with {len(param_values)} commands.")