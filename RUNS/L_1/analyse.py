import pickle
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_plot_data(pickle_file):
    """
    Reads a pickle file where keys represent the x-axis values, and values are 2x3 numpy arrays.
    Generates a plot with three lines corresponding to Y[0,0], Y[0,1], and Y[0,2].

    Parameters:
    pickle_file (str): Path to the pickle file containing the data.
    """
    # Load the data from the pickle file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    betas = np.array(data['betas'])  # Ensure x values are sorted
    Es = np.array(data['Es'][0,:])
    E_sq_s = np.array(data['Es'][1,:])
    Ns = np.array(data['Ns'][0,:])
    N_sq_s = np.array(data['Ns'][1,:])

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(1/betas, Es,'.',alpha=0.5,c='r',label='1')
    plt.plot(1/betas,((betas)**2)* (E_sq_s - Es**2),'.',alpha=0.5,c='b',label='s')
    #plt.plot(x_values, Y_values[:, 1], '.',alpha=0.5,c='g',label="6")
    #plt.plot(x_values, Y_values[:, 2],'.',c='b',alpha=0.5,label='30')

    # Formatting
    plt.xlabel("$\mu/U$",fontsize=12)
    plt.ylabel("$\\langle \hat{N} \\rangle $",fontsize=12)
    plt.title("$U=6,t=1,V=1$",fontsize=12)
    #plt.yticks([0,4,8,12,16,20,24])
    #plt.xticks([-4,-3,-2,-1,0,1,2,3,4])
    plt.legend()
    #plt.grid(axis='both')

    # Show the plot
    plt.savefig('ec.png')


# Example usage:
#analyze_and_plot_data("data.pkl")  # Assuming data.pkl follows the described format
