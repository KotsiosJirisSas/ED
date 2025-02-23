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

    # Extract x values (keys) and Y values (first row of each stored 2x3 array)
    x_values = np.array(sorted(data.keys()))  # Ensure x values are sorted
    Y_values = np.array([data[x][0] for x in x_values])  # Extract first row (shape: len(x_values) x 3)

    x_values = -3+6*x_values/400

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, Y_values[:, 0],'.',alpha=0.5,c='r',label='1')
    plt.plot(x_values, Y_values[:, 1], '.',alpha=0.5,c='g',label="6")
    plt.plot(x_values, Y_values[:, 2],'.',c='b',alpha=0.5,label='30')

    # Formatting
    plt.xlabel("$\mu/U$",fontsize=12)
    plt.ylabel("$\\langle \hat{N} \\rangle $",fontsize=12)
    plt.title("$U=6,t=1,V=0$",fontsize=12)
    plt.yticks([0,4,8,12,16,20,24])
    plt.legend()
    plt.grid(axis='both')

    plt.savefig('ec_new.png')

# Example usage:
analyze_and_plot_data("final_data.pkl")  # Assuming data.pkl follows the described format
