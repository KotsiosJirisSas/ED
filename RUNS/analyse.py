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

def analyze_and_plot_data2(dir):
    """
    Reads a pickle file where keys represent the x-axis values, and values are 2x3 numpy arrays.
    Generates a plot with three lines corresponding to Y[0,0], Y[0,1], and Y[0,2].

    Parameters:
    pickle_file (str): Path to the pickle file containing the data.
    """
    # Load the data from the pickle file
    
    file1 = dir+str('/data_1.pkl')
    file2 = dir+str('/data_2.pkl')
    file3 = dir+str('/data_3.pkl')
    file0 = dir+str('/data.pkl')
    #files = [file1,file2,file3]
    #files = [file3,file2,file1]
    #files = [file1,file3,file2]
    files = [file0]
    betas = []
    Es = []
    Ns = []
    Es_sq = []
    Ns_sq = []
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        betas.append(data['betas'])  
        Es.append(data['Es'][0,:])
        Ns.append(data['Ns'][0,:])
        Es_sq.append(data['Es'][1,:])
        Ns_sq.append(data['Ns'][1,:])
        t = data['params']['H_params']['t']
    betas = np.concatenate(np.array(betas))
    print(betas)
    Es = np.concatenate(np.array(Es))
    Es_sq = np.concatenate(np.array(Es_sq))
    Ns = np.concatenate(np.array(Ns))
    Ns_sq = np.concatenate(np.array(Ns_sq))
    for i,beta in enumerate(betas):
        if np.round(beta,2) == 4.90:
            print(i,beta,Ns_sq[i]-Ns[i]**2)
        if np.round(beta,2) == 4.60:
            print(i,beta,Ns_sq[i]-Ns[i]**2)
        if np.round(beta,2) == 4.06:
            print(i,beta,Ns_sq[i]-Ns[i]**2)
    quit()
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(np.log10(1/(betas*t)), Es,'.-',alpha=0.5,c='r',label='$\delta E$')
    plt.plot(np.log10(1/(betas*t)), Ns - Ns[-1],'.-',alpha=0.75,c='purple',label='$\delta N$')
    plt.plot(np.log10(1/(betas*t)),((betas)**2)*(Es_sq - Es**2),'.-',alpha=0.5,c='b',label='$c$')
    plt.plot(np.log10(1/(betas*t)),(Ns_sq - Ns**2),'.-',alpha=0.5,c='green',label='$\\chi_c$')

    # Formatting
    plt.xlabel("$T/t$",fontsize=12)
    plt.title('$\\nu = 2$',fontsize=12)
    #plt.title("$U=6,t=1,V=1$",fontsize=12)
    #plt.yticks([0,4,8,12,16,20,24])
    plt.xticks([-4,-3,-2,-1,0,1,2],['$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^0$','$10^{1}$','$10^{2}$'])
    plt.legend()
    #plt.grid(axis='both')

    # Show the plot
    file_out = dir + '/ec_new.svg'
    plt.savefig(file_out,dpi=600)

# Example usage:
#analyze_and_plot_data("data.pkl")  # Assuming data.pkl follows the described format
analyze_and_plot_data2("L_2_half_filling")
#analyze_and_plot_data2("L_2_Ne_8")
#analyze_and_plot_data2("L_2_Ne_8")
