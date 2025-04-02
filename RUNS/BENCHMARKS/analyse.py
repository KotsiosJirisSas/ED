'''
take benchmark pickles and turn into .txt for Dumitru's ease
'''
import pickle 
text_file = 'output.txt'
pickle_files = ['data_0.0_.pkl','data_-0.5_.pkl','data_0.5_.pkl','data_-1.0_.pkl','data_1.0_.pkl','data_-1.5_.pkl','data_1.5_.pkl','data_-2.0_.pkl','data_2.0_.pkl']
Es_GS = [-6.06490013848496,23.15296940595781,-36.84703059404219,49.3100736314966,-70.68992636850341,72.42225296055265,-107.57774703944735,92.19193812526039,-147.80806187473968]
for n,pickle_file in enumerate(pickle_files):
    E_GS = Es_GS[n]
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        with open(text_file, "a") as file:
            file.write(f"parameters:\t{ data['params']['H_params']}\n")
            file.write("beta\t <H> \t <H>^2 -<H>^2\t <n> \t <n^2> - <n>^2\n")  # Header
            for i in range(len(data['betas'])):
                beta = data['betas'][i]
                H = data['Es'][0,i] - E_GS
                Hsq = data['Es'][1,i] - data['Es'][0,i]**2
                N = data['Ns'][0,i]
                Nsq = data['Ns'][1,i] - data['Ns'][0,i]**2
                file.write(f"{beta}\t{H} \t{Hsq}\t{N} \t{Nsq} \n")  # Tab-separated values
            file.write("*"*100+"\n")
