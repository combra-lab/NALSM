import NALSM_GEN_SUPPORT as sup
import numpy as np
import os


SAVE_VER_INP = input('VERSION? [int]: ')
SAVE_VER = int(SAVE_VER_INP)
VER =  SAVE_VER


main_Path = os.getcwd()
data_path = main_Path + '/train_data/ver_' + str(SAVE_VER)
net_path = main_Path + '/networks'

for file in os.listdir(data_path):
    if file.endswith('.bf'):
        bf_file = file


names, data = sup.unpack_file(filename=bf_file, dataPath=data_path)
all_data = data[names.index('all_data')]


samples_bf_l=[]
for s in range(0,len(all_data)):
    neuron_ave_bf_l = []
    neuron_std_bf_l = []
    for n in range(0,len(all_data[s])):
        neuron_ave_bf_l.append(all_data[s][n][5])
    samples_bf_l.append(np.nanmean(neuron_ave_bf_l))


print('VERSION: '+str(SAVE_VER))
print('BRANCHING FACTOR: '+str(np.nanmean(samples_bf_l)))



