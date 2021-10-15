import numpy as np
import NALSM_GEN_SUPPORT as sup
import os

SAVE_VER_INP = input('VERSION? [int]: ')
SAVE_VER = int(SAVE_VER_INP)

LEN = -1

main_Path = os.getcwd()
dataPath_TRAIN = main_Path + '/train_data'
netPath = main_Path + '/networks'


PATH = dataPath_TRAIN+'/ver_' + str(SAVE_VER) + '/'

filename = 'VER_'+str(SAVE_VER)+'_params.params'
names, data = sup.unpack_file(filename=filename, dataPath=PATH)

filename = 'FILE_L_VAL_ACC_L.files'
names, data = sup.unpack_file(filename=filename,dataPath=PATH)

ACUM_VAL_ACC_l = data[names.index('ACUM_VAL_ACC_l')][0:LEN]
max_val_acc_idx = np.argmax(np.asarray(ACUM_VAL_ACC_l))

ACUM_TST_ACC_l = data[names.index('ACUM_TST_ACC_l')][0:LEN]

print('TEST ACCURACY FOR VERSION: '+str(SAVE_VER))
print('ACCURACY: '+str(ACUM_TST_ACC_l[max_val_acc_idx]))

print('TEST ACC AT PEAK VALIDATION,VAL PEAK, TRAIN: ')
