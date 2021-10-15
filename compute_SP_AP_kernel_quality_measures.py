import numpy as np
import NALSM_GEN_SUPPORT as sup
import os


SAVE_VER_INP = input('VERSION? [int]: ')
VER = int(SAVE_VER_INP)

main_Path = os.getcwd()
data_path = main_Path + '/train_data/ver_'+str(VER)
net_path = main_Path + '/networks'


SAVE_VER_INP = input('DATASET? [M: mnist, N:nmnist]: ')
ds = str(SAVE_VER_INP)

if ds=='M':
    noise_level = 125
elif ds=='N':
    noise_level = 10

num_samples_per_label=100
shuffles=1000

run_SP = True
run_AP = True


if run_SP==True:
    rank_per_shuffle_SP_l = []

    sagg_l = []
    labels_l = []
    for i in range(0,40):

        names, data = sup.unpack_file(filename='TEST_'+str(VER)+'_S_AGG_BATCH_'+str(i)+'.datasagg',dataPath=data_path+'/')

        sagg_l.extend(data[names.index('olc_S_AGG_RES')])
        labels_l.extend(data[names.index('scalar_labels_in_batch_arr')])

    AGG = np.asarray(sagg_l)
    LBL = np.asarray(labels_l)

    for s in range(0, shuffles):

        idx_l = []
        for i in range(0,10):

            idx_unshuffled=np.where(LBL==i)[0]
            idx = np.random.permutation(idx_unshuffled)[0:num_samples_per_label]
            idx_l.extend(idx)

        filt_sagg_sp = AGG[idx_l,:]

        rank_per_shuffle_SP_l.append(np.linalg.matrix_rank(np.transpose(filt_sagg_sp)))

    rank_sp = np.average(rank_per_shuffle_SP_l)

if run_AP==True:
    sagg_l = []
    labels_l = []
    for i in range(0,40):
        names, data = sup.unpack_file(filename='NOISE_'+str(noise_level)+'_TEST_' + str(VER) + '_S_AGG_BATCH_' + str(i) + '.datasagg',
                                      dataPath=data_path + '/')

        sagg_l.extend(data[names.index('olc_S_AGG_RES')])
        labels_l.extend(data[names.index('scalar_labels_in_batch_arr')])

    AGG = np.asarray(sagg_l)
    LBL = np.asarray(labels_l)

    rank_per_shuffle_AP_l = []
    for s in range(0,shuffles):

        idx_l = []
        for i in range(0,10):
            # print('digit '+str(i))
            idx_unshuffled=np.where(LBL==i)[0]
            idx = np.random.permutation(idx_unshuffled)[0:num_samples_per_label]
            idx_l.extend(idx)

        filt_sagg_ap = AGG[idx_l,:]

        rank_per_shuffle_AP_l.append(np.linalg.matrix_rank(np.transpose(filt_sagg_ap)))

    rank_ap = np.average(rank_per_shuffle_AP_l)



if run_SP==True:
    print('SP: '+str(rank_sp))

if run_AP==True:
    print('AP: '+str(rank_ap))


