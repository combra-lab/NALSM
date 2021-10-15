import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import sys
import time
import NALSM_CONSTRUCTOR_MNIST_NOISE as construct
import NALSM_PARALLEL_SAVE as parasaver
import NALSM_SIM_SUPPORT as runSupport
import NALSM_GEN_SUPPORT as sup



class run_sim:
    def __init__(self):
        self.main_Path = os.getcwd()
        self.dataPath_TRAIN = self.main_Path + '/train_data'
        self.netPath = self.main_Path + '/networks'

        self.num_save_processes = 4

    def train_loop_v0(self
                           , SAVE_VER_IN
                           , NET_NUM_IN
                           , W_RES_EXC_MIN_MAX_IN
                           , W_RES_INH_MIN_MAX_IN
                           , W_RES_INP_MIN_MAX_IN
                           , W_OUT_RES_MIN_MAX_IN

                           , N1_SAMPLE_INPUT_DURATION_MS_IN
                           , N_INI_SAMPLE_INPUT_DURATION_MS_IN

                           , N1_BATCH_SIZE_IN
                           , N_INI_BATCH_SIZE_IN
                           , N1_BATCHS_PER_BLOCK_IN
                           , N1_record_samples_in_batch_l_IN
                           , N_INI_record_samples_in_batch_l_IN

                           , STDP_POTENTIATION_LR_IN

                           , N1_RES_LR_DECAY_IN
                           , N_INI_RES_LR_DECAY_IN

                           , ASTRO_BIAS_OFFSET_PERCENTAGE_IN
                           , ASTRO_W_SCALING_IN
                           , ASTRO_TAU_IN
                           , INPUT_CURRENT_IN

                           , DATASET_SEED_IN
                           , INTIAL_ABS_W_SCLR_IN
                            , NP_SEED_IN
                            , TF_SEED_IN

                            , NOISE_VAR_IN

                           , N1_SPIKE_STORE_MOD_IN
                           , N1_NUM_TRAIN_VAL_TEST_L_IN
                           , N_INI_NUM_TRAIN_VAL_TEST_L_IN
                           , N_INI_SPIKE_STORE_MOD_IN=1000000000

                           , NUM_OF_TRAIN_CYLES_ON_DATASET_IN=5000
                            ):


        ## -SEEDSET- SET RANDOM SEED SEED
        if NP_SEED_IN==-1:
            np.random.seed()
            rnd_sd = np.random.randint(0, 1000000)
            np.random.seed(rnd_sd)
        else:
            rnd_sd=NP_SEED_IN
            np.random.seed(rnd_sd)

        tf.reset_default_graph()

        self.savePath = self.dataPath_TRAIN + '/ver_' + str(SAVE_VER_IN) + '/'
        net_name = 'Network_' + str(NET_NUM_IN)

        if DATASET_SEED_IN==-1:
            print('FULL DATASET BEING USED WITH: ')
            N1_NUM_TRAIN_VAL_TEST_L_IN = [50000,10000,10000]
            print('TRAINING SAMPLES: '+str(N1_NUM_TRAIN_VAL_TEST_L_IN[0]))
            print('VALIDATION SAMPLES: ' + str(N1_NUM_TRAIN_VAL_TEST_L_IN[1]))
            print('TRAINING SAMPLES: ' + str(N1_NUM_TRAIN_VAL_TEST_L_IN[2]))

        print('SAVING SIMULATION PARAMETERS')
        param_names = ['SAVE_VER_IN'
                        , 'NET_NUM_IN'
                        , 'W_RES_EXC_MIN_MAX_IN'
                        , 'W_RES_INH_MIN_MAX_IN'
                        , 'W_RES_INP_MIN_MAX_IN'
                        , 'W_OUT_RES_MIN_MAX_IN'
                        , 'N1_SAMPLE_INPUT_DURATION_MS_IN'
                        , 'N_INI_SAMPLE_INPUT_DURATION_MS_IN'
                        , 'N1_BATCH_SIZE_IN'
                        , 'N_INI_BATCH_SIZE_IN'
                        , 'N1_BATCHS_PER_BLOCK_IN'
                        , 'N1_record_samples_in_batch_l_IN'
                        , 'N_INI_record_samples_in_batch_l_IN'
                        , 'STDP_POTENTIATION_LR_IN'
                        , 'N1_RES_LR_DECAY_IN'
                        , 'N_INI_RES_LR_DECAY_IN'
                        , 'ASTRO_BIAS_OFFSET_PERCENTAGE_IN'
                        , 'ASTRO_W_SCALING_IN'
                        , 'ASTRO_TAU_IN'
                        , 'INPUT_CURRENT_IN'
                        , 'DATASET_SEED_IN'
                        , 'NP_SEED_IN'
                        , 'TF_SEED_IN'
                        , 'INTIAL_ABS_W_SCLR_IN'
                        , 'N1_SPIKE_STORE_MOD_IN'
                        , 'N1_NUM_TRAIN_VAL_TEST_L_IN'
                        , 'N_INI_NUM_TRAIN_VAL_TEST_L_IN'
                        , 'N_INI_SPIKE_STORE_MOD_IN'
                        , 'NUM_OF_TRAIN_CYLES_ON_DATASET_IN'
                       , 'NOISE_VAR_IN']

        param_values = [SAVE_VER_IN
            , NET_NUM_IN
            , W_RES_EXC_MIN_MAX_IN
            , W_RES_INH_MIN_MAX_IN
            , W_RES_INP_MIN_MAX_IN
            , W_OUT_RES_MIN_MAX_IN
            , N1_SAMPLE_INPUT_DURATION_MS_IN
            , N_INI_SAMPLE_INPUT_DURATION_MS_IN
            , N1_BATCH_SIZE_IN
            , N_INI_BATCH_SIZE_IN
            , N1_BATCHS_PER_BLOCK_IN
            , N1_record_samples_in_batch_l_IN
            , N_INI_record_samples_in_batch_l_IN
            , STDP_POTENTIATION_LR_IN
            , N1_RES_LR_DECAY_IN
            , N_INI_RES_LR_DECAY_IN
            , ASTRO_BIAS_OFFSET_PERCENTAGE_IN
            , ASTRO_W_SCALING_IN
            , ASTRO_TAU_IN
            , INPUT_CURRENT_IN
            , DATASET_SEED_IN
            , NP_SEED_IN
            , TF_SEED_IN
            , INTIAL_ABS_W_SCLR_IN
            , N1_SPIKE_STORE_MOD_IN
            , N1_NUM_TRAIN_VAL_TEST_L_IN
            , N_INI_NUM_TRAIN_VAL_TEST_L_IN
            , N_INI_SPIKE_STORE_MOD_IN
            , NUM_OF_TRAIN_CYLES_ON_DATASET_IN
            , NOISE_VAR_IN]
        sup.save_log_file_of_parameters(root_savename='VER_'+str(SAVE_VER_IN)+'_NOISE_'+str(NOISE_VAR_IN)+'_params',savepath=self.savePath,parameter_names=param_names,parameters_values=param_values)



        NUM_BATCHES_IN_FULL_TRAIN_DATASET = int(N1_NUM_TRAIN_VAL_TEST_L_IN[0] / N1_BATCH_SIZE_IN)

        if int(NUM_BATCHES_IN_FULL_TRAIN_DATASET / N1_BATCHS_PER_BLOCK_IN) == (
            NUM_BATCHES_IN_FULL_TRAIN_DATASET / N1_BATCHS_PER_BLOCK_IN):
            NUM_BLOCKS_OF_BATCHES_IN_FULL_DATASET = int(NUM_BATCHES_IN_FULL_TRAIN_DATASET / N1_BATCHS_PER_BLOCK_IN)
            print('NUM_BLOCKS_OF_BATCHES_IN_FULL_DATASET',NUM_BLOCKS_OF_BATCHES_IN_FULL_DATASET)
        else:
            print('BATCH BLOCK SIZE DOES NOT MULTIPLY INTO NUM OF BATCHES IN DATASET WHOLLY, QUITTING...')
            sys.exit(2)


        # INITIALIZE MODEL
        N1 = construct.simulation_constructor(
                 network_name=net_name
                ,network_path=self.netPath
                , source_path=self.savePath
                , save_path=self.savePath
                , W_RES_EXC_MIN_MAX_IN=W_RES_EXC_MIN_MAX_IN
                , W_RES_INH_MIN_MAX_IN=W_RES_INH_MIN_MAX_IN
                , W_RES_INP_MIN_MAX_IN=W_RES_INP_MIN_MAX_IN
                , W_OUT_RES_MIN_MAX_IN=W_OUT_RES_MIN_MAX_IN
                , SAMPLE_INPUT_DURATION_MS_IN=N1_SAMPLE_INPUT_DURATION_MS_IN
                , BATCH_SIZE_IN=N1_BATCH_SIZE_IN
                , record_samples_in_batch_l_IN=N1_record_samples_in_batch_l_IN

                , STDP_POTENTIATION_LR_IN=STDP_POTENTIATION_LR_IN
                , RES_LR_DECAY_IN=N1_RES_LR_DECAY_IN
                , ASTRO_BIAS_OFFSET_PERCENTAGE_IN=ASTRO_BIAS_OFFSET_PERCENTAGE_IN
                , ASTRO_W_SCALING_IN=ASTRO_W_SCALING_IN
                , ASTRO_TAU_IN=ASTRO_TAU_IN
                , INPUT_CURRENT_IN=INPUT_CURRENT_IN
                , RES_INI_COST_W_SCLR_IN = INTIAL_ABS_W_SCLR_IN
                , INITIALIZE_ASTRO=False
                , INITIALIZE_STDP=False
                )


        print('ASSEMBLING DATA FOR N1...')
        # GET DATA
        run_sup = runSupport.run_support(number_of_networks_IN=N1_BATCH_SIZE_IN)
        N1_input_data_names, N1_input_data_data = run_sup.assemble_data_w_NOISE(DATA_SEED_VAR_IN=DATASET_SEED_IN,num_train_val_test_l=N1_NUM_TRAIN_VAL_TEST_L_IN,var_IN=NOISE_VAR_IN)


        test_data_INPUT_VEC = N1_input_data_data[N1_input_data_names.index('test_data_INPUT_VEC')]
        test_data_LABEL_VEC = N1_input_data_data[N1_input_data_names.index('test_data_LABEL_VEC')]
        test_data_LABEL_SCALAR = N1_input_data_data[N1_input_data_names.index('test_data_LABEL_SCALAR')]


        print('INITIALIZING PARALLEL SAVER...')
        ### INITIALIZE PARALLEL SAVER ###
        smart_saver = parasaver.multi_process(num_processes=self.num_save_processes, save_path=self.savePath)

        st = time.time()
        ## -SEEDSET-
        if TF_SEED_IN==-1:
            rnd_sd_TF = np.random.randint(0, 100000)
            tf.set_random_seed(rnd_sd_TF)
        else:
            rnd_sd_TF = TF_SEED_IN
            tf.set_random_seed(rnd_sd_TF)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        list_of_uniq_weights = np.unique(sess.run(N1.W_DENSE))
        print(list_of_uniq_weights)


        # GENERATE NOISY SPIKE COUNTS
        print('TEST DATA: GENERATING S AGG DATA')
        tst_spike_files_l, tst_s_agg_files_l = N1.generate_validation_or_test_spike_data_STATIC(sess_IN=sess
                                                                                                , saver_object_IN=smart_saver
                                                                                                , valid_or_test_data_INPUT_VEC_IN=test_data_INPUT_VEC
                                                                                                , valid_or_test_data_LABEL_VEC_IN=test_data_LABEL_VEC
                                                                                                , valid_or_test_data_LABEL_SCALAR_IN=test_data_LABEL_SCALAR
                                                                                                , BATCH_SIZE_IN=N1_BATCH_SIZE_IN
                                                                                                , SAMPLE_INPUT_DURATION_MS_IN=N1_SAMPLE_INPUT_DURATION_MS_IN
                                                                                                , NUM_DATA_SAMPLES_IN=N1_NUM_TRAIN_VAL_TEST_L_IN[2]
                                                                                                , spike_store_batch_mod_IN=N1_SPIKE_STORE_MOD_IN
                                                                                                , record_samples_in_batch_l_IN=N1_record_samples_in_batch_l_IN
                                                                                                , root_save_name_IN=str('NOISE_'+str(NOISE_VAR_IN)+'_TEST_' + str(SAVE_VER_IN))
                                                                                                )





        smart_saver.save_data(signal=1
                              , names=['tst_spike_files_l', 'tst_s_agg_files_l'
                                    , 'rnd_sd', 'rnd_sd_TF', 'list_of_uniq_weights'
                                       ]
                              , data=[tst_spike_files_l, tst_s_agg_files_l
                                    , rnd_sd, rnd_sd_TF, list_of_uniq_weights
                                      ]
                              , save_filename='NOISE_'+str(NOISE_VAR_IN)+'_FILE_L_VAL_ACC_L.files'
                              )


        smart_saver.kill_workers(process_count=self.num_save_processes)

        print('TOTAL SIMULATION TIME: '+str(time.time()-st))


        sess.close()



if __name__ == "__main__":
    GLOBAL_VAR__GPU_NUM = input('GPU? ')
    # GLOBAL_VAR__GPU_NUM = 1

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GLOBAL_VAR__GPU_NUM)
    print('GPU: ' + str(os.environ["CUDA_VISIBLE_DEVICES"]))

    SAVE_VER_VAR_INP = input('VERSION? [int]: ')
    SAVE_VER_VAR = int(SAVE_VER_VAR_INP)

    NET_NUM_VAR_INP = input('NET_NUM_VAR? [int]: ')
    NET_NUM_VAR = int(NET_NUM_VAR_INP)

    N1_BATCH_SIZE_INP = input('BATCH_SIZE? [int](250 for smaller nets(1000 neurons)/ 50 for larger nets(8000 neurons): ')
    N1_BATCH_SIZE_VAR = int(N1_BATCH_SIZE_INP)

    N1_BATCHS_PER_BLOCK_INP = input('BATCHS_PER_BLOCK? [int](10 for smaller nets(1000 neurons)/ 50 for larger nets(8000 neurons): ')
    N1_BATCHS_PER_BLOCK_VAR = int(N1_BATCHS_PER_BLOCK_INP)

    # N1_SAMPLE_INPUT_DURATION_MS_INP = input('N1_SAMPLE_INPUT_DURATION_MS? [int]: ')
    # N1_SAMPLE_INPUT_DURATION_MS_VAR = int(N1_SAMPLE_INPUT_DURATION_MS_INP)
    N1_SAMPLE_INPUT_DURATION_MS_VAR = int(250)

    # ASTRO_BIAS_OFFSET_PERCENTAGE_INP = input('ASTRO_BIAS_OFFSET_PERCENTAGE? [float]: ')
    # ASTRO_BIAS_OFFSET_PERCENTAGE_VAR = float(ASTRO_BIAS_OFFSET_PERCENTAGE_INP)

    INTIAL_ABS_W_SCLR_INP = input('INTIAL_ABS_W_SCLR? [float]: ')
    INTIAL_ABS_W_SCLR_VAR = float(INTIAL_ABS_W_SCLR_INP)

    # DATASET_SEED_INP = input('DATASET_SEED? [int]: ')
    # DATASET_SEED_VAR = int(DATASET_SEED_INP)
    DATASET_SEED_VAR = int(-1)  # will use full dataset

    # NP_SEED_INP = input('NP_SEED? [int]: ')
    # NP_SEED_VAR = int(NP_SEED_INP)
    NP_SEED_VAR = int(-1)  # will generate random seed

    # TF_SEED_INP = input('TF_SEED? [int]: ')
    # TF_SEED_VAR = int(TF_SEED_INP)
    TF_SEED_VAR = int(-1)  # will generate random seed

    # NOISE_VAR_INP = input('NOISE_VAR? [int] (0-255): ')
    # NOISE_VAR_VAR = int(NOISE_VAR_INP)
    NOISE_VAR_VAR = int(125)

    N1_SPIKE_STORE_MOD_VAR = int(10000/N1_BATCH_SIZE_VAR)
    print('SPIKE_STORE_BATCH_MOD: ' + str(N1_SPIKE_STORE_MOD_VAR))

    spike_save_idxs = np.asarray(np.round(N1_BATCH_SIZE_VAR * np.divide(np.arange(20), 20), 0), dtype=np.int32)
    print('SPIKE_SAVE_IDX LIST: ', spike_save_idxs)

    TSL = run_sim()
    TSL.train_loop_v0(SAVE_VER_IN=SAVE_VER_VAR
                        , NET_NUM_IN=NET_NUM_VAR
                        , W_RES_EXC_MIN_MAX_IN=[0.0,3.0]
                        , W_RES_INH_MIN_MAX_IN=[-3.0,0.0]
                        , W_RES_INP_MIN_MAX_IN=[-3.0,3.0]
                        , W_OUT_RES_MIN_MAX_IN=[0.0,0.0]

                        , N1_SAMPLE_INPUT_DURATION_MS_IN=N1_SAMPLE_INPUT_DURATION_MS_VAR
                        , N_INI_SAMPLE_INPUT_DURATION_MS_IN=int(1000)


                        , N1_BATCH_SIZE_IN=N1_BATCH_SIZE_VAR#int(20)
                        , N_INI_BATCH_SIZE_IN=1#1
                        , N1_BATCHS_PER_BLOCK_IN = N1_BATCHS_PER_BLOCK_VAR
                        , N1_record_samples_in_batch_l_IN=spike_save_idxs
                        , N_INI_record_samples_in_batch_l_IN=[]

                        , STDP_POTENTIATION_LR_IN=0.15

                        , N1_RES_LR_DECAY_IN=0.999
                        , N_INI_RES_LR_DECAY_IN=1.0

                        , DATASET_SEED_IN=DATASET_SEED_VAR
                        , INTIAL_ABS_W_SCLR_IN = INTIAL_ABS_W_SCLR_VAR
                        , NP_SEED_IN=NP_SEED_VAR
                        , TF_SEED_IN=TF_SEED_VAR

                      , ASTRO_BIAS_OFFSET_PERCENTAGE_IN=0.0
                        , ASTRO_W_SCALING_IN=0.002
                        , ASTRO_TAU_IN=100.0
                        , INPUT_CURRENT_IN=100.0
                        , NOISE_VAR_IN = NOISE_VAR_VAR

                        , N1_SPIKE_STORE_MOD_IN=N1_SPIKE_STORE_MOD_VAR
                        , N1_NUM_TRAIN_VAL_TEST_L_IN=[2000,500,500]
                        , N_INI_NUM_TRAIN_VAL_TEST_L_IN=[100, 0, 0]
                      )














