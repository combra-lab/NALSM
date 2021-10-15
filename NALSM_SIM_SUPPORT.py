import numpy as np
import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import NALSM_GEN_SUPPORT as sup

class run_support:
    def __init__(self,number_of_networks_IN):

        self.main_Path = os.getcwd()
        self.dataPath_TRAIN = self.main_Path + '/train_data'
        self.netPath = self.main_Path + '/networks'
        self.datasets = self.main_Path + '/datasets'


        self.number_of_networks = number_of_networks_IN


    def transfer_W_STORE_from_single_batch_to_multi_batch_net(self,single_batch_W_IN,multi_batch_W_IN):

        return tf.assign(multi_batch_W_IN, tf.add(tf.scalar_mul(0.0, multi_batch_W_IN), single_batch_W_IN))

    def simulation_initializer(self
                               , inp_range_IN
                               , res_exc_range_IN
                               , res_inh_range_IN
                               , res_range_IN
                               , out_range_IN

                               , w_mask_sparse_single_net_np_IN
                               , record_nets_l_IN
                               , num_neurons_in_network_IN
                               , input_sample_duration_IN):


        gather_raw_idx_0 = np.where(w_mask_sparse_single_net_np_IN == 1)
        scatter_idx_w_dense_to_s_out_main = []
        for net in range(0, self.number_of_networks):
            current_row = -1
            row_counter = 0
            scatter_idx_w_to_ss = []
            for i in range(0, len(gather_raw_idx_0[0])):
                if current_row == gather_raw_idx_0[0][i]:

                    scatter_idx_w_to_ss.append([net, gather_raw_idx_0[0][i]])
                    row_counter = row_counter + 1
                else:
                    current_row = gather_raw_idx_0[0][i]
                    row_counter = 0

                    scatter_idx_w_to_ss.append([net, gather_raw_idx_0[0][i]])
                    row_counter = row_counter + 1
            scatter_idx_w_dense_to_s_out_main.append(scatter_idx_w_to_ss)
        scatter_idx_w_dense_to_s_out_main_tf = tf.constant(scatter_idx_w_dense_to_s_out_main)
        # OUTPUT: scatter_idx_w_dense_to_s_out_main_tf



        scatter_S_to_S_SUBSET_inputNeurons = []
        for nn in range(0, self.number_of_networks):
            scatter_S_to_S_SUBSET_inputNeurons_temp = []
            for cccc in range(inp_range_IN[0], inp_range_IN[1]):
                scatter_S_to_S_SUBSET_inputNeurons_temp.append([int(nn), int(cccc)])
            scatter_S_to_S_SUBSET_inputNeurons.append(scatter_S_to_S_SUBSET_inputNeurons_temp)
        scatter_S_to_S_SUBSET_inputNeurons_tf = tf.constant(scatter_S_to_S_SUBSET_inputNeurons)
        # OUTPUT: scatter_S_to_S_SUBSET_inputNeurons_tf

        gather_raw_idx_2 = np.where(w_mask_sparse_single_net_np_IN == 1)
        gather_idx_Sin_to_Wdense_prelim = gather_raw_idx_2[1]
        gather_idx_Sin_to_Wdense = []
        for net in range(0, self.number_of_networks):
            gather_idx_Sin_to_Wdense_temp = []
            for i in range(0, len(gather_idx_Sin_to_Wdense_prelim)):
                gather_idx_Sin_to_Wdense_temp.append([net, gather_idx_Sin_to_Wdense_prelim[i]])
            gather_idx_Sin_to_Wdense.append(gather_idx_Sin_to_Wdense_temp)
        gather_idx_Sin_to_Wdense_tf = tf.constant(gather_idx_Sin_to_Wdense)
        print(len(gather_idx_Sin_to_Wdense))
        print(len(gather_idx_Sin_to_Wdense[0]))


        gather_raw_idx_4 = np.where(w_mask_sparse_single_net_np_IN == 1)
        gather_idx_Sout_to_Wdense_prelim = gather_raw_idx_4[0]
        gather_idx_Sout_to_Wdense = []
        for net in range(0, self.number_of_networks):
            gather_idx_Sout_to_Wdense_temp = []
            for i in range(0, len(gather_idx_Sout_to_Wdense_prelim)):
                gather_idx_Sout_to_Wdense_temp.append([net, gather_idx_Sout_to_Wdense_prelim[i]])
            gather_idx_Sout_to_Wdense.append(gather_idx_Sout_to_Wdense_temp)
        gather_idx_Sout_to_Wdense_tf = tf.constant(gather_idx_Sout_to_Wdense)
        # OUTPUT: gather_idx_Sout_to_Wdense


        # S and W_dense masks
        S_exc_and_inp_mask_np = np.zeros((1, num_neurons_in_network_IN), dtype=np.float32)
        S_exc_and_inp_mask_np[0, res_exc_range_IN[0]:res_exc_range_IN[1]] = 1.0
        S_exc_and_inp_mask_np[0, inp_range_IN[0]:inp_range_IN[1]] = 1.0

        S_inh_mask_np = np.zeros((1, num_neurons_in_network_IN), dtype=np.float32)
        S_inh_mask_np[0, res_inh_range_IN[0]:res_inh_range_IN[1]] = 1.0

        w_mask_idx = np.where(w_mask_sparse_single_net_np_IN == 1.0)
        num_syns = len(w_mask_idx[0])

        S_res_out_mask_temp = np.zeros((num_neurons_in_network_IN, 1), dtype=np.float32)
        S_res_out_mask_temp[res_range_IN[0]:res_range_IN[1], 0] = 1.0

        # ASSEMBLE EXC TO RES CONNECTION MASK OF FORMAT W_DENSE
        S_exc_in_mask_temp = np.zeros((1, num_neurons_in_network_IN), dtype=np.float32)
        S_exc_in_mask_temp[0, res_exc_range_IN[0]:res_exc_range_IN[1]] = 1.0

        w_mask_filt_exc_to_res = np.multiply(w_mask_sparse_single_net_np_IN, np.multiply(S_res_out_mask_temp, S_exc_in_mask_temp))

        w_dense_exc_to_res_mask_temp_l = []
        for i in range(0, len(w_mask_idx[0])):
            w_dense_exc_to_res_mask_temp_l.append(w_mask_filt_exc_to_res[w_mask_idx[0][i], w_mask_idx[1][i]])

        w_dense_exc_to_res_mask_np = np.asarray(w_dense_exc_to_res_mask_temp_l).reshape((1, num_syns))
        # ASSEMBLE INH TO RES CONNECTION MASK OF FORMAT W_DENSE


        # ASSEMBLE INH TO RES CONNECTION MASK OF FORMAT W_DENSE
        S_inh_in_mask_temp = np.zeros((1, num_neurons_in_network_IN), dtype=np.float32)
        S_inh_in_mask_temp[0, res_inh_range_IN[0]:res_inh_range_IN[1]] = 1.0

        w_mask_filt_inh_to_res = np.multiply(w_mask_sparse_single_net_np_IN, np.multiply(S_res_out_mask_temp, S_inh_in_mask_temp))

        w_dense_inh_to_res_mask_temp_l = []
        for i in range(0, len(w_mask_idx[0])):
            w_dense_inh_to_res_mask_temp_l.append(w_mask_filt_inh_to_res[w_mask_idx[0][i], w_mask_idx[1][i]])

        w_dense_inh_to_res_mask_np = np.asarray(w_dense_inh_to_res_mask_temp_l).reshape((1, num_syns))
        # ASSEMBLE INH TO RES CONNECTION MASK OF FORMAT W_DENSE

        # ASSEMBLE INP TO RES CONNECTION MASK OF FORMAT W_DENSE
        S_inp_in_mask_temp = np.zeros((1, num_neurons_in_network_IN), dtype=np.float32)
        S_inp_in_mask_temp[0, inp_range_IN[0]:inp_range_IN[1]] = 1.0

        w_mask_filt_inp_to_res = np.multiply(w_mask_sparse_single_net_np_IN, np.multiply(S_res_out_mask_temp, S_inp_in_mask_temp))

        w_dense_inp_to_res_mask_temp_l = []
        for i in range(0, len(w_mask_idx[0])):
            w_dense_inp_to_res_mask_temp_l.append(w_mask_filt_inp_to_res[w_mask_idx[0][i], w_mask_idx[1][i]])

        w_dense_inp_to_res_mask_np = np.asarray(w_dense_inp_to_res_mask_temp_l).reshape((1, num_syns))
        # ASSEMBLE INH TO RES CONNECTION MASK OF FORMAT W_DENSE


        # ASSEMBLE RES TO OUT CONNECTION MASK OF FORMAT W_DENSE
        S_res_in_mask_temp = np.zeros((1, num_neurons_in_network_IN), dtype=np.float32)
        S_res_in_mask_temp[0, res_range_IN[0]:res_range_IN[1]] = 1.0

        S_out_out_mask_temp = np.zeros((num_neurons_in_network_IN, 1), dtype=np.float32)
        S_out_out_mask_temp[out_range_IN[0]:out_range_IN[1], 0] = 1.0

        w_mask_filt_res_to_out = np.multiply(w_mask_sparse_single_net_np_IN, np.multiply(S_out_out_mask_temp, S_res_in_mask_temp))

        w_dense_res_to_out_mask_temp_l = []
        for i in range(0, len(w_mask_idx[0])):
            w_dense_res_to_out_mask_temp_l.append(w_mask_filt_res_to_out[w_mask_idx[0][i], w_mask_idx[1][i]])

        w_dense_res_to_out_mask_np = np.asarray(w_dense_res_to_out_mask_temp_l).reshape((1, num_syns))
        # ASSEMBLE INH TO RES CONNECTION MASK OF FORMAT W_DENSE

        return ['scatter_idx_w_dense_to_s_out_main_tf'
            , 'scatter_S_to_S_SUBSET_inputNeurons_tf'
            , 'gather_idx_Sin_to_Wdense_tf'
            , 'gather_idx_Sout_to_Wdense_tf'

            , 'S_exc_and_inp_mask_np'
            , 'S_inh_mask_np'
            , 'w_dense_exc_to_res_mask_np'
            , 'w_dense_inh_to_res_mask_np'
            , 'w_dense_inp_to_res_mask_np'
            , 'w_dense_res_to_out_mask_np'],[

            scatter_idx_w_dense_to_s_out_main_tf
            , scatter_S_to_S_SUBSET_inputNeurons_tf
            , gather_idx_Sin_to_Wdense_tf
            , gather_idx_Sout_to_Wdense_tf

            , S_exc_and_inp_mask_np
            , S_inh_mask_np
            , w_dense_exc_to_res_mask_np
            , w_dense_inh_to_res_mask_np
            , w_dense_inp_to_res_mask_np
            , w_dense_res_to_out_mask_np]

    def assemble_data_for_fast_delivery_partial(self,input_data_IN, label_data_IN):

        reformatted_input_data_l_INTERNAL = []
        reformatted_label_data_l_INTERNAL = []
        reformatted_labels_l_INTERNAL = []
        for mnmn in range(np.shape(input_data_IN)[0]):


            poisson_values = np.expand_dims(np.squeeze(np.reshape(input_data_IN[mnmn,:,:], (1, -1))),axis=0)


            label = int(label_data_IN[mnmn])
            output_label_l = []
            for i in range(0, 10):
                if i == int(label):
                    # output_stream_l.append(ho)
                    # output_stream_l.append(lo)
                    output_label_l.append(1.0)
                    # output_label_l.append(-1.0)
                else:
                    # output_stream_l.append(lo)
                    # output_stream_l.append(ho)
                    output_label_l.append(-1.0)
                    # output_label_l.append(1.0)

            reformatted_input_data_l_INTERNAL.append(poisson_values)
            reformatted_label_data_l_INTERNAL.append(np.asarray(output_label_l))
            reformatted_labels_l_INTERNAL.append(label)

        return reformatted_input_data_l_INTERNAL, reformatted_label_data_l_INTERNAL, reformatted_labels_l_INTERNAL


    def assemble_data(self,DATA_SEED_VAR_IN,num_train_val_test_l):

        if DATA_SEED_VAR_IN==-1:
            FULL_DATA_SET_IN=True
        else:
            FULL_DATA_SET_IN = False

        PATH = self.datasets + '/mnist.npz'
        (orig_x_train, y_train), (orig_x_test, y_test) = tf.keras.datasets.mnist.load_data(path=PATH)

        sumed_orig_x_train = np.sum(orig_x_train, axis=(1, 2))
        normed_x_train = np.divide(orig_x_train, sumed_orig_x_train[:, np.newaxis, np.newaxis])
        rescale_factor = np.amax(normed_x_train)
        x_train = normed_x_train / rescale_factor

        print('DATA TRAIN REFORMATING STATS:')
        print('max ave: ' + str(np.amax(np.average(x_train, axis=(1, 2)))))
        print('min ave: ' + str(np.amin(np.average(x_train, axis=(1, 2)))))
        print('max sum: ' + str(np.amax(np.sum(x_train, axis=(1, 2)))))
        print('min sum: ' + str(np.amin(np.sum(x_train, axis=(1, 2)))))

        # reformat x_test
        sumed_orig_x_test = np.sum(orig_x_test, axis=(1, 2))
        normed_x_test = np.divide(orig_x_test, sumed_orig_x_test[:, np.newaxis, np.newaxis])
        rescale_factor_test = np.amax(normed_x_test)
        x_test = normed_x_test / rescale_factor_test

        print('DATA TEST REFORMATING STATS:')
        print('max ave: ' + str(np.amax(np.average(x_test, axis=(1, 2)))))
        print('min ave: ' + str(np.amin(np.average(x_test, axis=(1, 2)))))
        print('max sum: ' + str(np.amax(np.sum(x_test, axis=(1, 2)))))
        print('min sum: ' + str(np.amin(np.sum(x_test, axis=(1, 2)))))


        if FULL_DATA_SET_IN==True:
            print('DATA SET ASSEMBLY: USING FULL UNSHUFFLED DATASET.')
            x_valid = x_train[50000:60000]
            y_valid = y_train[50000:60000]

            x_train = x_train[0:50000]
            y_train = y_train[0:50000]

        else:

            np.random.seed(DATA_SEED_VAR_IN)

            NUM_TRAIN_SAMPLES_VAR = num_train_val_test_l[0]
            NUM_VALID_SAMPLES_VAR = num_train_val_test_l[1]
            NUM_TEST_SAMPLES_VAR = num_train_val_test_l[2]

            valid_idxs = np.random.permutation(50000 + np.arange(10000))[0:NUM_VALID_SAMPLES_VAR]

            x_valid = x_train[valid_idxs]
            y_valid = y_train[valid_idxs]

            training_idxs = np.random.permutation(np.arange(50000))[0:NUM_TRAIN_SAMPLES_VAR]

            x_train = x_train[training_idxs]
            y_train = y_train[training_idxs]

            test_idxs = np.random.permutation(np.arange(len(y_test)))[0:NUM_TEST_SAMPLES_VAR]

            x_test = x_test[test_idxs]
            y_test = y_test[test_idxs]

        reformatted_input_data_l, reformatted_label_data_l, reformatted_labels_l = self.assemble_data_for_fast_delivery_partial(
            input_data_IN=x_train, label_data_IN=y_train)

        reformatted_input_valid_data_l, reformatted_label_valid_data_l, reformatted_valid_labels_l = self.assemble_data_for_fast_delivery_partial(
            input_data_IN=x_valid, label_data_IN=y_valid)

        reformatted_input_test_data_l, reformatted_label_test_data_l, reformatted_test_labels_l = self.assemble_data_for_fast_delivery_partial(
            input_data_IN=x_test, label_data_IN=y_test)

        return ['train_data_INPUT_VEC',
                'train_data_LABEL_VEC',
                'train_data_LABEL_SCALAR',

                'valid_data_INPUT_VEC',
                'valid_data_LABEL_VEC',
                'valid_data_LABEL_SCALAR',

                'test_data_INPUT_VEC',
                'test_data_LABEL_VEC',
                'test_data_LABEL_SCALAR'
                ],[reformatted_input_data_l,
                    reformatted_label_data_l,
                    reformatted_labels_l,

                   reformatted_input_valid_data_l,
                   reformatted_label_valid_data_l,
                   reformatted_valid_labels_l,

                   reformatted_input_test_data_l,
                   reformatted_label_test_data_l,
                   reformatted_test_labels_l
                   ]

    def assemble_data_FMNIST(self,DATA_SEED_VAR_IN,num_train_val_test_l):

        if DATA_SEED_VAR_IN==-1:
            FULL_DATA_SET_IN=True
        else:
            FULL_DATA_SET_IN = False

        
        PATH = '/common/users/vai9/CNAN/datasets/'
        names, data = sup.unpack_file(filename='fmnist.ds', dataPath=PATH)

        orig_x_train = data[names.index('x_train')]
        y_train = data[names.index('y_train')]
        orig_x_test = data[names.index('x_test')]
        y_test = data[names.index('y_test')]

        # reformat x_train
        sumed_orig_x_train = np.sum(orig_x_train, axis=(1, 2))
        normed_x_train = np.divide(orig_x_train, sumed_orig_x_train[:, np.newaxis, np.newaxis])
        rescale_factor = np.amax(normed_x_train)
        x_train = normed_x_train / rescale_factor

        print('DATA TRAIN REFORMATING STATS:')
        print('max ave: ' + str(np.amax(np.average(x_train, axis=(1, 2)))))
        print('min ave: ' + str(np.amin(np.average(x_train, axis=(1, 2)))))
        print('max sum: ' + str(np.amax(np.sum(x_train, axis=(1, 2)))))
        print('min sum: ' + str(np.amin(np.sum(x_train, axis=(1, 2)))))

        # reformat x_test
        sumed_orig_x_test = np.sum(orig_x_test, axis=(1, 2))
        normed_x_test = np.divide(orig_x_test, sumed_orig_x_test[:, np.newaxis, np.newaxis])
        rescale_factor_test = np.amax(normed_x_test)
        x_test = normed_x_test / rescale_factor_test

        print('DATA TEST REFORMATING STATS:')
        print('max ave: ' + str(np.amax(np.average(x_test, axis=(1, 2)))))
        print('min ave: ' + str(np.amin(np.average(x_test, axis=(1, 2)))))
        print('max sum: ' + str(np.amax(np.sum(x_test, axis=(1, 2)))))
        print('min sum: ' + str(np.amin(np.sum(x_test, axis=(1, 2)))))


        if FULL_DATA_SET_IN==True:
            print('DATA SET ASSEMBLY: USING FULL UNSHUFFLED DATASET.')
            x_valid = x_train[50000:60000]
            y_valid = y_train[50000:60000]

            x_train = x_train[0:50000]
            y_train = y_train[0:50000]

        else:

            np.random.seed(DATA_SEED_VAR_IN)

            NUM_TRAIN_SAMPLES_VAR = num_train_val_test_l[0]
            NUM_VALID_SAMPLES_VAR = num_train_val_test_l[1]
            NUM_TEST_SAMPLES_VAR = num_train_val_test_l[2]

            valid_idxs = np.random.permutation(50000 + np.arange(10000))[0:NUM_VALID_SAMPLES_VAR]

            x_valid = x_train[valid_idxs]
            y_valid = y_train[valid_idxs]

            training_idxs = np.random.permutation(np.arange(50000))[0:NUM_TRAIN_SAMPLES_VAR]

            x_train = x_train[training_idxs]
            y_train = y_train[training_idxs]

            test_idxs = np.random.permutation(np.arange(len(y_test)))[0:NUM_TEST_SAMPLES_VAR]

            x_test = x_test[test_idxs]
            y_test = y_test[test_idxs]

        reformatted_input_data_l, reformatted_label_data_l, reformatted_labels_l = self.assemble_data_for_fast_delivery_partial(
            input_data_IN=x_train, label_data_IN=y_train)

        reformatted_input_valid_data_l, reformatted_label_valid_data_l, reformatted_valid_labels_l = self.assemble_data_for_fast_delivery_partial(
            input_data_IN=x_valid, label_data_IN=y_valid)

        reformatted_input_test_data_l, reformatted_label_test_data_l, reformatted_test_labels_l = self.assemble_data_for_fast_delivery_partial(
            input_data_IN=x_test, label_data_IN=y_test)

        return ['train_data_INPUT_VEC',
                'train_data_LABEL_VEC',
                'train_data_LABEL_SCALAR',

                'valid_data_INPUT_VEC',
                'valid_data_LABEL_VEC',
                'valid_data_LABEL_SCALAR',

                'test_data_INPUT_VEC',
                'test_data_LABEL_VEC',
                'test_data_LABEL_SCALAR'
                ],[reformatted_input_data_l,
                    reformatted_label_data_l,
                    reformatted_labels_l,

                   reformatted_input_valid_data_l,
                   reformatted_label_valid_data_l,
                   reformatted_valid_labels_l,

                   reformatted_input_test_data_l,
                   reformatted_label_test_data_l,
                   reformatted_test_labels_l
                   ]

    def sample_random_inputs(self,size_of_dataset_to_sample_from_IN,num_samples_IN):

        idx_ordered = np.arange(size_of_dataset_to_sample_from_IN)
        selected_idx = np.random.permutation(idx_ordered)[0:num_samples_IN]

        return selected_idx

    def sample_random_inputs_v2(self,size_of_dataset_to_sample_from_IN,num_samples_IN):

        num_of_dataset_duplicates = np.ceil(num_samples_IN/size_of_dataset_to_sample_from_IN)

        idx_ordered = np.mod(np.arange(num_of_dataset_duplicates*size_of_dataset_to_sample_from_IN),size_of_dataset_to_sample_from_IN)
        selected_idx = np.random.permutation(idx_ordered)[0:num_samples_IN]

        return selected_idx


    def get_data_lists_labels(self,data_type_IN):

        PATH = self.datasets + '/nmnist/' + str(data_type_IN) + '/'

        names, data = sup.unpack_file(filename='NMNIST_' + str(data_type_IN) + '_file_label_list', dataPath=PATH)

        file_name_label_list = data[names.index('file_name_label_list')]

        return file_name_label_list


    def get_nmnist_data_point(self,idx_IN,data_type_IN,data_list_IN,INPUT_DURATION_IN=300):


        PATH = self.datasets + '/nmnist/' + str(data_type_IN) + '/'

        filename = PATH+str(data_list_IN[idx_IN][0])
        label = int(data_list_IN[idx_IN][1])

        with open(filename, 'rb') as input_file:
            compact_file = input_file.read()

        event_data = self.transform_compact_file_to_array(empty_array=np.zeros((2, 34, 34, INPUT_DURATION_IN)), compact_file_data=compact_file)

        output_label_arr = -1.0*np.ones(10,dtype=np.float32)
        output_label_arr[label] = 1.0


        return event_data.reshape((2*34*34,INPUT_DURATION_IN)), output_label_arr, label


    def transform_compact_file_to_array(self, empty_array, compact_file_data):
        """
        Transform compact file data to a numpy array
        :param empty_array: empty numpy array
        :type empty_array: numpy array
        :param compact_file_data: compact file data
        :type compact_file_data: byte list
        :return: empty_array
        :rtype: numpy array
        """
        input_as_int = np.asarray([x for x in compact_file_data])
        x_event = input_as_int[0::5]
        y_event = input_as_int[1::5]
        p_event = input_as_int[2::5] >> 7
        t_event = ((input_as_int[2::5] << 16) | (input_as_int[3::5] << 8) | (input_as_int[4::5])) & 0x7FFFFF
        t_event = t_event / 1000  # convert spike times to ms
        # round event information
        x_event = np.round(x_event).astype(int)
        y_event = np.round(y_event).astype(int)
        p_event = np.round(p_event).astype(int)
        t_event = np.round(t_event).astype(int)
        valid_ind = np.argwhere((x_event < empty_array.shape[2]) &
                                (y_event < empty_array.shape[1]) &
                                (p_event < empty_array.shape[0]) &
                                (t_event < empty_array.shape[3]) &
                                (x_event >= 0) &
                                (y_event >= 0) &
                                (p_event >= 0) &
                                (t_event >= 0))
        empty_array[p_event[valid_ind],
                    y_event[valid_ind],
                    x_event[valid_ind],
                    t_event[valid_ind]] = 1
        return empty_array


    # with added guassian NOISE for kernel quality measures
    def assemble_data_w_NOISE(self,DATA_SEED_VAR_IN,num_train_val_test_l,var_IN=0):

        if DATA_SEED_VAR_IN==-1:
            FULL_DATA_SET_IN=True
        else:
            FULL_DATA_SET_IN = False


        PATH = self.datasets + '/mnist.npz'
        (orig_x_train_wo_noise, y_train), (orig_x_test_wo_noise, y_test) = tf.keras.datasets.mnist.load_data(path=PATH)


        orig_x_train = np.clip(
            orig_x_train_wo_noise + np.reshape(np.random.normal(0.0, var_IN, size=np.size(orig_x_train_wo_noise)),
                                              np.shape(orig_x_train_wo_noise)), 0,
            np.amax(orig_x_train_wo_noise))

        orig_x_test = np.clip(
            orig_x_test_wo_noise + np.reshape(np.random.normal(0.0, var_IN, size=np.size(orig_x_test_wo_noise)), np.shape(orig_x_test_wo_noise)), 0,
            np.amax(orig_x_test_wo_noise))


        # reformat x_train
        sumed_orig_x_train = np.sum(orig_x_train, axis=(1, 2))
        normed_x_train = np.divide(orig_x_train, sumed_orig_x_train[:, np.newaxis, np.newaxis])
        rescale_factor = np.amax(normed_x_train)
        x_train = normed_x_train / rescale_factor

        print('DATA TRAIN REFORMATING STATS:')
        print('max ave: ' + str(np.amax(np.average(x_train, axis=(1, 2)))))
        print('min ave: ' + str(np.amin(np.average(x_train, axis=(1, 2)))))
        print('max sum: ' + str(np.amax(np.sum(x_train, axis=(1, 2)))))
        print('min sum: ' + str(np.amin(np.sum(x_train, axis=(1, 2)))))

        # reformat x_test
        sumed_orig_x_test = np.sum(orig_x_test, axis=(1, 2))
        normed_x_test = np.divide(orig_x_test, sumed_orig_x_test[:, np.newaxis, np.newaxis])
        rescale_factor_test = np.amax(normed_x_test)
        x_test = normed_x_test / rescale_factor_test

        print('DATA TEST REFORMATING STATS:')
        print('max ave: ' + str(np.amax(np.average(x_test, axis=(1, 2)))))
        print('min ave: ' + str(np.amin(np.average(x_test, axis=(1, 2)))))
        print('max sum: ' + str(np.amax(np.sum(x_test, axis=(1, 2)))))
        print('min sum: ' + str(np.amin(np.sum(x_test, axis=(1, 2)))))


        if FULL_DATA_SET_IN==True:
            print('DATA SET ASSEMBLY: USING FULL UNSHUFFLED DATASET.')
            x_valid = x_train[50000:60000]
            y_valid = y_train[50000:60000]

            x_train = x_train[0:50000]
            y_train = y_train[0:50000]

        else:

            np.random.seed(DATA_SEED_VAR_IN)

            NUM_TRAIN_SAMPLES_VAR = num_train_val_test_l[0]
            NUM_VALID_SAMPLES_VAR = num_train_val_test_l[1]
            NUM_TEST_SAMPLES_VAR = num_train_val_test_l[2]

            valid_idxs = np.random.permutation(50000 + np.arange(10000))[0:NUM_VALID_SAMPLES_VAR]

            x_valid = x_train[valid_idxs]
            y_valid = y_train[valid_idxs]

            training_idxs = np.random.permutation(np.arange(50000))[0:NUM_TRAIN_SAMPLES_VAR]

            x_train = x_train[training_idxs]
            y_train = y_train[training_idxs]

            test_idxs = np.random.permutation(np.arange(len(y_test)))[0:NUM_TEST_SAMPLES_VAR]

            x_test = x_test[test_idxs]
            y_test = y_test[test_idxs]

        reformatted_input_data_l, reformatted_label_data_l, reformatted_labels_l = self.assemble_data_for_fast_delivery_partial(
            input_data_IN=x_train, label_data_IN=y_train)

        reformatted_input_valid_data_l, reformatted_label_valid_data_l, reformatted_valid_labels_l = self.assemble_data_for_fast_delivery_partial(
            input_data_IN=x_valid, label_data_IN=y_valid)

        reformatted_input_test_data_l, reformatted_label_test_data_l, reformatted_test_labels_l = self.assemble_data_for_fast_delivery_partial(
            input_data_IN=x_test, label_data_IN=y_test)

        return ['train_data_INPUT_VEC',
                'train_data_LABEL_VEC',
                'train_data_LABEL_SCALAR',

                'valid_data_INPUT_VEC',
                'valid_data_LABEL_VEC',
                'valid_data_LABEL_SCALAR',

                'test_data_INPUT_VEC',
                'test_data_LABEL_VEC',
                'test_data_LABEL_SCALAR'
                ],[reformatted_input_data_l,
                    reformatted_label_data_l,
                    reformatted_labels_l,

                   reformatted_input_valid_data_l,
                   reformatted_label_valid_data_l,
                   reformatted_valid_labels_l,

                   reformatted_input_test_data_l,
                   reformatted_label_test_data_l,
                   reformatted_test_labels_l
                   ]

    def get_nmnist_data_point_with_NOISE(self,idx_IN,data_type_IN,data_list_IN,INPUT_DURATION_IN=300,time_var_IN=10):

        PATH = self.datasets + '/nmnist/' + str(data_type_IN) + '/'

        filename = PATH+str(data_list_IN[idx_IN][0])
        label = int(data_list_IN[idx_IN][1])

        with open(filename, 'rb') as input_file:
            compact_file = input_file.read()

        event_data = self.transform_compact_file_to_array(empty_array=np.zeros((2, 34, 34, INPUT_DURATION_IN)), compact_file_data=compact_file)

        i = np.where(event_data == 1)
        # new_time_idx = np.clip(np.round(i[3] + np.random.normal(0, time_var_IN)).astype(dtype=np.int32), 0, INPUT_DURATION_IN-1)
        new_time_idx = np.absolute(np.subtract(np.absolute(np.multiply(-1.0,np.subtract(np.absolute(np.round(i[3] + np.random.normal(0, time_var_IN)).astype(dtype=np.int32)),INPUT_DURATION_IN-1))),INPUT_DURATION_IN-1)).astype(dtype=np.int32)

        new_i = (i[0], i[1], i[2], new_time_idx)

        new_event_data = np.zeros((2, 34, 34, INPUT_DURATION_IN),dtype=np.int32)
        new_event_data[new_i]=1

        output_label_arr = -1.0*np.ones(10,dtype=np.float32)
        output_label_arr[label] = 1.0


        return new_event_data.reshape((2*34*34,INPUT_DURATION_IN)), output_label_arr, label
