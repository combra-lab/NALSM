# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
import NALSM_GEN_SUPPORT as sup


class CUBA_LIF_network:
    def __init__(self,number_of_networks_IN):
        self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.dataPath = self.rootPath + '/dataFiles/P1/networks'
        self.codePath = self.rootPath + '/codeFiles'

        self.number_of_networks = number_of_networks_IN

    # used in network creating, hence kept as old verion. reinitializer function converts for parallel network ops
    def initialize_network_parameters_v1(self, tau_v,tau_u,v_thrsh,b,w,w_mask,r,t_rfr):

        '''
        INPUTS:
        a,b,c,d VECTORS 
        w WEIGHT MATRIX/CONNECTIVITY MATRIX : axis=0 is output neurons, axis=1 is input neurons

        :return: TENSORFLOW VARIABLES: A,B,C,D,V,U,W,I 
        '''

        assert (np.shape(tau_v) == np.shape(tau_u))
        assert (np.shape(tau_v) == np.shape(v_thrsh))
        assert (np.shape(tau_v) == np.shape(b))
        assert ((np.shape(tau_v)[0], np.shape(tau_v)[0]) == np.shape(w))
        assert (np.shape(w) == np.shape(w_mask))
        assert (np.shape(tau_v) == np.shape(r))
        assert (np.shape(tau_v) == np.shape(t_rfr))


        TAU_V = tf.Variable(tau_v, dtype=tf.float32, expected_shape=[len(tau_v)], name='TAU_V')

        tau_u_mat = np.matmul(np.expand_dims(tau_u, axis=1), np.ones((1, len(tau_u)), dtype=np.float32))

        TAU_U = tf.Variable(tau_u_mat, dtype=tf.float32,
                            expected_shape=[np.shape(tau_u_mat)[0], np.shape(tau_u_mat)[1]], name='TAU_U')
        V_THRSH = tf.Variable(v_thrsh, dtype=tf.float32, expected_shape=[len(v_thrsh)], name='V_THRSH')
        B = tf.Variable(b, dtype=tf.float32, expected_shape=[len(b)], name='B')

        W = tf.Variable(w, dtype=tf.float32, expected_shape=[np.shape(w)[0], np.shape(w)[1]], name='W')
        W_mask = tf.Variable(w_mask, dtype=tf.float32, expected_shape=[np.shape(w)[0], np.shape(w)[1]], name='W_mask')


        V = tf.Variable(0.0 * np.ones(np.shape(tau_v), dtype=np.float32), dtype=tf.float32, expected_shape=[len(tau_v)],
                        name='V')


        I = tf.Variable(np.zeros(np.shape(b), dtype=np.float32), dtype=tf.float32, expected_shape=[len(b)],
                        name='I')

        S = tf.Variable(np.zeros(len(b), dtype=np.float32), dtype=tf.float32, expected_shape=[len(b)], name='S')

        S_store = tf.Variable(np.zeros((len(b), 1000), dtype=np.float32), dtype=tf.float32,
                              expected_shape=[len(b), 1000],
                              name='S_store')
        R = tf.Variable(r, dtype=tf.float32, expected_shape=[len(r)], name='R')
        T_RFR = tf.Variable(t_rfr, dtype=tf.float32, expected_shape=[len(t_rfr)], name='T_RFR')
        T_RFR_STATE = tf.Variable(np.zeros(len(t_rfr),dtype=np.float32), dtype=tf.float32, expected_shape=[len(t_rfr)], name='T_RFR_STATE')

        t_lim = 20.0
        max_steps = 1000
        counter = 0
        while(np.exp((-t_lim)/np.amax(tau_u))>0.01):
            t_lim = t_lim + 5.0
            if max_steps == counter:
                print('max_steps reached, high enough tlim not found...check')
                break
            counter = counter + 1

        print('Automatic t_lim finder got:')
        print('t_lim: '+str(t_lim))
        print('max tau_u exp val: ' + str(np.exp((-t_lim)/np.amax(tau_u))))
        print('min tau_u exp val: ' + str(np.exp((-t_lim) / np.amin(tau_u))))

        SS_T_STATE = tf.Variable(t_lim*w_mask, dtype=tf.float32, expected_shape=[np.shape(w)[0], np.shape(w)[1]], name='SS_t_state')

        ### generate constants list
        tf_inds = []
        for i in range(0, 1000):
            ind = []
            for j in range(0, len(b)):
                ind.append([j, i])
            tf_inds.append(tf.constant(ind))

        return ['TAU_V', 'TAU_U', 'V_THRSH', 'B', 'V', 'I', 'W', 'W_mask', 'S', 'S_store', 'SS_T_STATE', 'R', 'T_RFR', 'T_RFR_STATE', 'tf_inds'], [TAU_V, TAU_U, V_THRSH, B, V, I, W, W_mask, S, S_store, SS_T_STATE, R, T_RFR, T_RFR_STATE, tf_inds]


    def reinitialize_network_parameters_v2(self, tau_v_IN, tau_u_IN, v_thrsh_IN, b_IN, v_IN, w_IN, r_IN, w_mask_IN, ss_t_state_IN, t_rfr_IN, num_input_neurons_IN):

        '''
        INPUTS:
        a,b,c,d VECTORS 
        w WEIGHT MATRIX/CONNECTIVITY MATRIX : axis=0 is output neurons, axis=1 is input neurons

        :return: TENSORFLOW VARIABLES: A,B,C,D,V,U,W,I 
        '''

        assert ((np.shape(tau_v_IN)[0], np.shape(tau_v_IN)[0]) == np.shape(tau_u_IN))
        assert ((np.shape(tau_v_IN)[0], np.shape(tau_v_IN)[0]) == np.shape(ss_t_state_IN))
        assert (np.shape(tau_v_IN) == np.shape(v_thrsh_IN))
        assert (np.shape(tau_v_IN) == np.shape(b_IN))
        assert ((np.shape(tau_v_IN)[0], np.shape(tau_v_IN)[0]) == np.shape(w_IN))
        assert (np.shape(w_IN) == np.shape(w_mask_IN))
        assert (np.shape(w_IN) == np.shape(tau_u_IN))
        assert (np.shape(tau_v_IN) == np.shape(r_IN))
        # assert (np.shape(w_IN)[0] == self.number_of_networks)
        # assert (np.shape(b_IN) == self.number_of_networks)
        # assert (np.shape(v_thrsh_IN) == self.number_of_networks)
        # assert (np.shape(tau_u_IN) == self.number_of_networks)
        # assert (np.shape(tau_v_IN) == self.number_of_networks)
        # assert (np.shape(w_mask_IN)[0] == self.number_of_networks)


        # assmeble idx list for w_sparse to w_dense conversion

        syn_idx = np.where(w_mask_IN == 1.0)

        num_synapses_in_network = len(syn_idx[0])
        num_neurons_in_network = len(tau_v_IN)

        ss_t_state_dense_one_net_l = []
        tau_u_dense_one_net_l = []
        w_dense_one_net_l = []
        for i in range(0, len(syn_idx[0])):
            w_dense_one_net_l.append(w_IN[syn_idx[0][i], syn_idx[1][i]])
            tau_u_dense_one_net_l.append(tau_u_IN[syn_idx[0][i], syn_idx[1][i]])
            ss_t_state_dense_one_net_l.append(ss_t_state_IN[syn_idx[0][i], syn_idx[1][i]])

        w_dense_np = np.broadcast_to(np.asarray(w_dense_one_net_l), shape=(self.number_of_networks, num_synapses_in_network))
        tau_u_dense_np = np.broadcast_to(np.asarray(tau_u_dense_one_net_l), shape=(self.number_of_networks, num_synapses_in_network))
        ss_t_state_dense_np = np.broadcast_to(np.asarray(ss_t_state_dense_one_net_l),
                                         shape=(self.number_of_networks, num_synapses_in_network))

        ss_t_state_reset_scalar = ss_t_state_dense_one_net_l[0]

        # shape = (num nets, num_syns in one network)
        TAU_U_DENSE = tf.constant(tau_u_dense_np, dtype=tf.float32, shape=[self.number_of_networks, num_synapses_in_network], name='TAU_U')
        W_DENSE = tf.Variable(w_dense_np, dtype=tf.float32, expected_shape=[self.number_of_networks, num_synapses_in_network], name='W_DENSE')
        SS_T_STATE = tf.Variable(ss_t_state_dense_np, dtype=tf.float32, expected_shape=[self.number_of_networks, num_synapses_in_network],name='SS_t_state')

        # shape = (num nets, num_neurons in one network)
        TAU_V = tf.constant(np.broadcast_to(tau_v_IN,shape=(self.number_of_networks,num_neurons_in_network)), dtype=tf.float32, shape=[self.number_of_networks,num_neurons_in_network], name='TAU_V')
        V_THRSH = tf.constant(np.broadcast_to(v_thrsh_IN,shape=(self.number_of_networks, num_neurons_in_network)), dtype=tf.float32, shape=[self.number_of_networks, num_neurons_in_network], name='V_THRSH')
        B = tf.constant(np.broadcast_to(b_IN,shape=(self.number_of_networks, num_neurons_in_network)), dtype=tf.float32, shape=[self.number_of_networks, num_neurons_in_network], name='B')
        V = tf.Variable(np.zeros((self.number_of_networks, num_neurons_in_network), dtype=np.float32), dtype=tf.float32, expected_shape=[self.number_of_networks, num_neurons_in_network], name='V')
        S = tf.Variable(np.zeros((self.number_of_networks, num_neurons_in_network), dtype=np.float32), dtype=tf.float32, expected_shape=[self.number_of_networks, num_neurons_in_network], name='S')
        T_RFR = tf.constant(np.broadcast_to(t_rfr_IN, shape=(self.number_of_networks, num_neurons_in_network)),
                            dtype=tf.float32, shape=[self.number_of_networks, num_neurons_in_network], name='T_RFR')
        T_RFR_STATE = tf.Variable(np.zeros((self.number_of_networks, num_neurons_in_network), dtype=np.float32),
                                  dtype=tf.float32, expected_shape=[self.number_of_networks, num_neurons_in_network],
                                  name='T_RFR_STATE')

        # shape = (num nets, num INPUT neurons in one network)
        I_PH = tf.placeholder(dtype=tf.float32, shape=[self.number_of_networks,num_input_neurons_IN], name='ext_I_ph')

        INPUT_POISSON_VALS = tf.Variable(np.zeros((self.number_of_networks,num_input_neurons_IN),dtype=np.float32),dtype=tf.float32,expected_shape=[self.number_of_networks,num_input_neurons_IN])

        return ['TAU_U_DENSE','W_DENSE','SS_T_STATE','TAU_V','V_THRSH','B','V','S','T_RFR','T_RFR_STATE','I_PH','INPUT_POISSON_VALS','ss_t_state_reset_scalar'], [TAU_U_DENSE,W_DENSE,SS_T_STATE,TAU_V,V_THRSH,B,V,S,T_RFR,T_RFR_STATE,I_PH,INPUT_POISSON_VALS,ss_t_state_reset_scalar]


    def initialize_network_masks(self
                                 , num_total_neurons_IN
                                 , num_synapses_in_network_IN

                                 , S_mask_exc_inp_np_IN
                                 , S_mask_inh_np_IN
                                 # , S_mask_res_np_IN

                                 , W_dense_mask_exc_to_res_np_IN
                                 , W_dense_mask_inh_to_res_np_IN
                                 , W_dense_mask_inp_to_res_np_IN
                                 , W_dense_mask_res_to_out_np_IN
                                 # , W_dense_mask_res_np_IN
                                 ):


        assert (np.shape(S_mask_exc_inp_np_IN) == (1,num_total_neurons_IN))
        assert (np.shape(S_mask_inh_np_IN) == (1,num_total_neurons_IN))
        # assert (np.shape(S_mask_res_np_IN) == (1,num_total_neurons_IN))

        S_MASK_EXC_INP = tf.constant(S_mask_exc_inp_np_IN,dtype=tf.float32, shape=[1, num_total_neurons_IN], name='S_MASK_EXC_INP')

        S_MASK_INH = tf.constant(S_mask_inh_np_IN, dtype=tf.float32, shape=[1, num_total_neurons_IN], name='S_MASK_INH')

        # S_MASK_RES = tf.constant(S_mask_res_np_IN, dtype=tf.float32, shape=[1, num_total_neurons_IN], name='S_MASK_RES')


        W_DENSE_MASK_EXC_TO_RES = tf.constant(W_dense_mask_exc_to_res_np_IN, dtype=tf.float32, shape=[1, num_synapses_in_network_IN], name='W_DENSE_MASK_EXC_TO_RES')

        W_DENSE_MASK_INH_TO_RES = tf.constant(W_dense_mask_inh_to_res_np_IN, dtype=tf.float32, shape=[1, num_synapses_in_network_IN], name='W_DENSE_MASK_INH_TO_RES')

        W_DENSE_MASK_INP_TO_RES = tf.constant(W_dense_mask_inp_to_res_np_IN, dtype=tf.float32, shape=[1, num_synapses_in_network_IN], name='W_DENSE_MASK_INP_TO_RES')

        W_DENSE_MASK_RES_TO_OUT = tf.constant(W_dense_mask_res_to_out_np_IN, dtype=tf.float32, shape=[1, num_synapses_in_network_IN], name='W_DENSE_MASK_RES_TO_OUT')

        return ['S_MASK_EXC_INP'
                   ,'S_MASK_INH'

                   ,'W_DENSE_MASK_EXC_TO_RES'
                   ,'W_DENSE_MASK_INH_TO_RES'
                   ,'W_DENSE_MASK_INP_TO_RES'
                   ,'W_DENSE_MASK_RES_TO_OUT'
                   ],[
                    S_MASK_EXC_INP
                    ,S_MASK_INH

                    ,W_DENSE_MASK_EXC_TO_RES
                    ,W_DENSE_MASK_INH_TO_RES
                    ,W_DENSE_MASK_INP_TO_RES
                    ,W_DENSE_MASK_RES_TO_OUT
                    ]



    # PROJECTION INDICES NEED MODIFICATION
    def initialize_spike_store(self, s_IN, input_duration_IN, record_nets_l_IN, num_neurons_in_single_network_IN):

        assert(len(record_nets_l_IN)>0)
        for iii in range(0,len(record_nets_l_IN)):
            assert (self.number_of_networks>record_nets_l_IN[iii])



        gath_ind = []
        for kk in range(0, len(record_nets_l_IN)):
            for j in range(0, num_neurons_in_single_network_IN):
                gath_ind.append([record_nets_l_IN[kk], j])
        gather_ind_tf = tf.constant(gath_ind)

        # print('input_duration_IN',input_duration_IN)
        # print('record_nets_l_IN', record_nets_l_IN)
        # print('num_neurons_in_single_network_IN', num_neurons_in_single_network_IN)

        scatter_inds = []
        for i in range(0, input_duration_IN):
            ind_temp = []
            for kk in range(0, len(record_nets_l_IN)):
                for j in range(0, num_neurons_in_single_network_IN):
                    ind_temp.append([kk, j, i])
            # print('scatter_ind t: ' + str(i))
            # print(ind_temp)
            scatter_inds.append(tf.constant(ind_temp))

        ## INT8 VAR TYPE shape (num_batch_nets, num of neurons in single network, duration of sampling in ms)
        S_STORE = tf.Variable(np.zeros((self.number_of_networks, num_neurons_in_single_network_IN, input_duration_IN), dtype=np.int8), dtype=tf.int8,
                              expected_shape=[self.number_of_networks, num_neurons_in_single_network_IN, input_duration_IN], name='S_store')

        spike_save_ops = []
        for i in range(0, input_duration_IN):
            # spike_save_ops.append(tf.gather_nd(params=s_ph, indices=tf.constant(gath_ind)))
            spike_save_ops.append(tf.assign(S_STORE, tf.add(S_STORE, tf.scatter_nd(scatter_inds[i], tf.cast(
                tf.gather_nd(params=s_IN, indices=gather_ind_tf), dtype=tf.int8), shape=[self.number_of_networks,
                                                                                                 num_neurons_in_single_network_IN,
                                                                                                 input_duration_IN]))))

        zero_out_S_STORE_op = tf.assign(S_STORE, tf.scalar_mul(0, S_STORE))

        # condenses 3d mat into d mat with columns: (batch sample, ft, fn)
        condense_spike_store_to_Fn_Ft_op = tf.where(tf.equal(tf.transpose(S_STORE,perm=[0,2,1]), 1))

        return ['spike_save_ops','zero_out_S_STORE_op','condense_spike_store_to_Fn_Ft_op'],[spike_save_ops,zero_out_S_STORE_op,condense_spike_store_to_Fn_Ft_op]

    def initialize_W_DENSE_store(self,W_DENSE_IN, num_synapses_in_network_IN):

        W_DENSE_STORE = tf.Variable(np.zeros((self.number_of_networks, num_synapses_in_network_IN), dtype=np.float32), dtype=tf.float32,
                                    expected_shape=[self.number_of_networks, num_synapses_in_network_IN], name='W_DENSE_STORE')

        save_W_op = tf.assign(W_DENSE_STORE,W_DENSE_IN)

        reset_W_from_saved_state_op = tf.assign(W_DENSE_IN,W_DENSE_STORE)

        return ['W_DENSE_STORE','save_W_op','reset_W_from_saved_state_op'],[W_DENSE_STORE,save_W_op,reset_W_from_saved_state_op]


    # SLIM UPDATED
    def update_neuron_states_wo_R(self, v_IN, tau_v_IN, tau_u_dense_IN, b_IN, ss_t_state_IN, W_dense_IN, I_poisson_IN, scatter_idx_W_dense_to_Neurons_IN,scatter_shp_IN,scatter_S_to_S_SUBSET_inputNeurons_IN,batch_by_num_input_neurons_l_IN,input_current_IN):
        ### FULLY UPDATED WITH 0 CAST [FLOAT32]: 1 SEC = 0.47

        # PARALLEL COMPATIBLE
        # COMPUTE CURRENT INPUT
        input_currents = tf.scalar_mul(input_current_IN,tf.ceil(tf.subtract(I_poisson_IN, tf.random_uniform(shape=[batch_by_num_input_neurons_l_IN[0], batch_by_num_input_neurons_l_IN[1]], dtype=tf.float32))))

        # PARALLEL COMPATIBLE
        v_leak = tf.multiply(tf.divide(v_IN, tau_v_IN), -1.0)

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION
        # this is projetion of tau_u constant from W_sparse to W_dense ----> THIS NEEDS TO BE REMOVED FROM HERE, JUST INITIALIZE TAU_U IN DENSE FORM FROM START********
        # project_tau_u_sparse_to_SS = tf.gather_nd(params=tau_u_sparse_IN,indices=gather_idx_Wsparse_to_Wdense_IN)

        # print(project_tau_u_Neuron_to_SS)
        # currents_debug1 = tf.exp(tf.divide(tf.scalar_mul(-1.0, ss_t_state_IN), project_tau_u_Neuron_to_SS))
        # currents_debug2 = tf.divide(currents_debug1,project_tau_u_Neuron_to_SS)

        # PARALLEL COMPATIBLE
        currents = tf.divide(tf.exp(tf.divide(tf.scalar_mul(-1.0, ss_t_state_IN), tau_u_dense_IN)), tau_u_dense_IN)

        # condense_w_to_ss = tf.gather_nd(params=w_IN, indices=gath_w_to_ss_idx_IN)

        # PARALLEL COMPATIBLE
        synaptic_input = tf.multiply(W_dense_IN, currents)

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_scatternd_0.py********
        project_ss_to_v = tf.scatter_nd(indices=scatter_idx_W_dense_to_Neurons_IN,
                                        updates=synaptic_input, shape=scatter_shp_IN)

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION ---> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_scatternd_1.py********
        project_inputNeuron_I_into_allNeuron_vector = tf.scatter_nd(indices=scatter_S_to_S_SUBSET_inputNeurons_IN,updates=input_currents,shape=scatter_shp_IN)

        # PARALLEL COMPATIBLE
        new_v = tf.add_n([v_IN, project_ss_to_v, project_inputNeuron_I_into_allNeuron_vector, b_IN, v_leak])

        return tf.assign(v_IN,new_v)

    # with gaussian noise
    def update_neuron_states_wo_R_w_NOISE(self, v_IN, tau_v_IN, tau_u_dense_IN, b_IN, ss_t_state_IN, W_dense_IN, I_poisson_IN, scatter_idx_W_dense_to_Neurons_IN,scatter_shp_IN,scatter_S_to_S_SUBSET_inputNeurons_IN,batch_by_num_input_neurons_l_IN,input_current_IN):
        ### FULLY UPDATED WITH 0 CAST [FLOAT32]: 1 SEC = 0.47

        # PARALLEL COMPATIBLE
        # COMPUTE CURRENT INPUT
        poisson_with_noise = tf.add(I_poisson_IN,tf.random.normal(mean=0.0,stddev=0.025,shape=[batch_by_num_input_neurons_l_IN[0], batch_by_num_input_neurons_l_IN[1]], dtype=tf.float32))
        input_currents = tf.scalar_mul(input_current_IN,tf.ceil(tf.subtract(poisson_with_noise, tf.random_uniform(shape=[batch_by_num_input_neurons_l_IN[0], batch_by_num_input_neurons_l_IN[1]], dtype=tf.float32))))

        # PARALLEL COMPATIBLE
        v_leak = tf.multiply(tf.divide(v_IN, tau_v_IN), -1.0)

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION
        # this is projetion of tau_u constant from W_sparse to W_dense ----> THIS NEEDS TO BE REMOVED FROM HERE, JUST INITIALIZE TAU_U IN DENSE FORM FROM START********
        # project_tau_u_sparse_to_SS = tf.gather_nd(params=tau_u_sparse_IN,indices=gather_idx_Wsparse_to_Wdense_IN)

        # print(project_tau_u_Neuron_to_SS)
        # currents_debug1 = tf.exp(tf.divide(tf.scalar_mul(-1.0, ss_t_state_IN), project_tau_u_Neuron_to_SS))
        # currents_debug2 = tf.divide(currents_debug1,project_tau_u_Neuron_to_SS)

        # PARALLEL COMPATIBLE
        currents = tf.divide(tf.exp(tf.divide(tf.scalar_mul(-1.0, ss_t_state_IN), tau_u_dense_IN)), tau_u_dense_IN)

        # condense_w_to_ss = tf.gather_nd(params=w_IN, indices=gath_w_to_ss_idx_IN)

        # PARALLEL COMPATIBLE
        synaptic_input = tf.multiply(W_dense_IN, currents)

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_scatternd_0.py********
        project_ss_to_v = tf.scatter_nd(indices=scatter_idx_W_dense_to_Neurons_IN,
                                        updates=synaptic_input, shape=scatter_shp_IN)

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION ---> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_scatternd_1.py********
        project_inputNeuron_I_into_allNeuron_vector = tf.scatter_nd(indices=scatter_S_to_S_SUBSET_inputNeurons_IN,updates=input_currents,shape=scatter_shp_IN)

        # PARALLEL COMPATIBLE
        new_v = tf.add_n([v_IN, project_ss_to_v, project_inputNeuron_I_into_allNeuron_vector, b_IN, v_leak])

        return tf.assign(v_IN,new_v)



    # # SLIM UPDATED
    # def update_neuron_states_w_R(self, v_IN, tau_v_IN, tau_u_sparse_IN, b_IN, r_IN, ss_t_state_IN, W_dense_IN, i_IN,
    #                               scatter_idx_W_dense_to_Neurons_IN, scatter_shp_IN,
    #                               gather_idx_Wsparse_to_Wdense_IN, scatter_S_to_S_SUBSET_inputNeurons_IN):
    #     ### FULLY UPDATED WITH 0 CAST [FLOAT32]: 1 SEC = 0.47
    #
    #     # v_leak = tf.multiply(tf.divide(v_IN, tau_v_IN), -1.0)
    #     # v_leak = tf.multiply(-1.0, v_IN)
    #
    #     # this is projetion of tau_u constant from W_sparse to W_dense
    #     project_tau_u_sparse_to_SS = tf.gather_nd(params=tau_u_sparse_IN, indices=gather_idx_Wsparse_to_Wdense_IN)
    #
    #     # print(project_tau_u_Neuron_to_SS)
    #     # currents_debug1 = tf.exp(tf.divide(tf.scalar_mul(-1.0, ss_t_state_IN), project_tau_u_Neuron_to_SS))
    #     # currents_debug2 = tf.divide(currents_debug1,project_tau_u_Neuron_to_SS)
    #
    #     currents = tf.divide(tf.exp(tf.divide(tf.scalar_mul(-1.0, ss_t_state_IN), project_tau_u_sparse_to_SS)),
    #                          project_tau_u_sparse_to_SS)
    #
    #     # condense_w_to_ss = tf.gather_nd(params=w_IN, indices=gath_w_to_ss_idx_IN)
    #
    #
    #     synaptic_input = tf.multiply(W_dense_IN, currents)
    #
    #     project_ss_to_v = tf.scatter_nd(indices=scatter_idx_W_dense_to_Neurons_IN,
    #                                     updates=synaptic_input, shape=scatter_shp_IN)
    #
    #     project_inputNeuron_I_into_allNeuron_vector = tf.scatter_nd(indices=scatter_S_to_S_SUBSET_inputNeurons_IN,
    #                                                                 updates=i_IN, shape=scatter_shp_IN)
    #
    #     new_v = tf.add(v_IN,tf.divide(tf.subtract(tf.multiply(r_IN,tf.add_n([project_ss_to_v, project_inputNeuron_I_into_allNeuron_vector, b_IN])), v_IN),tau_v_IN))
    #     # new_v = tf.add_n([v_IN, project_ss_to_v, project_inputNeuron_I_into_allNeuron_vector, b_IN, v_leak])
    #
    #     return tf.assign(v_IN, new_v)



    # SLIM UPDATED
    def propagate_spikes(self, spikes_IN, ss_t_state_IN, gather_idx_Sin_to_Wdense_IN):
        #  1 sec = 0.186

        # PARALLEL COMPATIBLE
        spikes_rev = tf.add(tf.multiply(spikes_IN, -1.0), 1.0)

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_2.py
        project_s_to_ss = tf.gather_nd(params=spikes_rev, indices=gather_idx_Sin_to_Wdense_IN)

        # PARALLEL COMPATIBLE
        select_zero_out_t_syn_ref = tf.multiply(ss_t_state_IN, project_s_to_ss)

        return tf.assign(ss_t_state_IN,select_zero_out_t_syn_ref)

    # PARALLEL COMPATIBLE
    # SLIM UPDATED
    def evolve_input_currents(self, ss_t_state_IN, integration_step_size_IN=1.0, max_dt_val=100.0):
        # 1 SEC = 0.177

        # PARALLEL COMPATIBLE
        new_ss_t_state = tf.clip_by_value(tf.add(ss_t_state_IN, integration_step_size_IN),
                                          clip_value_min=0.0, clip_value_max=max_dt_val)

        return tf.assign(ss_t_state_IN,new_ss_t_state)

    # PARALLEL COMPATIBLE
    # SLIM UPDATED
    def register_spikes(self, v_IN, v_trsh_IN, s_IN, t_rfr_state_IN):
        # 1 sec = 0.19 with slight modifications

        # PARALLEL COMPATIBLE
        # t_rfr_state represents count down to 0 where neuron is out of rfr period
        # binary t_rfr_state reversed: +0 -> 0, 0-> 1
        binary_rev_rfr = tf.add(tf.multiply(tf.clip_by_value(t_rfr_state_IN, 0.0, 1.0), -1.0), 1.0)  # FASTER

        # PARALLEL COMPATIBLE
        # 0 FOR SPIKE, 1 FOR NO SPIKE
        spike_vec_filt = tf.subtract(1.0,tf.ceil(tf.clip_by_value(tf.subtract(v_trsh_IN, tf.multiply(v_IN, binary_rev_rfr)), 0.0, 1.0))) # FASTER

        # 1 FOR SPIKE, 0 FOR NO SPIKE
        return tf.assign(s_IN, spike_vec_filt)

    # PARALLEL COMPATIBLE
    # SLIM UPDATED
    def reset_spiked_neurons(self, s_IN,v_IN):
        # PARALLEL COMPATIBLE
        # 1 sec = 0.15 sec
        # 0 FOR SPIKE, 1 FOR NO SPIKE
        inv_spike_vec = tf.add(tf.multiply(s_IN, -1.0), 1.0) # - CHANGE THIS IS FASTER WAY

        # PARALLEL COMPATIBLE
        # computes new v,u for spiked neurons
        new_v = tf.multiply(inv_spike_vec, v_IN)

        return tf.assign(v_IN,new_v)

    # PARALLEL COMPATIBLE
    def update_refractory_var(self, t_rfr_IN, t_rfr_state_IN, s_IN):
        # inverse spike vec 1 -> 0, 0->1
        # inv_S = tf.abs(tf.subtract(S, 1.0))
        inv_S = tf.add(tf.multiply(s_IN, -1.0),1.0)

        # PARALLEL COMPATIBLE
        # decay by 1 ms and setting new refractory period for new spikes
        new_T_RFR_STATE = tf.add(tf.multiply(tf.subtract(t_rfr_state_IN, 1.0), inv_S), tf.multiply(s_IN, t_rfr_IN))

        return tf.assign(t_rfr_state_IN, new_T_RFR_STATE)

    def set_new_input_batch(self,I_PH_IN,INPUT_POISSON_VALS_IN):

        return tf.assign(INPUT_POISSON_VALS_IN,I_PH_IN)

    def zero_out_V(self,V_IN):

        return tf.assign(V_IN,tf.scalar_mul(0.0,V_IN))

    def zero_out_S(self, S_IN):

        return tf.assign(S_IN, tf.scalar_mul(0.0, S_IN))

    def zero_out_T_RFR_STATE(self, T_RFR_STATE_IN):

        return tf.assign(T_RFR_STATE_IN, tf.scalar_mul(0.0, T_RFR_STATE_IN))

    def reset_ss_t_state_op(self,SS_T_STATE_IN,ss_t_state_reset_scalar_IN):
        return tf.assign(SS_T_STATE_IN,tf.add(ss_t_state_reset_scalar_IN,tf.scalar_mul(0.0,SS_T_STATE_IN)))


    def open_network_stuctures(self,net_name,netPath):
        # data_fn = os.path.abspath(os.path.join(netPath, net_name))
        names,data = sup.unpack_file(net_name,netPath)

        # 0     1       2        3  4  5  6  7
        # tau_v, tau_u, v_thrsh, b, v, i, w, w_mask
        # 0      1      2        3  4  5  6  7       8  9        10          11           12
        # TAU_V, TAU_U, V_THRSH, B, V, I, W, W_mask, S, S_store, SS_T_STATE, tf_inds
        # TAU_V, TAU_U, V_THRSH, B, V, I, W, W_mask, S, S_store, SS_T_STATE, ext_I_ph, tf_inds
        # NEW VERSION
        # tau_v, tau_u, v_thrsh, b, v, i, w, w_mask, ss_t_state, r, t_rfr
        # TAU_V, TAU_U, V_THRSH, B, V, I, W, W_mask, S, S_store, SS_T_STATE, ext_I_ph, R, T_RFR, T_RFR_STATE, tf_inds

        TAU_V = data[names.index('TAU_V')]
        TAU_U = data[names.index('TAU_U')]
        V_THRSH = data[names.index('V_THRSH')]
        B = data[names.index('B')]
        V = data[names.index('V')]
        # I = data[names.index('I')]
        W = data[names.index('W')]
        W_MASK = data[names.index('W_mask')]
        SS_T_STATE = data[names.index('SS_T_STATE')]
        R = data[names.index('R')]
        T_RFR = data[names.index('T_RFR')]

        neuron_ranges_dict = dict(data[names.index('neuron_ranges')])
        inp_range = neuron_ranges_dict['inp_range']

        num_input_neurons = inp_range[1]-inp_range[0]

        names_t,data_t = self.reinitialize_network_parameters_v2(tau_v_IN=TAU_V,tau_u_IN=TAU_U,v_thrsh_IN=V_THRSH,b_IN=B,v_IN=V,r_IN=R,w_IN=W,w_mask_IN=W_MASK,ss_t_state_IN=SS_T_STATE, t_rfr_IN = T_RFR, num_input_neurons_IN=num_input_neurons)

        num_neurons = len(data[0])

        return names_t,data_t,num_neurons,names,data


    def open_network_stuctures_w_custom_W_set(self,net_name,netPath,set_w_ee_ei_ie_ii_inp_IN):
        # data_fn = os.path.abspath(os.path.join(netPath, net_name))
        names,data = sup.unpack_file(net_name,netPath)

        # 0     1       2        3  4  5  6  7
        # tau_v, tau_u, v_thrsh, b, v, i, w, w_mask
        # 0      1      2        3  4  5  6  7       8  9        10          11           12
        # TAU_V, TAU_U, V_THRSH, B, V, I, W, W_mask, S, S_store, SS_T_STATE, tf_inds
        # TAU_V, TAU_U, V_THRSH, B, V, I, W, W_mask, S, S_store, SS_T_STATE, ext_I_ph, tf_inds
        # NEW VERSION
        # tau_v, tau_u, v_thrsh, b, v, i, w, w_mask, ss_t_state, r, t_rfr
        # TAU_V, TAU_U, V_THRSH, B, V, I, W, W_mask, S, S_store, SS_T_STATE, ext_I_ph, R, T_RFR, T_RFR_STATE, tf_inds

        TAU_V = data[names.index('TAU_V')]
        TAU_U = data[names.index('TAU_U')]
        V_THRSH = data[names.index('V_THRSH')]
        B = data[names.index('B')]
        V = data[names.index('V')]
        # I = data[names.index('I')]
        W = data[names.index('W')]
        W_MASK = data[names.index('W_mask')]
        SS_T_STATE = data[names.index('SS_T_STATE')]
        R = data[names.index('R')]
        T_RFR = data[names.index('T_RFR')]

        sign_W = np.sign(data[names.index('W')])

        neuron_ranges_dict = dict(data[names.index('neuron_ranges')])
        res_range = neuron_ranges_dict['res_range']
        res_exc_range = neuron_ranges_dict['res_exc_range']
        res_inh_range = neuron_ranges_dict['res_inh_range']
        inp_range = neuron_ranges_dict['inp_range']
        out_range = neuron_ranges_dict['out_range']

        # setting E->E weights
        W[res_exc_range[0]:res_exc_range[1], res_exc_range[0]:res_exc_range[1]] = set_w_ee_ei_ie_ii_inp_IN[0]
        # setting E->I weights
        W[res_inh_range[0]:res_inh_range[1], res_exc_range[0]:res_exc_range[1]] = set_w_ee_ei_ie_ii_inp_IN[1]
        # setting I->E weights
        W[res_exc_range[0]:res_exc_range[1], res_inh_range[0]:res_inh_range[1]] = set_w_ee_ei_ie_ii_inp_IN[2]
        # setting I->I weights
        W[res_inh_range[0]:res_inh_range[1], res_inh_range[0]:res_inh_range[1]] = set_w_ee_ei_ie_ii_inp_IN[3]
        # setting inp->res weights
        W[res_range[0]:res_range[1], inp_range[0]:inp_range[1]] = set_w_ee_ei_ie_ii_inp_IN[4]

        # new_W = np.multiply(new_temp_W,signed_mask_W)
        new_W = np.multiply(np.multiply(W, sign_W),W_MASK)

        print('got new_W')

        # E -> E
        unique_w_ee, counts_unique_w_ee =  np.unique(new_W[res_exc_range[0]:res_exc_range[1], res_exc_range[0]:res_exc_range[1]],return_counts=True)
        print('EE_unique w and counts: ', unique_w_ee, counts_unique_w_ee)
        # E -> I
        unique_w_ei, counts_unique_w_ei = np.unique(
            new_W[res_inh_range[0]:res_inh_range[1], res_exc_range[0]:res_exc_range[1]], return_counts=True)
        print('EI_unique w and counts: ', unique_w_ei, counts_unique_w_ei)
        # I -> E
        unique_w_ie, counts_unique_w_ie = np.unique(
            new_W[res_exc_range[0]:res_exc_range[1], res_inh_range[0]:res_inh_range[1]], return_counts=True)
        print('IE_unique w and counts: ', unique_w_ie, counts_unique_w_ie)
        # I -> I
        unique_w_ii, counts_unique_w_ii = np.unique(
            new_W[res_inh_range[0]:res_inh_range[1], res_inh_range[0]:res_inh_range[1]], return_counts=True)
        print('IE_unique w and counts: ', unique_w_ii, counts_unique_w_ii)
        # INP -> E
        unique_w_inp_res, counts_unique_w_inp_res = np.unique(
            new_W[res_range[0]:res_range[1], inp_range[0]:inp_range[1]], return_counts=True)
        print('INP_RES_unique w and counts: ', unique_w_inp_res, counts_unique_w_inp_res)

        num_input_neurons = inp_range[1]-inp_range[0]

        names_t,data_t = self.reinitialize_network_parameters_v2(tau_v_IN=TAU_V,tau_u_IN=TAU_U,v_thrsh_IN=V_THRSH,b_IN=B,v_IN=V,r_IN=R,w_IN=new_W,w_mask_IN=W_MASK,ss_t_state_IN=SS_T_STATE, t_rfr_IN = T_RFR, num_input_neurons_IN=num_input_neurons)

        num_neurons = len(data[0])

        return names_t,data_t,num_neurons,names,data

