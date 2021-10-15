import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os


class stdp:
    def __init__(self,
                 stdp_LR_fixed_potentiation,
                 res_LR_decay_factor,
                 number_of_networks_IN,

                 w_res_exc_min_max,
                 w_res_inh_min_max,
                 w_res_inp_min_max,
                 w_out_res_min_max,

                 stdp_a_const=0.1,
                 stdp_tau=10.0,
                 ):

        self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.dataPath = self.rootPath + '/dataFiles/crit_nan/networks'
        self.codePath = self.rootPath + '/codeFiles'

        # self.a_minus = a_minus_plus[0]
        self.a_const = stdp_a_const
        self.number_of_networks = number_of_networks_IN
        self.stdp_LR_fixed_potentiation = stdp_LR_fixed_potentiation
        self.res_LR_decay_factor = res_LR_decay_factor

        # tm = self.approximate_stdp_decay_multiplier(tau=int(tau_minus_plus[0]))
        # self.tau_minus_decay = tm

        tp = self.approximate_stdp_decay_multiplier(tau=int(stdp_tau))
        self.stdp_tau = tp

        self.min_w_res_exc = w_res_exc_min_max[0]
        self.max_w_res_exc = w_res_exc_min_max[1]
        self.min_w_res_inh = w_res_inh_min_max[0]
        self.max_w_res_inh = w_res_inh_min_max[1]
        self.min_w_res_inp = w_res_inp_min_max[0]
        self.max_w_res_inp = w_res_inp_min_max[1]
        self.min_w_out_res = w_out_res_min_max[0]
        self.max_w_out_res = w_out_res_min_max[1]




    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    def approximate_stdp_decay_multiplier(self, tau, plot_approx='on'):
        # tau = 20

        ratio_to_reach = (1 / np.e)

        x = np.linspace(0, tau, tau * 100)
        y = np.exp(-x / tau)

        approx_mults = np.linspace(0.01, 0.9999, 1000)
        state = np.ones(np.shape(approx_mults))
        for i in range(0, tau):
            state = np.multiply(approx_mults, state)

        na, ind = self.find_nearest(state, y[-1])

        approx_multiple = approx_mults[ind]

        dec_val = 1.0
        y_list = [dec_val]
        for i in range(0, tau):
            dec_val = dec_val * approx_multiple
            y_list.append(dec_val)

        print(approx_multiple, na)

        return approx_multiple

    def initialize_learning_parameters_v0(self, num_total_neurons_IN):

        '''
        INPUTS:
        trace_mask VECTOR 
        w WEIGHT MATRIX/CONNECTIVITY MATRIX : axis=0 is output neurons, axis=1 is input neurons

        :return: TENSORFLOW VARIABLES: A,B,C,D,V,U,W,I 
        '''

        # shape: (num of nets in batch, num of neurons in single network)
        STDP_TRACE = tf.Variable(np.zeros((self.number_of_networks,num_total_neurons_IN),dtype=np.float32), dtype=tf.float32, expected_shape=[self.number_of_networks,num_total_neurons_IN], name='TRACE')

        # scalar learning rate
        STDP_POTENTIATION_LR = tf.Variable(self.stdp_LR_fixed_potentiation, dtype=tf.float32, expected_shape=[], name='STDP_POTENTIATION_LR')
        STDP_POTENTIATION_LR_STORE = tf.constant(self.stdp_LR_fixed_potentiation, dtype=tf.float32, shape=[],name='STDP_POTENTIATION_LR_STORE')


        return ['STDP_TRACE', 'STDP_POTENTIATION_LR', 'STDP_POTENTIATION_LR_STORE'], [STDP_TRACE, STDP_POTENTIATION_LR, STDP_POTENTIATION_LR_STORE]

    def update_trace(self,trace_IN,S_IN):

        return tf.assign(trace_IN,tf.scalar_mul(self.stdp_tau,tf.add(tf.scalar_mul(self.a_const,S_IN),trace_IN)))



    def STDP_w_astro_depression(self
                               , W_dense_IN
                               , S_IN

                               # STDP INPUTS
                               , trace_IN

                               , W_dense_mask_exc_to_res_IN
                               , W_dense_mask_inh_to_res_IN
                               , W_dense_mask_inp_to_res_IN
                               , W_dense_mask_res_to_out_IN

                               , S_mask_exc_and_inp_IN
                               , S_mask_inh_IN

                               , astro_LR_IN
                               # , LR_fixed_potentiation_IN

                               , gather_idx_Sin_to_Wdense_IN
                               , gather_idx_Sout_to_Wdense_IN

                               , stdp_potentiation_LR_IN
                                ):


        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION
        #### NEW DEBUGGED STDP ####
        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_2.py
        project_spikes_in_to_W_dense = tf.gather_nd(params=S_IN, indices=gather_idx_Sin_to_Wdense_IN)
        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_4.py
        project_spikes_out_to_W_dense = tf.gather_nd(params=S_IN, indices=gather_idx_Sout_to_Wdense_IN)

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_2.py
        project_trace_in_to_W_dense = tf.gather_nd(params=trace_IN, indices=gather_idx_Sin_to_Wdense_IN)
        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_4.py
        project_trace_out_to_W_dense = tf.gather_nd(params=trace_IN, indices=gather_idx_Sout_to_Wdense_IN)

        # PARALLEL INCOMPATIBLE
        s_post_x_t_pre_lr = tf.add(tf.scalar_mul(stdp_potentiation_LR_IN, S_mask_exc_and_inp_IN),
                                   tf.multiply(tf.negative(astro_LR_IN), S_mask_inh_IN))
        s_pre_x_t_post_lr = tf.add(tf.multiply(tf.negative(astro_LR_IN), S_mask_exc_and_inp_IN),
                                   tf.scalar_mul(stdp_potentiation_LR_IN, S_mask_inh_IN))

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_2.py
        project_s_post_x_t_pre_lr_to_W_dense = tf.gather_nd(params=s_post_x_t_pre_lr, indices=gather_idx_Sin_to_Wdense_IN)
        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_2.py
        project_s_pre_x_t_post_lr_to_W_dense = tf.gather_nd(params=s_pre_x_t_post_lr, indices=gather_idx_Sin_to_Wdense_IN)

        # PARALLEL COMPATIBLE
        s_post_x_t_pre = tf.multiply(project_trace_in_to_W_dense, project_spikes_out_to_W_dense)
        s_pre_x_t_post = tf.multiply(project_trace_out_to_W_dense, project_spikes_in_to_W_dense)

        # PARALLEL COMPATIBLE
        delta_w__s_post_x_t_pre = tf.multiply(project_s_post_x_t_pre_lr_to_W_dense, s_post_x_t_pre)
        delta_w__s_pre_x_t_post = tf.multiply(project_s_pre_x_t_post_lr_to_W_dense, s_pre_x_t_post)

        # PARALLEL COMPATIBLE
        new_W_raw = tf.add_n([W_dense_IN, delta_w__s_post_x_t_pre, delta_w__s_pre_x_t_post])

        # PARALLEL COMPATIBLE
        new_W_res_exc = tf.multiply(W_dense_mask_exc_to_res_IN, tf.clip_by_value(new_W_raw, self.min_w_res_exc, self.max_w_res_exc))
        new_W_res_inh = tf.multiply(W_dense_mask_inh_to_res_IN, tf.clip_by_value(new_W_raw, self.min_w_res_inh, self.max_w_res_inh))
        new_W_res_inp = tf.multiply(W_dense_mask_inp_to_res_IN, tf.clip_by_value(new_W_raw, self.min_w_res_inp, self.max_w_res_inp))
        new_W_out_res = tf.multiply(W_dense_mask_res_to_out_IN, W_dense_IN)

        ######## STDP OPS ######### END

        # PARALLEL COMPATIBLE
        new_W_dense_final = tf.add_n([new_W_res_exc, new_W_res_inh, new_W_res_inp, new_W_out_res])

        return tf.assign(W_dense_IN,new_W_dense_final)




    # PARALLEL COMPATIBLE
    # def decay_output_lr(self, readout_lr_IN):
    #
    #     # DECAY FACTOR RANGES 0-1, WITH NO DECAY WHEN ITS 1
    #     new_readout_lr = tf.clip_by_value(tf.scalar_mul(self.ro_LR_decay_factor,readout_lr_IN),self.min_readout_lr_IN,10000.0)
    #
    #     return tf.assign(readout_lr_IN,new_readout_lr)

    # PARALLEL COMPATIBLE
    def decay_res_lr(self, res_lr_IN):

        # DECAY FACTOR RANGES 0-1, WITH NO DECAY WHEN ITS 1
        new_readout_lr = tf.clip_by_value(tf.scalar_mul(self.res_LR_decay_factor,res_lr_IN),0.0,10000.0)

        return tf.assign(res_lr_IN,new_readout_lr)

    # PARALLEL COMPATIBLE
    def reset_res_lr(self, res_lr_IN, res_lr_store_IN):
        return tf.assign(res_lr_IN, res_lr_store_IN)



    ###### PURE STDP #####
    def STDP_PURE(self
                               , W_dense_IN
                               , S_IN

                               # STDP INPUTS
                               , trace_IN

                               , W_dense_mask_exc_to_res_IN
                               , W_dense_mask_inh_to_res_IN
                               , W_dense_mask_inp_to_res_IN
                               , W_dense_mask_res_to_out_IN

                               , S_mask_exc_and_inp_IN
                               , S_mask_inh_IN

                               # , astro_LR_IN
                               # , LR_fixed_potentiation_IN

                               , gather_idx_Sin_to_Wdense_IN
                               , gather_idx_Sout_to_Wdense_IN

                               , stdp_potentiation_LR_IN
                                ):


        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION
        #### NEW DEBUGGED STDP ####
        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_2.py
        project_spikes_in_to_W_dense = tf.gather_nd(params=S_IN, indices=gather_idx_Sin_to_Wdense_IN)
        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_4.py
        project_spikes_out_to_W_dense = tf.gather_nd(params=S_IN, indices=gather_idx_Sout_to_Wdense_IN)

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_2.py
        project_trace_in_to_W_dense = tf.gather_nd(params=trace_IN, indices=gather_idx_Sin_to_Wdense_IN)
        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_4.py
        project_trace_out_to_W_dense = tf.gather_nd(params=trace_IN, indices=gather_idx_Sout_to_Wdense_IN)

        # PARALLEL INCOMPATIBLE
        s_post_x_t_pre_lr = tf.add(tf.scalar_mul(stdp_potentiation_LR_IN, S_mask_exc_and_inp_IN),
                                   tf.multiply(tf.negative(stdp_potentiation_LR_IN), S_mask_inh_IN))
        s_pre_x_t_post_lr = tf.add(tf.multiply(tf.negative(stdp_potentiation_LR_IN), S_mask_exc_and_inp_IN),
                                   tf.scalar_mul(stdp_potentiation_LR_IN, S_mask_inh_IN))

        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_2.py
        project_s_post_x_t_pre_lr_to_W_dense = tf.gather_nd(params=s_post_x_t_pre_lr, indices=gather_idx_Sin_to_Wdense_IN)
        # PARALLEL INCOMPATIBLE - PROJECTION INDICES NEED MODIFICATION --------> tested new index in codeFiles/slim_parallel_code/tst_mapps_funcs/funcTst_LIF_CORE2_gathernd_2.py
        project_s_pre_x_t_post_lr_to_W_dense = tf.gather_nd(params=s_pre_x_t_post_lr, indices=gather_idx_Sin_to_Wdense_IN)

        # PARALLEL COMPATIBLE
        s_post_x_t_pre = tf.multiply(project_trace_in_to_W_dense, project_spikes_out_to_W_dense)
        s_pre_x_t_post = tf.multiply(project_trace_out_to_W_dense, project_spikes_in_to_W_dense)

        # PARALLEL COMPATIBLE
        delta_w__s_post_x_t_pre = tf.multiply(project_s_post_x_t_pre_lr_to_W_dense, s_post_x_t_pre)
        delta_w__s_pre_x_t_post = tf.multiply(project_s_pre_x_t_post_lr_to_W_dense, s_pre_x_t_post)

        # PARALLEL COMPATIBLE
        new_W_raw = tf.add_n([W_dense_IN, delta_w__s_post_x_t_pre, delta_w__s_pre_x_t_post])

        # PARALLEL COMPATIBLE
        new_W_res_exc = tf.multiply(W_dense_mask_exc_to_res_IN, tf.clip_by_value(new_W_raw, self.min_w_res_exc, self.max_w_res_exc))
        new_W_res_inh = tf.multiply(W_dense_mask_inh_to_res_IN, tf.clip_by_value(new_W_raw, self.min_w_res_inh, self.max_w_res_inh))
        new_W_res_inp = tf.multiply(W_dense_mask_inp_to_res_IN, tf.clip_by_value(new_W_raw, self.min_w_res_inp, self.max_w_res_inp))
        new_W_out_res = tf.multiply(W_dense_mask_res_to_out_IN, W_dense_IN)

        ######## STDP OPS ######### END

        # PARALLEL COMPATIBLE
        new_W_dense_final = tf.add_n([new_W_res_exc, new_W_res_inh, new_W_res_inp, new_W_out_res])

        return tf.assign(W_dense_IN,new_W_dense_final)
