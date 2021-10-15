import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

'''
-NEW BRANCH CREATED FROM P1NAN_ASTRO_CORE_SLIMv0_local.py
made for main run file v5_2 with astrocyte bias optimization

DIRECT COPY OF P1NAN_ASTRO_CORE_SLIMv1_local.py
- HERE FIXED MODEL OF ASTROCYTE WHERE TOTAL SUM OF SPIKES IS USED WITH OUT ANY SPECIAL WEIGHTING FOR RATIO OF INP/RES N

direct copy of v2
 - here added decay function for astro to decay during single sample training, compatible for P1NAN_TRAIN_MNIST_SLIMv8_4_2_XXX.py

'''



class ASTRO_LR_CONTROLLER:
    def __init__(self, astro_LR_decay_factor_IN, number_of_networks_IN):
        self.astro_LR_decay_factor = astro_LR_decay_factor_IN
        self.number_of_networks = number_of_networks_IN

    # PARALLEL INCOMPATIBLE
    def astro_initializer(self, num_total_neurons_IN, inp_range_IN, res_range_IN, a_bias_offset_percentage_IN, w_scaling_IN, a_initial_val_IN,a_tau_IN):

        # ASSUMES EXACT SAME NETWORK FOR ALL NETWORKS, THIS INITIALIZER IS DESIGNED FOR BATCH TRAINING SAME NETWORK, NOT FOR DIFFERENT NETWORKS IN BATCH

        # PARALLEL INCOMPATIBLE - change shape
        # shape: (nets,1)
        ASTRO_STDP_LR = tf.Variable(a_initial_val_IN*np.ones((self.number_of_networks,1),dtype=np.float32), dtype=tf.float32, expected_shape=[self.number_of_networks,1], name='ASTRO_STDP_LR')
        # shape: (nets,1)
        ASTRO_BIAS = tf.Variable((a_initial_val_IN+(a_initial_val_IN*a_bias_offset_percentage_IN))*np.ones((self.number_of_networks,1),dtype=np.float32), dtype=tf.float32, expected_shape=[self.number_of_networks,1], name='ASTRO_BIAS')

        # shape: (nets,1)
        ASTRO_STDP_LR_STORE = tf.constant(a_initial_val_IN*np.ones((self.number_of_networks,1),dtype=np.float32), dtype=tf.float32, shape=[self.number_of_networks,1], name='ASTRO_STDP_LR_STORE')
        # shape: (nets,1)
        ASTRO_BIAS_STORE = tf.constant((a_initial_val_IN+(a_initial_val_IN*a_bias_offset_percentage_IN))*np.ones((self.number_of_networks,1),dtype=np.float32), dtype=tf.float32, shape=[self.number_of_networks,1], name='ASTRO_BIAS_STORE')

        # shape: (nets,1)
        ASTRO_W = tf.Variable(w_scaling_IN * np.ones((self.number_of_networks, 1), dtype=np.float32), dtype=tf.float32,
                              expected_shape=[self.number_of_networks, 1], name='ASTRO_W')
        # shape: (nets,1)
        ASTRO_W_STORE = tf.constant(w_scaling_IN * np.ones((self.number_of_networks, 1), dtype=np.float32), dtype=tf.float32, shape=[self.number_of_networks, 1],
                                    name='ASTRO_W_STORE')

        # for astro_w_mask
        astro_vector_np = np.zeros(num_total_neurons_IN, dtype=np.float32)
        # input neuron weight
        astro_vector_np[inp_range_IN[0]:inp_range_IN[1]] = -1.0
        # res neuron weight
        astro_vector_np[res_range_IN[0]:res_range_IN[1]] = 1.0
        # shape: (nets,num_neurons)
        ASTRO_W_MASK = tf.constant(np.broadcast_to(astro_vector_np,shape=(self.number_of_networks,num_total_neurons_IN)), dtype=tf.float32, shape=[self.number_of_networks,num_total_neurons_IN], name='ASTRO_W_MASK')

        # shape: (nets,1)
        ASTRO_TAU = tf.constant(a_tau_IN * np.ones((self.number_of_networks, 1), dtype=np.float32),
                                    dtype=tf.float32, shape=[self.number_of_networks, 1],
                                    name='ASTRO_TAU')


        return ['ASTRO_STDP_LR', 'ASTRO_W', 'ASTRO_BIAS', 'ASTRO_STDP_LR_STORE','ASTRO_BIAS_STORE', 'ASTRO_W_STORE','ASTRO_W_MASK','ASTRO_TAU'], [ASTRO_STDP_LR, ASTRO_W, ASTRO_BIAS, ASTRO_STDP_LR_STORE, ASTRO_BIAS_STORE, ASTRO_W_STORE,ASTRO_W_MASK,ASTRO_TAU]


    def astro_initializer_SPARSE_W_MASK(self, num_total_neurons_IN, inp_range_IN, res_range_IN, a_bias_offset_percentage_IN,
                          w_scaling_IN, a_initial_val_IN, a_tau_IN,W_MASK_DENSITY_IN):
        # ASSUMES EXACT SAME NETWORK FOR ALL NETWORKS, THIS INITIALIZER IS DESIGNED FOR BATCH TRAINING SAME NETWORK, NOT FOR DIFFERENT NETWORKS IN BATCH

        # PARALLEL INCOMPATIBLE - change shape
        # shape: (nets,1)
        ASTRO_STDP_LR = tf.Variable(a_initial_val_IN * np.ones((self.number_of_networks, 1), dtype=np.float32),
                                    dtype=tf.float32, expected_shape=[self.number_of_networks, 1],
                                    name='ASTRO_STDP_LR')
        # shape: (nets,1)
        ASTRO_BIAS = tf.Variable((a_initial_val_IN + (a_initial_val_IN * a_bias_offset_percentage_IN)) * np.ones(
            (self.number_of_networks, 1), dtype=np.float32), dtype=tf.float32,
                                 expected_shape=[self.number_of_networks, 1], name='ASTRO_BIAS')

        # shape: (nets,1)
        ASTRO_STDP_LR_STORE = tf.constant(
            a_initial_val_IN * np.ones((self.number_of_networks, 1), dtype=np.float32), dtype=tf.float32,
            shape=[self.number_of_networks, 1], name='ASTRO_STDP_LR_STORE')
        # shape: (nets,1)
        ASTRO_BIAS_STORE = tf.constant(
            (a_initial_val_IN + (a_initial_val_IN * a_bias_offset_percentage_IN)) * np.ones(
                (self.number_of_networks, 1), dtype=np.float32), dtype=tf.float32,
            shape=[self.number_of_networks, 1], name='ASTRO_BIAS_STORE')

        # shape: (nets,1)
        ASTRO_W = tf.Variable(w_scaling_IN * np.ones((self.number_of_networks, 1), dtype=np.float32),
                              dtype=tf.float32,
                              expected_shape=[self.number_of_networks, 1], name='ASTRO_W')
        # shape: (nets,1)
        ASTRO_W_STORE = tf.constant(w_scaling_IN * np.ones((self.number_of_networks, 1), dtype=np.float32),
                                    dtype=tf.float32, shape=[self.number_of_networks, 1],
                                    name='ASTRO_W_STORE')

        # for astro_w_mask
        astro_vector_np = np.zeros(num_total_neurons_IN, dtype=np.float32)
        # input neuron weight
        # astro_vector_np[inp_range_IN[0]:inp_range_IN[1]] = -1.0
        # res neuron weight
        # astro_vector_np[res_range_IN[0]:res_range_IN[1]] = 1.0
        # shape: (nets,num_neurons)
        num_input_connections = int(np.ceil(W_MASK_DENSITY_IN*(inp_range_IN[1]-inp_range_IN[0])))
        num_res_connections = int(np.ceil(W_MASK_DENSITY_IN * (res_range_IN[1] - res_range_IN[0])))
        input_connection_idx = np.random.permutation(inp_range_IN[0]+np.arange(inp_range_IN[1]-inp_range_IN[0]))[0:num_input_connections]
        res_connection_idx = np.random.permutation(res_range_IN[0] + np.arange(res_range_IN[1] - res_range_IN[0]))[0:num_res_connections]
        astro_vector_np[input_connection_idx] = -1.0
        astro_vector_np[res_connection_idx] = 1.0


        ASTRO_W_MASK = tf.constant(
            np.broadcast_to(astro_vector_np, shape=(self.number_of_networks, num_total_neurons_IN)),
            dtype=tf.float32, shape=[self.number_of_networks, num_total_neurons_IN], name='ASTRO_W_MASK')

        # shape: (nets,1)
        ASTRO_TAU = tf.constant(a_tau_IN * np.ones((self.number_of_networks, 1), dtype=np.float32),
                                dtype=tf.float32, shape=[self.number_of_networks, 1],
                                name='ASTRO_TAU')

        return ['ASTRO_STDP_LR', 'ASTRO_W', 'ASTRO_BIAS', 'ASTRO_STDP_LR_STORE', 'ASTRO_BIAS_STORE',
                'ASTRO_W_STORE', 'ASTRO_W_MASK', 'ASTRO_TAU'], [ASTRO_STDP_LR, ASTRO_W, ASTRO_BIAS,
                                                                ASTRO_STDP_LR_STORE, ASTRO_BIAS_STORE,
                                                                ASTRO_W_STORE, ASTRO_W_MASK, ASTRO_TAU]


    # PARALLEL INCOMPATIBLE
    def update_astro_state(self, ASTRO_STDP_LR_IN, ASTRO_BIAS_IN, ASTRO_W_IN, ASTRO_W_MASK_IN, S_IN, ASTRO_TAU_IN):

        '''
        
        :param ASTRO_STDP_LR_IN: SHAPE: (nets,1)
        :param ASTRO_BIAS_IN:  SHAPE: (nets,1)
        :param ASTRO_W_IN:  SHAPE: (nets,1)
        :param ASTRO_W_MASK_IN: SHAPE: (nets,NUM_NEURONS_IN_NETWORK)
        :param S_IN:  SHAPE: (nets,NUM_NEURONS_IN_NETWORK)
        :param ASTRO_TAU_IN: SHAPE: (nets,1) 
        :return: 
        '''

        # PARALLEL COMPATIBLE
        new_A = tf.add(ASTRO_STDP_LR_IN, tf.divide(tf.subtract(tf.add(tf.reduce_sum(tf.multiply(tf.multiply(ASTRO_W_IN,ASTRO_W_MASK_IN), S_IN),axis=1,keep_dims=True), ASTRO_BIAS_IN), ASTRO_STDP_LR_IN), ASTRO_TAU_IN))

        return tf.assign(ASTRO_STDP_LR_IN, new_A)

    # PARALLEL COMPATIBLE
    def decay_astro_stdp_lr(self,astro_lr_IN):

        new_readout_lr = tf.scalar_mul(self.astro_LR_decay_factor, astro_lr_IN)

        return tf.assign(astro_lr_IN, new_readout_lr)

    # PARALLEL COMPATIBLE
    def decay_astro_w(self,astro_w_IN):

        new_astro_w = tf.scalar_mul(self.astro_LR_decay_factor, astro_w_IN)

        return tf.assign(astro_w_IN, new_astro_w)

    # PARALLEL COMPATIBLE
    def decay_astro_bias(self,astro_bias_IN):

        new_astro_bias = tf.scalar_mul(self.astro_LR_decay_factor, astro_bias_IN)

        return tf.assign(astro_bias_IN, new_astro_bias)

    # PARALLEL COMPATIBLE
    def reset_astro_stdp_lr(self,astro_lr_IN,astro_lr_store_IN):
        return tf.assign(astro_lr_IN, astro_lr_store_IN)

    # PARALLEL COMPATIBLE
    def reset_astro_w(self, astro_w_IN, astro_w_store_IN):
        return tf.assign(astro_w_IN, astro_w_store_IN)

    # PARALLEL COMPATIBLE
    def reset_astro_bias(self, astro_bias_IN, astro_bias_store_IN):
        return tf.assign(astro_bias_IN, astro_bias_store_IN)