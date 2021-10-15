# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np


class CUBA_LIF_network:
    def __init__(self):
        self.main_Path = os.getcwd()
        self.dataPath_TRAIN = self.main_Path + '/train_data'
        self.netPath = self.main_Path + '/networks'


    def initialize_network_parameters_v1(self, tau_v,tau_u,v_thrsh,b,w,w_mask,r,t_rfr):

        '''
        INITIALIZES NETWORK STRUCTURES
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
