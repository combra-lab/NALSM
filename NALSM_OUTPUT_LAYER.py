import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class output_linear_layer:
    def __init__(self,output_layer_size_IN):
        self.dataPath_TRAIN = 'V:/CNAN/dataFiles/P1/train'
        self.netPath = 'V:/CNAN/dataFiles/P1/networks'
        self.codePath = 'V:/CNAN/codeFiles'

        self.layer_size = output_layer_size_IN

    def initialize_layer_structures(self,num_res_neurons_IN,initial_batch_size_IN):

        inDim = num_res_neurons_IN
        outDim = self.layer_size

        label_Inp = tf.placeholder(tf.int64, shape=[initial_batch_size_IN])
        target_output_ph = tf.placeholder(tf.float32, shape=[initial_batch_size_IN, outDim])
        W1_ph = tf.placeholder(tf.float32, shape=[inDim, outDim])
        b1_ph = tf.placeholder(tf.float32, shape=[outDim])

        W1 = tf.Variable(self.weight_constant([inDim, outDim]), name='W1')
        b1 = tf.Variable(self.bias_variable([outDim]), name='b1')
        BS = tf.constant(initial_batch_size_IN,shape=[], name='BS',dtype=tf.int64)

        S_AGG_RES = tf.Variable(tf.zeros((initial_batch_size_IN,inDim),dtype=np.float32),dtype=tf.float32,expected_shape=[initial_batch_size_IN,inDim])
        S_AGG_RES_PH = tf.placeholder(tf.float32, shape=[initial_batch_size_IN,inDim])



        return ['label_Inp', 'target_output_ph','W1_ph', 'b1_ph','W1','b1','BS','S_AGG_RES','S_AGG_RES_PH'],[label_Inp, target_output_ph, W1_ph, b1_ph, W1,b1,BS,S_AGG_RES,S_AGG_RES_PH]


    def initialize_layer_ops(self,S_AGG_RES_IN,target_output_ph_IN,label_Inp_IN,W1_IN,b1_IN,BS_IN,beta_IN=0.0000000005,lr_IN=5e-1):

        xInp_temp = tf.divide(S_AGG_RES_IN,tf.norm(S_AGG_RES_IN,ord=1,axis=1,keep_dims=True))

        xInp = tf.divide(xInp_temp,tf.reduce_max(xInp_temp))

        y1 = tf.matmul(xInp, W1_IN) + b1_IN

        regularizers = tf.nn.l2_loss(W1_IN)
        error_for_batch = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y1, labels=tf.nn.softmax(target_output_ph_IN)))
        error_plus_reg = tf.reduce_mean(error_for_batch + beta_IN * regularizers)
        # train_step = tf.train.GradientDescentOptimizer(2e-2).minimize(error_plus_reg)
        train_step = tf.train.AdadeltaOptimizer(lr_IN).minimize(error_plus_reg)
        # adamoptimzer lr - 2e-4 , 2.5e-2

        accuracy_eval_per_batch = tf.divide(tf.count_nonzero(tf.equal(tf.argmax(tf.nn.softmax(y1), axis=1), label_Inp_IN)), BS_IN)

        return ['train_step','accuracy_eval_per_batch'],[train_step,accuracy_eval_per_batch]


    def assign_previously_saved_S_AGG(self,S_AGG_RES_IN,S_AGG_RES_PH_IN):
        return tf.assign(S_AGG_RES_IN,S_AGG_RES_PH_IN)

    def aggregate_spikes_for_output_layer_op(self,S_AGG_RES_IN,S_IN,res_range_IN):
        return tf.assign(S_AGG_RES_IN,tf.add(S_AGG_RES_IN, S_IN[:,res_range_IN[0]:res_range_IN[1]]))

    def initialize_saved_output_layer_weights(self,W_IN,b_IN,new_W_IN,new_b_IN):

        set_new_W_op = tf.assign(W_IN,new_W_IN)
        set_new_b_op = tf.assign(b_IN, new_b_IN)

        return ['set_new_W_op', 'set_new_b_op'], [set_new_W_op, set_new_b_op]

    def zero_out_S_AGG_RES_op(self,S_AGG_RES_IN):
        return tf.assign(S_AGG_RES_IN,tf.scalar_mul(0.0,S_AGG_RES_IN))


    def weight_constant(self,shape):
        # initial = tf.truncated_normal(shape, stddev=1.0)
        initial = tf.constant(0.0, shape=shape)
        return initial


    def bias_variable(self,shape):
        # initial = tf.truncated_normal(shape, stddev=1.0)
        initial = tf.constant(0.0, shape=shape)
        return initial