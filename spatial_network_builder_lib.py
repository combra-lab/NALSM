import numpy as np
import os
import sys
import NALSM_GEN_SUPPORT as sup
import GEN_NET_LIF as cLIF


class create_network:
    def __init__(self,seed_IN):
        self.main_Path = os.getcwd()
        self.codePath = self.main_Path
        self.dataPath = self.main_Path + '/networks'

        np.random.seed(seed_IN)



    def create_mask_based_on_3d_space_v0(self,res_exc_size,res_inh_size,input_size,output_size,res_3_dims_l, C_EE_EI_IE_II_l,lamb,input_connection_density,output_connection_density):

        reservoir_size = res_exc_size + res_inh_size
        assert (reservoir_size > 0)
        assert (input_size > 0)
        assert (output_size > 0)
        assert (res_3_dims_l[0]*res_3_dims_l[1]*res_3_dims_l[2] == reservoir_size)

        res_exc_range = [0,res_exc_size]
        res_inh_range = [res_exc_size,res_exc_size+res_inh_size]

        # GENERATE RESERVOIR BLOCK - NON-ZERO
        # block_0_0_rows = np.random.randint(0, reservoir_size, size=int(reservoir_size * reservoir_size * reservoir_connection_density))
        # block_0_0_cols = np.random.randint(0, reservoir_size, size=int(reservoir_size * reservoir_size * reservoir_connection_density))

        block_0_0 = np.zeros((reservoir_size, reservoir_size), dtype=np.float32)
        # candidate_idxs = np.where(block_0_0==0.0)
        # sel_idx = np.random.permutation(np.arange(len(candidate_idxs[0])))[0:int(reservoir_size * reservoir_size * reservoir_connection_density)]
        # block_0_0[candidate_idxs[0][sel_idx], candidate_idxs[1][sel_idx]] = 1.0

        positions = np.arange(reservoir_size)
        neuron_perms = np.random.permutation(positions)

        neuron = []
        x_coor = []
        y_coor = []
        z_coor = []
        counter = 0
        for i in range(0, res_3_dims_l[0]):
            for j in range(0, res_3_dims_l[1]):
                for k in range(0, res_3_dims_l[2]):
                    x_coor.append(i)
                    y_coor.append(j)
                    z_coor.append(k)
                    neuron.append(counter)
                    counter = counter + 1

        X = np.asarray(x_coor)[neuron_perms]
        Y = np.asarray(y_coor)[neuron_perms]
        Z = np.asarray(z_coor)[neuron_perms]

        xt = np.matmul(np.ones((np.size(X), 1)), np.expand_dims(X, axis=0))
        yt = np.matmul(np.ones((np.size(Y), 1)), np.expand_dims(Y, axis=0))
        zt = np.matmul(np.ones((np.size(Z), 1)), np.expand_dims(Z, axis=0))

        dif_x = (xt - np.transpose(xt)) ** 2
        dif_y = (yt - np.transpose(yt)) ** 2
        dif_z = (zt - np.transpose(zt)) ** 2

        D = np.sqrt(np.add(np.add(dif_x, dif_y), dif_z))

        C = np.zeros((reservoir_size, reservoir_size))
        C[res_exc_range[0]:res_exc_range[1], res_exc_range[0]:res_exc_range[1]] = C_EE_EI_IE_II_l[0]
        C[res_inh_range[0]:res_inh_range[1], res_exc_range[0]:res_exc_range[1]] = C_EE_EI_IE_II_l[1]
        C[res_exc_range[0]:res_exc_range[1], res_inh_range[0]:res_inh_range[1]] = C_EE_EI_IE_II_l[2]
        C[res_inh_range[0]:res_inh_range[1], res_inh_range[0]:res_inh_range[1]] = C_EE_EI_IE_II_l[3]

        P = C * np.exp(-1.0 * (D / lamb) ** 2)

        sample = np.clip(np.ceil(np.subtract(P, np.random.random(np.shape(P)))), 0, 1)
        conn = np.multiply(sample, np.abs(np.identity(np.shape(P)[0]) - 1))

        block_0_0[np.where(conn==1.0)] = 1.0

        assert(np.sum(conn) == np.sum(block_0_0))
        print('number of syns in reservoir: '+str(np.sum(block_0_0)))

        coordinates = [X,Y,Z]


        # GENERATE BLOCK 0_1 RESERVOIR INPUT - NON-ZERO
        block_0_1 = np.zeros((reservoir_size, input_size), dtype=np.float32)
        candidate_idxs = np.where(block_0_1 == 0.0)
        sel_idx = np.random.permutation(np.arange(len(candidate_idxs[0])))[
                  0:int(reservoir_size * input_size * input_connection_density)]
        block_0_1[candidate_idxs[0][sel_idx], candidate_idxs[1][sel_idx]] = 1.0

        # block_0_1_rows = np.random.randint(0, reservoir_size,
        #                                    size=int(reservoir_size * input_size * input_connection_density))
        # block_0_1_cols = np.random.randint(0, input_size,
        #                                    size=int(reservoir_size * input_size * input_connection_density))
        #
        # block_0_1 = np.zeros((reservoir_size, input_size), dtype=np.float32)
        # block_0_1[block_0_1_rows, block_0_1_cols] = 1.0

        # GENERATE BLOCK 2_0 OUTPUT RESERVOIR - NON-ZERO
        block_2_0 = np.zeros((output_size, reservoir_size), dtype=np.float32)
        candidate_idxs = np.where(block_2_0 == 0.0)
        sel_idx = np.random.permutation(np.arange(len(candidate_idxs[0])))[
                  0:int(output_size * reservoir_size * output_connection_density)]
        block_2_0[candidate_idxs[0][sel_idx], candidate_idxs[1][sel_idx]] = 1.0

        #
        # block_2_0_rows = np.random.randint(0, output_size,
        #                                    size=int(output_size * reservoir_size * output_connection_density))
        # block_2_0_cols = np.random.randint(0, reservoir_size,
        #                                    size=int(output_size * reservoir_size * output_connection_density))
        #
        # block_2_0 = np.zeros((output_size, reservoir_size), dtype=np.float32)
        # block_2_0[block_2_0_rows, block_2_0_cols] = 1.0

        # GENERATE BLOCK 1_0 INPUT RESERVOIR - ZERO
        block_1_0 = np.zeros((input_size, reservoir_size), dtype=np.float32)

        # GENERATE BLOCK 1_1 INPUT INPUT - ZERO
        block_1_1 = np.zeros((input_size, input_size), dtype=np.float32)

        # GENERATE BLOCK 0_2 RESERVOIR OUTPUT - ZERO
        block_0_2 = np.zeros((reservoir_size, output_size), dtype=np.float32)

        # GENERATE BLOCK 1_2 INPUT OUTPUT - ZERO
        block_1_2 = np.zeros((input_size, output_size), dtype=np.float32)

        # GENERATE BLOCK 2_1 OUTPUT INPUT - ZERO
        block_2_1 = np.zeros((output_size, input_size), dtype=np.float32)

        # GENERATE BLOCK 2_2 OUTPUT OUTPUT - ZERO
        block_2_2 = np.zeros((output_size, output_size), dtype=np.float32)


        block_col_0 = np.concatenate([block_0_0,block_1_0,block_2_0],axis=0)
        block_col_1 = np.concatenate([block_0_1, block_1_1, block_2_1], axis=0)
        block_col_2 = np.concatenate([block_0_2, block_1_2, block_2_2], axis=0)

        mask = np.concatenate([block_col_0,block_col_1,block_col_2],axis=1)

        return mask, coordinates



    ## EXACTLY NUMBER OF CONNECTIONS
    ## copied from create_mask from crit_nan_create_RANDOM_network_v1.py
    def create_mask_v0(self,reservoir_size,input_size,output_size,reservoir_connection_density,input_connection_density,output_connection_density):

        assert (reservoir_size > 0)
        assert (input_size > 0)
        assert (output_size > 0)

        # GENERATE RESERVOIR BLOCK - NON-ZERO
        # block_0_0_rows = np.random.randint(0, reservoir_size, size=int(reservoir_size * reservoir_size * reservoir_connection_density))
        # block_0_0_cols = np.random.randint(0, reservoir_size, size=int(reservoir_size * reservoir_size * reservoir_connection_density))

        block_0_0 = np.zeros((reservoir_size, reservoir_size), dtype=np.float32)
        candidate_idxs = np.where(block_0_0==0.0)
        sel_idx = np.random.permutation(np.arange(len(candidate_idxs[0])))[0:int(reservoir_size * reservoir_size * reservoir_connection_density)]
        block_0_0[candidate_idxs[0][sel_idx], candidate_idxs[1][sel_idx]] = 1.0

        print('number of syns in reservoir: '+str(np.sum(block_0_0)))

        # if np.sum(block_0_0)<int(reservoir_size * reservoir_size * reservoir_connection_density):
        #     syns_left_to_fill = int(reservoir_size * reservoir_size * reservoir_connection_density)-np.sum(block_0_0)
        #     candidate_slots = np.where(block_0_0==0.0)
        #     for s in range(0,syns_left_to_fill):
        #         np.random.randint(0,len(candidate_slots[0]))



        # GENERATE BLOCK 0_1 RESERVOIR INPUT - NON-ZERO
        block_0_1 = np.zeros((reservoir_size, input_size), dtype=np.float32)
        candidate_idxs = np.where(block_0_1 == 0.0)
        sel_idx = np.random.permutation(np.arange(len(candidate_idxs[0])))[
                  0:int(reservoir_size * input_size * input_connection_density)]
        block_0_1[candidate_idxs[0][sel_idx], candidate_idxs[1][sel_idx]] = 1.0

        # block_0_1_rows = np.random.randint(0, reservoir_size,
        #                                    size=int(reservoir_size * input_size * input_connection_density))
        # block_0_1_cols = np.random.randint(0, input_size,
        #                                    size=int(reservoir_size * input_size * input_connection_density))
        #
        # block_0_1 = np.zeros((reservoir_size, input_size), dtype=np.float32)
        # block_0_1[block_0_1_rows, block_0_1_cols] = 1.0

        # GENERATE BLOCK 2_0 OUTPUT RESERVOIR - NON-ZERO
        block_2_0 = np.zeros((output_size, reservoir_size), dtype=np.float32)
        candidate_idxs = np.where(block_2_0 == 0.0)
        sel_idx = np.random.permutation(np.arange(len(candidate_idxs[0])))[
                  0:int(output_size * reservoir_size * output_connection_density)]
        block_2_0[candidate_idxs[0][sel_idx], candidate_idxs[1][sel_idx]] = 1.0

        #
        # block_2_0_rows = np.random.randint(0, output_size,
        #                                    size=int(output_size * reservoir_size * output_connection_density))
        # block_2_0_cols = np.random.randint(0, reservoir_size,
        #                                    size=int(output_size * reservoir_size * output_connection_density))
        #
        # block_2_0 = np.zeros((output_size, reservoir_size), dtype=np.float32)
        # block_2_0[block_2_0_rows, block_2_0_cols] = 1.0

        # GENERATE BLOCK 1_0 INPUT RESERVOIR - ZERO
        block_1_0 = np.zeros((input_size, reservoir_size), dtype=np.float32)

        # GENERATE BLOCK 1_1 INPUT INPUT - ZERO
        block_1_1 = np.zeros((input_size, input_size), dtype=np.float32)

        # GENERATE BLOCK 0_2 RESERVOIR OUTPUT - ZERO
        block_0_2 = np.zeros((reservoir_size, output_size), dtype=np.float32)

        # GENERATE BLOCK 1_2 INPUT OUTPUT - ZERO
        block_1_2 = np.zeros((input_size, output_size), dtype=np.float32)

        # GENERATE BLOCK 2_1 OUTPUT INPUT - ZERO
        block_2_1 = np.zeros((output_size, input_size), dtype=np.float32)

        # GENERATE BLOCK 2_2 OUTPUT OUTPUT - ZERO
        block_2_2 = np.zeros((output_size, output_size), dtype=np.float32)


        block_col_0 = np.concatenate([block_0_0,block_1_0,block_2_0],axis=0)
        block_col_1 = np.concatenate([block_0_1, block_1_1, block_2_1], axis=0)
        block_col_2 = np.concatenate([block_0_2, block_1_2, block_2_2], axis=0)

        mask = np.concatenate([block_col_0,block_col_1,block_col_2],axis=1)

        return mask








    # FOR LOGNORMAL DISTRIBUTIONS FOR all weight groups: reservoir, input, output
    ## copied from create_weight_matrix_v1 from crit_nan_create_RANDOM_network_v1.py
    def create_weight_matrix_v0(self,w_mask,reservoir_size,
                                input_size,
                                output_size,
                                num_res_exc_neurons,
                                num_res_inh_neurons,
                                res_res_exc_conn_dist,
                                res_res_inh_conn_dist,
                                res_inp_conn_dist,
                                out_res_conn_dist,
                                skew=1.0,
                                conn_dist_types=[('res_res','log_norm'),('res_inp','log_norm'),('out_res','log_norm')]
                                ):



        assert (reservoir_size > 0)
        assert (input_size > 0)
        assert (output_size > 0)

        conn_dist_types_dict = dict(conn_dist_types)


        # GENERATE RESERVOIR BLOCK - NON-ZERO - lognormal distributions
        if conn_dist_types_dict['res_res']=='log_norm':
            block_0_0 = np.concatenate([np.random.lognormal(res_res_exc_conn_dist[0],res_res_exc_conn_dist[1],size=(reservoir_size,num_res_exc_neurons)),-skew*np.random.lognormal(res_res_inh_conn_dist[0],res_res_inh_conn_dist[1],size=(reservoir_size,num_res_inh_neurons))],axis=1)
        elif conn_dist_types_dict['res_res']=='constant':
            block_0_0 = np.concatenate([res_res_exc_conn_dist*np.ones((reservoir_size,num_res_exc_neurons)),-skew*res_res_inh_conn_dist*np.ones((reservoir_size,num_res_inh_neurons))],axis=1)
        else:
            print('unspecified distribution, exiting')
            sys.exit()

        # GENERATE BLOCK 0_1 RESERVOIR INPUT - NON-ZERO
        if conn_dist_types_dict['res_inp'] == 'log_norm':
            block_0_1 = np.random.lognormal(res_inp_conn_dist[0], res_inp_conn_dist[1],
                                            size=(reservoir_size, input_size))
        elif conn_dist_types_dict['res_inp'] == 'constant':
            block_0_1 = res_inp_conn_dist*np.ones((reservoir_size, input_size),dtype=np.float32)
        elif conn_dist_types_dict['res_inp'] == 'constant_randomize_+/-':
            block_0_1 = (res_inp_conn_dist * np.ceil((2.0*(np.reshape(np.random.randint(0,2,size=reservoir_size*input_size),(reservoir_size, input_size))))-1.0)).astype(dtype=np.float32)
        else:
            print('unspecified distribution, exiting')
            sys.exit()

        # GENERATE BLOCK 2_0 OUTPUT RESERVOIR - NON-ZERO
        if conn_dist_types_dict['out_res'] == 'log_norm':
            block_2_0 = np.random.lognormal(out_res_conn_dist[0], out_res_conn_dist[1],
                                            size=(output_size, reservoir_size))
        elif conn_dist_types_dict['out_res'] == 'constant':
            block_2_0 = out_res_conn_dist*np.ones((output_size, reservoir_size),dtype=np.float32)
        elif conn_dist_types_dict['res_inp'] == 'constant_randomize_+/-':
            block_2_0 = (out_res_conn_dist * np.ceil((2.0 * (
            np.reshape(np.random.randint(0, 2, size=output_size * reservoir_size),
                       (output_size, reservoir_size)))) - 1.0)).astype(dtype=np.float32)
        else:
            print('unspecified distribution, exiting')
            sys.exit()

        # GENERATE BLOCK 1_0 INPUT RESERVOIR - ZERO
        block_1_0 = np.zeros((input_size, reservoir_size))

        # GENERATE BLOCK 1_1 INPUT INPUT - ZERO
        block_1_1 = np.zeros((input_size, input_size))

        # GENERATE BLOCK 0_2 RESERVOIR OUTPUT - ZERO
        block_0_2 = np.zeros((reservoir_size, output_size))

        # GENERATE BLOCK 1_2 INPUT OUTPUT - ZERO
        block_1_2 = np.zeros((input_size, output_size))

        # GENERATE BLOCK 2_1 OUTPUT INPUT - ZERO
        block_2_1 = np.zeros((output_size, input_size))

        # GENERATE BLOCK 2_2 OUTPUT OUTPUT - ZERO
        block_2_2 = np.zeros((output_size, output_size))

        block_col_0 = np.concatenate([block_0_0, block_1_0, block_2_0], axis=0)
        block_col_1 = np.concatenate([block_0_1, block_1_1, block_2_1], axis=0)
        block_col_2 = np.concatenate([block_0_2, block_1_2, block_2_2], axis=0)

        weights = np.concatenate([block_col_0, block_col_1, block_col_2], axis=1).astype(np.float32)

        return np.multiply(w_mask,weights)

    def assemble_neuron_ranges(self, num_neurons_in_res,
                                         num_input_neurons,
                                         num_output_neurons,
                                         num_res_exc_neurons,
                                         num_res_inh_neurons):

        res_range = [0,num_neurons_in_res]
        res_exc_range = [0,num_res_exc_neurons]
        res_inh_range = [num_res_exc_neurons, num_res_exc_neurons + num_res_inh_neurons]
        inp_range = [num_neurons_in_res,num_neurons_in_res+num_input_neurons]
        out_range = [num_neurons_in_res+num_input_neurons,num_neurons_in_res+num_input_neurons+num_output_neurons]

        return [('res_range',res_range),('res_exc_range',res_exc_range),('res_inh_range',res_inh_range),('inp_range',inp_range),('out_range',out_range)]




    def create_network_v1(self,net_num,tau_v,tau_u,v_thrsh,b,r,t_rfr,w,w_mask,neuron_ranges,coordinates):

        net = cLIF.CUBA_LIF_network()

        names,data = net.initialize_network_parameters_v1(tau_v=tau_v,tau_u=tau_u,v_thrsh=v_thrsh,b=b,w=w,w_mask=w_mask,r=r,t_rfr=t_rfr)

        net_name = 'Network_'+str(net_num)
        sup.save_tf_nontf_data(names=names,data=data,names_nontf=['neuron_ranges','coordinates'],data_nontf=[neuron_ranges,coordinates],filename=net_name,savePath=self.dataPath)

        print('Saved '+str(net_name)+' to '+str(self.dataPath))


    def create_network_NP_v1(self,net_num,tau_v,tau_u,v_thrsh,b,r,t_rfr,w,w_mask,neuron_ranges,coordinates):

        net = cLIF.CUBA_LIF_network()

        names,data = net.initialize_network_parameters_v1(tau_v=tau_v,tau_u=tau_u,v_thrsh=v_thrsh,b=b,w=w,w_mask=w_mask,r=r,t_rfr=t_rfr)

        net_name = 'Network_'+str(net_num)
        sup.save_tf_nontf_data(names=names,data=data,names_nontf=['neuron_ranges','coordinates'],data_nontf=[neuron_ranges,coordinates],filename=net_name,savePath=self.dataPath)

        print('Saved '+str(net_name)+' to '+str(self.dataPath))