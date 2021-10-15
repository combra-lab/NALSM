import os
import sys
import numpy as np
import NALSM_GEN_SUPPORT as sup
import spatial_network_builder_lib as snb


class build_networks:
    def __init__(self):
        self.main_Path = os.getcwd()
        self.codePath = self.main_Path
        self.dataPath = self.main_Path + '/networks'


        np.random.seed()
        rnd_sd = np.random.randint(0, 1000000)
        np.random.seed(rnd_sd)

        self.seed = rnd_sd


    def build_3D_LIF_network(self, net_num,
                             num_neurons_in_res,
                             num_input_neurons,
                             num_output_neurons,
                             num_res_exc_neurons,
                             num_res_inh_neurons,

                             res_3_dims_l,
                             C_EE_EI_IE_II_l,
                             lamb,

                             conn_density_input,
                             conn_density_output,
                             dist_exc_w_res_res,
                             dist_inh_w_res_res,
                             dist_w_res_inp,
                             dist_w_out_res,
                             param_tau_v,
                             param_tau_u,
                             param_v_thrsh,
                             param_b,
                             param_r,
                             param_t_rfr,
                             param_dist_types=[('tau_v', 'constant'),
                                               ('tau_u', 'constant'),
                                               ('v_thrsh', 'constant'),
                                               ('b', 'constant'),
                                               ('r', 'constant'),
                                               ('t_rfr', 'constant')],
                             w_dist_types=[('res_res', 'log_norm'), ('res_inp', 'log_norm'),
                                           ('out_res', 'log_norm')]
                             ):



        param_dist_types_dict = dict(param_dist_types)

        cn = snb.create_network(seed_IN=self.seed)

        # INITIALIZE NETWORK PARAMETERS
        if param_dist_types_dict['tau_v'] == 'constant':

            tau_v = np.concatenate([param_tau_v[0] * np.ones(num_res_exc_neurons, dtype=np.float32),
                                    param_tau_v[1] * np.ones(num_res_inh_neurons, dtype=np.float32),
                                    param_tau_v[2] * np.ones(num_input_neurons, dtype=np.float32),
                                    param_tau_v[3] * np.ones(num_output_neurons, dtype=np.float32)])
        elif param_dist_types_dict['tau_v'] == 'normal':
            tau_v = np.concatenate([
                np.random.normal(param_tau_v[0][0], param_tau_v[0][1], num_res_exc_neurons),
                np.random.normal(param_tau_v[1][0], param_tau_v[1][1], num_res_inh_neurons),
                np.random.normal(param_tau_v[2][0], param_tau_v[2][1], num_input_neurons),
                np.random.normal(param_tau_v[3][0], param_tau_v[3][1], num_output_neurons)])
        elif param_dist_types_dict['tau_v'] == 'uniform_discrete':
            tau_v = np.concatenate([
                np.random.randint(param_tau_v[0][0], param_tau_v[0][1], num_res_exc_neurons),
                np.random.randint(param_tau_v[1][0], param_tau_v[1][1], num_res_inh_neurons),
                np.random.randint(param_tau_v[2][0], param_tau_v[2][1], num_input_neurons),
                np.random.randint(param_tau_v[3][0], param_tau_v[3][1], num_output_neurons)])


        else:
            print('unspecified distribution, exiting')
            sys.exit()

        if param_dist_types_dict['tau_u'] == 'constant':

            tau_u = np.concatenate([param_tau_u[0] * np.ones(num_res_exc_neurons, dtype=np.float32),
                                    param_tau_u[1] * np.ones(num_res_inh_neurons, dtype=np.float32),
                                    param_tau_u[2] * np.ones(num_input_neurons, dtype=np.float32),
                                    param_tau_u[3] * np.ones(num_output_neurons, dtype=np.float32)])
        elif param_dist_types_dict['tau_u'] == 'normal':
            tau_u = np.concatenate([
                np.random.normal(param_tau_u[0][0], param_tau_u[0][1], num_res_exc_neurons),
                np.random.normal(param_tau_u[1][0], param_tau_u[1][1], num_res_inh_neurons),
                np.random.normal(param_tau_u[2][0], param_tau_u[2][1], num_input_neurons),
                np.random.normal(param_tau_u[3][0], param_tau_u[3][1], num_output_neurons)])
        elif param_dist_types_dict['tau_u'] == 'uniform_discrete':
            tau_u = np.concatenate([
                np.random.randint(param_tau_u[0][0], param_tau_u[0][1], num_res_exc_neurons),
                np.random.randint(param_tau_u[1][0], param_tau_u[1][1], num_res_inh_neurons),
                np.random.randint(param_tau_u[2][0], param_tau_u[2][1], num_input_neurons),
                np.random.randint(param_tau_u[3][0], param_tau_u[3][1], num_output_neurons)])
        else:
            print('unspecified distribution, exiting')
            sys.exit()

        if param_dist_types_dict['v_thrsh'] == 'constant':

            v_thrsh = np.concatenate([
                param_v_thrsh[0] * np.ones(num_res_exc_neurons, dtype=np.float32),
                param_v_thrsh[1] * np.ones(num_res_inh_neurons, dtype=np.float32),
                param_v_thrsh[2] * np.ones(num_input_neurons, dtype=np.float32),
                param_v_thrsh[3] * np.ones(num_output_neurons, dtype=np.float32)])
        elif param_dist_types_dict['v_thrsh'] == 'normal':
            v_thrsh = np.concatenate([
                np.random.normal(param_v_thrsh[0][0], param_v_thrsh[0][1], num_res_exc_neurons),
                np.random.normal(param_v_thrsh[1][0], param_v_thrsh[1][1], num_res_inh_neurons),
                np.random.normal(param_v_thrsh[2][0], param_v_thrsh[2][1], num_input_neurons),
                np.random.normal(param_v_thrsh[3][0], param_v_thrsh[3][1], num_output_neurons)])
        elif param_dist_types_dict['v_thrsh'] == 'uniform_discrete':
            v_thrsh = np.concatenate([
                np.random.randint(param_v_thrsh[0][0], param_v_thrsh[0][1], num_res_exc_neurons),
                np.random.randint(param_v_thrsh[1][0], param_v_thrsh[1][1], num_res_inh_neurons),
                np.random.randint(param_v_thrsh[2][0], param_v_thrsh[2][1], num_input_neurons),
                np.random.randint(param_v_thrsh[3][0], param_v_thrsh[3][1], num_output_neurons)])


        else:
            print('unspecified distribution, exiting')
            sys.exit()

        if param_dist_types_dict['b'] == 'constant':

            b = np.concatenate([param_b[0] * np.ones(num_res_exc_neurons, dtype=np.float32),
                                param_b[1] * np.ones(num_res_inh_neurons, dtype=np.float32),
                                param_b[2] * np.ones(num_input_neurons, dtype=np.float32),
                                param_b[3] * np.ones(num_output_neurons, dtype=np.float32)])
        elif param_dist_types_dict['b'] == 'normal':
            b = np.concatenate([
                np.random.normal(param_b[0][0], param_b[0][1], num_res_exc_neurons),
                np.random.normal(param_b[1][0], param_b[1][1], num_res_inh_neurons),
                np.random.normal(param_b[2][0], param_b[2][1], num_input_neurons),
                np.random.normal(param_b[3][0], param_b[3][1], num_output_neurons)])
        elif param_dist_types_dict['b'] == 'uniform_discrete':
            b = np.concatenate([
                np.random.randint(param_b[0][0], param_b[0][1], num_res_exc_neurons),
                np.random.randint(param_b[1][0], param_b[1][1], num_res_inh_neurons),
                np.random.randint(param_b[2][0], param_b[2][1], num_input_neurons),
                np.random.randint(param_b[3][0], param_b[3][1], num_output_neurons)])
        else:
            print('unspecified distribution, exiting')
            sys.exit()

        ## R
        if param_dist_types_dict['r'] == 'constant':

            r = np.concatenate([param_r[0] * np.ones(num_res_exc_neurons, dtype=np.float32),
                                param_r[1] * np.ones(num_res_inh_neurons, dtype=np.float32),
                                param_r[2] * np.ones(num_input_neurons, dtype=np.float32),
                                param_r[3] * np.ones(num_output_neurons, dtype=np.float32)])
        elif param_dist_types_dict['r'] == 'normal':
            r = np.concatenate([
                np.random.normal(param_r[0][0], param_r[0][1], num_res_exc_neurons),
                np.random.normal(param_r[1][0], param_r[1][1], num_res_inh_neurons),
                np.random.normal(param_r[2][0], param_r[2][1], num_input_neurons),
                np.random.normal(param_r[3][0], param_r[3][1], num_output_neurons)])
        elif param_dist_types_dict['r'] == 'uniform_discrete':
            r = np.concatenate([
                np.random.randint(param_r[0][0], param_r[0][1], num_res_exc_neurons),
                np.random.randint(param_r[1][0], param_r[1][1], num_res_inh_neurons),
                np.random.randint(param_r[2][0], param_r[2][1], num_input_neurons),
                np.random.randint(param_r[3][0], param_r[3][1], num_output_neurons)])
        else:
            print('unspecified distribution, exiting')
            sys.exit()

        ## R
        if param_dist_types_dict['t_rfr'] == 'constant':

            t_rfr = np.concatenate([param_t_rfr[0] * np.ones(num_res_exc_neurons, dtype=np.float32),
                                param_t_rfr[1] * np.ones(num_res_inh_neurons, dtype=np.float32),
                                param_t_rfr[2] * np.ones(num_input_neurons, dtype=np.float32),
                                param_t_rfr[3] * np.ones(num_output_neurons, dtype=np.float32)])
        elif param_dist_types_dict['t_rfr'] == 'normal':
            t_rfr = np.concatenate([
                np.random.normal(param_t_rfr[0][0], param_t_rfr[0][1], num_res_exc_neurons),
                np.random.normal(param_t_rfr[1][0], param_t_rfr[1][1], num_res_inh_neurons),
                np.random.normal(param_t_rfr[2][0], param_t_rfr[2][1], num_input_neurons),
                np.random.normal(param_t_rfr[3][0], param_t_rfr[3][1], num_output_neurons)])
        elif param_dist_types_dict['t_rfr'] == 'uniform_discrete':
            t_rfr = np.concatenate([
                np.random.randint(param_t_rfr[0][0], param_t_rfr[0][1], num_res_exc_neurons),
                np.random.randint(param_t_rfr[1][0], param_t_rfr[1][1], num_res_inh_neurons),
                np.random.randint(param_t_rfr[2][0], param_t_rfr[2][1], num_input_neurons),
                np.random.randint(param_t_rfr[3][0], param_t_rfr[3][1], num_output_neurons)])
        else:
            print('unspecified distribution, exiting')
            sys.exit()

        # CREATE CONNECTIONS BASED ON 3D EUCLIDEAN DISTANCE
        w_mask, coordinates = cn.create_mask_based_on_3d_space_v0(res_exc_size=num_res_exc_neurons,
                                                     res_inh_size=num_res_inh_neurons,
                                                     input_size=num_input_neurons,
                                                     output_size=num_output_neurons,
                                                     res_3_dims_l=res_3_dims_l,
                                                     C_EE_EI_IE_II_l=C_EE_EI_IE_II_l,
                                                     lamb=lamb,
                                                     input_connection_density=conn_density_input,
                                                     output_connection_density=conn_density_output)

        # CREATE WEIGHTS FOR LIQUID AND INPUT CONNECTIONS
        weights = cn.create_weight_matrix_v0(w_mask=w_mask,
                                             reservoir_size=num_neurons_in_res,
                                             input_size=num_input_neurons,
                                             output_size=num_output_neurons,
                                             num_res_exc_neurons=num_res_exc_neurons,
                                             num_res_inh_neurons=num_res_inh_neurons,
                                             res_res_exc_conn_dist=dist_exc_w_res_res,
                                             res_res_inh_conn_dist=dist_inh_w_res_res,
                                             res_inp_conn_dist=dist_w_res_inp,
                                             out_res_conn_dist=dist_w_out_res,
                                             conn_dist_types=w_dist_types
                                             )

        # COMPUTE NEURON INDEX RANGES FOR INPUT NEURONS, LIQUID NEURONS
        neuron_ranges = cn.assemble_neuron_ranges(num_neurons_in_res=num_neurons_in_res,
                                                  num_input_neurons=num_input_neurons,
                                                  num_output_neurons=num_output_neurons,
                                                  num_res_exc_neurons=num_res_exc_neurons,
                                                  num_res_inh_neurons=num_res_inh_neurons)


        cn.create_network_v1(net_num=net_num,
                             tau_v=tau_v,
                             tau_u=tau_u,
                             v_thrsh=v_thrsh,
                             b=b,
                             r=r,
                             t_rfr=t_rfr,
                             w=weights,
                             w_mask=w_mask,
                             neuron_ranges=neuron_ranges,
                             coordinates=coordinates)

        # SAVE LOG STATISTICS ABOUT NETWORK
        log_filename = 'Network_' + str(net_num) + '_Log.txt'
        log_fn = os.path.abspath(os.path.join(self.dataPath, log_filename))
        with open(log_fn, 'w') as f:
            f.write('LOG___NETWORK_' + str(net_num) + '\n\n')

            f.write('NETWORK STATS:\n\n')
            f.write('   num_neurons_in_res:     ' + str(num_neurons_in_res) + '\n')
            f.write('   num_exc_neurons_in_res:     ' + str(num_res_exc_neurons) + '\n')
            f.write('   num_inh_neurons_in_res:     ' + str(num_res_inh_neurons) + '\n')
            f.write('   num_input_neurons:     ' + str(num_input_neurons) + '\n')
            f.write('   num_output_neurons:     ' + str(num_output_neurons) + '\n')
            f.write('\n')
            # f.write('   conn_density_res:     ' + str(conn_density_res) + '\n')
            f.write('   conn_density_input:     ' + str(conn_density_input) + '\n')
            f.write('   conn_density_output:     ' + str(conn_density_output) + '\n')
            f.write('\n')

            f.write('                  res_3_dims_l:     ' + str(res_3_dims_l) + '\n')
            f.write('               C_EE_EI_IE_II_l:     ' + str(C_EE_EI_IE_II_l) + '\n')
            f.write('                          lamb:     ' + str(lamb) + '\n')

            f.write('\n')
            f.write('   dist_exc_w_res_res:     ' + str(dist_exc_w_res_res) + '\n')
            f.write('   dist_inh_w_res_res:     ' + str(dist_inh_w_res_res) + '\n')
            f.write('   dist_w_res_inp:     ' + str(dist_w_res_inp) + '\n')
            f.write('   dist_w_out_res:     ' + str(dist_w_out_res) + '\n')
            f.write('\n')
            f.write('   param_tau_v:     ' + str(param_tau_v) + '\n')
            f.write('   param_tau_u:     ' + str(param_tau_u) + '\n')
            f.write('   param_v_thrsh:     ' + str(param_v_thrsh) + '\n')
            f.write('   param_b:     ' + str(param_b) + '\n')
            f.write('\n')
            f.write('   param_dist_types:     ' + str(param_dist_types) + '\n')
            f.write('   w_dist_types:     ' + str(w_dist_types) + '\n')

            mean_exc = np.average(weights[0:num_neurons_in_res, 0:num_res_exc_neurons])
            var_exc = np.var(weights[0:num_neurons_in_res, 0:num_res_exc_neurons])

            mean_inh = np.average(weights[0:num_neurons_in_res, num_res_exc_neurons:num_neurons_in_res])
            var_inh = np.var(weights[0:num_neurons_in_res, num_res_exc_neurons:num_neurons_in_res])

            mean_inp = np.average(
                weights[0:num_neurons_in_res, num_neurons_in_res:num_neurons_in_res + num_input_neurons])
            var_inp = np.var(
                weights[0:num_neurons_in_res, num_neurons_in_res:num_neurons_in_res + num_input_neurons])

            mean_out = np.average(
                weights[
                num_neurons_in_res + num_input_neurons:num_neurons_in_res + num_input_neurons + num_output_neurons,
                0:num_neurons_in_res])
            var_out = np.var(weights[
                             num_neurons_in_res + num_input_neurons:num_neurons_in_res + num_input_neurons + num_output_neurons,
                             0:num_neurons_in_res])

            num_exc_conns = np.sum(w_mask[0:num_neurons_in_res, 0:num_res_exc_neurons])
            num_exc_conns_den = num_exc_conns / np.size(w_mask[0:num_neurons_in_res, 0:num_res_exc_neurons])

            num_inh_conns = np.sum(w_mask[0:num_neurons_in_res, num_res_exc_neurons:num_neurons_in_res])
            num_inh_conns_den = num_inh_conns / np.size(
                w_mask[0:num_neurons_in_res, num_res_exc_neurons:num_neurons_in_res])

            num_inp_conns = np.sum(
                w_mask[0:num_neurons_in_res, num_neurons_in_res:num_neurons_in_res + num_input_neurons])
            num_inp_conns_den = num_inp_conns / np.size(
                w_mask[0:num_neurons_in_res, num_neurons_in_res:num_neurons_in_res + num_input_neurons])

            num_out_conns = np.sum(
                w_mask[
                num_neurons_in_res + num_input_neurons:num_neurons_in_res + num_input_neurons + num_output_neurons,
                0:num_neurons_in_res])
            num_out_conns_den = num_out_conns / np.size(
                w_mask[
                num_neurons_in_res + num_input_neurons:num_neurons_in_res + num_input_neurons + num_output_neurons,
                0:num_neurons_in_res])

            ###### NEW DENSITY CALCS #####

            num_ee_conns = np.sum(w_mask[0:num_res_exc_neurons, 0:num_res_exc_neurons])
            num_ee_conns_den = num_ee_conns / np.size(w_mask[0:num_res_exc_neurons, 0:num_res_exc_neurons])

            num_ei_conns = np.sum(w_mask[num_res_exc_neurons:num_res_exc_neurons+num_res_inh_neurons, 0:num_res_exc_neurons])
            num_ei_conns_den = num_ei_conns / np.size(w_mask[num_res_exc_neurons:num_res_exc_neurons+num_res_inh_neurons, 0:num_res_exc_neurons])

            num_ie_conns = np.sum(w_mask[0:num_res_exc_neurons, num_res_exc_neurons:num_res_exc_neurons+num_res_inh_neurons])
            num_ie_conns_den = num_ie_conns / np.size(w_mask[0:num_res_exc_neurons, num_res_exc_neurons:num_res_exc_neurons+num_res_inh_neurons])

            num_ii_conns = np.sum(
                w_mask[num_res_exc_neurons:num_res_exc_neurons+num_res_inh_neurons, num_res_exc_neurons:num_res_exc_neurons+num_res_inh_neurons])
            num_ii_conns_den = num_ii_conns / np.size(
                w_mask[num_res_exc_neurons:num_res_exc_neurons+num_res_inh_neurons, num_res_exc_neurons:num_res_exc_neurons+num_res_inh_neurons])

            f.write('\n')
            f.write('   seed :                       ' + str(self.seed) + '\n')
            f.write('   mean_exc:                    ' + str(mean_exc) + '\n')
            f.write('   mean_inh:                    ' + str(mean_inh) + '\n')
            f.write('   mean_inp:                    ' + str(mean_inp) + '\n')
            f.write('   mean_out:                    ' + str(mean_out) + '\n')
            f.write('   var_exc:                     ' + str(var_exc) + '\n')
            f.write('   var_inh:                     ' + str(var_inh) + '\n')
            f.write('   var_inp:                     ' + str(var_inp) + '\n')
            f.write('   var_out:                     ' + str(var_out) + '\n')
            f.write('   num_exc_conns + density:     ' + str(num_exc_conns) + '__' + str(num_exc_conns_den) + '\n')
            f.write('   num_inh_conns + density:     ' + str(num_inh_conns) + '__' + str(num_inh_conns_den) + '\n')
            f.write('   num_inp_conns + density:     ' + str(num_inp_conns) + '__' + str(num_inp_conns_den) + '\n')
            f.write('   num_out_conns + density:     ' + str(num_out_conns) + '__' + str(num_out_conns_den) + '\n')

            f.write('   NEW DENSITY METRICS:     \n\n')

            f.write('   E->E num_conns + density:     ' + str(num_ee_conns) + '__' + str(num_ee_conns_den) + '\n')
            f.write('   E->I num_conns + density:     ' + str(num_ei_conns) + '__' + str(num_ei_conns_den) + '\n')
            f.write('   I->E num_conns + density:     ' + str(num_ie_conns) + '__' + str(num_ie_conns_den) + '\n')
            f.write('   I->I num_conns + density:     ' + str(num_ii_conns) + '__' + str(num_ii_conns_den) + '\n')
            f.write('   num_inp_conns + density:     ' + str(num_inp_conns) + '__' + str(num_inp_conns_den) + '\n')

            # SAVE NETWORK
            sup.save_non_tf_data(
                names=['mean_exc', 'mean_inh', 'mean_inp', 'mean_out', 'var_exc', 'var_inh', 'var_inp', 'var_out'],
                data=[mean_exc, mean_inh, mean_inp, mean_out, var_exc, var_inh, var_inp, var_out],
                filename='Network_' + str(net_num) + '_W_stats', savePath=self.dataPath)
