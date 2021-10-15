import numpy as np
import high_lvl_build_network_funcs as bnf


if __name__ == "__main__":


    NET_TYPE_VAR_INP = input('WHICH_DATASET_TO_GENERATE_NETWORK_FOR? [TYPE M FOR MNIST/ N FOR NMNIST]: ')
    NET_TYPE_VAR = str(NET_TYPE_VAR_INP)

    NET_NUM_VAR_INP = input('NETWORK_NUMBER_TO_CREATE? [int]: ')
    NET_NUM_VAR = int(NET_NUM_VAR_INP)

    RES_DIM_1_INP = input('SIZE_OF_LIQUID_DIMENSION_1? [int]: ')
    RES_DIM_1_VAR = int(RES_DIM_1_INP)

    RES_DIM_2_INP = input('SIZE_OF_LIQUID_DIMENSION_2? [int]: ')
    RES_DIM_2_VAR = int(RES_DIM_2_INP)

    RES_DIM_3_INP = input('SIZE_OF_LIQUID_DIMENSION_3? [int]: ')
    RES_DIM_3_VAR = int(RES_DIM_3_INP)

    bn = bnf.build_networks()


    if NET_TYPE_VAR=='M':


        # NET_IN =101510007
        # res_size_of_single_dim = 7

        NET_IN = NET_NUM_VAR

        # NET_IN = 101510010
        res_size_of_single_dim_1=RES_DIM_1_VAR
        res_size_of_single_dim_2=RES_DIM_2_VAR
        res_size_of_single_dim_3=RES_DIM_3_VAR

        # paramater order: res_exc, res_inh, input,output
        bn.build_3D_LIF_network(net_num=NET_IN
                                ,num_neurons_in_res=int(res_size_of_single_dim_1*res_size_of_single_dim_2*res_size_of_single_dim_3)
                                ,num_res_exc_neurons=int(np.round(res_size_of_single_dim_1*res_size_of_single_dim_2*res_size_of_single_dim_3*0.8))
                                ,num_res_inh_neurons=int(np.round(res_size_of_single_dim_1*res_size_of_single_dim_2*res_size_of_single_dim_3*0.2))
                                ,num_input_neurons=784
                                ,num_output_neurons=1
                                ,res_3_dims_l=[res_size_of_single_dim_1,res_size_of_single_dim_2,res_size_of_single_dim_3]
                                ,C_EE_EI_IE_II_l=[0.2, 0.1, 0.3, 0.05]
                                ,lamb=3.0
                                ,conn_density_input=0.15
                                ,conn_density_output=0.02
                                ,dist_exc_w_res_res=1.0
                                ,dist_inh_w_res_res=1.0
                                ,dist_w_res_inp=1.0
                                ,dist_w_out_res=0.0
                                ,param_tau_v=[64.0, 64.0, 64.0, 64.0]
                                ,param_tau_u=[[1, 2], [1, 2], [1, 2], [1, 2]]
                                ,param_v_thrsh=[20.0, 20.0, 20.0, 20.0]
                                ,param_b=[0.0, 0.0, 0.0, 0.0]
                                ,param_r=[64.0, 64.0, 64.0, 64.0]
                                ,param_t_rfr=[2.0, 2.0, 2.0, 2.0]
                                ,param_dist_types=[('tau_v', 'constant'),
                                                       ('tau_u', 'uniform_discrete'),
                                                       ('v_thrsh', 'constant'),
                                                       ('b', 'constant'),
                                                       ('r', 'constant'),
                                                       ('t_rfr', 'constant')],
                                 w_dist_types=[('res_res', 'constant'), ('res_inp', 'constant_randomize_+/-'),
                                               ('out_res', 'constant')]

                                )

    elif NET_TYPE_VAR=='N':


        # #  FOR NMNIST
        #
        #
        NET_IN = NET_NUM_VAR
        # res_size_of_single_dim=10

        res_size_of_single_dim_1 = RES_DIM_1_VAR
        res_size_of_single_dim_2 = RES_DIM_2_VAR
        res_size_of_single_dim_3 = RES_DIM_3_VAR

        # paramater order: res_exc, res_inh, input,output
        bn.build_3D_LIF_network(net_num=NET_IN
                                ,num_neurons_in_res=int(res_size_of_single_dim_1*res_size_of_single_dim_2*res_size_of_single_dim_3)
                                ,num_res_exc_neurons=int(np.round(res_size_of_single_dim_1*res_size_of_single_dim_2*res_size_of_single_dim_3*0.8))
                                ,num_res_inh_neurons=int(np.round(res_size_of_single_dim_1*res_size_of_single_dim_2*res_size_of_single_dim_3*0.2))
                                ,num_input_neurons=2*34*34
                                ,num_output_neurons=1
                                ,res_3_dims_l=[res_size_of_single_dim_1,res_size_of_single_dim_2,res_size_of_single_dim_3]
                                ,C_EE_EI_IE_II_l=[0.2, 0.1, 0.3, 0.05]
                                ,lamb=3.0
                                ,conn_density_input=0.15
                                ,conn_density_output=0.02
                                ,dist_exc_w_res_res=1.0
                                ,dist_inh_w_res_res=1.0
                                ,dist_w_res_inp=1.0
                                ,dist_w_out_res=0.0
                                ,param_tau_v=[64.0, 64.0, 64.0, 64.0]
                                ,param_tau_u=[[1, 2], [1, 2], [1, 2], [1, 2]]
                                ,param_v_thrsh=[20.0, 20.0, 20.0, 20.0]
                                ,param_b=[0.0, 0.0, 0.0, 0.0]
                                ,param_r=[64.0, 64.0, 64.0, 64.0]
                                ,param_t_rfr=[2.0, 2.0, 2.0, 2.0]
                                ,param_dist_types=[('tau_v', 'constant'),
                                                       ('tau_u', 'uniform_discrete'),
                                                       ('v_thrsh', 'constant'),
                                                       ('b', 'constant'),
                                                       ('r', 'constant'),
                                                       ('t_rfr', 'constant')],
                                 w_dist_types=[('res_res', 'constant'), ('res_inp', 'constant_randomize_+/-'),
                                               ('out_res', 'constant')]

                                )

