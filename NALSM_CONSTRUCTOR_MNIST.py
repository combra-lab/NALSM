import numpy as np
import time
# updated
import NALSM_GEN_SUPPORT as sup
import NALSM_SIM_SUPPORT as runSupport
import NALSM_OUTPUT_LAYER as out_lin_csfr
import NALSM_STDP as plasticity
import NALSM_ASTRO as astrocyte
import NALSM_LIF_v1 as neuron


class simulation_constructor:
    def __init__(self
                 , network_name
                 , network_path
                 , source_path
                 , save_path
                , W_RES_EXC_MIN_MAX_IN
                , W_RES_INH_MIN_MAX_IN
                , W_RES_INP_MIN_MAX_IN
                , W_OUT_RES_MIN_MAX_IN
                , SAMPLE_INPUT_DURATION_MS_IN
                , BATCH_SIZE_IN
                , record_samples_in_batch_l_IN

                , STDP_POTENTIATION_LR_IN
                , RES_LR_DECAY_IN
                , ASTRO_BIAS_OFFSET_PERCENTAGE_IN
                , ASTRO_W_SCALING_IN
                , ASTRO_TAU_IN
                , INPUT_CURRENT_IN
                , READOUT_LR_IN = 0.1

                , RES_INI_COST_W_SCLR_IN = -1

                , SPIKE_STORE = True
                , INITIALIZE_ASTRO = True
                , INITIALIZE_STDP = True
                , INITIALIZE_OUTPUT_LAYER=True
                , INITIALIZE_RES_OPS=True
                ):

        self.source_path = source_path
        self.save_path = save_path

        # _____############################################################################################______
        #      ####################### INITIALIZE STRUCTURES ############################################## START
        # _____############################################################################################______
        lifs = neuron.CUBA_LIF_network(number_of_networks_IN=BATCH_SIZE_IN)
        run_sup = runSupport.run_support(number_of_networks_IN=BATCH_SIZE_IN)
        if INITIALIZE_ASTRO == True:
            astrs = astrocyte.ASTRO_LR_CONTROLLER(astro_LR_decay_factor_IN=RES_LR_DECAY_IN
                                                  , number_of_networks_IN=BATCH_SIZE_IN
                                                  )
        if INITIALIZE_STDP == True:
            stdp = plasticity.stdp(stdp_LR_fixed_potentiation=STDP_POTENTIATION_LR_IN
                                   , res_LR_decay_factor=RES_LR_DECAY_IN
                                   , number_of_networks_IN=BATCH_SIZE_IN
                                   , w_res_exc_min_max=W_RES_EXC_MIN_MAX_IN
                                   , w_res_inh_min_max=W_RES_INH_MIN_MAX_IN
                                   , w_res_inp_min_max=W_RES_INP_MIN_MAX_IN
                                   , w_out_res_min_max=W_OUT_RES_MIN_MAX_IN
                                   )
        if INITIALIZE_OUTPUT_LAYER==True:
            olc = out_lin_csfr.output_linear_layer(output_layer_size_IN=10)

        # ASSUMES MAXIMAL WEIGHT INITIALIZATION
        if RES_INI_COST_W_SCLR_IN == -1:
            print('INITIALIZATION NETWORKS WITH MAXIMAL WEIGHTS')
            names_TF, data_TF, num_of_neurons, orig_names, orig_data = lifs.open_network_stuctures_w_custom_W_set(
                net_name=network_name,
                netPath=network_path,
                set_w_ee_ei_ie_ii_inp_IN=[W_RES_EXC_MIN_MAX_IN[1],W_RES_EXC_MIN_MAX_IN[1],W_RES_EXC_MIN_MAX_IN[1],W_RES_EXC_MIN_MAX_IN[1],W_RES_EXC_MIN_MAX_IN[1]])
        else:
            print('INITIALIZATION NETWORKS WITH SCALR ABS W: '+str(RES_INI_COST_W_SCLR_IN))
            names_TF, data_TF, num_of_neurons, orig_names, orig_data = lifs.open_network_stuctures_w_custom_W_set(
                net_name=network_name,
                netPath=network_path,
                set_w_ee_ei_ie_ii_inp_IN=[RES_INI_COST_W_SCLR_IN, RES_INI_COST_W_SCLR_IN, RES_INI_COST_W_SCLR_IN,
                                          RES_INI_COST_W_SCLR_IN, RES_INI_COST_W_SCLR_IN])

        self.TAU_U_DENSE = data_TF[names_TF.index('TAU_U_DENSE')]
        self.W_DENSE = data_TF[names_TF.index('W_DENSE')]
        self.SS_T_STATE = data_TF[names_TF.index('SS_T_STATE')]
        self.TAU_V = data_TF[names_TF.index('TAU_V')]
        self.V_THRSH = data_TF[names_TF.index('V_THRSH')]
        self.B = data_TF[names_TF.index('B')]
        self.V = data_TF[names_TF.index('V')]
        self.S = data_TF[names_TF.index('S')]
        self.T_RFR = data_TF[names_TF.index('T_RFR')]
        self.T_RFR_STATE = data_TF[names_TF.index('T_RFR_STATE')]
        self.I_PH = data_TF[names_TF.index('I_PH')]
        self.INPUT_POISSON_VALS = data_TF[names_TF.index('INPUT_POISSON_VALS')]
        self.ss_t_state_reset_scalar = data_TF[names_TF.index('ss_t_state_reset_scalar')]

        neuron_ranges_dict = dict(orig_data[orig_names.index('neuron_ranges')])
        self.res_range = neuron_ranges_dict['res_range']
        self.res_exc_range = neuron_ranges_dict['res_exc_range']
        self.res_inh_range = neuron_ranges_dict['res_inh_range']
        self.inp_range = neuron_ranges_dict['inp_range']
        self.out_range = neuron_ranges_dict['out_range']

        self.w_mask_np = orig_data[orig_names.index('W_mask')]
        self.num_total_neurons = np.shape(self.w_mask_np)[0]
        # num_output_neurons = out_range[1] - out_range[0]
        self.num_res_neurons = self.res_range[1] - self.res_range[0]
        self.num_inp_neurons = self.inp_range[1] - self.inp_range[0]
        self.num_total_syns = len(np.where(self.w_mask_np == 1.0)[0])

        print('num of neurons in network: ' + str(self.num_total_neurons))
        print('num of res neurons in network: ' + str(self.num_res_neurons))
        print('num of input neurons in network: ' + str(self.num_inp_neurons))
        print('num of syns in network: ' + str(self.num_total_syns))

        ini_names, ini_data = run_sup.simulation_initializer(inp_range_IN=self.inp_range
                                                             , res_exc_range_IN=self.res_exc_range
                                                             , res_inh_range_IN=self.res_inh_range
                                                             , res_range_IN=self.res_range
                                                             , out_range_IN=self.out_range

                                                             , w_mask_sparse_single_net_np_IN=self.w_mask_np
                                                             , record_nets_l_IN=record_samples_in_batch_l_IN
                                                             , num_neurons_in_network_IN=self.num_total_neurons
                                                             , input_sample_duration_IN=SAMPLE_INPUT_DURATION_MS_IN
                                                             )

        scatter_idx_w_dense_to_s_out_main_tf = ini_data[ini_names.index('scatter_idx_w_dense_to_s_out_main_tf')]
        scatter_S_to_S_SUBSET_inputNeurons_tf = ini_data[ini_names.index('scatter_S_to_S_SUBSET_inputNeurons_tf')]
        gather_idx_Sin_to_Wdense_tf = ini_data[ini_names.index('gather_idx_Sin_to_Wdense_tf')]
        # save_spikes_gath_ind_tf = ini_data[ini_names.index('save_spikes_gath_ind_tf')]
        # save_spikes_scatter_inds_l_of_tfs = ini_data[ini_names.index('save_spikes_scatter_inds_l_of_tfs')]
        gather_idx_Sout_to_Wdense_tf = ini_data[ini_names.index('gather_idx_Sout_to_Wdense_tf')]
        S_exc_and_inp_mask_np = ini_data[ini_names.index('S_exc_and_inp_mask_np')]
        S_inh_mask_np = ini_data[ini_names.index('S_inh_mask_np')]
        w_dense_exc_to_res_mask_np = ini_data[ini_names.index('w_dense_exc_to_res_mask_np')]
        w_dense_inh_to_res_mask_np = ini_data[ini_names.index('w_dense_inh_to_res_mask_np')]
        w_dense_inp_to_res_mask_np = ini_data[ini_names.index('w_dense_inp_to_res_mask_np')]
        w_dense_res_to_out_mask_np = ini_data[ini_names.index('w_dense_res_to_out_mask_np')]

        net_masks_names, net_masks_data = lifs.initialize_network_masks(num_total_neurons_IN=self.num_total_neurons
                                                                        , num_synapses_in_network_IN=self.num_total_syns

                                                                        , S_mask_exc_inp_np_IN=S_exc_and_inp_mask_np
                                                                        , S_mask_inh_np_IN=S_inh_mask_np

                                                                        ,
                                                                        W_dense_mask_exc_to_res_np_IN=w_dense_exc_to_res_mask_np
                                                                        ,
                                                                        W_dense_mask_inh_to_res_np_IN=w_dense_inh_to_res_mask_np
                                                                        ,
                                                                        W_dense_mask_inp_to_res_np_IN=w_dense_inp_to_res_mask_np
                                                                        ,
                                                                        W_dense_mask_res_to_out_np_IN=w_dense_res_to_out_mask_np
                                                                        )

        S_MASK_EXC_INP = net_masks_data[net_masks_names.index('S_MASK_EXC_INP')]
        S_MASK_INH = net_masks_data[net_masks_names.index('S_MASK_INH')]
        W_DENSE_MASK_EXC_TO_RES = net_masks_data[net_masks_names.index('W_DENSE_MASK_EXC_TO_RES')]
        W_DENSE_MASK_INH_TO_RES = net_masks_data[net_masks_names.index('W_DENSE_MASK_INH_TO_RES')]
        W_DENSE_MASK_INP_TO_RES = net_masks_data[net_masks_names.index('W_DENSE_MASK_INP_TO_RES')]
        W_DENSE_MASK_RES_TO_OUT = net_masks_data[net_masks_names.index('W_DENSE_MASK_RES_TO_OUT')]


        if SPIKE_STORE == True:
            spike_save_names, spike_save_data = lifs.initialize_spike_store(s_IN=self.S
                                                                            , input_duration_IN=SAMPLE_INPUT_DURATION_MS_IN
                                                                            , record_nets_l_IN=record_samples_in_batch_l_IN
                                                                            ,
                                                                            num_neurons_in_single_network_IN=self.num_total_neurons
                                                                            )

            self.spike_save_ops = spike_save_data[spike_save_names.index('spike_save_ops')]
            self.zero_out_S_STORE_op = spike_save_data[spike_save_names.index('zero_out_S_STORE_op')]
            self.condense_spike_store_to_Fn_Ft_op = spike_save_data[spike_save_names.index('condense_spike_store_to_Fn_Ft_op')]

        w_store_names, w_store_data = lifs.initialize_W_DENSE_store(W_DENSE_IN=self.W_DENSE
                                                                    , num_synapses_in_network_IN=self.num_total_syns
                                                                    )

        self.W_DENSE_STORE = w_store_data[w_store_names.index('W_DENSE_STORE')]
        self.save_W_op = w_store_data[w_store_names.index('save_W_op')]
        self.reset_W_from_saved_state_op = w_store_data[w_store_names.index('reset_W_from_saved_state_op')]

        if INITIALIZE_ASTRO == True:
            astro_names, astro_data = astrs.astro_initializer(num_total_neurons_IN=self.num_total_neurons
                                                              , inp_range_IN=self.inp_range
                                                              , res_range_IN=self.res_range
                                                              , a_bias_offset_percentage_IN=ASTRO_BIAS_OFFSET_PERCENTAGE_IN
                                                              , w_scaling_IN=ASTRO_W_SCALING_IN
                                                              , a_initial_val_IN=STDP_POTENTIATION_LR_IN
                                                              , a_tau_IN=ASTRO_TAU_IN
                                                              )

            self.ASTRO_STDP_LR = astro_data[astro_names.index('ASTRO_STDP_LR')]
            self.ASTRO_W = astro_data[astro_names.index('ASTRO_W')]
            self.ASTRO_BIAS = astro_data[astro_names.index('ASTRO_BIAS')]
            self.ASTRO_STDP_LR_STORE = astro_data[astro_names.index('ASTRO_STDP_LR_STORE')]
            self.ASTRO_BIAS_STORE = astro_data[astro_names.index('ASTRO_BIAS_STORE')]
            self.ASTRO_W_STORE = astro_data[astro_names.index('ASTRO_W_STORE')]
            self.ASTRO_W_MASK = astro_data[astro_names.index('ASTRO_W_MASK')]
            self.ASTRO_TAU = astro_data[astro_names.index('ASTRO_TAU')]

        if INITIALIZE_STDP == True:
            stdp_names, stdp_data = stdp.initialize_learning_parameters_v0(num_total_neurons_IN=self.num_total_neurons)

            self.STDP_TRACE = stdp_data[stdp_names.index('STDP_TRACE')]
            self.STDP_POTENTIATION_LR = stdp_data[stdp_names.index('STDP_POTENTIATION_LR')]
            self.STDP_POTENTIATION_LR_STORE = stdp_data[stdp_names.index('STDP_POTENTIATION_LR_STORE')]

        if INITIALIZE_OUTPUT_LAYER == True:
            olc_names, olc_data = olc.initialize_layer_structures(num_res_neurons_IN=self.num_res_neurons
                                                                  , initial_batch_size_IN=BATCH_SIZE_IN
                                                                  )

            self.olc_label_Inp = olc_data[olc_names.index('label_Inp')]
            self.olc_target_output_ph = olc_data[olc_names.index('target_output_ph')]
            self.olc_W1_ph = olc_data[olc_names.index('W1_ph')]
            self.olc_b1_ph = olc_data[olc_names.index('b1_ph')]
            self.olc_W1 = olc_data[olc_names.index('W1')]
            self.olc_b1 = olc_data[olc_names.index('b1')]
            self.olc_BS = olc_data[olc_names.index('BS')]
            self.olc_S_AGG_RES = olc_data[olc_names.index('S_AGG_RES')]
            self.olc_S_AGG_RES_PH = olc_data[olc_names.index('S_AGG_RES_PH')]

        # _____############################################################################################______
        #      ####################### INITIALIZE STRUCTURES ############################################## END
        # _____############################################################################################______

        # _____############################################################################################______
        #      ####################### INITIALIZE TF OPS ############################################## START
        # _____############################################################################################______

        # NEURON OPS
        if INITIALIZE_RES_OPS == True:
            self.LIF_update_neuron_states_op = lifs.update_neuron_states_wo_R(v_IN=self.V
                                                                         , tau_v_IN=self.TAU_V
                                                                         , tau_u_dense_IN=self.TAU_U_DENSE
                                                                         , b_IN=self.B
                                                                         , ss_t_state_IN=self.SS_T_STATE
                                                                         , W_dense_IN=self.W_DENSE
                                                                         , I_poisson_IN=self.INPUT_POISSON_VALS
                                                                         ,
                                                                         scatter_idx_W_dense_to_Neurons_IN=scatter_idx_w_dense_to_s_out_main_tf
                                                                         , scatter_shp_IN=[BATCH_SIZE_IN, self.num_total_neurons]
                                                                         ,
                                                                         scatter_S_to_S_SUBSET_inputNeurons_IN=scatter_S_to_S_SUBSET_inputNeurons_tf
                                                                         , batch_by_num_input_neurons_l_IN=[BATCH_SIZE_IN,
                                                                                                            self.num_inp_neurons]
                                                                         , input_current_IN=INPUT_CURRENT_IN
                                                                         )
            self.LIF_propagate_spike_op = lifs.propagate_spikes(spikes_IN=self.S
                                                           , ss_t_state_IN=self.SS_T_STATE
                                                           , gather_idx_Sin_to_Wdense_IN=gather_idx_Sin_to_Wdense_tf)
            self.LIF_evolveCurr_op = lifs.evolve_input_currents(ss_t_state_IN=self.SS_T_STATE)
            self.LIF_compute_spikes_op = lifs.register_spikes(v_IN=self.V, v_trsh_IN=self.V_THRSH, s_IN=self.S, t_rfr_state_IN=self.T_RFR_STATE)
            self.LIF_reset_v_op = lifs.reset_spiked_neurons(s_IN=self.S, v_IN=self.V)
            self.LIF_update_refractory_state_op = lifs.update_refractory_var(t_rfr_IN=self.T_RFR, t_rfr_state_IN=self.T_RFR_STATE, s_IN=self.S)
            self.LIF_update_input_poisson_vals_op = lifs.set_new_input_batch(I_PH_IN=self.I_PH
                                                                        , INPUT_POISSON_VALS_IN=self.INPUT_POISSON_VALS)
            self.LIF_reset_V_to_zero_op = lifs.zero_out_V(V_IN=self.V)
            self.LIF_reset_S_to_zero_op = lifs.zero_out_S(S_IN=self.S)
            self.LIF_reset_T_RFR_STATE_to_zero_op = lifs.zero_out_T_RFR_STATE(T_RFR_STATE_IN=self.T_RFR_STATE)
            self.LIF_reset_SS_T_STATE_op = lifs.reset_ss_t_state_op(SS_T_STATE_IN=self.SS_T_STATE
                                                               , ss_t_state_reset_scalar_IN=self.ss_t_state_reset_scalar)

        # ASTRO OPS
        if INITIALIZE_ASTRO == True:
            self.update_astro_op = astrs.update_astro_state(ASTRO_STDP_LR_IN=self.ASTRO_STDP_LR
                                                       , ASTRO_BIAS_IN=self.ASTRO_BIAS
                                                       , ASTRO_W_IN=self.ASTRO_W
                                                       , ASTRO_W_MASK_IN=self.ASTRO_W_MASK
                                                       , S_IN=self.S
                                                       , ASTRO_TAU_IN=self.ASTRO_TAU)

            self.ASTRO_decay_astro_res_lr_op = astrs.decay_astro_stdp_lr(astro_lr_IN=self.ASTRO_STDP_LR)
            self.ASTRO_decay_astro_w_op = astrs.decay_astro_w(astro_w_IN=self.ASTRO_W)
            self.ASTRO_decay_astro_bias_op = astrs.decay_astro_bias(astro_bias_IN=self.ASTRO_BIAS)

            self.ASTRO_reset_astro_res_lr_op = astrs.reset_astro_stdp_lr(astro_lr_IN=self.ASTRO_STDP_LR,
                                                                    astro_lr_store_IN=self.ASTRO_STDP_LR_STORE)
            self.ASTRO_reset_astro_w_op = astrs.reset_astro_w(astro_w_IN=self.ASTRO_W, astro_w_store_IN=self.ASTRO_W_STORE)
            self.ASTRO_reset_astro_bias_op = astrs.reset_astro_bias(astro_bias_IN=self.ASTRO_BIAS,
                                                               astro_bias_store_IN=self.ASTRO_BIAS_STORE)

        # STDP OPS
        if INITIALIZE_STDP == True:
            self.PLASTICITY_update_stdp_only_op = stdp.STDP_w_astro_depression(
                                                                W_dense_IN=self.W_DENSE
                                                                , S_IN=self.S

                                                                # STDP INPUTS
                                                                , trace_IN=self.STDP_TRACE

                                                                , W_dense_mask_exc_to_res_IN=W_DENSE_MASK_EXC_TO_RES
                                                                , W_dense_mask_inh_to_res_IN=W_DENSE_MASK_INH_TO_RES
                                                                , W_dense_mask_inp_to_res_IN=W_DENSE_MASK_INP_TO_RES
                                                                , W_dense_mask_res_to_out_IN=W_DENSE_MASK_RES_TO_OUT

                                                                , S_mask_exc_and_inp_IN=S_MASK_EXC_INP
                                                                , S_mask_inh_IN=S_MASK_INH

                                                                , astro_LR_IN=self.ASTRO_STDP_LR

                                                                , gather_idx_Sin_to_Wdense_IN=gather_idx_Sin_to_Wdense_tf
                                                                , gather_idx_Sout_to_Wdense_IN=gather_idx_Sout_to_Wdense_tf

                                                                , stdp_potentiation_LR_IN=self.STDP_POTENTIATION_LR
                                                            )

            self.PLASTICITY_update_trace_op = stdp.update_trace(trace_IN=self.STDP_TRACE, S_IN=self.S)
            self.PLASTICITY_decay_res_lr_op = stdp.decay_res_lr(res_lr_IN=self.STDP_POTENTIATION_LR)
            self.PLASTICITY_reset_res_lr_op = stdp.reset_res_lr(res_lr_IN=self.STDP_POTENTIATION_LR,
                                                           res_lr_store_IN=self.STDP_POTENTIATION_LR_STORE)

        # output layer ops
        if INITIALIZE_OUTPUT_LAYER == True:
            olc_ops_names, olc_ops_data = olc.initialize_layer_ops(S_AGG_RES_IN=self.olc_S_AGG_RES
                                                                   , label_Inp_IN=self.olc_label_Inp
                                                                   , target_output_ph_IN=self.olc_target_output_ph
                                                                   , W1_IN=self.olc_W1
                                                                   , b1_IN=self.olc_b1
                                                                   , BS_IN=self.olc_BS
                                                                   , lr_IN=READOUT_LR_IN
                                                                   )
            self.OLC_train_step_op = olc_ops_data[olc_ops_names.index('train_step')]
            self.OLC_accuracy_eval_per_batch_op = olc_ops_data[olc_ops_names.index('accuracy_eval_per_batch')]

            self.OLC_agg_spikes_op = olc.aggregate_spikes_for_output_layer_op(S_AGG_RES_IN=self.olc_S_AGG_RES
                                                                      , S_IN=self.S
                                                                      , res_range_IN=self.res_range
                                                                      )
            self.OLC_zero_out_S_AGG_RES_op = olc.zero_out_S_AGG_RES_op(S_AGG_RES_IN=self.olc_S_AGG_RES)

            self.OLC_set_S_AGG_to_previously_saved_batch_op = olc.assign_previously_saved_S_AGG(S_AGG_RES_IN=self.olc_S_AGG_RES,S_AGG_RES_PH_IN=self.olc_S_AGG_RES_PH)

            w_ini_names, w_ini_data = olc.initialize_saved_output_layer_weights(W_IN=self.olc_W1
                                                                              , b_IN=self.olc_b1
                                                                              , new_W_IN=self.olc_W1_ph
                                                                              , new_b_IN=self.olc_b1_ph
                                                                                )
            self.OLC_set_new_W_op = w_ini_data[w_ini_names.index('set_new_W_op')]
            self.OLC_set_new_b_op = w_ini_data[w_ini_names.index('set_new_b_op')]


            # _____############################################################################################______
        #      ####################### INITIALIZE TF OPS ############################################## END
        # _____############################################################################################______

        #########################################################################################################
        ############################## INITIALIZE BATCH NETWORK ############################################# END
        #########################################################################################################




    ################### VAL/TST GENERATE OPS ################################################################# START

    def generate_validation_or_test_spike_data(self
                                               , sess_IN
                                               , saver_object_IN
                                               , valid_or_test_data_INPUT_VEC_IN
                                               , valid_or_test_data_LABEL_VEC_IN
                                               , valid_or_test_data_LABEL_SCALAR_IN
                                               , BATCH_SIZE_IN
                                               , SAMPLE_INPUT_DURATION_MS_IN
                                               , NUM_DATA_SAMPLES_IN
                                               , spike_store_batch_mod_IN
                                               , record_samples_in_batch_l_IN
                                               , root_save_name_IN):

        '''
        GENERATES VALIDATION/TEST SPIKE COUNT DATA FOR NALSM MODEL
        
        :param sess_IN: TF SESSION
        :param saver_object_IN: OJECT FOR SAVING DATA
        :param valid_or_test_data_INPUT_VEC_IN: INPUT DATA
        :param valid_or_test_data_LABEL_VEC_IN: LABEL DATA IN ONE HOT VECTOR FORM
        :param valid_or_test_data_LABEL_SCALAR_IN: RAW OUTPUT DATA(0,1,2,3...) SCALARS
        :param BATCH_SIZE_IN: SIZE OF BATCH
        :param SAMPLE_INPUT_DURATION_MS_IN: TIME OF PRESENTING EACH SAMPLE TO LIQUID 
        :param NUM_DATA_SAMPLES_IN: TOTAL NUMBER OF TETS OR VALIDATION SAMPLES
        :param spike_store_batch_mod_IN: WHICH BATCH TO STORE SPIKE DATA FROM
        :param record_samples_in_batch_l_IN: SAMPLES TO STORE SPIKE DATA FROM IN BATCH
        :param root_save_name_IN: SAVE NAME
        :return: 
        '''


        ITERATIONS = int(np.floor(NUM_DATA_SAMPLES_IN/BATCH_SIZE_IN))
        print('VALIDATION/TESTING RUNNING FOR '+str(ITERATIONS)+' ITERATIONS WITH BATCH SIZE: '+str(BATCH_SIZE_IN))

        spikes_files_l = []
        s_agg_files_l = []

        for ITER in range(0, ITERATIONS):
            if ITER > 0:
                print('VAL/TST Epoch ' + str(ITER - 1) + ' competed in ' + str(time.time() - st_time))
            st_time = time.time()

            scalar_labels_in_batch_for_spike_data_l = []
            batch_samples_agg_temp = []
            scalar_labels_in_batch_l = []
            target_output_for_linear_layer_labels_in_batch_l = []
            for bt in range(0, BATCH_SIZE_IN):
                sample_idx = ((ITER * BATCH_SIZE_IN) + bt)
                batch_samples_agg_temp.append(valid_or_test_data_INPUT_VEC_IN[sample_idx])
                target_output_for_linear_layer_labels_in_batch_l.append(valid_or_test_data_LABEL_VEC_IN[sample_idx])
                scalar_labels_in_batch_l.append(valid_or_test_data_LABEL_SCALAR_IN[sample_idx])
            if ITER % spike_store_batch_mod_IN == spike_store_batch_mod_IN-1:
                for bt_s in range(0,len(record_samples_in_batch_l_IN)):
                    sample_idx = ((ITER * BATCH_SIZE_IN) + bt_s)
                    scalar_labels_in_batch_for_spike_data_l.append(valid_or_test_data_LABEL_SCALAR_IN[sample_idx])

            sess_IN.run(self.LIF_update_input_poisson_vals_op, feed_dict={self.I_PH: np.vstack(batch_samples_agg_temp)})

            sess_IN.run(self.LIF_reset_V_to_zero_op)
            sess_IN.run(self.LIF_reset_S_to_zero_op)
            sess_IN.run(self.LIF_reset_T_RFR_STATE_to_zero_op)
            sess_IN.run(self.LIF_reset_SS_T_STATE_op)
            sess_IN.run(self.ASTRO_reset_astro_res_lr_op)
            sess_IN.run(self.ASTRO_reset_astro_w_op)
            sess_IN.run(self.ASTRO_reset_astro_bias_op)
            sess_IN.run(self.PLASTICITY_reset_res_lr_op)
            sess_IN.run(self.reset_W_from_saved_state_op)
            sess_IN.run(self.OLC_zero_out_S_AGG_RES_op)

            for t in range(0, SAMPLE_INPUT_DURATION_MS_IN):
                ############### LIF FUNCTIONS ###################### START
                sess_IN.run(self.LIF_compute_spikes_op)
                sess_IN.run(self.LIF_reset_v_op)

                # SPIKE RECORD OP
                if ITER%spike_store_batch_mod_IN==spike_store_batch_mod_IN-1:
                    sess_IN.run(self.spike_save_ops[t])
                sess_IN.run(self.OLC_agg_spikes_op)

                sess_IN.run(self.LIF_propagate_spike_op)
                sess_IN.run(self.LIF_update_neuron_states_op)
                sess_IN.run(self.LIF_evolveCurr_op)
                sess_IN.run(self.LIF_update_refractory_state_op)
                ############### MAIN LIF FUNCTIONS ###################### END

                ############### PLASTICITY FUNCTIONS ###################### START
                sess_IN.run(self.PLASTICITY_update_stdp_only_op)
                sess_IN.run(self.PLASTICITY_update_trace_op)
                ############### PLASTICITY FUNCTIONS ###################### END

                ############### ASTRO FUNCTIONS ########################## START
                sess_IN.run(self.update_astro_op)
                ############### ASTRO FUNCTIONS ############################ END

                ############### PLASTICITY ASTRO DECAYS ###################### START
                sess_IN.run(self.PLASTICITY_decay_res_lr_op)
                sess_IN.run(self.ASTRO_decay_astro_res_lr_op)
                sess_IN.run(self.ASTRO_decay_astro_w_op)
                sess_IN.run(self.ASTRO_decay_astro_bias_op)
                ############### PLASTICITY ASTRO DECAYS ###################### END

            ################### EXTRACT SPIKE DATA ############### START
            if ITER % spike_store_batch_mod_IN == spike_store_batch_mod_IN-1:
                spike_rec_per_batch = sess_IN.run(self.condense_spike_store_to_Fn_Ft_op)
                sess_IN.run(self.zero_out_S_STORE_op)

                spikes_save_filename = str(root_save_name_IN) + '_SPIKES_BATCH_' + str(ITER) + '.spikes'

                saver_object_IN.save_data(signal=1
                                      , names=['ITER','spike_rec_per_batch', 'scalar_labels_in_batch_for_spike_data_l']
                                      , data=[ITER,spike_rec_per_batch, scalar_labels_in_batch_for_spike_data_l]
                                      , save_filename=spikes_save_filename
                                      )

                spike_rec_per_batch = 0
                scalar_labels_in_batch_for_spike_data_l.clear()
                spikes_files_l.append(spikes_save_filename)
            ################### EXTRACT SPIKE DATA ############### END

            ################### SAVE SPIKE COUNT DATA ############### START
            s_agg_save_filename = str(root_save_name_IN) + '_S_AGG_BATCH_' + str(ITER) + '.datasagg'

            saver_object_IN.save_data(signal=1
                                  , names=['ITER','olc_S_AGG_RES', 'target_output_for_linear_layer_labels_in_batch_arr','scalar_labels_in_batch_arr']
                                  , data=[ITER,sess_IN.run(self.olc_S_AGG_RES), np.vstack(target_output_for_linear_layer_labels_in_batch_l),np.asarray(scalar_labels_in_batch_l)]
                                  , save_filename=s_agg_save_filename
                                  )
            s_agg_files_l.append(s_agg_save_filename)
            target_output_for_linear_layer_labels_in_batch_l.clear()
            scalar_labels_in_batch_l.clear()
            ################### SAVE SPIKE COUNT DATA ############### END

        return spikes_files_l,s_agg_files_l

    def generate_validation_or_test_spike_data_STATIC(self
                                               , sess_IN
                                               , saver_object_IN
                                               , valid_or_test_data_INPUT_VEC_IN
                                               , valid_or_test_data_LABEL_VEC_IN
                                               , valid_or_test_data_LABEL_SCALAR_IN
                                               , BATCH_SIZE_IN
                                               , SAMPLE_INPUT_DURATION_MS_IN
                                               , NUM_DATA_SAMPLES_IN
                                               , spike_store_batch_mod_IN
                                               , record_samples_in_batch_l_IN
                                               , root_save_name_IN):

        '''
        GENERATES VALIDATION/TEST SPIKE COUNT DATA FOR LSM MODEL
        
        :param sess_IN: TF SESSION
        :param saver_object_IN: OJECT FOR SAVING DATA
        :param valid_or_test_data_INPUT_VEC_IN: INPUT DATA
        :param valid_or_test_data_LABEL_VEC_IN: LABEL DATA IN ONE HOT VECTOR FORM
        :param valid_or_test_data_LABEL_SCALAR_IN: RAW OUTPUT DATA(0,1,2,3...) SCALARS
        :param BATCH_SIZE_IN: SIZE OF BATCH
        :param SAMPLE_INPUT_DURATION_MS_IN: TIME OF PRESENTING EACH SAMPLE TO LIQUID 
        :param NUM_DATA_SAMPLES_IN: TOTAL NUMBER OF TETS OR VALIDATION SAMPLES
        :param spike_store_batch_mod_IN: WHICH BATCH TO STORE SPIKE DATA FROM
        :param record_samples_in_batch_l_IN: SAMPLES TO STORE SPIKE DATA FROM IN BATCH
        :param root_save_name_IN: SAVE NAME
        :return: 
        '''

        ITERATIONS = int(np.floor(NUM_DATA_SAMPLES_IN/BATCH_SIZE_IN))
        print('VALIDATION/TESTING RUNNING FOR '+str(ITERATIONS)+' ITERATIONS WITH BATCH SIZE: '+str(BATCH_SIZE_IN))

        spikes_files_l = []
        s_agg_files_l = []

        for ITER in range(0, ITERATIONS):
            if ITER > 0:
                print('VAL/TST Epoch ' + str(ITER - 1) + ' competed in ' + str(time.time() - st_time))
            st_time = time.time()

            scalar_labels_in_batch_for_spike_data_l = []
            batch_samples_agg_temp = []
            scalar_labels_in_batch_l = []
            target_output_for_linear_layer_labels_in_batch_l = []
            for bt in range(0, BATCH_SIZE_IN):
                sample_idx = ((ITER * BATCH_SIZE_IN) + bt)
                batch_samples_agg_temp.append(valid_or_test_data_INPUT_VEC_IN[sample_idx])
                target_output_for_linear_layer_labels_in_batch_l.append(valid_or_test_data_LABEL_VEC_IN[sample_idx])
                scalar_labels_in_batch_l.append(valid_or_test_data_LABEL_SCALAR_IN[sample_idx])
            if ITER % spike_store_batch_mod_IN == spike_store_batch_mod_IN-1:
                for bt_s in range(0,len(record_samples_in_batch_l_IN)):
                    sample_idx = ((ITER * BATCH_SIZE_IN) + bt_s)
                    scalar_labels_in_batch_for_spike_data_l.append(valid_or_test_data_LABEL_SCALAR_IN[sample_idx])

            sess_IN.run(self.LIF_update_input_poisson_vals_op, feed_dict={self.I_PH: np.vstack(batch_samples_agg_temp)})

            sess_IN.run(self.LIF_reset_V_to_zero_op)
            sess_IN.run(self.LIF_reset_S_to_zero_op)
            sess_IN.run(self.LIF_reset_T_RFR_STATE_to_zero_op)
            sess_IN.run(self.LIF_reset_SS_T_STATE_op)
            # sess_IN.run(self.ASTRO_reset_astro_res_lr_op)
            # sess_IN.run(self.ASTRO_reset_astro_w_op)
            # sess_IN.run(self.ASTRO_reset_astro_bias_op)
            # sess_IN.run(self.PLASTICITY_reset_res_lr_op)
            # sess_IN.run(self.reset_W_from_saved_state_op)
            sess_IN.run(self.OLC_zero_out_S_AGG_RES_op)

            for t in range(0, SAMPLE_INPUT_DURATION_MS_IN):
                ############### LIF FUNCTIONS ###################### START
                sess_IN.run(self.LIF_compute_spikes_op)
                sess_IN.run(self.LIF_reset_v_op)

                # SPIKE RECORD OP
                if ITER%spike_store_batch_mod_IN==spike_store_batch_mod_IN-1:
                    sess_IN.run(self.spike_save_ops[t])
                sess_IN.run(self.OLC_agg_spikes_op)

                sess_IN.run(self.LIF_propagate_spike_op)
                sess_IN.run(self.LIF_update_neuron_states_op)
                sess_IN.run(self.LIF_evolveCurr_op)
                sess_IN.run(self.LIF_update_refractory_state_op)
                ############### MAIN LIF FUNCTIONS ###################### END

                ############### PLASTICITY FUNCTIONS ###################### START
                # sess_IN.run(self.PLASTICITY_update_stdp_only_op)
                # sess_IN.run(self.PLASTICITY_update_trace_op)
                ############### PLASTICITY FUNCTIONS ###################### END

                ############### ASTRO FUNCTIONS ########################## START
                # sess_IN.run(self.update_astro_op)
                ############### ASTRO FUNCTIONS ############################ END

                ############### PLASTICITY ASTRO DECAYS ###################### START
                # sess_IN.run(self.PLASTICITY_decay_res_lr_op)
                # sess_IN.run(self.ASTRO_decay_astro_res_lr_op)
                # sess_IN.run(self.ASTRO_decay_astro_w_op)
                # sess_IN.run(self.ASTRO_decay_astro_bias_op)
                ############### PLASTICITY ASTRO DECAYS ###################### END

            ################### EXTRACT SPIKE DATA ############### START
            if ITER % spike_store_batch_mod_IN == spike_store_batch_mod_IN-1:
                spike_rec_per_batch = sess_IN.run(self.condense_spike_store_to_Fn_Ft_op)
                sess_IN.run(self.zero_out_S_STORE_op)

                spikes_save_filename = str(root_save_name_IN) + '_SPIKES_BATCH_' + str(ITER) + '.spikes'

                saver_object_IN.save_data(signal=1
                                      , names=['ITER','spike_rec_per_batch', 'scalar_labels_in_batch_for_spike_data_l']
                                      , data=[ITER,spike_rec_per_batch, scalar_labels_in_batch_for_spike_data_l]
                                      , save_filename=spikes_save_filename
                                      )

                spike_rec_per_batch = 0
                scalar_labels_in_batch_for_spike_data_l.clear()
                spikes_files_l.append(spikes_save_filename)
            ################### EXTRACT SPIKE DATA ############### END

            ################### SAVE SPIKE COUNT DATA ############### START
            s_agg_save_filename = str(root_save_name_IN) + '_S_AGG_BATCH_' + str(ITER) + '.datasagg'

            saver_object_IN.save_data(signal=1
                                  , names=['ITER','olc_S_AGG_RES', 'target_output_for_linear_layer_labels_in_batch_arr','scalar_labels_in_batch_arr']
                                  , data=[ITER,sess_IN.run(self.olc_S_AGG_RES), np.vstack(target_output_for_linear_layer_labels_in_batch_l),np.asarray(scalar_labels_in_batch_l)]
                                  , save_filename=s_agg_save_filename
                                  )
            s_agg_files_l.append(s_agg_save_filename)
            target_output_for_linear_layer_labels_in_batch_l.clear()
            scalar_labels_in_batch_l.clear()
            ################### SAVE SPIKE COUNT DATA ############### END

        return spikes_files_l,s_agg_files_l

    ################### VAL/TST GENERATE OPS ################################################################# END
    def evaluate_output_layer_on_validation_or_test_s_agg_data(self
                                                               , sess_IN
                                                               , data_batched_INPUT_VEC_IN
                                                               , data_batched_LABEL_VEC_IN
                                                               , data_batched_LABEL_SCALAR_IN
                                                               ):
        '''
        EVALUATES OUTPUT LAYER ON SINGLE BATCH OF DATA
        
        :param sess_IN: TF SESSION
        :param data_batched_INPUT_VEC_IN: INPUT DATA 
        :param data_batched_LABEL_VEC_IN: OUTPUT DATA ONE HOT FORMAT
        :param data_batched_LABEL_SCALAR_IN: SCALAR LABELS
        :return: 
        '''

        accuracy_l = []
        for batch in range(0,len(data_batched_INPUT_VEC_IN)):
            sess_IN.run(self.OLC_set_S_AGG_to_previously_saved_batch_op, feed_dict={self.olc_S_AGG_RES_PH:data_batched_INPUT_VEC_IN[batch]})
            accuracy_l.append(sess_IN.run(self.OLC_accuracy_eval_per_batch_op, feed_dict={self.olc_label_Inp:data_batched_LABEL_SCALAR_IN[batch]}))

        return accuracy_l

    def evaluate_output_layer_on_FULL_validation_or_test_s_agg_data(self
                                                                    , sess_IN
                                                                    , saver_object_IN
                                                                    , list_of_data_files_IN
                                                                    , num_of_simultaneous_open_files=2000):

        '''
        EVALUATES PERFORMANCE OF OUTPUT LAYER ON VALIDATION OR TEST DATA
        
        :param sess_IN: TF SESSION
        :param saver_object_IN: OBJECT FOR SAVING DATA
        :param list_of_data_files_IN: SPECIFIES DATA TO TEST/VALIDATE ON
        :param num_of_simultaneous_open_files: NUMBER OF BATCHES TO LOAD AT ONE TIME
        :return: 
        '''

        accuracy_list_full = []
        data_batched_INPUT_VEC_l = []
        data_batched_LABEL_VEC_l = []
        data_batched_LABEL_SCALAR_l = []
        for fn in range(0,len(list_of_data_files_IN)):
            names, data = sup.unpack_file(filename=list_of_data_files_IN[fn],dataPath=self.source_path)
            data_batched_INPUT_VEC_l.append(data[names.index('olc_S_AGG_RES')])
            data_batched_LABEL_VEC_l.append(data[names.index('target_output_for_linear_layer_labels_in_batch_arr')])
            data_batched_LABEL_SCALAR_l.append(data[names.index('scalar_labels_in_batch_arr')])
            if len(data_batched_INPUT_VEC_l)==num_of_simultaneous_open_files or (fn==(len(list_of_data_files_IN)-1) and len(data_batched_INPUT_VEC_l)>0):
                accuracy_list_full.extend(self.evaluate_output_layer_on_validation_or_test_s_agg_data(sess_IN=sess_IN
                                                                           , data_batched_INPUT_VEC_IN=data_batched_INPUT_VEC_l
                                                                           , data_batched_LABEL_VEC_IN=data_batched_LABEL_VEC_l
                                                                           , data_batched_LABEL_SCALAR_IN=data_batched_LABEL_SCALAR_l
                                                                            ))
                data_batched_INPUT_VEC_l.clear()
                data_batched_LABEL_VEC_l.clear()
                data_batched_LABEL_SCALAR_l.clear()
        if len(accuracy_list_full)==len(list_of_data_files_IN):
            print('ALL EVALUATION FILES PROCESSED AND ACCOUNTED FOR TOTALING: '+str(len(list_of_data_files_IN)))
        else:
            print('ERROR: NOT ALL EVALUTATION FILES PROCEED AND ACCOUNTED, ACCOUNTED ONLY: '+str(len(accuracy_list_full))+' out of '+str(len(list_of_data_files_IN)))

        average_accuracy = np.average(accuracy_list_full)

        return average_accuracy



    ################### TRAINING OPS ######################################################################## START

    def train_on_set_number_of_batches(self
                                       , sess_IN
                                       , saver_object_IN
                                       , training_iteration_range_IN
                                       , training_dataset_size_IN
                                       , list_of_data_files_IN
                                       , BATCH_SIZE_IN
                                       , root_save_name_IN
                                       , train_data_INPUT_VEC_IN
                                       , train_data_LABEL_VEC_IN
                                       , train_data_LABEL_SCALAR_IN
                                       , spike_store_batch_mod_IN
                                       , record_samples_in_batch_l_IN
                                       , SAMPLE_INPUT_DURATION_MS_IN
                                       , REPEATS_ON_ITER_RANGE_IN=1
                                       ):

        '''
        GENERATES SPIKE COUNT DATA FOR EACH SAMPLE AND TRAINS THE OUTPUT LAYER ONCE ON ALL GENERATED BATCHS
        THIS FUNCTION USED FOR NALSM MODEL, WITH ASTROCYTE MODULATED STDP

        :param sess_IN: TF SESSION
        :param saver_object_IN: OBJECT FOR SAVING FILES
        :param training_iteration_range_IN: BATCHES TO GENERATE AND TRAIN
        :param training_dataset_size_IN: FULL SIZE OF TRAINING SET
        :param list_of_data_files_IN: ALREADY GENERATED SPIKE COUNT FILES (S_AGG FILES)
        :param BATCH_SIZE_IN: BATCH SIZE
        :param root_save_name_IN: CORE SAVE NAME
        :param train_data_INPUT_VEC_IN: INPUT DATA
        :param train_data_LABEL_VEC_IN: LABELS IN FORMAT FOR TRAINING OUTPUT LAYER(ONE HOT VECTOR)
        :param train_data_LABEL_SCALAR_IN: RAW LABELS (0,1,2,3,4..)
        :param spike_store_batch_mod_IN: WHEN TO STORE ALL SPIKES FROM LIQUID
        :param record_samples_in_batch_l_IN: WHICH SAMPLES TO STORE SPIKE DATA FROM IN BATCH
        :param SAMPLE_INPUT_DURATION_MS_IN: DURATION OF SAMPLE INPUT PRESENTED TO LIQUID (250 ms)
        :param REPEATS_ON_ITER_RANGE_IN: ALWAYS 1
        :return: 
        '''


        max_num_of_save_files = int(training_dataset_size_IN/BATCH_SIZE_IN)
        print('max number of training data save files: '+str(max_num_of_save_files))

        raw_iters_arr = training_iteration_range_IN[0]+np.arange(training_iteration_range_IN[1]-training_iteration_range_IN[0])
        all_iters_arr = np.mod(raw_iters_arr,max_num_of_save_files)
        idx = np.where(raw_iters_arr< max_num_of_save_files)[0]
        iters_with_no_existing_data_l = raw_iters_arr[idx]

        print('-----------------------------------: '+str(raw_iters_arr))
        print('------------------------all_iters_l: ' + str(all_iters_arr))
        print('-----------idx for no_existing_data: '+str(idx))
        print('------iters_with_no_existing_data_l: ' + str(iters_with_no_existing_data_l))

        spikes_files_l = []
        s_agg_files_l = []
        st_time_train_total = time.time()
        if len(iters_with_no_existing_data_l)>0:
            print('PROCEEDING TO GENERATE NEW S AGG RES DATA')
            for ITER in iters_with_no_existing_data_l:
                print('SPIKE DATA NEEDS TO BE GENERATED, NOW GENERATING SPIKE DATA FOR BATCH: ' + str(ITER))
                # iter_ned is the batch number or the ITER from eval function

                if ITER > iters_with_no_existing_data_l[0]:
                    print('Epoch ' + str(ITER - 1) + ' competed in ' + str(time.time() - st_time))
                st_time = time.time()

                scalar_labels_in_batch_for_spike_data_l = []
                batch_samples_agg_temp = []
                scalar_labels_in_batch_l = []
                target_output_for_linear_layer_labels_in_batch_l = []
                for bt in range(0, BATCH_SIZE_IN):
                    sample_idx = ((ITER * BATCH_SIZE_IN) + bt)
                    batch_samples_agg_temp.append(train_data_INPUT_VEC_IN[sample_idx])
                    target_output_for_linear_layer_labels_in_batch_l.append(train_data_LABEL_VEC_IN[sample_idx])
                    scalar_labels_in_batch_l.append(train_data_LABEL_SCALAR_IN[sample_idx])
                if ITER % spike_store_batch_mod_IN == spike_store_batch_mod_IN - 1:
                    for bt_s in range(0, len(record_samples_in_batch_l_IN)):
                        sample_idx = ((ITER * BATCH_SIZE_IN) + bt_s)
                        scalar_labels_in_batch_for_spike_data_l.append(train_data_LABEL_SCALAR_IN[sample_idx])

                sess_IN.run(self.LIF_update_input_poisson_vals_op,
                            feed_dict={self.I_PH: np.vstack(batch_samples_agg_temp)})

                sess_IN.run(self.LIF_reset_V_to_zero_op)
                sess_IN.run(self.LIF_reset_S_to_zero_op)
                sess_IN.run(self.LIF_reset_T_RFR_STATE_to_zero_op)
                sess_IN.run(self.LIF_reset_SS_T_STATE_op)
                sess_IN.run(self.ASTRO_reset_astro_res_lr_op)
                sess_IN.run(self.ASTRO_reset_astro_w_op)
                sess_IN.run(self.ASTRO_reset_astro_bias_op)
                sess_IN.run(self.PLASTICITY_reset_res_lr_op)
                sess_IN.run(self.reset_W_from_saved_state_op)
                sess_IN.run(self.OLC_zero_out_S_AGG_RES_op)

                for t in range(0, SAMPLE_INPUT_DURATION_MS_IN):
                    ############### LIF FUNCTIONS ###################### START
                    sess_IN.run(self.LIF_compute_spikes_op)
                    sess_IN.run(self.LIF_reset_v_op)

                    # SPIKE RECORD OP
                    if ITER % spike_store_batch_mod_IN == spike_store_batch_mod_IN - 1:
                        sess_IN.run(self.spike_save_ops[t])
                    sess_IN.run(self.OLC_agg_spikes_op)

                    sess_IN.run(self.LIF_propagate_spike_op)
                    sess_IN.run(self.LIF_update_neuron_states_op)
                    sess_IN.run(self.LIF_evolveCurr_op)
                    sess_IN.run(self.LIF_update_refractory_state_op)
                    ############### MAIN LIF FUNCTIONS ###################### END

                    ############### PLASTICITY FUNCTIONS ###################### START
                    sess_IN.run(self.PLASTICITY_update_stdp_only_op)
                    sess_IN.run(self.PLASTICITY_update_trace_op)
                    ############### PLASTICITY FUNCTIONS ###################### END

                    ############### ASTRO FUNCTIONS ########################## START
                    sess_IN.run(self.update_astro_op)
                    ############### ASTRO FUNCTIONS ############################ END

                    ############### PLASTICITY ASTRO DECAYS ###################### START
                    sess_IN.run(self.PLASTICITY_decay_res_lr_op)
                    sess_IN.run(self.ASTRO_decay_astro_res_lr_op)
                    sess_IN.run(self.ASTRO_decay_astro_w_op)
                    sess_IN.run(self.ASTRO_decay_astro_bias_op)
                    ############### PLASTICITY ASTRO DECAYS ###################### END

                ################### EXTRACT SPIKE DATA ############### START
                if ITER % spike_store_batch_mod_IN == spike_store_batch_mod_IN - 1:
                    spike_rec_per_batch = sess_IN.run(self.condense_spike_store_to_Fn_Ft_op)
                    sess_IN.run(self.zero_out_S_STORE_op)

                    saver_object_IN.save_data(signal=1
                                              , names=['ITER', 'spike_rec_per_batch',
                                                       'scalar_labels_in_batch_for_spike_data_l']
                                              , data=[ITER, spike_rec_per_batch,
                                                      scalar_labels_in_batch_for_spike_data_l]
                                              , save_filename=str(root_save_name_IN) + '_BATCH_' + str(
                            ITER) + '.spikes'
                                              )

                    spike_rec_per_batch = 0
                    scalar_labels_in_batch_for_spike_data_l.clear()
                    spikes_files_l.append(str(root_save_name_IN) + '_BATCH_' + str(ITER) + '.spikes')
                ################### EXTRACT SPIKE DATA ############### END

                ################### SAVE SPIKE COUNTS DATA ############### START
                if ITER<iters_with_no_existing_data_l[-1]:
                    saver_object_IN.save_data(signal=1
                                              , names=['ITER', 'olc_S_AGG_RES',
                                                       'target_output_for_linear_layer_labels_in_batch_arr',
                                                       'scalar_labels_in_batch_arr']
                                              , data=[ITER, sess_IN.run(self.olc_S_AGG_RES),
                                                      np.vstack(target_output_for_linear_layer_labels_in_batch_l),
                                                      np.asarray(scalar_labels_in_batch_l)]
                                              ,
                                              save_filename=str(root_save_name_IN) + '_BATCH_' + str(ITER) + '.datasagg'
                                              )
                else:
                    sup.save_non_tf_data(names=['ITER', 'olc_S_AGG_RES',
                                                       'target_output_for_linear_layer_labels_in_batch_arr',
                                                       'scalar_labels_in_batch_arr']
                                              , data=[ITER, sess_IN.run(self.olc_S_AGG_RES),
                                                      np.vstack(target_output_for_linear_layer_labels_in_batch_l),
                                                      np.asarray(scalar_labels_in_batch_l)]
                                              , filename=str(root_save_name_IN) + '_BATCH_' + str(ITER) + '.datasagg'
                                              , savePath=self.save_path)
                    print('LAST S_AGG BATCH OF BLOCK GENERATED AND SAVED TO: '+str(str(root_save_name_IN) + '_BATCH_' + str(ITER) + '.datasagg'))

                s_agg_files_l.append(str(root_save_name_IN) + '_BATCH_' + str(ITER) + '.datasagg')
                target_output_for_linear_layer_labels_in_batch_l.clear()
                scalar_labels_in_batch_l.clear()
            ################### SAVE SPIKE COUNTS DATA ############### END

        ################### OPEN AND TRAIN OUTPUT LAYER ON GENERATED BATCHES ############### START
        print('TIME TO GENERATE SPIKE NEW: '+str(time.time()-st_time_train_total))
        print('PROCEEDING TO ASSEMBLE DATA FOR TRAINING OUTPUT LAYER ON S AGG DATA')
        # extract all data files
        st_time_train_output_total = time.time()
        tp2_target_output_l = []
        tp2_S_AGG_RES_input_l = []
        for iter_ed in all_iters_arr:
            print('SPIKE DATA WAS ALREADY GENEATED, OPENING FILE WITH DATA USING BATCH NUMBER: ' + str(iter_ed))

            tp2_names, tp2_data = sup.unpack_file(
                filename=str(root_save_name_IN) + '_BATCH_' + str(iter_ed) + '.datasagg',dataPath=self.source_path)
            tp2_target_output_l.append(
                tp2_data[tp2_names.index('target_output_for_linear_layer_labels_in_batch_arr')])
            tp2_S_AGG_RES_input_l.append(tp2_data[tp2_names.index('olc_S_AGG_RES')])

        print('ASSEMBLED ' + str(len(tp2_target_output_l)) + ' DATA BATCHES FROM SAVED FILES, PROCEEDING TO TRAIN...')

        for rir in range(0,REPEATS_ON_ITER_RANGE_IN):
            print('WHOLE_ITER_SUBSET_RANGE TRAINING iteration: '+str(rir))
            for train_iter in range(0, len(tp2_target_output_l)):
                sess_IN.run(self.OLC_set_S_AGG_to_previously_saved_batch_op,
                            feed_dict={self.olc_S_AGG_RES_PH: tp2_S_AGG_RES_input_l[train_iter]})
                sess_IN.run(self.OLC_train_step_op,
                            feed_dict={self.olc_target_output_ph: tp2_target_output_l[train_iter]})

        tp2_target_output_l.clear()
        tp2_S_AGG_RES_input_l.clear()

        print('TRAINING OUTPUT LAYER ON BATCH BLOCK COMPLETE, TIME TAKE: '+str(time.time()-st_time_train_output_total))
        ################### OPEN AND TRAING OUTPUT LAYER ON GENERATED BATCHES ############### END


        return spikes_files_l, s_agg_files_l


    def train_on_set_number_of_batches_STATIC(self
                                       , sess_IN
                                       , saver_object_IN
                                       , training_iteration_range_IN
                                       , training_dataset_size_IN
                                       , list_of_data_files_IN
                                       , BATCH_SIZE_IN
                                       , root_save_name_IN
                                       , train_data_INPUT_VEC_IN
                                       , train_data_LABEL_VEC_IN
                                       , train_data_LABEL_SCALAR_IN
                                       , spike_store_batch_mod_IN
                                       , record_samples_in_batch_l_IN
                                       , SAMPLE_INPUT_DURATION_MS_IN
                                       , REPEATS_ON_ITER_RANGE_IN=1
                                       ):

        '''
        GENERATES SPIKE COUNT DATA FOR EACH SAMPLE AND TRAINS THE OUTPUT LAYER ONCE ON ALL GENERATED BATCHS
        THIS FUNCTION USED FOR LSM MODEL, WITH STATIC/FIXED WEIGHTS
        
        :param sess_IN: TF SESSION
        :param saver_object_IN: OBJECT FOR SAVING FILES
        :param training_iteration_range_IN: BATCHES TO GENERATE AND TRAIN
        :param training_dataset_size_IN: FULL SIZE OF TRAINING SET
        :param list_of_data_files_IN: ALREADY GENERATED SPIKE COUNT FILES (S_AGG FILES)
        :param BATCH_SIZE_IN: BATCH SIZE
        :param root_save_name_IN: CORE SAVE NAME
        :param train_data_INPUT_VEC_IN: INPUT DATA
        :param train_data_LABEL_VEC_IN: LABELS IN FORMAT FOR TRAINING OUTPUT LAYER(ONE HOT VECTOR)
        :param train_data_LABEL_SCALAR_IN: RAW LABELS (0,1,2,3,4..)
        :param spike_store_batch_mod_IN: WHEN TO STORE ALL SPIKES FROM LIQUID
        :param record_samples_in_batch_l_IN: WHICH SAMPLES TO STORE SPIKE DATA FROM IN BATCH
        :param SAMPLE_INPUT_DURATION_MS_IN: DURATION OF SAMPLE INPUT PRESENTED TO LIQUID (250 ms)
        :param REPEATS_ON_ITER_RANGE_IN: ALWAYS 1
        :return: 
        '''


        max_num_of_save_files = int(training_dataset_size_IN/BATCH_SIZE_IN)
        print('max number of training data save files: '+str(max_num_of_save_files))

        raw_iters_arr = training_iteration_range_IN[0]+np.arange(training_iteration_range_IN[1]-training_iteration_range_IN[0])
        all_iters_arr = np.mod(raw_iters_arr,max_num_of_save_files)
        idx = np.where(raw_iters_arr< max_num_of_save_files)[0]
        iters_with_no_existing_data_l = raw_iters_arr[idx]

        print('-----------------------------------: '+str(raw_iters_arr))
        print('------------------------all_iters_l: ' + str(all_iters_arr))
        print('-----------idx for no_existing_data: '+str(idx))
        print('------iters_with_no_existing_data_l: ' + str(iters_with_no_existing_data_l))

        spikes_files_l = []
        s_agg_files_l = []
        st_time_train_total = time.time()
        if len(iters_with_no_existing_data_l)>0:
            print('PROCEEDING TO GENERATE NEW S AGG RES DATA')
            for ITER in iters_with_no_existing_data_l:
                print('SPIKE DATA NEEDS TO BE GENERATED, NOW GENERATING SPIKE DATA FOR BATCH: ' + str(ITER))
                # iter_ned is the batch number or the ITER from eval function

                #  edit the iter time printing....
                if ITER > iters_with_no_existing_data_l[0]:
                    print('Epoch ' + str(ITER - 1) + ' competed in ' + str(time.time() - st_time))
                st_time = time.time()

                scalar_labels_in_batch_for_spike_data_l = []
                batch_samples_agg_temp = []
                scalar_labels_in_batch_l = []
                target_output_for_linear_layer_labels_in_batch_l = []
                for bt in range(0, BATCH_SIZE_IN):
                    sample_idx = ((ITER * BATCH_SIZE_IN) + bt)
                    batch_samples_agg_temp.append(train_data_INPUT_VEC_IN[sample_idx])
                    target_output_for_linear_layer_labels_in_batch_l.append(train_data_LABEL_VEC_IN[sample_idx])
                    scalar_labels_in_batch_l.append(train_data_LABEL_SCALAR_IN[sample_idx])
                if ITER % spike_store_batch_mod_IN == spike_store_batch_mod_IN - 1:
                    for bt_s in range(0, len(record_samples_in_batch_l_IN)):
                        sample_idx = ((ITER * BATCH_SIZE_IN) + bt_s)
                        scalar_labels_in_batch_for_spike_data_l.append(train_data_LABEL_SCALAR_IN[sample_idx])

                sess_IN.run(self.LIF_update_input_poisson_vals_op,
                            feed_dict={self.I_PH: np.vstack(batch_samples_agg_temp)})

                sess_IN.run(self.LIF_reset_V_to_zero_op)
                sess_IN.run(self.LIF_reset_S_to_zero_op)
                sess_IN.run(self.LIF_reset_T_RFR_STATE_to_zero_op)
                sess_IN.run(self.LIF_reset_SS_T_STATE_op)
                # sess_IN.run(self.ASTRO_reset_astro_res_lr_op)
                # sess_IN.run(self.ASTRO_reset_astro_w_op)
                # sess_IN.run(self.ASTRO_reset_astro_bias_op)
                # sess_IN.run(self.PLASTICITY_reset_res_lr_op)
                # sess_IN.run(self.reset_W_from_saved_state_op)
                sess_IN.run(self.OLC_zero_out_S_AGG_RES_op)

                for t in range(0, SAMPLE_INPUT_DURATION_MS_IN):
                    ############### LIF FUNCTIONS ###################### START
                    sess_IN.run(self.LIF_compute_spikes_op)
                    sess_IN.run(self.LIF_reset_v_op)

                    # SPIKE RECORD OP
                    if ITER % spike_store_batch_mod_IN == spike_store_batch_mod_IN - 1:
                        sess_IN.run(self.spike_save_ops[t])
                    sess_IN.run(self.OLC_agg_spikes_op)

                    sess_IN.run(self.LIF_propagate_spike_op)
                    sess_IN.run(self.LIF_update_neuron_states_op)
                    sess_IN.run(self.LIF_evolveCurr_op)
                    sess_IN.run(self.LIF_update_refractory_state_op)
                    ############### MAIN LIF FUNCTIONS ###################### END

                    ############### PLASTICITY FUNCTIONS ###################### START
                    # sess_IN.run(self.PLASTICITY_update_stdp_only_op)
                    # sess_IN.run(self.PLASTICITY_update_trace_op)
                    ############### PLASTICITY FUNCTIONS ###################### END

                    ############### ASTRO FUNCTIONS ########################## START
                    # sess_IN.run(self.update_astro_op)
                    ############### ASTRO FUNCTIONS ############################ END

                    ############### PLASTICITY ASTRO DECAYS ###################### START
                    # sess_IN.run(self.PLASTICITY_decay_res_lr_op)
                    # sess_IN.run(self.ASTRO_decay_astro_res_lr_op)
                    # sess_IN.run(self.ASTRO_decay_astro_w_op)
                    # sess_IN.run(self.ASTRO_decay_astro_bias_op)
                    ############### PLASTICITY ASTRO DECAYS ###################### END

                ################### EXTRACT SPIKE DATA ############### START
                if ITER % spike_store_batch_mod_IN == spike_store_batch_mod_IN - 1:
                    spike_rec_per_batch = sess_IN.run(self.condense_spike_store_to_Fn_Ft_op)
                    sess_IN.run(self.zero_out_S_STORE_op)

                    saver_object_IN.save_data(signal=1
                                              , names=['ITER', 'spike_rec_per_batch',
                                                       'scalar_labels_in_batch_for_spike_data_l']
                                              , data=[ITER, spike_rec_per_batch,
                                                      scalar_labels_in_batch_for_spike_data_l]
                                              , save_filename=str(root_save_name_IN) + '_BATCH_' + str(
                            ITER) + '.spikes'
                                              )

                    spike_rec_per_batch = 0
                    scalar_labels_in_batch_for_spike_data_l.clear()
                    spikes_files_l.append(str(root_save_name_IN) + '_BATCH_' + str(ITER) + '.spikes')
                ################### EXTRACT SPIKE DATA ############### END

                ################### SAVE SPIKE COUNTS DATA ############### START
                if ITER<iters_with_no_existing_data_l[-1]:
                    saver_object_IN.save_data(signal=1
                                              , names=['ITER', 'olc_S_AGG_RES',
                                                       'target_output_for_linear_layer_labels_in_batch_arr',
                                                       'scalar_labels_in_batch_arr']
                                              , data=[ITER, sess_IN.run(self.olc_S_AGG_RES),
                                                      np.vstack(target_output_for_linear_layer_labels_in_batch_l),
                                                      np.asarray(scalar_labels_in_batch_l)]
                                              ,
                                              save_filename=str(root_save_name_IN) + '_BATCH_' + str(ITER) + '.datasagg'
                                              )
                else:
                    sup.save_non_tf_data(names=['ITER', 'olc_S_AGG_RES',
                                                       'target_output_for_linear_layer_labels_in_batch_arr',
                                                       'scalar_labels_in_batch_arr']
                                              , data=[ITER, sess_IN.run(self.olc_S_AGG_RES),
                                                      np.vstack(target_output_for_linear_layer_labels_in_batch_l),
                                                      np.asarray(scalar_labels_in_batch_l)]
                                              , filename=str(root_save_name_IN) + '_BATCH_' + str(ITER) + '.datasagg'
                                              , savePath=self.save_path)
                    print('LAST S_AGG BATCH OF BLOCK GENERATED AND SAVED TO: '+str(str(root_save_name_IN) + '_BATCH_' + str(ITER) + '.datasagg'))

                s_agg_files_l.append(str(root_save_name_IN) + '_BATCH_' + str(ITER) + '.datasagg')
                target_output_for_linear_layer_labels_in_batch_l.clear()
                scalar_labels_in_batch_l.clear()
                ################### SAVE SPIKE COUNTS DATA ############### END

        ################### OPEN AND TRAIN OUTPUT LAYER ON GENERATED BATCHES ############### START
        print('TIME TO GENERATE SPIKE NEW: '+str(time.time()-st_time_train_total))
        print('PROCEEDING TO ASSEMBLE DATA FOR TRAINING OUTPUT LAYER ON S AGG DATA')
        # extract all data files
        st_time_train_output_total = time.time()
        tp2_target_output_l = []
        tp2_S_AGG_RES_input_l = []
        for iter_ed in all_iters_arr:
            print('SPIKE DATA WAS ALREADY GENEATED, OPENING FILE WITH DATA USING BATCH NUMBER: ' + str(iter_ed))

            tp2_names, tp2_data = sup.unpack_file(
                filename=str(root_save_name_IN) + '_BATCH_' + str(iter_ed) + '.datasagg',dataPath=self.source_path)
            tp2_target_output_l.append(
                tp2_data[tp2_names.index('target_output_for_linear_layer_labels_in_batch_arr')])
            tp2_S_AGG_RES_input_l.append(tp2_data[tp2_names.index('olc_S_AGG_RES')])

        print('ASSEMBLED ' + str(len(tp2_target_output_l)) + ' DATA BATCHES FROM SAVED FILES, PROCEEDING TO TRAIN...')

        for rir in range(0,REPEATS_ON_ITER_RANGE_IN):
            print('WHOLE_ITER_SUBSET_RANGE TRAINING iteration: '+str(rir))
            for train_iter in range(0, len(tp2_target_output_l)):
                sess_IN.run(self.OLC_set_S_AGG_to_previously_saved_batch_op,
                            feed_dict={self.olc_S_AGG_RES_PH: tp2_S_AGG_RES_input_l[train_iter]})
                sess_IN.run(self.OLC_train_step_op,
                            feed_dict={self.olc_target_output_ph: tp2_target_output_l[train_iter]})

        tp2_target_output_l.clear()
        tp2_S_AGG_RES_input_l.clear()

        print('TRAINING OUTPUT LAYER ON BATCH BLOCK COMPLETE, TIME TAKE: '+str(time.time()-st_time_train_output_total))
        ################### OPEN AND TRAIN OUTPUT LAYER ON GENERATED BATCHES ############### END

        return spikes_files_l, s_agg_files_l

    ################### TRAINING OPS ######################################################################## END

    def run_W_DENSE_initialization_loop(self
                                        , sess_IN
                                        , INI_NUM_TRAIN_VAL_TEST_L_IN
                                        , INI_BATCH_SIZE_IN
                                        , INI_initialization_idx_l
                                        , INI_SAMPLE_INPUT_DURATION_MS_IN
                                        , train_data_INPUT_VEC
                                        , train_data_LABEL_SCALAR
                                        ):

        '''
        RUNS INITIALIZATION OF LIQUID WEIGHTS BY PRESENTING SERIES OF SNAPSHOTS TO LIQUID WITH ASTRO-STDP
        
        :param sess_IN: TF SESSION
        :param INI_NUM_TRAIN_VAL_TEST_L_IN: NUMBER OF TRAINING DATA TO USE
        :param INI_BATCH_SIZE_IN: BATCH SIZE = 1
        :param INI_initialization_idx_l: INDEXES OF RANDOMLY ORDERED TRAINING SAMPLES
        :param INI_SAMPLE_INPUT_DURATION_MS_IN: SNAPSHOT DURATION
        :param train_data_INPUT_VEC: INPUT DATA
        :param train_data_LABEL_SCALAR: LABELS
        :return: 
        '''


        for INI_ITER in range(0,INI_NUM_TRAIN_VAL_TEST_L_IN[0]):

            print('INITIALIZATION ITERATION '+str(INI_ITER)+' of '+str(INI_NUM_TRAIN_VAL_TEST_L_IN[0]))

            batch_samples_agg_temp = []
            scalar_labels_in_batch_l = []
            for bt in range(0, INI_BATCH_SIZE_IN):
                sample_idx = ((INI_ITER * INI_BATCH_SIZE_IN) + bt)
                batch_samples_agg_temp.append(train_data_INPUT_VEC[INI_initialization_idx_l[sample_idx]])
                scalar_labels_in_batch_l.append(train_data_LABEL_SCALAR[INI_initialization_idx_l[sample_idx]])

            sess_IN.run(self.LIF_update_input_poisson_vals_op, feed_dict={self.I_PH: np.vstack(batch_samples_agg_temp)})

            sess_IN.run(self.LIF_reset_V_to_zero_op)
            sess_IN.run(self.LIF_reset_S_to_zero_op)
            sess_IN.run(self.LIF_reset_T_RFR_STATE_to_zero_op)
            sess_IN.run(self.LIF_reset_SS_T_STATE_op)
            sess_IN.run(self.ASTRO_reset_astro_res_lr_op)
            sess_IN.run(self.ASTRO_reset_astro_w_op)
            sess_IN.run(self.ASTRO_reset_astro_bias_op)
            sess_IN.run(self.PLASTICITY_reset_res_lr_op)

            for t in range(0,INI_SAMPLE_INPUT_DURATION_MS_IN):
                ############### LIF FUNCTIONS ###################### START
                sess_IN.run(self.LIF_compute_spikes_op)
                sess_IN.run(self.LIF_reset_v_op)

                # SPIKE RECORD OP
                # sess_IN.run(spike_save_ops[t])
                # sess_IN.run(OLC_agg_spikes)

                sess_IN.run(self.LIF_propagate_spike_op)
                sess_IN.run(self.LIF_update_neuron_states_op)
                sess_IN.run(self.LIF_evolveCurr_op)
                sess_IN.run(self.LIF_update_refractory_state_op)
                ############### MAIN LIF FUNCTIONS ###################### END

                ############### PLASTICITY FUNCTIONS ###################### START
                sess_IN.run(self.PLASTICITY_update_stdp_only_op)
                sess_IN.run(self.PLASTICITY_update_trace_op)
                ############### PLASTICITY FUNCTIONS ###################### END

                ############### ASTRO FUNCTIONS ########################## START
                sess_IN.run(self.update_astro_op)
                ############### ASTRO FUNCTIONS ############################ END

                ############### PLASTICITY ASTRO DECAYS ###################### START
                # sess_IN.run(PLASTICITY_decay_res_lr_op)
                # sess_IN.run(ASTRO_decay_astro_res_lr_op)
                # sess_IN.run(ASTRO_decay_astro_w_op)
                # sess_IN.run(ASTRO_decay_astro_bias_op)
                ############### PLASTICITY ASTRO DECAYS ###################### END

        print('W_DENSE INTIALIZATION COMPLETE.')


