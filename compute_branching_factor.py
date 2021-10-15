import numpy as np
import NALSM_GEN_SUPPORT as sup
import multiprocessing as mp
import os

class bf_comp:
    def __init__(self,ver,num_processes=10,buffer_IN=25):

        self.main_Path = os.getcwd()
        self.data_path = self.main_Path + '/train_data/ver_'+str(ver)
        self.net_path = self.main_Path + '/networks'


        for file in os.listdir(self.data_path):
            if file.endswith(".spikes"):
                self.spike_file = file
                end_idx = file.index('.spikes')
                self.save_file = file[0:end_idx]+'.bf'

        print('PROCESSING FILE: '+str(self.spike_file))
        print('    OUTPUT FILE: ' + str(self.save_file))

        params_filename='VER_'+str(ver)+'_params.params'
        names_params, data_params = sup.unpack_file(filename=params_filename,dataPath=self.data_path)
        network_num = data_params[names_params.index('NET_NUM_IN')]
        sample_input_duration = data_params[names_params.index('N1_SAMPLE_INPUT_DURATION_MS_IN')]


        net_name='Network_'+str(network_num)
        names_net, data_net = sup.unpack_file(filename=net_name,dataPath=self.net_path)
        W_mask = data_net[names_net.index('W_mask')]
        neuron_ranges_dict = dict(data_net[names_net.index('neuron_ranges')])
        inp_range = neuron_ranges_dict['inp_range']
        res_range = neuron_ranges_dict['res_range']
        self.neuron_range = res_range


        pre_list,post_list = self.network_pre_post_tuple_lists(W_mask,self.neuron_range)

        self.pre_dict = dict(pre_list)
        self.post_dict = dict(post_list)


        names_spike, data_spike = sup.unpack_file(filename=self.spike_file,dataPath=self.data_path)
        spike_data_orig = data_spike[names_spike.index('spike_rec_per_batch')]

        self.ft_fn_by_sample_in_batch = self.split_spike_data_by_batch(spike_data_orig,self.neuron_range,sample_input_duration,buffer=buffer_IN)


        self.start_time = buffer_IN
        self.end_time = sample_input_duration-buffer_IN

        self.num_of_processors = num_processes

        ######### MULTI PROCESS INITIALIZATION #################
        # Define IPC managers
        manager1 = mp.Manager()
        # Define lists (queue) for tasks and computation results
        self.data_feed1 = manager1.Queue()
        self.status1 = manager1.Queue()
        self.spike_to_avalanche_dict = manager1.dict()
        self.processes1 = []
        # activate full branch saving processes
        for i in range(self.num_of_processors):
            # Set process name
            process_name = 'Pb%i' % i
            # Create the process, and connect it to the worker function
            new_process = mp.Process(target=self.compute_bf_for_neuron, args=(process_name, self.data_feed1, self.status1))
            # Add new process to the list of processes
            self.processes1.append(new_process)
            # Start the process
            new_process.start()
        ######### MULTI PROCESS INITIALIZATION #################

    def add_inputs_to_parallel_feed(self,signal, neuron, dt, start_time, end_time, sample):

        self.data_feed1.put([signal, neuron, dt, start_time, end_time, sample])

    def network_pre_post_tuple_lists(self, W_mask_IN, neuron_range_IN):
        
        pre_list = []
        post_list = []
        for n in range(neuron_range_IN[0],neuron_range_IN[1]):
            idx_all = np.where(W_mask_IN[:,n]==1)[0]
            idx_filt1 = np.where(idx_all >=neuron_range_IN[0])[0]
            idx_filt2 = np.where(idx_all < neuron_range_IN[1])[0]
            idx_of_idx_all = np.intersect1d(idx_filt1,idx_filt2)
            post_list.append((n,idx_all[idx_of_idx_all]))

            idx_all_pre = np.where(W_mask_IN[n,:] == 1)[0]
            idx_filt1_pre = np.where(idx_all_pre >= neuron_range_IN[0])[0]
            idx_filt2_pre = np.where(idx_all_pre < neuron_range_IN[1])[0]
            idx_of_idx_all_pre = np.intersect1d(idx_filt1_pre, idx_filt2_pre)
            pre_list.append((n, idx_all_pre[idx_of_idx_all_pre]))

        return pre_list,post_list
            

    def compute_bf_for_neuron(self, process_name,data_feed,status):

        print('[%s] Avalanche processor launched, waiting for data' % process_name)

        while True:
            data = data_feed.get()

            if data[0] == -1:
                print('[%s] Avalanche process terminated' % process_name)
                status.put(1)
                break
            else:

                neuron_idx = data[1]
                dt = data[2]
                start_time = data[3]
                end_time = data[4]
                sample = data[5]

                ft = self.ft_fn_by_sample_in_batch[sample][0]
                fn = self.ft_fn_by_sample_in_batch[sample][1]


                post_spikes_over_time_list = []
                pre_spikes_over_time_list = []
                t_list = []
                for t in range(start_time+dt,end_time-dt):

                    idx_t = np.where(ft == t)[0]

                    if len(np.where(fn[idx_t] == neuron_idx)[0])>0:
                        idx0 = np.where(ft>=(t-dt))[0]
                        idx1 = np.where(ft < t)[0]
                        filt_time_idx = np.intersect1d(idx0, idx1)
                        fn_filt = fn[filt_time_idx]
                        pre_spike_count = 0
                        for n in self.pre_dict[neuron_idx]:
                            pre_spike_count += len(np.where(fn_filt==n)[0])

                        idx00 = np.where(ft >= (t + 1))[0]
                        idx11 = np.where(ft < (t + 1 + dt))[0]
                        filt_time_idx1 = np.intersect1d(idx00, idx11)
                        fn_filt1 = fn[filt_time_idx1]
                        post_spike_count = 0
                        for n in self.post_dict[neuron_idx]:
                            post_spike_count += len(np.where(fn_filt1==n)[0])

                        pre_spikes_over_time_list.append(pre_spike_count)
                        post_spikes_over_time_list.append(post_spike_count)
                        t_list.append(t)

                # compute diff series
                pre_spikes_over_time_arr = np.clip(np.asarray(pre_spikes_over_time_list),1.0,(((self.neuron_range[1]-self.neuron_range[0])*dt)+100))
                post_spikes_over_time_arr = np.asarray(post_spikes_over_time_list)
                bf_series = np.divide(post_spikes_over_time_arr,pre_spikes_over_time_arr)
                ave_bf = np.average(bf_series)

                status.put([neuron_idx,pre_spikes_over_time_arr,post_spikes_over_time_arr,t_list,bf_series,ave_bf])


    def kill_workers(self,process_count):
        for i in range(0, process_count):
            print('KILL SWITCH SENT FOR PROCESS '+str(i))
            self.add_inputs_to_parallel_feed(signal=-1, neuron=0, dt=0, start_time=0, end_time=0, sample=0)

        sum1 = 0
        while sum1 != process_count:
            temp = self.status1.get()
            if type(temp) == int:

                sum1 = sum1 + temp
            else:
                # print(temp)
                print('Found other stuff in queue....check')


    def split_spike_data_by_batch(self,spike_data_orig_IN, neuron_range_IN, duration_ms, buffer=25):

        unique_idxs = np.unique(np.asarray(spike_data_orig_IN)[:, 0])


        ft_fn_list = []
        for u in unique_idxs:
            idx = np.where(np.asarray(spike_data_orig_IN)[:, 0] == u)

            Ft = np.squeeze(np.asarray(spike_data_orig_IN)[idx, 1])
            Fn = np.squeeze(np.asarray(spike_data_orig_IN)[idx, 2])

            idx1 = np.where(Ft >= buffer)[0]
            idx2 = np.where(Ft < duration_ms-buffer)[0]
            filt_idx = np.intersect1d(idx1, idx2)

            Ft_filt = Ft[filt_idx]
            Fn_filt = Fn[filt_idx]

            idx1_0 = np.where(Fn_filt >= neuron_range_IN[0])[0]
            idx2_0 = np.where(Fn_filt < neuron_range_IN[1])[0]
            filt_idx0 = np.intersect1d(idx1_0, idx2_0)

            Ft_filt_f = Ft_filt[filt_idx0]
            Fn_filt_f = Fn_filt[filt_idx0]

            ft_fn_list.append([Ft_filt_f,Fn_filt_f])

        return ft_fn_list


    def compute_bf_for_all_batches(self,dt_IN=4):

        all_data_AGG_list =[]
        for i in range(0,len(self.ft_fn_by_sample_in_batch)):
            print('Processing Batch: '+str(i))
            for n in range(self.neuron_range[0],self.neuron_range[1]):
                self.add_inputs_to_parallel_feed(signal=0, neuron=n, dt=dt_IN, start_time=self.start_time, end_time=self.end_time, sample=i)
            counter = 0
            all_data_list = []
            neurons_list = []
            while (self.neuron_range[1]-self.neuron_range[0]) != len(neurons_list):
                temp = self.status1.get()
                if type(temp) == list:
                    neurons_list.append(temp[0])
                    all_data_list.append(temp)
                    counter = counter + 1
            ns,cs = np.unique(neurons_list,return_counts=True)
            if np.amax(cs)==1:
                print('num of neurons processed: '+str(len(neurons_list))+'__no duplicates found')
            else:
                print('num of neurons processed: ' + str(len(neurons_list)) + '__DUPLIATES FUOUND!!!!!!!!!!!!!!!!!!!!!!')
            all_data_AGG_list.append(all_data_list)

        sup.save_non_tf_data(names=['all_data'],data=[all_data_AGG_list],filename=self.save_file,savePath=self.data_path)

        print('SAVED_COMPUTED_BF_DATA_TO: '+str(self.save_file))

        self.kill_workers(process_count=self.num_of_processors)




if __name__ == '__main__':

    SAVE_VER_INP = input('VERSION? [int]: ')
    SAVE_VER = int(SAVE_VER_INP)

    bfc = bf_comp(ver=SAVE_VER,num_processes=5)
    bfc.compute_bf_for_all_batches()














