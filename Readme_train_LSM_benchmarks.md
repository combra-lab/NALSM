# Instructions for Training LSM Benchmarks

This file contains instructions for training all LSM benchmarks presented in the paper, which include models:
1. [NALSM](#1-NALSM)
2. [LSM+STDP](#2-LSMSTDP)
3. [LSM](#3-LSM)
4. [LSM+AP-STDP](#4-LSMAP_STDP)


## 1. NALSM

### 1.1 MNIST
To train NALSM model on MNIST, enter the following command: 
	
	python NALSM_RUN_MAIN_SIM_MNIST.py

This will prompt for the following inputs:
* `GPU?` : enter an integer specifying the gpu to use for training.
* `VERSION? [int]` : enter an integer to label the training simulation.
* `NET_NUM_VAR? [int]` : enter the number of the network.
* `BATCH_SIZE? [int]` : specify the number of samples to train at same time (batch), for liquids with 1000 neurons, batch size of 250 will work on a 12gb gpu. For larger liquids(8000), smaller batch sizes of 50 should work.
* `BATCHS_PER_BLOCK? [int]` : specify number of batchs to keep in memory for training output layer, we found 2500 samples works well in terms of speed and memory (so for batch size of 250, this should be set to 10 (10 x 250 = 2500), for batch size 50 set this to 50 (50 x 50 = 2500).
* `ASTRO_W_SCALING? [float]` : specify the astrocyte weight detailed in equation 7 of paper. We used 0.015 for all 1000 neuron liquids, and 0.0075 for 8000 neuron liquids. Generally accuracy peaks with a value around 0.01 (See Appendix).

This will generate all output in sub-directory `<Dir>/<Project Name>/train_data/ver_XX/` where `XX` is `VERSION` number.

### 1.2 N-MNIST
To train NALSM model on N-MNIST, enter the following command: 

	python NALSM_RUN_MAIN_SIM_N_MNIST.py

All input prompts and output are the same as described above for run file `NALSM_RUN_MAIN_SIM_MNIST.py`.

### 1.3 Fashion-MNIST

To train NALSM model on Fashion-MNIST, enter the following command: 

	python NALSM_RUN_MAIN_SIM_F_MNIST.py

All input prompts and output are the same as described above for run file `NALSM_RUN_MAIN_SIM_MNIST.py`.


## 2. LSM+STDP

### 2.1 MNIST
To train LSM+STDP model on MNIST, enter the following command: 

	python LSM_STDP_RUN_MAIN_SIM_MNIST.py

All input prompts and output are the same as described above in [Section 1.1](#11-mnist) for run file `NALSM_RUN_MAIN_SIM_MNIST.py`.

### 2.2 N-MNIST
To train LSM+STDP model on N-MNIST, enter the following command: 

	python LSM_STDP_RUN_MAIN_SIM_N_MNIST.py

All input prompts and output are the same as described above in [Section 1.1](#11-mnist) for run file `NALSM_RUN_MAIN_SIM_MNIST.py`.


## 3. LSM

### 3.1 MNIST
To train LSM model on MNIST, enter the following command: 

	python LSM_RUN_MAIN_SIM_MNIST.py
	
All input prompts and output are the same as described above in [Section 1.1](#11-mnist) for run file `NALSM_RUN_MAIN_SIM_MNIST.py`, except for input prompt:
* `INTIAL_ABS_W_SCLR? [float]` : specify a value for the absolute synaptic weight of all connections in the fixed weight liquid.
	
### 3.2 N-MNIST
To train LSM model on N-MNIST, enter the following command:

	python LSM_RUN_MAIN_SIM_N_MNIST.py

All input prompts and output are the same as described above for run file `LSM_RUN_MAIN_SIM_MNIST.py`.


## 4. LSM+AP_STDP

### 4.1 MNIST
To train LSM+AP_STDP model on MNIST, enter the following command:

	python LSM_APSTDP_RUN_MAIN_SIM_MNIST.py
	
All input prompts and output are the same as described above in [Section 1.1](#11-mnist) for run file `NALSM_RUN_MAIN_SIM_MNIST.py`, except for input prompts:
* `STDP_NEURON_RATE_THESHOLD? [float]` : specify the rate (Hz) above which LTP and below which LTD will be applied to all input connections of all neurons (See Paper Appendix).
* `STDP_NEURON_RATE_RANGE? [float]` : specify the rate range above and below STDP_NEURON_RATE_THESHOLD in which LTP and LTD will be applied to connections, respectively (See Paper Appendix).  
* `INTIAL_ABS_W_SCLR? [float]` : specify the value of all liquid weights before liquid weights are initialized with AP-STDP (See Paper Appendix).

### 4.2 N-MNIST
To train LSM+AP_STDP model on N-MNIST, enter the following command: 

	python LSM_APSTDP_RUN_MAIN_SIM_N_MNIST.py

All input prompts and output are the same as described above for run file `LSM_APSTDP_RUN_MAIN_SIM_MNIST.py`.
