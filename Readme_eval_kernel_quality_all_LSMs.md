# Instructions for Evaluating Model Kernel Quality

This file contains instructions for evaluation of LSM kernel quality presented in the paper for models:
1. [NALSM](#1-NALSM)
2. [LSM+STDP](#2-LSMSTDP)
3. [LSM](#3-LSM)
4. [LSM+AP-STDP](#4-LSMAP_STDP)


Model liquid kernel quality was calculated from metrics:
* SP: linear separation of liquid
* AP: generalization of liquid

which was done for both MNIST and N-MNIST datasets. To compute SP and AP metrics, first noisy spike counts must be generated for the AP metric, as follows. 

## Generating Noisy Spike Counts
### 1. NALSM

#### 1.1 MNIST
To generate noisy spike counts for NALSM model on MNIST, enter the following command:

	python NALSM_RUN_MAIN_SIM_MNIST_NOISE.py

The run file requires a `W_INI.wdata` file (the initialized weights), which should have been generated during model training.

The run file will prompt for the following inputs:
* `GPU?` : enter an integer to select the gpu for the training simulation.
* `VERSION? [int]` : enter the version number of the trained model.
* `NET_NUM_VAR? [int]` : enter the network number of the trained model.
* `BATCH_SIZE? [int]` : use the same value used for training the model.
* `BATCHS_PER_BLOCK? [int]` : use the same value used for training the model.
	
The run file will generate all output in sub-directory `<Dir>/<Project Name>/train_data/ver_XX/` where `XX` is VERSION number.

#### 1.2 N-MNIST
To generate noisy spike counts for NALSM model on N-MNIST, enter the following command:

	python NALSM_RUN_MAIN_SIM_N_MNIST_NOISE.py

As above, the run file requires 'W_INI.wdata' file. All input prompts and output are the same as in [Section 1.1](#11-mnist).


### 2. LSM+STDP

#### 2.1 MNIST
To generate noisy spike counts for LSM+STDP model on MNIST, enter the following command:

	python LSM_STDP_RUN_MAIN_SIM_MNIST_NOISE.py
	
The run file requires a `W_INI.wdata` file. All input prompts and output are the same as in [Section 1.1](#11-mnist).

#### 2.2 N-MNIST
To generate noisy spike counts for LSM+STDP model on N-MNIST, enter the following command:

	python LSM_STDP_RUN_MAIN_SIM_N_MNIST_NOISE.py
	
As above, the run file requires `W_INI.wdata` file. All input prompts and output are the same as in [Section 1.1](#11-mnist).


### 3. LSM

#### 3.1 MNIST
To generate noisy spike counts for LSM model on MNIST, enter the following command:

	python LSM_RUN_MAIN_SIM_MNIST_NOISE.py
	
The run file requires a `W_INI.wdata` file. All input prompts and output are the same as in [Section 1.1](#11-mnist), except for input prompts:
* `INTIAL_ABS_W_SCLR? [float]` : use the same value used for training the model.

#### 3.2 N-MNIST
To generate noisy spike counts for LSM model on N-MNIST, enter the following command:

	python LSM_RUN_MAIN_SIM_N_MNIST_NOISE.py
	
All input prompts and output are the same as in [Section 1.1](#11-mnist).


### 4. LSM+AP_STDP

#### 4.1 MNIST
To generate noisy spike counts for LSM+AP_STDP model on MNIST, enter the following command:

	python LSM_APSTDP_STDP_RUN_MAIN_SIM_MNIST_NOISE.py
	
All input prompts and output are the same as in [Section 1.1](#11-mnist).
	
#### 4.2 N-MNIST
To generate noisy spike counts for LSM+AP-STDP model on N-MNIST, enter the following command:

	python LSM_APSTDP_RUN_MAIN_SIM_N_MNIST_NOISE.py
	
All input prompts and output are the same as in [Section 1.1](#11-mnist).


## Computating SP and AP metrics

After generating the noisy spike counts, to compute the SP and AP metrics for each trained model enter the following command:

	python compute_SP_AP_kernel_quality_measures.py
	
The run file will prompt for inputs:
* `VERSION? [int]` : enter the version number of the trained model.
* `DATASET_MODEL_WAS_TRAINED_ON? [TYPE M FOR MNIST/ N FOR NMNIST]` : enter the dataset the model was trained on.
The run file will print out the SP and AP metrics.
