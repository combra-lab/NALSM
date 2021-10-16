# NALSM: Neuron-Astrocyte Liquid State Machine

This package is a Tensorflow implementation of the **N**euron-**A**strocyte **L**iquid **S**tate **M**achine (**NALSM**) that introduces astrocyte-modulated STDP to the Liquid State Machine learning framework for improved accuracy performance and minimal tuning. 

The paper has been accepted at NeurIPS 2021. The arXiv preprint is available **here**

## Software Installation

* Python 3.6.9
* Tensorflow 2.1 (with CUDA 11.2 using tensorflow.compat.v1)
* Numpy
* Multiprocessing

## Usage

This code performs the following functions:
1. [Generate the 3D network](#1-generate-3d-network)
2. [Train NALSM](#2-train-nalsm)  
3. [Evaluate trained model accuracy](#3-evaluate-trained-model-accuracy)
4. [Evaluate trained model branching factor](#4-evaluate-model-branching-factor)
5. [Evaluate model kernel quality](#5-evaluate-model-kernel-quality)

Instructions for obtaining/setting up datasets can be accessed **here**.

Overview of all files can be accessed **here**.

### 1. Generate 3D Network

To generate the 3D network, enter the following command:
	
	python generate_spatial_network.py

This will prompt for following inputs:
* `WHICH_DATASET_TO_GENERATE_NETWORK_FOR? [TYPE M FOR MNIST/ N FOR NMNIST]` : enter `M` to make a network with an input layer sized for MNIST/Fashion-MNIST or `N` for N-MNIST.
* `NETWORK_NUMBER_TO_CREATE? [int]` : enter an integer to label the network.
* `SIZE_OF_LIQUID_DIMENSION_1? [int]` : enter an integer representing the number of neurons to be in dimension 1 of liquid.
* `SIZE_OF_LIQUID_DIMENSION_2? [int]` : enter an integer representing the number of neurons to be in dimension 2 of liquid.
* `SIZE_OF_LIQUID_DIMENSION_3? [int]` : enter an integer representing the number of neurons to be in dimension 3 of liquid.

The run file will generate the network and associated log file containing data about the liquid (i.e. connection densities) in sub-directory `<Dir>/<Project Name>/networks/`.

### 2. Train NALSM

#### 2.1 MNIST
To train NALSM model on MNIST, enter the following command: 
	
	python NALSM_RUN_MAIN_SIM_MNIST.py

This will prompt for the following inputs:
* `GPU?` : enter an integer specifying the gpu to use for training.
* `VERSION? [int]` : enter an integer to label the training simulation.
* `NET_NUM_VAR? [int]` : enter the number of the network created in Section 1.
* `BATCH_SIZE? [int]` : specify the number of samples to train at same time (batch), for liquids with 1000 neurons, batch size of 250 will work on a 12gb gpu. For larger liquids(8000), smaller batch sizes of 50 should work.
* `BATCHS_PER_BLOCK? [int]` : specify number of batchs to keep in memory for training output layer, we found 2500 samples works well in terms of speed and memory (so for batch size of 250, this should be set to 10 (10 x 250 = 2500), for batch size 50 set this to 50 (50 x 50 = 2500).
* `ASTRO_W_SCALING? [float]` : specify the astrocyte weight detailed in equation 7 of paper. We used 0.015 for all 1000 neuron liquids, and 0.0075 for 8000 neuron liquids. Generally accuracy peaks with a value around 0.01 (See Appendix).

This will generate all output in sub-directory `<Dir>/<Project Name>/train_data/ver_XX/` where `XX` is `VERSION` number.

#### 2.2 N-MNIST

To train NALSM model on N-MNIST, enter the following command: 

	python NALSM_RUN_MAIN_SIM_N_MNIST.py

All input prompts and output are the same as described above for run file `NALSM_RUN_MAIN_SIM_MNIST.py`.

#### 2.3 Fashion-MNIST

To train NALSM model on Fashion-MNIST, enter the following command: 

	python NALSM_RUN_MAIN_SIM_F_MNIST.py

All input prompts and output are the same as described above for run file `NALSM_RUN_MAIN_SIM_MNIST.py`.


Instructions for training other benchmarked LSM models can be accessed **here**

### 3. Evaluate Trained Model Accuracy

To get accuracy of a trained model, enter the following command:

   python get_test_accuracy.py

The run file will prompt for following inputs:
* `VERSION? [int]` : enter the version number of the trained model

This will find the epoch with maximum validation accuracy and return the test accuracy for that epoch.

### 4. Evaluate Model Branching Factor

To compute the branching factor of a trained model, enter the following command:
	
	python compute_branching_factor.py

The run file will prompt for following inputs:
* `VERSION? [int]` : enter the version number of the trained model.

The trained model directory must have atleast one `.spikes` file, which contains millisecond spike data of each neuron for 20 arbitrarily selected input samples in a batch.
The run file will generate a `.bf` file with same name as the `.spikes` file. 

To read the generated `.bf` file, enter the following command:

	python get_branching_factor.py

The run file will prompt for following inputs:
* `VERSION? [int]` : enter the version number of the trained model.

The run file will print the average branching factor over the 20 samples.

### 5. Evaluate Model Kernel Quality

Model liquid kernel quality was calculated from the linear speration (SP) and generalization (AP) metrics for MNIST and N-MNIST datasets. To compute SP and AP metrics, first noisy spike counts must be generated for the AP metric, as follows. 


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

To generate noisy spike counts for NALSM model on N-MNIST, enter the following command:

	python NALSM_RUN_MAIN_SIM_N_MNIST_NOISE.py

As above, the run file requires 'W_INI.wdata' file. All input prompts and output are the same as described above for run file `NALSM_RUN_MAIN_SIM_MNIST_NOISE.py`.

After generating the noisy spike counts, to compute the SP and AP metrics for each trained model enter the following command:

	python compute_SP_AP_kernel_quality_measures.py
	
The run file will prompt for inputs:
* `VERSION? [int]` : enter the version number of the trained model.
* `DATASET_MODEL_WAS_TRAINED_ON? [TYPE M FOR MNIST/ N FOR NMNIST]` : enter dataset the model was trained on.
The run file will print out the SP and AP metrics.


Instructions for evaluating kernel quality for other benchmarked LSM models can be accessed **here**


# Citation

Vladimir A. Ivanov and Konstantinos P. Michmizos. "Increasing Liquid State Machine Performance with Edge-of-Chaos Dynamics Organized by Astrocyte-modulated Plasticity." *35th Conference on Neural Information Processing Systems* (NeurIPS 2021).

	@inproceedings{ivanov_2021,
	author = {Ivanov, Vladimir A. and Michmizos, Konstantinos P.},
	title = {Increasing Liquid State Machine Performance with Edge-of-Chaos Dynamics Organized by Astrocyte-modulated Plasticity},
	year = {2021},
	pages={1--10},
	booktitle = {35th Conference on Neural Information Processing Systems (NeurIPS 2021)}
	}
