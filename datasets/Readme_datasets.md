# Setting up datasets

This file contains instructions for downloading/setting up dataset directories.


## MNIST

The MNIST dataset is included as `mnist.npz` file in `<Dir>/<Project Name>/datasets/`.


## Fashion-MNIST

The Fashion-MNIST dataset can be downloaded at: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)  
The file should be named as `fmnist.ds` and added to `<Dir>/<Project Name>/datasets/`. 


## N-MNIST

The N-MNIST dataset can be downloaded at: [https://www.garrickorchard.com/datasets/n-mnist](https://www.garrickorchard.com/datasets/n-mnist)  
To set up N-MNIST for training distribute N-MNIST files in the appropriate sub-directories as follows:
* `/nmnist/test_all/` : directory for all test files. Test files starting from `00001.bin` to `10000.bin` should go here. File `NMNIST_test_all_file_label_list` already exists in this directory and is used by training code to access the test files.  
* `/nmnist/valid_all/` : directory for all validation files. Taken from the training files, validation files starting from `50001.bin` to `60000.bin` should go here. File `NMNIST_valid_all_file_label_list` already exists in this directory and is used by training code to access the validation files.  
* `/nmnist/train_all/` : directory for all training files. Training files starting from `00001.bin` to `50000.bin` should go here. File `NMNIST_train_all_file_label_list` already exists in this directory and is used by training code to access the training files.  
