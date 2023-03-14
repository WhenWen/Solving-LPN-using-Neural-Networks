# Solving LPN Using Neural Networks.

This repository is the official codebase for the paper Practically Solving LPN in High Noise Regimes Faster Using Neural Networks.

## Environment Setup

With [Anaconda | The World&#39;s Most Popular Data Science Platform](https://www.anaconda.com/) installed, one can setup the environment for reproducing the experiments by the following commands.

```bash
conda env create -f environment.yml
```

You can then activate this virtual environment by the following command.

```bash
conda activate LPN
```

## Setup Hyperparameters

To perform an experiment, one need to specify the hyperparameters in a config file. We provide some examples in directories `config` and `hyperconfig`. One can view `config/baseline` for a detailed explaination.

## Data Preparation

To prepare the data for an experiment, one should run one of the following commands,

```bash
python pyscript/data.py -c config/EXPERIMENT_NAME.config
python pyscript/data.py -c hyper_config/EXPERIMENT_NAME.config
```

Data will then be generated in directory `data`.

## Train Neural Networks to solve LPN problems

To train a neural network to solve LPN problems, one should run the following command.

```bash
python pyscript/lpn.py -c config/EXPERIMENT_NAME.config
```

## Perform Sample Complexity Search with GPU parallelization

To try binary search the minimal sample complexity for a hyperparameter to solve LPN problems, one should run the following command.

```bash
python pyscript/data.py -c hyper_config/EXPERIMENT_NAME.config
```

## Performing Gaussian Elimination based on the Predictions of the Neural Network

We also provide the code to perform Gaussian Elimination based on a trained neural network.

To do so, one need to first generate the pool of data where Gaussian Elimination source data is sampled, as well as the test data, by the neural network. One should run the following command.

```bash
python pyscript/network_generate.py --network_path PATH_TO_NETWORK --secret_path PATH_TO_SECRET --pool_data_size SIZE_OF_GAUSSIAN_POOL --test_data_size SIZE_OF_TESTING_NUMBER
```

One can then use the code in `cppscript/Gaussian` to decode the secret. One should first change the variables `SECRET_PATH` in line 56 of `LPN_Oracle_hack.cpp` and the variable `TEST_QUERY_PATH` in line 188 and `QUERY_PATH` in line 298 of `Prange_hack.cpp`. Then one can modify `Prange.h` accordingly, see the file for a detailed explaination for the hyperparameters. After setting the parameters, one should run the following command to compile

```bash
gcc -g Prange_hack.cpp LPN_Oracle_hack.cpp -lm -lstdc++ -o try -std=c++11 -fopenmp -O3
./try
```

## Acknowledgment

The authors would like to thank [Memphisd/LPN-decoded (github.com)](https://github.com/Memphisd/LPN-decoded) for providing the base code for Gaussian Elimination.

## Citation

If the code help you, please cite our paper:

Welcome to shoot any questions in Github Issues!
