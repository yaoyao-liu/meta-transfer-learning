# Meta-Transfer Learning PyTorch
[![LICENSE](https://img.shields.io/github/license/y2l/meta-transfer-learning-tensorflow.svg)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.0-orange.svg)](https://pytorch.org/)

#### Summary

* [Installation](#installation)
* [Project Architecture](#project-architecture)
* [Running Experiments](#running-experiments)
* [Performance](#performance)


## Installation

In order to run this repository, we advise you to install python 3.5 and PyTorch 0.4.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:

```bash
conda create --name mtl-pytorch python=3.5
conda activate mtl-pytorch
conda install pytorch=0.4.0 
conda install torchvision -c pytorch
```

Install other requirements:
```bash
pip install tqdm tensorboardX
```

Clone this repository:

```bash
git clone https://github.com/y2l/meta-transfer-learning-pytorch.git 
cd meta-transfer-learning-pytorch
```

## Project Architecture

```
.
├── data_generator              
|   ├── dataset_loader.py       # data loader for all datasets
|   └── meta_data_generator.py  # samplers for meta train
├── models                      
|   ├── mtl.py                  # meta-transfer class
|   ├── resnet_mtl.py           # resnet class
|   └── conv2d_mtl.py           # meta-transfer convolution class
├── trainer                     
|   ├── pre.py                  # pre-train trainer class
|   └── meta.py                 # meta-train trainer class
├── utils                       
|   ├── gpu_tools.py            # GPU tool functions
|   └── misc.py                 # miscellaneous tool functions
├── main.py                     # the python file with main function and parameter settings
├── run_pre.py                  # the script to run pre-train phase
└── run_meta.py                 # the script to run meta-train and meta-test phases
```

## Running Experiments

Run pretrain phase:
```bash
python run_pre.py
```
Run meta-train and meta-test phase:
```bash
python run_meta.py
```

### Hyperparameters and Options
Hyperparameters and options in `main.py`.

- `model_type` The network architecture
- `dataset` Meta dataset
- `phase` pre-train, meta-train or meta-eval
- `seed` Manual seed for PyTorch, "0" means using random seed
- `gpu` GPU id
- `dataset_dir` Directory for the images
- `max_epoch` Epoch number for meta-train phase
- `num_batch` The number for different tasks used for meta-train
- `shot` Shot number, how many samples for one class in a task
- `way` Way number, how many classes in a task
- `train_query` The number of training samples for each class in a task 
- `val_query` The number of test samples for each class in a task
- `meta_lr1` Learning rate for SS weights
- `meta_lr2` Learning rate for FC weights
- `base_lr` Learning rate for the inner loop
- `update_step` The number of updates for the inner loop
- `step_size` The number of epochs to reduce the meta learning rates
- `gamma` Gamma for the meta-train learning rate decay
- `init_weights` The pretained weights for meta-train phase
- `eval_weights` The meta-trained weights for meta-eval phase
- `meta_label` Additional label for meta-train
- `pre_max_epoch` Epoch number for pre-train pahse
- `pre_batch_size` Batch size for pre-train pahse
- `pre_lr` Learning rate for pre-train pahse
- `pre_gamma` Gamma for the preteain learning rate decay
- `pre_step_size` The number of epochs to reduce the pre-train learning rate
- `pre_custom_weight_decay` Weight decay for the optimizer during pre-train

## Performance

|          (%)           | 𝑚𝑖𝑛𝑖 1-shot  | 𝑚𝑖𝑛𝑖 5-shot  | FC100 1-shot | FC100 5-shot | 𝒕𝒊𝒆𝒓𝒆𝒅 1-shot | 𝒕𝒊𝒆𝒓𝒆𝒅 5-shot |
| ---------------------- | ------------ | ------------ | ------------ | ------------ |------------ | ------------ |
| `MTL Paper`            | `60.2 ± 1.8` | `74.3 ± 0.9` | `43.6 ± 1.8` | `55.4 ± 0.9` | <center>—</center> | <center>—</center> |
| `This Repo`            | `62.1 ± 1.8` | `78.5 ± 0.9` | `45.3 ± 1.8` | `59.7 ± 1.0` | `67.8 ± 1.8` | `83.0 ± 1.0` |

