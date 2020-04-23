# Meta-Transfer Learning TensorFlow
[![Python](https://img.shields.io/badge/python-2.7%20%7C%203.5-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-1.3.0-orange.svg)](https://github.com/y2l/meta-transfer-learning/tree/master/tensorflow)

#### Summary

* [Installation](#installation)
* [Project Architecture](#project-architecture)
* [Running Experiments](#running-experiments)


## Installation

In order to run this repository, we advise you to install python 2.7 or 3.5 and TensorFlow 1.3.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install tensorflow on it:

```bash
conda create --name mtl-tf python=2.7
conda activate mtl-tf
conda install tensorflow-gpu=1.3.0
```

Install other requirements:
```bash
pip install scipy tqdm opencv-python pillow matplotlib miniimagenettools
```

Clone this repository:

```bash
git clone https://github.com/yaoyao-liu/meta-transfer-learning.git 
cd meta-transfer-learning/tensorflow
```

## Project Architecture

```
.
├── data_generator              # dataset generator 
|   ├── pre_data_generator.py   # data genertor for pre-train phase
|   └── meta_data_generator.py  # data genertor for meta-train phase
├── models                      # tensorflow model files 
|   ├── resnet12.py             # resnet12 class
|   ├── resnet18.py             # resnet18 class
|   ├── pre_model.py            # pre-train model class
|   └── meta_model.py           # meta-train model class
├── trainer                     # tensorflow trianer files  
|   ├── pre.py                  # pre-train trainer class
|   └── meta.py                 # meta-train trainer class
├── utils                       # a series of tools used in this repo
|   └── misc.py                 # miscellaneous tool functions
├── main.py                     # the python file with main function and parameter settings
└── run_experiment.py           # the script to run the whole experiment
```

## Running Experiments

### Training from Scratch
Run pre-train phase:
```bash
python run_experiment.py PRE
```
Run meta-train and meta-test phase:
```bash
python run_experiment.py META
```

### Hyperparameters and Options
You may edit the `run_experiment.py` file to change the hyperparameters and options. 

- `LOG_DIR` Name of the folder to save the log files
- `GPU_ID` GPU device id
- `NET_ARCH` Network backbone (resnet12 or resnet18)
- `PRE_TRA_LABEL` Additional label for pre-train model
- `PRE_TRA_ITER_MAX` Iteration number for the pre-train phase
- `PRE_TRA_DROP` Dropout keep rate for the pre-train phase
- `PRE_DROP_STEP` Iteration number for the pre-train learning rate reducing
- `PRE_LR` Pre-train learning rate
- `SHOT_NUM` Sample number for each class
- `WAY_NUM` Class number for the few-shot tasks
- `MAX_MAX_ITER` Iteration number for meta-train phase
- `META_BATCH_SIZE` Meta batch size 
- `PRE_ITER` Iteration number for the pre-train model used in the meta-train phase
- `UPDATE_NUM` Epoch number for the base learning
- `SAVE_STEP` Iteration number to save the meta model
- `META_LR` Meta learning rate
- `META_LR_MIN` Meta learning rate min value
- `LR_DROP_STEP` Iteration number for the meta learning rate reducing
- `BASE_LR` Base learning rate
- `PRE_TRA_DIR` Directory for the pre-train phase images
- `META_TRA_DIR` Directory for the meta-train images
- `META_VAL_DIR` Directory for the meta-validation images
- `META_TES_DIR` Directory for the meta-test images

The file `run_experiment.py` is just a script to generate commands for `main.py`. If you want to change other settings, please see the comments and descriptions in `main.py`.

### Using Downloaded Models
In the default setting, if you run `python run_experiment.py`, the pretrain process will be conducted before the meta-train phase starts. If you want to use the model pretrained by us, you may download the model by the following link. To run experiments with the downloaded model, please make sure you are using python 2.7.

Comparison of the original paper and the open-source code in terms of test set accuracy:

|          (%)           | 𝑚𝑖𝑛𝑖 1-shot  | 𝑚𝑖𝑛𝑖 5-shot  | FC100 1-shot | FC100 5-shot |
| ---------------------- | ------------ | ------------ | ------------ | ------------ |
| `MTL Paper`            | `60.2 ± 1.8` | `74.3 ± 0.9` | `43.6 ± 1.8` | `55.4 ± 0.9` |
| `This Repo`            | `60.8 ± 1.8` | `74.3 ± 0.9` | `44.3 ± 1.8` | `56.8 ± 1.0` |

Download models: [\[Google Drive\]](https://drive.google.com/drive/folders/1MzH2enwLKuzmODYAEATnyiP_602zrdrE?usp=sharing)

Move the downloaded npy files to `./logs/download_weights` (e.g. 𝑚𝑖𝑛𝑖ImageNet, 1-shot):
```bash
mkdir -p ./logs/download_weights
mv ~/downloads/mini-1shot/*.npy ./logs/download_weights
```

Run meta-train with downloaded model:
```bash
python run_experiment.py META_LOAD
```

Run meta-test with downloaded model:
```bash
python run_experiment.py TEST_LOAD
```

