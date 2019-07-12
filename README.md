# Meta-Transfer Learning TensorFlow
[![LICENSE](https://img.shields.io/github/license/y2l/meta-transfer-learning-tensorflow.svg?style=flat-square)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-2.7%20%7C%203.5-blue.svg?style=flat-square)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-1.3.0-orange.svg?style=flat-square)](https://www.tensorflow.org/)
[![CodeFactor](https://www.codefactor.io/repository/github/y2l/meta-transfer-learning-tensorflow/badge?style=flat-square)](https://www.codefactor.io/repository/github/y2l/meta-transfer-learning-tensorflow)

This repository contains the TensorFlow implementation for [CVPR 2019](http://cvpr2019.thecvf.com/) Paper ["Meta-Transfer Learning for Few-Shot Learning"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf) by [Qianru Sun](https://www.comp.nus.edu.sg/~sunqr/)\*, [Yaoyao Liu](https://yaoyao-liu.com)\*, [Tat-Seng Chua](https://www.chuatatseng.com/) and [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/).

If you have any questions on this repository or the related paper, feel free to create an issue or send me an email. (Email address: yaoyaoliu@outlook.com)

#### Summary

* [Introduction](#introduction)
* [Installation](#installation)
* [Datasets](#datasets)
* [Repo Architecture](#repo-architecture)
* [Running Experiments](#running-experiments)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)


## Introduction

Meta-learning has been proposed as a framework to address the challenging few-shot learning setting. The key idea is to leverage a large number of similar few-shot tasks in order to learn how to adapt a base-learner to a new task for which only a few labeled samples are available. As deep neural networks (DNNs) tend to overfit using a few samples only, meta-learning typically uses shallow neural networks (SNNs), thus limiting its effectiveness. In this paper we propose a novel few-shot learning method called ***meta-transfer learning (MTL)*** which learns to adapt a ***deep NN*** for ***few shot learning tasks***. Specifically, meta refers to training multiple tasks, and transfer is achieved by learning scaling and shifting functions of DNN weights for each task. We conduct experiments using (5-class, 1-shot) and (5-class, 5-shot) recognition tasks on two challenging few-shot learning benchmarks: 𝑚𝑖𝑛𝑖ImageNet and Fewshot-CIFAR100. 

<p align="center">
    <img src="https://meta-transfer-learning.yaoyao-liu.com/images/ss.png" width="400"/>
</p>

> Figure: Meta-Transfer Learning. (a) Parameter-level fine-tuning (FT) is a conventional meta-training operation, e.g. in MAML. Its update works for all neuron parameters, 𝑊 and 𝑏. (b) Our neuron-level scaling and shifting (SS) operations in meta-transfer learning. They reduce the number of learning parameters and avoid overfitting problems. In addition, they keep large-scale trained parameters (in yellow) frozen, preventing “catastrophic forgetting”.

## Installation

In order to run this repository, we advise you to install python 2.7 or 3.5 and TensorFlow 1.3.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install tensorflow on it:

```bash
conda create --name mtl python=2.7
conda activate mtl
conda install tensorflow-gpu=1.3.0
```

Install other requirements:
```bash
pip install scipy tqdm opencv-python pillow matplotlib
```

Clone this repository:

```bash
git clone https://github.com/y2l/meta-transfer-learning-tensorflow.git 
cd meta-transfer-learning-tensorflow
```

## Datasets

### 𝒎𝒊𝒏𝒊ImageNet

The 𝑚𝑖𝑛𝑖ImageNet dataset was proposed by [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) for few-shot learning evaluation. Its complexity is high due to the use of ImageNet images but requires fewer resources and infrastructure than running on the full [ImageNet dataset](https://arxiv.org/pdf/1409.0575.pdf). In total, there are 100 classes with 600 samples of 84×84 color images per class. These 100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test.

To generate this dataset from ImageNet, you may use the repository [𝑚𝑖𝑛𝑖ImageNet tools](https://github.com/y2l/mini-imagenet-tools). You may also directly download processed images. [\[Download Page\]](https://meta-transfer-learning.yaoyao-liu.com/download/)

### Fewshot-CIFAR100

Fewshot-CIFAR100 (FC100) is based on the popular object classification dataset CIFAR100. The splits were
proposed by [TADAM](https://arxiv.org/pdf/1805.10123.pdf). It offers a more challenging scenario with lower image resolution and more challenging meta-training/test splits that are separated according to object super-classes. It contains 100 object classes and each class has 600 samples of 32 × 32 color images. The 100 classes belong to 20 super-classes. Meta-training data are from 60 classes belonging to 12 super-classes. Meta-validation and meta-test sets contain 20 classes belonging to 4 super-classes, respectively.

You may directly download processed images. [\[Download Page\]](https://meta-transfer-learning.yaoyao-liu.com/download/)

### 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet

The [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet](https://arxiv.org/pdf/1803.00676.pdf) dataset is a larger subset of ILSVRC-12 with 608 classes (779,165 images) grouped into 34 higher-level nodes in the ImageNet human-curated hierarchy. 

To generate this dataset from ImageNet, you may use the repository 𝑡𝑖𝑒𝑟𝑒𝑑ImageNet dataset: [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet tools](https://github.com/y2l/tiered-imagenet-tools). You may also directly download processed images. [\[Download Page\]](https://meta-transfer-learning.yaoyao-liu.com/download/)

(P.S. We apply data augmentation strategy like horizontal flipping for pre-train phase. The augmented images are not included in the datasets provided.)

## Repo Architecture

```
.
├── data_generator              # dataset generator 
|   ├── pre_data_generator.py   # data genertor for pre-train phase
|   └── meta_data_generator.py  # data genertor for meta-train phase
├── models                      # tensorflow model files 
|   ├── models.py               # basic model class
|   ├── pre_model.py.py         # pre-train model class
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


## Todo

- [ ] 𝐇𝐚𝐫𝐝 𝐭𝐚𝐬𝐤 𝐦𝐞𝐭𝐚-𝐛𝐚𝐭𝐜𝐡.
  The implementation of hard task meta-batch is not included in the published code. I still need time to rewrite the hard task meta batch code for the current framework.
- [ ] 𝐌𝐨𝐫𝐞 𝐧𝐞𝐭𝐰𝐨𝐫𝐤 𝐚𝐫𝐜𝐡𝐢𝐭𝐞𝐜𝐭𝐮𝐫𝐞𝐬.
  We will add new backbones to the framework like ResNet18 and ResNet34.
- [ ] 𝐏𝐲𝐓𝐨𝐫𝐜𝐡 𝐯𝐞𝐫𝐬𝐢𝐨𝐧.
  We will release the code for MTL on pytorch. It may takes several months to be completed.

## Citation

Please cite our paper if it is helpful to your work:

```
@inproceedings{sun2019mtl,
  title={Meta-Transfer Learning for Few-Shot Learning},
  author={Qianru Sun and Yaoyao Liu and Tat{-}Seng Chua and Bernt Schiele},
  booktitle={CVPR},
  year={2019}
}
```

## Acknowledgements

Our implementation uses the source code from the following repositories:

[Model-Agnostic Meta-Learning](https://github.com/cbfinn/maml)

[Optimization as a Model for Few-Shot Learning](https://github.com/gitabcworld/FewShotLearning)
