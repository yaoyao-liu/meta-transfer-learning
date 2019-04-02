# Meta-Transfer Learning Tensorflow
[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)

This repo is the TensorFlow implementation of CVPR 2019 Paper [Meta-Transfer Learning for Few-Shot Learning](https://arxiv.org/pdf/1812.02391.pdf) by [Qianru Sun](https://www.comp.nus.edu.sg/~sunqr/), [Yaoyao Liu](https://yaoyao-liu.com), [Tat-Seng Chua](https://www.chuatatseng.com/) and [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/).

<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/meta-transfer-learning-tensorflow/master/docs/framework.png" width="700"/>
</p>

#### Summary:

* [Introduction](#introduction)
* [Installation](#installation)
* [Dataset](#Dataset)
* [Usage](#usage)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)


## Introduction

Meta-learning has been proposed as a framework to address the challenging few-shot learning setting. The key idea is to leverage a large number of similar few-shot tasks in order to learn how to adapt a base-learner to a new task for which only a few labeled samples are available. As deep neural networks (DNNs) tend to overfit using a few samples only, meta-learning typically uses shallow neural networks (SNNs), thus limiting its effectiveness. In this paper we propose a novel few-shot learning method called **meta-transfer learning (MTL)** which learns to adapt a **deep NN** for **few shot learning tasks**. Specifically, meta refers to training multiple tasks, and transfer is achieved by learning scaling and shifting functions of DNN weights for each task. In addition, we introduce the **hard task (HT) meta-batch** scheme as an effective learning curriculum for MTL. We conduct experiments using (5-class, 1-shot) and (5-class, 5-shot) recognition tasks on two challenging few-shot learning benchmarks: miniImageNet and Fewshot-CIFAR100. Extensive comparisons to related works validate that our **meta-transfer learning** approach trained with the proposed **HT meta-batch** scheme achieves top performance. An ablation study also shows that both components contribute to fast convergence and high accuracy.

## Installation

In order to run this repo, we advise you to install python 2.7 and TensorFlow 1.3.0 with Anaconda.

You may download Anaconda and read the installation instrucation on their offical website:
[https://www.anaconda.com/download/](https://www.anaconda.com/download/)

Create a new environment and install tensorflow on it:

```Bash
conda create --name tensorflow_1.3.0_gpu python=2.7
source activate tensorflow_1.3.0_gpu
pip install --ignore-installed --upgrade https://mirrors.tuna.tsinghua.edu.cn/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl
```

Clone the repo:

```Bash
git clone https://github.com/y2l/meta-transfer-learning-tensorflow.git 
cd meta-transfer-learning-tensorflow-master
```

Install the requirements:
```
pip install scipy
pip install tqdm
pip install opencv-python
```

## Dataset

### miniImageNet

Mini-ImageNet dataset was proposed by [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) for few-shot learning evaluation. Its complexity is high due to the use of ImageNet images but requires fewer resources and infrastructure than running on the full [ImageNet dataset](https://arxiv.org/pdf/1409.0575.pdf). In total, there are 100 classes with 600 samples of 84×84 color images per class. These 100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test.

To generate this dataset, your may use the repo [miniImageNet Tools](https://github.com/y2l/mini-imagenet-tools).

### FC100

For Few-shot CIFAR100 dataset, we will release the code to generate this dataset soon. You may also generate it yourself with the splits provided by [TADAM](https://arxiv.org/pdf/1805.10123.pdf).

## Uasge

To run the experiment:
```bash
python run_experiment.py
```
You may edit the `run_experiment.py` file to change the hyperparameters and default settings.

## Citation

Please cite our paper if it is helpful to your work:

```
@inproceedings{Sun2019MTL,
  title={Meta-Transfer Learning for Few-Shot Learning},
  author={Sun, Qianru and Liu, Yaoyao and Chua, Tat-Seng and Schiele, Bernt},
  booktitle={CVPR},
  year={2019}
}
```

## Acknowledgements

This repo use the source code from the following repos:

[MAML](https://github.com/cbfinn/maml)

[Optimization as a Model for Few-Shot Learning](https://github.com/gitabcworld/FewShotLearning)
