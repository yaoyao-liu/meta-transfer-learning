# Meta-Transfer Learning
[![LICENSE](https://img.shields.io/github/license/y2l/meta-transfer-learning-tensorflow.svg)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-2.7%20%7C%203.5-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-1.3.0-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.0-%237732a8)](https://pytorch.org/)
[![CodeFactor](https://www.codefactor.io/repository/github/y2l/meta-transfer-learning/badge)](https://www.codefactor.io/repository/github/y2l/meta-transfer-learning)

This repository contains the TensorFlow and PyTorch implementations for [CVPR 2019](http://cvpr2019.thecvf.com/) Paper ["Meta-Transfer Learning for Few-Shot Learning"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf) by [Qianru Sun](https://sites.google.com/view/qianrusun/home)\*, [Yaoyao Liu](https://yyliu.net)\*, [Tat-Seng Chua](https://www.chuatatseng.com/) and [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/) (\*equal contribution).

If you have any questions on this repository or the related paper, feel free to create an issue or send me an email. 

Email address: yaoyaoliu@outlook.com

#### Summary

* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [Datasets](#datasets)
* [Performance](#performance)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)


## Introduction

Meta-learning has been proposed as a framework to address the challenging few-shot learning setting. The key idea is to leverage a large number of similar few-shot tasks in order to learn how to adapt a base-learner to a new task for which only a few labeled samples are available. As deep neural networks (DNNs) tend to overfit using a few samples only, meta-learning typically uses shallow neural networks (SNNs), thus limiting its effectiveness. In this paper we propose a novel few-shot learning method called ***meta-transfer learning (MTL)*** which learns to adapt a ***deep NN*** for ***few shot learning tasks***. Specifically, meta refers to training multiple tasks, and transfer is achieved by learning scaling and shifting functions of DNN weights for each task. We conduct experiments using (5-class, 1-shot) and (5-class, 5-shot) recognition tasks on two challenging few-shot learning benchmarks: 𝑚𝑖𝑛𝑖ImageNet and Fewshot-CIFAR100. 

<p align="center">
    <img src="https://meta-transfer-learning.yaoyao-liu.com/images/ss.png" width="400"/>
</p>

> Figure: Meta-Transfer Learning. (a) Parameter-level fine-tuning (FT) is a conventional meta-training operation, e.g. in MAML. Its update works for all neuron parameters, 𝑊 and 𝑏. (b) Our neuron-level scaling and shifting (SS) operations in meta-transfer learning. They reduce the number of learning parameters and avoid overfitting problems. In addition, they keep large-scale trained parameters (in yellow) frozen, preventing “catastrophic forgetting”.

## Getting Started

Please see the `README.md` files in the corresponding folders:

* TensorFlow: [\[Docs\]](https://github.com/y2l/meta-transfer-learning/blob/master/tensorflow/README.md)
* PyTorch: [\[Docs\]](https://github.com/y2l/meta-transfer-learning/blob/master/pytorch/README.md)

## Datasets

Directly download processed images: [\[Download Page\]](https://meta-transfer-learning.yyliu.net/download/)

### 𝒎𝒊𝒏𝒊ImageNet

The 𝑚𝑖𝑛𝑖ImageNet dataset was proposed by [Vinyals et al.](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf) for few-shot learning evaluation. Its complexity is high due to the use of ImageNet images but requires fewer resources and infrastructure than running on the full [ImageNet dataset](https://arxiv.org/pdf/1409.0575.pdf). In total, there are 100 classes with 600 samples of 84×84 color images per class. These 100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test.To generate this dataset from ImageNet, you may use the repository [𝑚𝑖𝑛𝑖ImageNet tools](https://github.com/y2l/mini-imagenet-tools).

### Fewshot-CIFAR100

Fewshot-CIFAR100 (FC100) is based on the popular object classification dataset CIFAR100. The splits were
proposed by [TADAM](https://arxiv.org/pdf/1805.10123.pdf). It offers a more challenging scenario with lower image resolution and more challenging meta-training/test splits that are separated according to object super-classes. It contains 100 object classes and each class has 600 samples of 32 × 32 color images. The 100 classes belong to 20 super-classes. Meta-training data are from 60 classes belonging to 12 super-classes. Meta-validation and meta-test sets contain 20 classes belonging to 4 super-classes, respectively.

### 𝒕𝒊𝒆𝒓𝒆𝒅ImageNet

The [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet](https://arxiv.org/pdf/1803.00676.pdf) dataset is a larger subset of ILSVRC-12 with 608 classes (779,165 images) grouped into 34 higher-level nodes in the ImageNet human-curated hierarchy. To generate this dataset from ImageNet, you may use the repository 𝑡𝑖𝑒𝑟𝑒𝑑ImageNet dataset: [𝑡𝑖𝑒𝑟𝑒𝑑ImageNet tools](https://github.com/y2l/tiered-imagenet-tools). 


## Performance

|          (%)           | 𝑚𝑖𝑛𝑖 1-shot  | 𝑚𝑖𝑛𝑖 5-shot  | FC100 1-shot | FC100 5-shot | 𝒕𝒊𝒆𝒓𝒆𝒅 1-shot | 𝒕𝒊𝒆𝒓𝒆𝒅 5-shot |
| ---------------------- | ------------ | ------------ | ------------ | ------------ |------------ | ------------ |
| `MTL Paper`            | `60.2 ± 1.8` | `74.3 ± 0.9` | `43.6 ± 1.8` | `55.4 ± 0.9` | — | — |
| `TensorFlow`           | `60.8 ± 1.8` | `74.3 ± 0.9` | `44.3 ± 1.8` | `56.8 ± 1.0` | `65.6 ± 1.8` | `80.6 ± 0.9` |
| `PyTorch`              | `62.1 ± 1.8` | `78.5 ± 0.9` | `45.3 ± 1.8` | `59.7 ± 1.0` | `67.8 ± 1.8` | `83.0 ± 1.0` |

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

* [Model-Agnostic Meta-Learning](https://github.com/cbfinn/maml)

* [Optimization as a Model for Few-Shot Learning](https://github.com/gitabcworld/FewShotLearning)

* [FEAT](https://github.com/Sha-Lab/FEAT)

* [@icoz69](https://github.com/icoz69)

* [@CookieLau](https://github.com/CookieLau)
