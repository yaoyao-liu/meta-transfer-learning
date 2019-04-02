# Meta-Transfer Learning Tensorflow
[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)

This repo is the TensorFlow implementation of CVPR 2019 Paper [Meta-Transfer Learning for Few-Shot Learning](https://arxiv.org/pdf/1812.02391.pdf) by [Qianru Sun](https://www.comp.nus.edu.sg/~sunqr/), [Yaoyao Liu](https://yaoyao-liu.com), [Tat-Seng Chua](https://www.chuatatseng.com/) and [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/).

#### Summary:

* [Introduction](#introduction)
* [Installation](#installation)
* [Dataset](#Dataset)
* [Usage](#usage)
* [Citation](#citation)

## Introduction

Meta-learning has been proposed as a framework to address the challenging few-shot learning setting. The key idea is to leverage a large number of similar few-shot tasks in order to learn how to adapt a base-learner to a new task for which only a few labeled samples are available. As deep neural networks (DNNs) tend to overfit using a few samples only, meta-learning typically uses shallow neural networks (SNNs), thus limiting its effectiveness. In this paper we propose a novel few-shot learning method called **meta-transfer learning (MTL)** which learns to adapt a **deep NN** for **few shot learning tasks**. Specifically, meta refers to training multiple tasks, and transfer is achieved by learning scaling and shifting functions of DNN weights for each task. In addition, we introduce the **hard task (HT) meta-batch** scheme as an effective learning curriculum for MTL. We conduct experiments using (5-class, 1-shot) and (5-class, 5-shot) recognition tasks on two challenging few-shot learning benchmarks: miniImageNet and Fewshot-CIFAR100. Extensive comparisons to related works validate that our **meta-transfer learning** approach trained with the proposed **HT meta-batch** scheme achieves top performance. An ablation study also shows that both components contribute to fast convergence and high accuracy.

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
