## Introduction

Meta-learning has been proposed as a framework to address the challenging few-shot learning setting. The key idea is to leverage a large number of similar few-shot tasks in order to learn how to adapt a base-learner to a new task for which only a few labeled samples are available. As deep neural networks (DNNs) tend to overfit using a few samples only, meta-learning typically uses shallow neural networks (SNNs), thus limiting its effectiveness. In this paper we propose a novel few-shot learning method called **meta-transfer learning (MTL)** which learns to adapt a **deep NN** for **few shot learning tasks**. Specifically, meta refers to training multiple tasks, and transfer is achieved by learning scaling and shifting functions of DNN weights for each task. In addition, we introduce the **hard task (HT) meta-batch** scheme as an effective learning curriculum for MTL. We conduct experiments using (5-class, 1-shot) and (5-class, 5-shot) recognition tasks on two challenging few-shot learning benchmarks: miniImageNet and Fewshot-CIFAR100. Extensive comparisons to related works validate that our **meta-transfer learning** approach trained with the proposed **HT meta-batch** scheme achieves top performance. An ablation study also shows that both components contribute to fast convergence and high accuracy.

### Contributions:
- A novel MTL method that learns to transfer large-scale pre-trained DNN weights for solving few-shot learning tasks. 
- A novel HT meta-batch learning strategy that forces meta-transfer to “grow faster and stronger through hardship”.
- Extensive experiments on miniImageNet and FC100, and achieving the state-of-the-art performance.

## Pipeline
<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/meta-transfer-learning-tensorflow/master/docs/framework.png" width="700"/>
</p>

The pipeline of our proposed few-shot learning method, including three phases: (a) DNN training on large-scale data, i.e. using all training datapoints; (b) Meta-transfer learning (MTL) that learns the parameters of Scaling and Shifting (SS), based on the pre-trained feature extractor. Learning is scheduled by the proposed HT meta-batch; and (c) meta-test is done for an unseen task which consists of a base-learner Fine-Tuning stage and a final evaluation stage. Input data are along with arrows. Modules with names in bold get updated at corresponding phases. Specifically, SS parameters are learned by meta-training but fixed during meta-test. Base-learner parameters are optimized for every task.

## Performance

### miniImageNet
<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/meta-transfer-learning-tensorflow/master/docs/mini.png" width="700"/>
</p>

### FC100
<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/meta-transfer-learning-tensorflow/master/docs/fc100.png" width="700"/>
</p>
