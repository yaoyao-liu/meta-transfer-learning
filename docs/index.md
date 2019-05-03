## Abstract

Meta-learning has been proposed as a framework to address the challenging few-shot learning setting. The key idea is to leverage a large number of similar few-shot tasks in order to learn how to adapt a base-learner to a new task for which only a few labeled samples are available. As deep neural networks (DNNs) tend to overfit using a few samples only, meta-learning typically uses shallow neural networks (SNNs), thus limiting its effectiveness. In this paper we propose a novel few-shot learning method called ***meta-transfer learning (MTL)*** which learns to adapt a ***deep NN*** for ***few shot learning tasks***. Specifically, meta refers to training multiple tasks, and transfer is achieved by learning scaling and shifting functions of DNN weights for each task. In addition, we introduce the ***hard task (HT) meta-batch*** scheme as an effective learning curriculum for MTL. We conduct experiments using (5-class, 1-shot) and (5-class, 5-shot) recognition tasks on two challenging few-shot learning benchmarks: miniImageNet and Fewshot-CIFAR100. Extensive comparisons to related works validate that our ***meta-transfer learning*** approach trained with the proposed ***HT meta-batch*** scheme achieves top performance. An ablation study also shows that both components contribute to fast convergence and high accuracy.

### Contributions
- A novel MTL method that learns to transfer large-scale pre-trained DNN weights for solving few-shot learning tasks. 
- A novel HT meta-batch learning strategy that forces meta-transfer to “grow faster and stronger through hardship”.
- Extensive experiments on *mini*ImageNet and Fewshot-CIFAR100, and achieving the state-of-the-art performance.

## Pipeline
<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/meta-transfer-learning-tensorflow/master/docs/framework.png" width="750"/>
</p>

The pipeline of our proposed few-shot learning method, including three phases: (a) DNN training on large-scale data, *i.e.* using all training datapoints; (b) Meta-transfer learning (MTL) that learns the parameters of scaling and shifting (SS), based on the pre-trained feature extractor. Learning is scheduled by the proposed HT meta-batch; and (c) meta-test is done for an unseen task which consists of a base-learner Fine-Tuning stage and a final evaluation stage. Input data are along with arrows. Modules with names in bold get updated at corresponding phases. Specifically, SS parameters are learned by meta-training but fixed during meta-test. Base-learner parameters are optimized for every task.

## Meta-Transfer Learning
<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/meta-transfer-learning-tensorflow/master/docs/ss.png" width="450"/>
</p>

(a) Parameter-level fine-tuning (FT) is a conventional meta-training operation, e.g. in MAML \[1\]. Its update works for all neuron parameters, W and b. (b) Our neuron-level scaling and shifting (SS) operations in meta-transfer learning. They reduce the number of learning parameters and avoid overfitting problems. In addition, they keep large-scale trained parameters (in yellow) frozen, preventing “catastrophic forgetting”.

## Hard Task Meta Batch

In our meta-training pipeline, we intentionally pick up failure cases in each task and re-compose their data to be harder tasks for adverse re-training. We aim to force our meta-learner to “grow up through hardness”.
<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/meta-transfer-learning-tensorflow/master/docs/ht.png" width="450"/>
</p>

The figure shows the performance gap between with and without hard task meta-batch in terms of accuracy and converging speed. (a)(b) 1-shot and 5-shot on *mini*ImageNet; 
(c)(d)(e) 1-shot, 5-shot and 10-shot on Fewshot-CIFAR100. 
<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/meta-transfer-learning-tensorflow/master/docs/acc_plot.png" width="900"/>
</p>

## Experiments

### *mini*ImageNet
<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/meta-transfer-learning-tensorflow/master/docs/mini_acc.png" width="620"/>
</p>

### Fewshot-CIFAR100 
<p align="center">
    <img src="https://raw.githubusercontent.com/y2l/meta-transfer-learning-tensorflow/master/docs/fc100_acc.png" width="700"/>
</p>

## Citation

Please cite our paper if it is helpful to your work:

```
@inproceedings{Sun2019MTL,
  title={Meta-Transfer Learning for Few-Shot Learning},
  author={Qianru Sun and Yaoyao Liu and Tat{-}Seng Chua and Bernt Schiele},
  booktitle={CVPR},
  year={2019}
}
```

### References

\[1\] Chelsea et al. “Model-agnostic meta-learning for fast adaptation of deep networks.” In *ICML 2017*;
<br>
\[2\] Oreshkin et al. “TADAM: Task dependent adaptive metric for improved few-shot learning.” In *NeurIPS 2018*.
