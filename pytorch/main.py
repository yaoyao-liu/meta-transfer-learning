##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Main function for this repo. """
import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.meta import MetaTrainer
from trainer.pre import PreTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='ResNet', choices=['ResNet']) # The network architecture
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['miniImageNet', 'tieredImageNet', 'FC100']) # Dataset
    parser.add_argument('--phase', type=str, default='meta_train', choices=['pre_train', 'meta_train', 'meta_eval']) # Phase
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', default='1') # GPU id
    parser.add_argument('--dataset_dir', type=str, default='./data/mini/') # Dataset folder

    # Parameters for meta-train phase
    parser.add_argument('--max_epoch', type=int, default=100) # Epoch number for meta-train phase
    parser.add_argument('--num_batch', type=int, default=100) # The number for different tasks used for meta-train
    parser.add_argument('--shot', type=int, default=1) # Shot number, how many samples for one class in a task
    parser.add_argument('--way', type=int, default=5) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=15) # The number of training samples for each class in a task
    parser.add_argument('--val_query', type=int, default=15) # The number of test samples for each class in a task
    parser.add_argument('--meta_lr1', type=float, default=0.0001) # Learning rate for SS weights
    parser.add_argument('--meta_lr2', type=float, default=0.001) # Learning rate for FC weights
    parser.add_argument('--base_lr', type=float, default=0.01) # Learning rate for the inner loop
    parser.add_argument('--update_step', type=int, default=50) # The number of updates for the inner loop
    parser.add_argument('--step_size', type=int, default=10) # The number of epochs to reduce the meta learning rates
    parser.add_argument('--gamma', type=float, default=0.5) # Gamma for the meta-train learning rate decay
    parser.add_argument('--init_weights', type=str, default=None) # The pre-trained weights for meta-train phase
    parser.add_argument('--eval_weights', type=str, default=None) # The meta-trained weights for meta-eval phase
    parser.add_argument('--meta_label', type=str, default='exp1') # Additional label for meta-train

    # Parameters for pretain phase
    parser.add_argument('--pre_max_epoch', type=int, default=100) # Epoch number for pre-train phase
    parser.add_argument('--pre_batch_size', type=int, default=128) # Batch size for pre-train phase
    parser.add_argument('--pre_lr', type=float, default=0.1) # Learning rate for pre-train phase
    parser.add_argument('--pre_gamma', type=float, default=0.2) # Gamma for the pre-train learning rate decay
    parser.add_argument('--pre_step_size', type=int, default=30) # The number of epochs to reduce the pre-train learning rate
    parser.add_argument('--pre_custom_momentum', type=float, default=0.9) # Momentum for the optimizer during pre-train
    parser.add_argument('--pre_custom_weight_decay', type=float, default=0.0005) # Weight decay for the optimizer during pre-train

    # Set and print the parameters
    args = parser.parse_args()
    pprint(vars(args))

    # Set the GPU id
    set_gpu(args.gpu)

    # Set manual seed for PyTorch
    if args.seed==0:
        print ('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print ('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Start trainer for pre-train, meta-train or meta-eval
    if args.phase=='meta_train':
        trainer = MetaTrainer(args)
        trainer.train()
    elif args.phase=='meta_eval':
        trainer = MetaTrainer(args)
        trainer.eval()
    elif args.phase=='pre_train':
        trainer = PreTrainer(args)
        trainer.train()
    else:
        raise ValueError('Please set correct phase.')
