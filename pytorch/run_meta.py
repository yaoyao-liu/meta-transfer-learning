##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Generate commands for meta-train phase. """
import os

def run_exp(num_batch=1000, shot=1, query=15, lr1=0.0001, lr2=0.001, base_lr=0.01, update_step=10, gamma=0.5):
    max_epoch = 100
    way = 5
    step_size = 10
    gpu = 1
       
    the_command = 'python3 main.py' \
        + ' --max_epoch=' + str(max_epoch) \
        + ' --num_batch=' + str(num_batch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --meta_lr1=' + str(lr1) \
        + ' --meta_lr2=' + str(lr2) \
        + ' --step_size=' + str(step_size) \
        + ' --gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --update_step=' + str(update_step) 

    os.system(the_command + ' --phase=meta_train')
    os.system(the_command + ' --phase=meta_eval')

run_exp(num_batch=100, shot=1, query=15, lr1=0.0001, lr2=0.001, base_lr=0.01, update_step=100, gamma=0.5)
run_exp(num_batch=100, shot=5, query=15, lr1=0.0001, lr2=0.001, base_lr=0.01, update_step=100, gamma=0.5)
