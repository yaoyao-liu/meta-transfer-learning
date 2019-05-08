##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Tianjin University
## Email: liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os

def run_experiment(MAX_ITER=10000, SHOT_NUM=1, PHASE='META'):
    """The function to generate commands to run the experiments.
    Args:
      MAX_ITER: the iteration number for meta-train.
      SHOT_NUM: the shot number for the few-shot tasks.
      PHASE: the phase for MTL. 'PRE' means pre-train phase, and 'META' means meta-train and meta-test phases.
    """
    GPU_ID = 0 # The GPU device id 
    META_BATCH_SIZE = 2 # The meta batch size 
    PRE_ITER = 10000 # The iteration number for the pre-train model used in the meta-train phase
    UPDATE_NUM = 20 # The epoch number for the base learning
    WAY_NUM = 5 # The class number for the few-shot tasks
    GPU_MODE = 'False' # If 'GPU_MODE' is true, it will occupy all the GPU memory when the tensorflow session starts
    LOG_DIR = 'experiment_results' # The name of the folder to save the log files
    PRE_TRA_ITER_MAX = 20000 # The iteration number for the pre-train phase
    PRE_TRA_DROP = 0.9 # The dropout keep rate for the pre-train phase
    SAVE_STEP = 1000 # The iteration number to save the meta model
    LR_DROP_STEP = 1000 # The iteration number for the meta learning rate reducing
    PRE_TRA_FLD = './data/meta-train/train' # The directory for the pre-train phase images
    PRE_TRA_LAB = 'mini_normal' # The additional label for pre-train model 

    """More settings are in the main.py file.
    """

    # generate the base command for main.py
    base_command = 'python main.py' \
        + ' --metatrain_iterations=' + str(MAX_ITER) \
        + ' --meta_batch_size=' + str(META_BATCH_SIZE) \
        + ' --shot_num=' + str(SHOT_NUM) \
        + ' --base_lr=0.01' \
        + ' --train_base_epoch_num=' + str(UPDATE_NUM) \
        + ' --way_num=' + str(WAY_NUM) \
        + ' --exp_log_label=' + LOG_DIR \
        + ' --pretrain_dropout_keep=' + str(PRE_TRA_DROP) \
        + ' --activation=leaky_relu' \
        + ' --pre_lr=0.001' \
        + ' --pre_lr_dropstep=5000' \
        + ' --meta_save_step=' + str(SAVE_STEP) \
        + ' --lr_drop_step=' + str(LR_DROP_STEP) \
        + ' --pretrain_folders=' + PRE_TRA_FLD \
        + ' --pretrain_label=' + PRE_TRA_LAB \
        + ' --full_gpu_memory_mode=' + GPU_MODE \
        + ' --device_id=' + str(GPU_ID)

    def process_test_command(TEST_STEP, in_command):
    """The function to adapt the base command to the meta-test phase.
    Args:
      TEST_STEP: the iteration number for the meta model to be loaded.
      in_command: the input base command.
    Return:
      Processed command.
    """
        output_test_command = in_command \
            + ' --phase=meta' \
            + ' --pretrain_iterations=' + str(PRE_ITER) \
            + ' --metatrain=False' \
            + ' --test_iter=' + str(TEST_STEP)
        return output_test_command

    if PHASE=='PRE':
        print('****** Start Pre-train Phase ******')
        pre_command = base_command + ' --phase=pre' + ' --pretrain_iterations=' + str(PRE_TRA_ITER_MAX)
        os.system(pre_command)

    if PHASE=='META':
        print('****** Start Meta-train Phase ******')
        meta_train_command = base_command + ' --phase=meta' + ' --pretrain_iterations=' + str(PRE_ITER)
        os.system(meta_train_command)

        print('****** Start Meta-test Phase ******')
        for idx in range(MAX_ITER):
            if idx % SAVE_STEP == 0:
                print('[*] Runing meta-test, load model for ' + str(idx) + ' iterations')
                test_command = process_test_command(idx, base_command)
                os.system(test_command)
                
# run pre-train phase
run_experiment(PHASE='PRE')

# run meta-train and meta-test phase
run_experiment(MAX_ITER=20000, SHOT_NUM=1, PHASE='META')
