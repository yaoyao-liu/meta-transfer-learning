##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/cbfinn/maml
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

""" Generate commands for main.py """
import os
import sys

def run_experiment(PHASE='META'):
    """The function to generate commands to run the experiments.
    Arg:
      PHASE: the phase for MTL. 'PRE' means pre-train phase, and 'META' means meta-train and meta-test phases.
    """
    # Some important options
    # Please note that not all the options are shown here. For more detailed options, please edit main.py

    # Basic options
    LOG_DIR = 'experiment_results' # Name of the folder to save the log files
    GPU_ID = 1 # GPU device id
    NET_ARCH = 'resnet12' # Additional label for pre-train model

    # Pre-train phase options
    PRE_TRA_LABEL = 'normal' # Additional label for pre-train model
    PRE_TRA_ITER_MAX = 20000 # Iteration number for the pre-train phase
    PRE_TRA_DROP = 0.9 # Dropout keep rate for the pre-train phase
    PRE_DROP_STEP = 5000 # Iteration number for the pre-train learning rate reducing
    PRE_LR = 0.001 # Pre-train learning rate

    # Meta options
    SHOT_NUM = 1 # Shot number for the few-shot tasks
    WAY_NUM = 5 # Class number for the few-shot tasks
    MAX_ITER = 20000 # Iteration number for meta-train
    META_BATCH_SIZE = 2 # Meta batch size 
    PRE_ITER = 10000 # Iteration number for the pre-train model used in the meta-train phase
    UPDATE_NUM = 20 # Epoch number for the base learning
    SAVE_STEP = 100 # Iteration number to save the meta model
    META_LR = 0.001 # Meta learning rate
    META_LR_MIN = 0.0001 # Meta learning rate min value
    LR_DROP_STEP = 1000 # The iteration number for the meta learning rate reducing
    BASE_LR = 0.001 # Base learning rate

    # Data directories
    PRE_TRA_DIR = './data/mini-imagenet/train' # Directory for the pre-train phase images
    META_TRA_DIR = './data/mini-imagenet/train' # Directory for the meta-train images
    META_VAL_DIR = './data/mini-imagenet/val' # Directory for the meta-validation images
    META_TES_DIR = './data/mini-imagenet/test' # Directory for the meta-test images

    # Generate the base command for main.py
    base_command = 'python main.py' \
        + ' --backbone_arch=' + str(NET_ARCH) \
        + ' --metatrain_iterations=' + str(MAX_ITER) \
        + ' --meta_batch_size=' + str(META_BATCH_SIZE) \
        + ' --shot_num=' + str(SHOT_NUM) \
        + ' --meta_lr=' + str(META_LR) \
        + ' --min_meta_lr=' + str(META_LR_MIN) \
        + ' --base_lr=' + str(BASE_LR)\
        + ' --train_base_epoch_num=' + str(UPDATE_NUM) \
        + ' --way_num=' + str(WAY_NUM) \
        + ' --exp_log_label=' + LOG_DIR \
        + ' --pretrain_dropout_keep=' + str(PRE_TRA_DROP) \
        + ' --activation=leaky_relu' \
        + ' --pre_lr=' + str(PRE_LR)\
        + ' --pre_lr_dropstep=' + str(PRE_DROP_STEP) \
        + ' --meta_save_step=' + str(SAVE_STEP) \
        + ' --lr_drop_step=' + str(LR_DROP_STEP) \
        + ' --pretrain_folders=' + PRE_TRA_DIR \
        + ' --pretrain_label=' + PRE_TRA_LABEL \
        + ' --device_id=' + str(GPU_ID) \
        + ' --metatrain_dir=' + META_TRA_DIR \
        + ' --metaval_dir=' + META_VAL_DIR \
        + ' --metatest_dir=' + META_TES_DIR

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

    if PHASE=='META_LOAD':
        print('****** Start Meta-train Phase with Downloaded Weights ******')
        meta_train_command = base_command + ' --phase=meta' + ' --pretrain_iterations=' + str(PRE_ITER) + ' --load_saved_weights=True'
        os.system(meta_train_command)

    if PHASE=='TEST_LOAD':
        print('****** Start Meta-test Phase with Downloaded Weights ******')
        test_command = process_test_command(0, base_command) + ' --load_saved_weights=True'
        os.system(test_command)
       
THE_INPUT_PHASE = sys.argv[1]
run_experiment(PHASE=THE_INPUT_PHASE)
