import os

def run_exp(MAX_ITER=7000, SHOT_NUM=1, PRE_ITER=10000, UPDATE_NUM=20, GPU_MODE='False', PHASE='META'):
    GPU_ID = 0
    META_BATCH_SIZE = 2
    #UPDATE_NUM = 20
    WAY_NUM = 5
    LOG_DIR = 'experiment_1'
    PRE_TRA_ITER_MAX = 12000
    #PRE_TRA_ITER_LOAD = 8000
    PRE_TRA_DROP = 0.9
    SAVE_STEP = 1000
    LR_DROP_STEP = 1000
    PRE_TRA_FLD = './data/pre-train/mini_nh/train'
    PRE_TRA_LAB = 'mini_nh' 

    base_command = 'CUDA_VISIBLE_DEVICES=' + str(GPU_ID) + ' python main.py --metatrain_iterations=' + str(MAX_ITER) + ' --meta_batch_size=' + str(META_BATCH_SIZE) +' --update_batch_size=' + str(SHOT_NUM) + ' --update_lr=0.01 --num_updates=' + str(UPDATE_NUM) + ' --num_classes=' + str(WAY_NUM) + ' --logdir_label=' + LOG_DIR + ' --pretrain_dropout=' + str(PRE_TRA_DROP) + ' --activation=leaky_relu --pre_lr=0.001 --pre_lr_dropstep=5000 --save_step=' + str(SAVE_STEP) + ' --lr_drop_step=' + str(LR_DROP_STEP) + ' --pretrain_folders=' + PRE_TRA_FLD + ' --pretrain_label=' + PRE_TRA_LAB + ' --full_gpu_memory_mode=' + GPU_MODE

    def process_test_command(TEST_STEP, in_command):
        output_test_command = in_command + ' --phase=meta' + ' --pretrain_resnet_iterations=' + str(PRE_ITER) + ' --metatrain=False --test_set=True --test_iter=' + str(TEST_STEP)
        return output_test_command


    if PHASE=='PRE':
        print('****** Start Pre-training Phase ******')
        pre_command = base_command + ' --phase=pre' + ' --pretrain_resnet_iterations=15000'
        os.system(pre_command)

    if PHASE=='META':
        print('****** Start Meta-training Phase ******')
        meta_train_command = base_command + ' --phase=meta' + ' --pretrain_resnet_iterations=' + str(PRE_ITER)
        os.system(meta_train_command)

    if PHASE=='META_TE':
        print('****** Start Meta-test Phase ******')

        test_command = process_test_command(0, base_command)
        os.system(test_command)
        for idx in range(MAX_ITER-1):
            if idx!=0 and idx%SAVE_STEP==0:
                test_command = process_test_command(idx, base_command)
                os.system(test_command)

        test_command = process_test_command(MAX_ITER, base_command)
        os.system(test_command)
   
#run_exp(MAX_ITER=7000, SHOT_NUM=1, PRE_ITER=10000, UPDATE_NUM=20, GPU_MODE='False', PHASE='PRE')
run_exp(MAX_ITER=7000, SHOT_NUM=1, PRE_ITER=10000, UPDATE_NUM=20, GPU_MODE='False', PHASE='META')
run_exp(MAX_ITER=7000, SHOT_NUM=1, PRE_ITER=10000, UPDATE_NUM=20, GPU_MODE='False', PHASE='META_TE')
