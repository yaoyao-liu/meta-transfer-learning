import os

def run_exp(MAX_ITER, SHOT_NUM, PRE_TRA_ITER_LOAD, UPDATE_NUM, GPU_MODE, PRETRAIN, TRAIN, TEST):
    GPU_ID = 0
    META_BATCH_SIZE = 2
    #UPDATE_NUM = 20
    WAY_NUM = 5
    LOG_DIR = 'logs/experiment_1'
    PRE_TRA_ITER_MAX = 12000
    #PRE_TRA_ITER_LOAD = 8000
    PRE_TRA_DROP = 0.9
    SAVE_STEP = 1000
    LR_DROP_STEP = 1000
    PRE_TRA_FLD = './data/pre-train/mini_nh/train'
    PRE_TRA_LAB = 'mini_nh'    

    pretrain_command = 'CUDA_VISIBLE_DEVICES=' + str(GPU_ID) + ' python main.py --metatrain_iterations=' + str(MAX_ITER) + ' --meta_batch_size=' + str(META_BATCH_SIZE) +' --update_batch_size=' + str(SHOT_NUM) + ' --update_lr=0.01 --num_updates=' + str(UPDATE_NUM) + ' --num_classes=' + str(WAY_NUM) + ' --logdir=' + LOG_DIR + ' --pretrain=True --pretrain_resnet_iterations=' + str(PRE_TRA_ITER_MAX) + ' --pretrain_dropout=' + str(PRE_TRA_DROP) + ' --activation=leaky_relu --pre_lr=0.001 --pre_lr_dropstep=5000 --save_step=' + str(SAVE_STEP) + ' --lr_drop_step=' + str(LR_DROP_STEP) + ' --pretrain_folders=' + PRE_TRA_FLD + ' --pretrain_label=' + PRE_TRA_LAB + ' --full_gpu_memory_mode=' + GPU_MODE
    train_command = 'CUDA_VISIBLE_DEVICES=' + str(GPU_ID) + ' python main.py --metatrain_iterations=' + str(MAX_ITER) + ' --meta_batch_size=' + str(META_BATCH_SIZE) +' --update_batch_size=' + str(SHOT_NUM) + ' --update_lr=0.01 --num_updates=' + str(UPDATE_NUM) + ' --num_classes=' + str(WAY_NUM) + ' --logdir=' + LOG_DIR + ' --pretrain=False --pretrain_resnet_iterations=' + str(PRE_TRA_ITER_LOAD) + ' --pretrain_dropout=' + str(PRE_TRA_DROP) + ' --activation=leaky_relu --pre_lr=0.001 --pre_lr_dropstep=5000 --save_step=' + str(SAVE_STEP) + ' --lr_drop_step=' + str(LR_DROP_STEP) + ' --pretrain_folders=' + PRE_TRA_FLD + ' --pretrain_label=' + PRE_TRA_LAB + ' --full_gpu_memory_mode=' + GPU_MODE

    def process_test_command(TEST_STEP):
        output_test_command = train_command + ' --train=False --test_set=True --test_iter=' + str(TEST_STEP)
        return output_test_command

    if PRETRAIN:
        print('****** Start Pretrain Phase ******')
        os.system(pretrain_command)

    if TRAIN:
        print('****** Start Train Phase ******')
        os.system(train_command)

    if TEST:
        print('****** Start Test Phase ******')
        test_command = process_test_command(0)
        os.system(test_command)

        for idx in range(MAX_ITER-1):
            if idx!=0 and idx%SAVE_STEP==0:
                test_command = process_test_command(idx)
                os.system(test_command)

        test_command = process_test_command(MAX_ITER)
        os.system(test_command)

run_exp(MAX_ITER=7000, SHOT_NUM=1, PRE_TRA_ITER_LOAD=10000, UPDATE_NUM=20, GPU_MODE='True', PRETRAIN=True, TRAIN=True, TEST=True)
