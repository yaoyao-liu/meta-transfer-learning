import os

def run_exp(PRE_TRA_ITER_LOAD, LR_DROP_STEP, UPDATE_NUM):
    GPU_ID = 1
    MAX_ITER = 20000
    META_BATCH_SIZE = 2
    SHOT_NUM = 1
    #UPDATE_NUM = 20
    WAY_NUM = 5
    LOG_DIR = 'logs/experiment_1'
    PRE_TRA_ITER_MAX = 12000
    #PRE_TRA_ITER_LOAD = 8000
    PRE_TRA_DROP = 0.9
    SAVE_STEP = 1000
    #LR_DROP_STEP = 1000
    PRE_TRA_FLD = './data/pre-train/mini_nh/train'
    PRE_TRA_LAB = 'mini_nh'
    TRAIN = True
    PRETRAIN = False
    TEST = True

    pretrain_command = 'CUDA_VISIBLE_DEVICES=' + str(GPU_ID) + ' python main.py --metatrain_iterations=' + str(MAX_ITER) + ' --meta_batch_size=' + str(META_BATCH_SIZE) +' --update_batch_size=' + str(SHOT_NUM) + ' --update_lr=0.01 --num_updates=' + str(UPDATE_NUM) + ' --num_classes=' + str(WAY_NUM) + ' --logdir=' + LOG_DIR + ' --max_pool=True --pretrain=True --pretrain_resnet_iterations=' + str(PRE_TRA_ITER_MAX) + ' --pretrain_dropout=' + str(PRE_TRA_DROP) + ' --activation=leaky_relu --pre_lr=0.001 --pre_lr_dropstep=5000 --save_step=' + str(SAVE_STEP) + ' --lr_drop_step=' + str(LR_DROP_STEP) + ' --pretrain_folders=' + PRE_TRA_FLD + ' --pretrain_label=' + PRE_TRA_LAB
    train_command = 'CUDA_VISIBLE_DEVICES=' + str(GPU_ID) + ' python main.py --metatrain_iterations=' + str(MAX_ITER) + ' --meta_batch_size=' + str(META_BATCH_SIZE) +' --update_batch_size=' + str(SHOT_NUM) + ' --update_lr=0.01 --num_updates=' + str(UPDATE_NUM) + ' --num_classes=' + str(WAY_NUM) + ' --logdir=' + LOG_DIR + ' --max_pool=True --pretrain=False --pretrain_resnet_iterations=' + str(PRE_TRA_ITER_LOAD) + ' --pretrain_dropout=' + str(PRE_TRA_DROP) + ' --activation=leaky_relu --pre_lr=0.001 --pre_lr_dropstep=5000 --save_step=' + str(SAVE_STEP) + ' --lr_drop_step=' + str(LR_DROP_STEP) + ' --pretrain_folders=' + PRE_TRA_FLD + ' --pretrain_label=' + PRE_TRA_LAB

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


PRE_ITER_LIST = [10000]
LR_DROP_STEP_LIST = [1000]
UPDATE_NUM_LIST = [20]

for PRE_TRA_ITER_LOAD in PRE_ITER_LIST:
    for LR_DROP_STEP in LR_DROP_STEP_LIST:
        for UPDATE_NUM in UPDATE_NUM_LIST:
            run_exp(PRE_TRA_ITER_LOAD, LR_DROP_STEP, UPDATE_NUM)
