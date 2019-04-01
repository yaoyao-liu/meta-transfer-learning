##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyaoliu@outlook.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import csv
import pdb
import glob
import numpy as np

log_dir = './logs/experiment_results_meta_batch_4'
all_exp_folder_list = os.listdir(log_dir)

all_1shot_acc=[]
all_1shot_ci95=[]
all_1shot_info = []
all_5shot_acc=[]
all_5shot_ci95=[]
all_5shot_info = []

specific_epoch_num_mode = True
test_epoch_num = 100

for this_exp_folder in all_exp_folder_list:
    print('***** Processing ' + this_exp_folder)
    this_exp_folder_dir = os.path.join(log_dir, this_exp_folder)
    all_csv_dir = glob.glob(this_exp_folder_dir + '/result*.csv')
    step_list = []
    acc_list = []
    ci95_list = []
    for this_csv_dir in all_csv_dir:
        csv_reader = csv.reader(open(this_csv_dir))
        row_list = [this_row for this_row in csv_reader]
        acc_row = [float(acc) for acc in row_list[1]]
        ci95_row = [float(ci95) for ci95 in row_list[3]]
        if specific_epoch_num_mode:
            if len(acc_row)<test_epoch_num:
                acc_row = np.zeros(test_epoch_num)
            else:
                acc_row = acc_row[0:test_epoch_num]
        max_idx = np.argmax(acc_row)
        acc_max = acc_row[max_idx]
        ci95_max = ci95_row[max_idx]
        start_idx = this_csv_dir.find('shot_')
        end_idx = this_csv_dir.find('.csv')
        iter_str = this_csv_dir[start_idx+5:end_idx]
        #print('Step: ' + iter_str + ' Max_Acc: ' + str(acc_max) + ' CI95: ' + str(ci95_max))
        step_list.append(iter_str)
        acc_list.append(acc_max)
        ci95_list.append(ci95_max)
        if '5shot' in this_csv_dir:
            all_5shot_acc.append(acc_max)
            all_5shot_ci95.append(ci95_max)
            all_5shot_info.append(this_exp_folder + '_Step_' + iter_str)
        elif '1shot' in this_csv_dir:            
            all_1shot_acc.append(acc_max)
            all_1shot_ci95.append(ci95_max)
            all_1shot_info.append(this_exp_folder + '_Step_' + iter_str)
        else:
            print('Shot Num Error')

    result_dict_list = [{'step':int(step_list[idx]), 'acc':acc_list[idx], 'ci95':ci95_list[idx]} for idx in range(len(step_list))]
    result_dict_list = sorted(result_dict_list, key=lambda x:x['step'])
    print('***** Step')
    for idx in range(len(result_dict_list)):
        print(result_dict_list[idx]['step'])
    print('***** Acc')
    for idx in range(len(result_dict_list)):
        print(str(result_dict_list[idx]['acc']))

if len(all_1shot_acc) is not 0:
    final_max_1shot_acc_idx = np.argmax(all_1shot_acc)
    final_max_1shot_acc = all_1shot_acc[final_max_1shot_acc_idx]
    final_max_1shot_ci95 = all_1shot_ci95[final_max_1shot_acc_idx]
    final_max_1shot_info = all_1shot_info[final_max_1shot_acc_idx]
    print('Best 1shot: Acc: ' + str(final_max_1shot_acc) + ' CI95: ' + str(final_max_1shot_ci95))
    print('Info: ' + final_max_1shot_info)

if len(all_5shot_acc) is not 0:
    final_max_5shot_acc_idx = np.argmax(all_5shot_acc)
    final_max_5shot_acc = all_5shot_acc[final_max_5shot_acc_idx]
    final_max_5shot_ci95 = all_5shot_ci95[final_max_5shot_acc_idx]
    final_max_5shot_info = all_5shot_info[final_max_5shot_acc_idx]
    print('Best 5shot: Acc: ' + str(final_max_5shot_acc) + ' CI95: ' + str(final_max_5shot_ci95))
    print(' Info: ' + final_max_5shot_info)
