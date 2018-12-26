import os
import csv
import pdb
import glob
import numpy as np

log_dir = './logs/test_1'
all_exp_folder_list = os.listdir(log_dir)

all_1shot_acc=[]
all_1shot_ci95=[]
all_1shot_info = []
all_5shot_acc=[]
all_5shot_ci95=[]
all_5shot_info = []

for this_exp_folder in all_exp_folder_list:
    print('***** Processing ' + this_exp_folder)
    this_exp_folder_dir = os.path.join(log_dir, this_exp_folder)
    all_csv_dir = glob.glob(this_exp_folder_dir + '/*.csv')
    step_list = []
    acc_list = []
    ci95_list = []
    for this_csv_dir in all_csv_dir:
        csv_reader = csv.reader(open(this_csv_dir))
        row_list = [this_row for this_row in csv_reader]
        acc_row = [float(acc) for acc in row_list[1]]
        ci95_row = [float(ci95) for ci95 in row_list[3]]
        max_idx = np.argmax(acc_row)
        acc_max = acc_row[max_idx]
        ci95_max = ci95_row[max_idx]
        start_idx = this_csv_dir.find('testiter')
        end_idx = this_csv_dir.find('.csv')
        iter_str = this_csv_dir[start_idx+8:end_idx]
        #print('Step: ' + iter_str + ' Max_Acc: ' + str(acc_max) + ' CI95: ' + str(ci95_max))
        step_list.append(iter_str)
        acc_list.append(acc_max)
        ci95_list.append(ci95_max)
        if acc_max>=0.65:
            all_5shot_acc.append(acc_max)
            all_5shot_ci95.append(ci95_max)
            all_5shot_info.append(this_exp_folder + '_Step_' + iter_str)
        else:            
            all_1shot_acc.append(acc_max)
            all_1shot_ci95.append(ci95_max)
            all_1shot_info.append(this_exp_folder + '_Step_' + iter_str)

    result_dict_list = [{'step':int(step_list[idx]), 'acc':acc_list[idx], 'ci95':ci95_list[idx]} for idx in range(len(step_list))]
    result_dict_list = sorted(result_dict_list, key=lambda x:x['step'])
    print('***** Step')
    for idx in range(len(result_dict_list)):
        print(result_dict_list[idx]['step'])
    print('***** Acc')
    for idx in range(len(result_dict_list)):
        print(str(result_dict_list[idx]['acc']))

final_max_1shot_acc_idx = np.argmax(all_1shot_acc)
final_max_1shot_acc = all_1shot_acc[final_max_1shot_acc_idx]
final_max_1shot_ci95 = all_1shot_ci95[final_max_1shot_acc_idx]
final_max_1shot_info = all_1shot_info[final_max_1shot_acc_idx]
print('Best 1shot: Acc: ' + str(final_max_1shot_acc) + ' CI95: ' + str(final_max_1shot_ci95) + ' Info: ' + final_max_1shot_info)

final_max_5shot_acc_idx = np.argmax(all_5shot_acc)
final_max_5shot_acc = all_5shot_acc[final_max_5shot_acc_idx]
final_max_5shot_ci95 = all_5shot_ci95[final_max_5shot_acc_idx]
final_max_5shot_info = all_5shot_info[final_max_5shot_acc_idx]
print('Best 5shot: Acc: ' + str(final_max_5shot_acc) + ' CI95: ' + str(final_max_5shot_ci95) + ' Info: ' + final_max_5shot_info)
