import numpy as np
import pdb

weights = np.load('/BS/sun_project_meta/work/mlq_project/yyliu_project/meta_learning_project/maml_plus_200.0.1/logs/experiment_1/cls_5.mbs_2.ubs_1.numstep20.updatelr0.01.pretrainstep10000.dropout0.9.actifuncleaky_relu.lrdrop1000.pretrmini_nhmaxpoolbatchnorm/weights_1000.npy').tolist()

hyper_weights = np.load('/BS/sun_project_meta/work/mlq_project/yyliu_project/meta_learning_project/maml_plus_200.0.1/logs/experiment_1/cls_5.mbs_2.ubs_1.numstep20.updatelr0.01.pretrainstep10000.dropout0.9.actifuncleaky_relu.lrdrop1000.pretrmini_nhmaxpoolbatchnorm/hyper_weights_1000.npy').tolist()

fc_weights = np.load('/BS/sun_project_meta/work/mlq_project/yyliu_project/meta_learning_project/maml_plus_200.0.1/logs/experiment_1/cls_5.mbs_2.ubs_1.numstep20.updatelr0.01.pretrainstep10000.dropout0.9.actifuncleaky_relu.lrdrop1000.pretrmini_nhmaxpoolbatchnorm/fc_weights_1000.npy').tolist()

def count_size(input_weights):
    input_weights_size = 0
    for key in input_weights.keys():
        this_size = input_weights[key].size
        input_weights_size += this_size
    return input_weights_size

weights_size = count_size(weights)
hyper_weights_size = count_size(hyper_weights)
fc_weights_size = count_size(fc_weights)

print('Weights Size:' + str(weights_size))
print('Hyper Weights Size:' + str(hyper_weights_size))
print('FC Weights Size:' + str(fc_weights_size))
    
