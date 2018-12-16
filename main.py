import os
import csv
import pickle
import random
import numpy as np
import tensorflow as tf
import cv2
import pdb

from tqdm import trange
from data_generator import DataGenerator
from models import MODELS
from tensorflow.python.platform import flags
from utils import process_batch

# mac test

FLAGS = flags.FLAGS

## Options
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') 
flags.DEFINE_integer('meta_batch_size', 2, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_float('pre_lr', 0.001, 'the pretrain learning rate')
flags.DEFINE_integer('pre_lr_dropstep', 5000, '-')
flags.DEFINE_integer('pretrain_class_num', 64, '-')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') 
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_bool('pretrain', True, 'pretrain resnet classifier')
flags.DEFINE_integer('pretrain_resnet_iterations', 30000, 'number of pretraining iterations.')
flags.DEFINE_integer('lr_drop_step', 5000, '-')
flags.DEFINE_integer('save_step', 500, '-')
flags.DEFINE_float('pretrain_dropout', 0.9, 'step size alpha for inner gradient update.')
flags.DEFINE_string('activation', 'leaky_relu', 'leaky_relu, relu, or None')
flags.DEFINE_integer('test_num_updates', 100, 'number of inner gradient updates during training.')
flags.DEFINE_string('pretrain_folders', './minia/mini_nh/train', '-')
flags.DEFINE_string('pretrain_label', 'mini_nh', '-')
flags.DEFINE_bool('pre_lr_stop', False, '-')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', 1000, 'iteration to load model')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_string('pretrain_dir', './pretrain_weights', '-')
flags.DEFINE_bool('full_gpu_memory_mode', False, '-')
flags.DEFINE_float('gpu_rate', 0.8, '-')

def pretrain_main():

    pre_train_data_generator = DataGenerator(FLAGS.update_batch_size, FLAGS.meta_batch_size)
    pretrain_input, pretrain_label = pre_train_data_generator.make_data_tensor_for_pretrain_classifier()
    pre_train_input_tensors = {'pretrain_input': pretrain_input, 'pretrain_label': pretrain_label}

    dim_output = pre_train_data_generator.dim_output
    dim_input = pre_train_data_generator.dim_input
    model = MODELS(dim_input, dim_output)

    model.construct_pretrain_model(input_tensors=pre_train_input_tensors)
    model.pretrain_summ_op = tf.summary.merge_all()

    if FLAGS.full_gpu_memory_mode:
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
        sess = tf.InteractiveSession(config=gpu_config)
    else:
        sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    pre_train(model, sess)
    

def pre_train(model, sess):
    SUMMARY_INTERVAL = 10
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 1000
    LR_DROP_STEP = FLAGS.pre_lr_dropstep
    pretrain_iterations = FLAGS.pretrain_resnet_iterations
    weights_save_dir_base = FLAGS.pretrain_dir
    if not os.path.exists(weights_save_dir_base):
        os.mkdir(weights_save_dir_base)

    weights_save_dir = os.path.join(weights_save_dir_base, "pretrain_dropout_{}_activation_{}_prelr_{}_dropstep_{}".format(FLAGS.pretrain_dropout, FLAGS.activation, FLAGS.pre_lr, FLAGS.pre_lr_dropstep) + FLAGS.pretrain_label)
    if FLAGS.pre_lr_stop:
        weights_save_dir = weights_save_dir+'_prelrstop'
    if not os.path.exists(weights_save_dir):
        os.mkdir(weights_save_dir)

    pretrain_writer = tf.summary.FileWriter(weights_save_dir, sess.graph)
    pre_lr = FLAGS.pre_lr
    
    for itr in trange(pretrain_iterations):
        feed_dict = {model.pretrain_lr: pre_lr}
        input_tensors = [model.pretrain_op, model.pretrain_summ_op]
        input_tensors.extend([model.pretrain_task_loss, model.pretrain_task_accuracy])
        result = sess.run(input_tensors, feed_dict)
        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = '[*] Loss: ' + str(result[-2]) + ', Acc: ' + str(result[-1])
            print(print_str)

        if itr % SUMMARY_INTERVAL == 0:
            pretrain_writer.add_summary(result[1], itr)

        if (itr!=0) and itr % LR_DROP_STEP == 0:
            pre_lr = pre_lr * 0.5
            if FLAGS.pre_lr_stop:
                if pre_lr<0.0001:
                    pre_lr = 0.0001
                
        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            print('Saving pretrain weights to npy.')
            weights = sess.run(model.weights)
            np.save(os.path.join(weights_save_dir, "weights_{}.npy".format(itr)), weights)


def train(model, sess, exp_string):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = FLAGS.save_step
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    LR_DROP_STEP = FLAGS.lr_drop_step

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training')
    maml_loss, maml_acc = [], []

    num_classes = FLAGS.num_classes
    multitask_weights, reg_weights = [], []

    train_lr = 0.001

    num_samples_per_class = FLAGS.update_batch_size+15
    task_num = FLAGS.num_classes * num_samples_per_class

    num_samples_per_class_test = FLAGS.update_batch_size*2
    test_task_num = FLAGS.num_classes * num_samples_per_class_test

    epitr_sample_num = FLAGS.update_batch_size
    epite_sample_num = 15
    test_task_sample_num = FLAGS.update_batch_size
    dim_input = 21168

    all_npy_dir = './filenames_and_labels/'
    npy_base_dir = all_npy_dir + str(FLAGS.update_batch_size) + 'shot_' + str(FLAGS.num_classes) + 'way/'
    all_filenames = np.load(npy_base_dir + 'train_filenames.npy').tolist()
    labels = np.load(npy_base_dir + 'train_labels.npy').tolist()

    all_test_filenames = np.load(npy_base_dir + 'val_filenames.npy').tolist()
    test_labels = np.load(npy_base_dir + 'val_labels.npy').tolist()

    test_idx = 0

    for train_idx in trange(FLAGS.metatrain_iterations):

        inputa = []
        labela = []
        inputb = []
        labelb = []
        for meta_batch_idx in range(FLAGS.meta_batch_size):

            this_task_filenames = all_filenames[(train_idx*FLAGS.meta_batch_size+meta_batch_idx)*task_num:(train_idx*FLAGS.meta_batch_size+meta_batch_idx+1)*task_num]

            this_task_tr_filenames = []
            this_task_tr_labels = []
            this_task_te_filenames = []
            this_task_te_labels = []
            for class_k in range(FLAGS.num_classes):
                this_class_filenames = this_task_filenames[class_k*num_samples_per_class:(class_k+1)*num_samples_per_class]
                this_class_label = labels[class_k*num_samples_per_class:(class_k+1)*num_samples_per_class]
                this_task_tr_filenames += this_class_filenames[0:epitr_sample_num]
                this_task_tr_labels += this_class_label[0:epitr_sample_num]
                this_task_te_filenames += this_class_filenames[epitr_sample_num:]
                this_task_te_labels += this_class_label[epitr_sample_num:]

            this_inputa, this_labela = process_batch(this_task_tr_filenames, this_task_tr_labels, dim_input, epitr_sample_num, reshape_with_one=False)
            this_inputb, this_labelb = process_batch(this_task_te_filenames, this_task_te_labels, dim_input, epite_sample_num, reshape_with_one=False)
            inputa.append(this_inputa)
            labela.append(this_labela)
            inputb.append(this_inputb)
            labelb.append(this_labelb)

        inputa = np.array(inputa)
        labela = np.array(labela)
        inputb = np.array(inputb)
        labelb = np.array(labelb)

        feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: train_lr}

        input_tensors = [model.metatrain_op]

        if (train_idx % SUMMARY_INTERVAL == 0 or train_idx % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss])
            input_tensors.extend([model.total_accuracy])

        result = sess.run(input_tensors, feed_dict)

        if train_idx % SUMMARY_INTERVAL == 0:
            maml_loss.append(result[2])
            if FLAGS.log:
                train_writer.add_summary(result[1], train_idx)
            maml_acc.append(result[3])

        if (train_idx!=0) and train_idx % PRINT_INTERVAL == 0:
            print_str = 'Iteration:' + str(train_idx)
            print_str += ' Loss:' + str(np.mean(maml_loss)) + ' Acc:' + str(np.mean(maml_acc))
            print(print_str)
            maml_loss, maml_acc = [], []

        if (train_idx!=0) and train_idx % SAVE_INTERVAL == 0:
            weights = sess.run(model.weights)
            ss_weights = sess.run(model.ss_weights)
            fc_weights = sess.run(model.fc_weights)
            np.save(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(train_idx) + '.npy', weights)
            np.save(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(train_idx) + '.npy', ss_weights)
            np.save(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(train_idx) + '.npy', fc_weights)

        if (train_idx!=0) and train_idx % TEST_PRINT_INTERVAL == 0:
            test_loss = []
            test_accs = []

            for test_itr in range(10):            

                this_test_task_filenames = all_test_filenames[test_idx*test_task_num:(test_idx+1)*test_task_num]
                this_test_task_tr_filenames = []
                this_test_task_tr_labels = []
                this_test_task_te_filenames = []
                this_test_task_te_labels = []
                for class_k in range(FLAGS.num_classes):
                    this_test_class_filenames = this_test_task_filenames[class_k*num_samples_per_class_test:(class_k+1)*num_samples_per_class_test]
                    this_test_class_label = test_labels[class_k*num_samples_per_class_test:(class_k+1)*num_samples_per_class_test]
                    this_test_task_tr_filenames += this_test_class_filenames[0:test_task_sample_num]
                    this_test_task_tr_labels += this_test_class_label[0:test_task_sample_num]
                    this_test_task_te_filenames += this_test_class_filenames[test_task_sample_num:]
                    this_test_task_te_labels += this_test_class_label[test_task_sample_num:]

                test_inputa, test_labela = process_batch(this_test_task_tr_filenames, this_test_task_tr_labels, dim_input, test_task_sample_num)
                test_inputb, test_labelb = process_batch(this_test_task_te_filenames, this_test_task_te_labels, dim_input, test_task_sample_num)

                test_feed_dict = {model.inputa: test_inputa, model.inputb: test_inputb, model.labela: test_labela, model.labelb: test_labelb, model.meta_lr: 0.0}
                test_input_tensors = [model.total_loss, model.total_accuracy]

                test_result = sess.run(test_input_tensors, test_feed_dict)

                test_loss.append(test_result[0])
                test_accs.append(test_result[1])
            
                test_idx += 1

            print_str = '[***] Val Loss:' + str(np.mean(test_loss)*FLAGS.meta_batch_size) + ' Val Acc:' + str(np.mean(test_accs)*FLAGS.meta_batch_size)
            print(print_str)
                    

        if (train_idx!=0) and train_idx % LR_DROP_STEP == 0:
            train_lr = train_lr * 0.5
            if train_lr < 0.0001:
                train_lr = 0.0001
            print('Train LR: {}'.format(train_lr))

        if train_idx > 20000:
            SAVE_INTERVAL = 5000

    weights = sess.run(model.weights)
    ss_weights = sess.run(model.ss_weights)
    fc_weights = sess.run(model.fc_weights)
    np.save(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(train_idx+1) + '.npy', weights)
    np.save(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(train_idx+1) + '.npy', ss_weights)
    np.save(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(train_idx+1) + '.npy', fc_weights)


def test(model, sess, exp_string):

    NUM_TEST_POINTS = 600   
    num_classes = FLAGS.num_classes

    np.random.seed(1)

    metaval_accuracies = []

    num_samples_per_class = FLAGS.update_batch_size*2
    task_num = FLAGS.num_classes * num_samples_per_class
    half_num_samples = num_samples_per_class/2
    dim_input = 21168

    all_npy_dir = './filenames_and_labels/'
    npy_base_dir = all_npy_dir + str(FLAGS.update_batch_size) + 'shot_' + str(FLAGS.num_classes) + 'way/'
    all_filenames = np.load(npy_base_dir + 'test_filenames.npy').tolist()
    labels = np.load(npy_base_dir + 'test_labels.npy').tolist()

    for test_idx in trange(NUM_TEST_POINTS):

        this_task_filenames = all_filenames[test_idx*task_num:(test_idx+1)*task_num]

        this_task_tr_filenames = []
        this_task_tr_labels = []
        this_task_te_filenames = []
        this_task_te_labels = []
        for class_k in range(FLAGS.num_classes):
            this_class_filenames = this_task_filenames[class_k*num_samples_per_class:(class_k+1)*num_samples_per_class]
            this_class_label = labels[class_k*num_samples_per_class:(class_k+1)*num_samples_per_class]
            this_task_tr_filenames += this_class_filenames[0:half_num_samples]
            this_task_tr_labels += this_class_label[0:half_num_samples]
            this_task_te_filenames += this_class_filenames[half_num_samples:]
            this_task_te_labels += this_class_label[half_num_samples:]

        inputa, labela = process_batch(this_task_tr_filenames, this_task_tr_labels, dim_input, half_num_samples)
        inputb, labelb = process_batch(this_task_te_filenames, this_task_te_labels, dim_input, half_num_samples)

        feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        result = sess.run(model.metaval_total_accuracies, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    max_idx = np.argmax(means)
    max_acc = np.max(means)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)
    max_ci95 = ci95[max_idx]

    print('Mean validation accuracy and confidence intervals')
    print((means, ci95))

    print('***** Best Acc: '+ str(max_acc) + ' CI95: ' + str(max_ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'new_test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '_testiter' + str(FLAGS.test_iter) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'new_test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '_testiter' + str(FLAGS.test_iter) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def main():

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.train:
        data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  
    test_data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size) 

    if FLAGS.train:
        dim_output = data_generator.dim_output
        dim_input = data_generator.dim_input
    else:
        dim_output = test_data_generator.dim_output
        dim_input = test_data_generator.dim_input

    model = MODELS(dim_input, dim_output)

    if FLAGS.train:
        print('Building train model')
        model.construct_model(prefix='metatrain_')
    else:
        print('Building test mdoel')
        model.construct_test_model(prefix='metaval_')
    model.summ_op = tf.summary.merge_all()
    if FLAGS.full_gpu_memory_mode:
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_rate
        sess = tf.InteractiveSession(config=gpu_config)
    else:
        sess = tf.InteractiveSession()

    if FLAGS.train:
        random.seed(5)
        data_generator.make_data_list(train=True)

    random.seed(6)
    test_data_generator.make_data_list(train=False)

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.update_lr) + '.pretrainstep' + str(FLAGS.pretrain_resnet_iterations) + '.dropout' + str(FLAGS.pretrain_dropout) +'.actifunc' + FLAGS.activation +'.lrdrop' + str(FLAGS.lr_drop_step) + '.pretr' + FLAGS.pretrain_label

    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized')

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.train or FLAGS.test_iter==0:
        print('Loading pretrain weights pretrain_dropout_{}_activation_{}_prelr_{}_dropstep_{}_label_{}_itr_{}'.format(FLAGS.pretrain_dropout, FLAGS.activation, FLAGS.pre_lr, FLAGS.pre_lr_dropstep, FLAGS.pretrain_label, FLAGS.pretrain_resnet_iterations))
        weights_save_dir_base = FLAGS.pretrain_dir
        weights_save_dir = os.path.join(weights_save_dir_base, "pretrain_dropout_{}_activation_{}_prelr_{}_dropstep_{}".format(FLAGS.pretrain_dropout, FLAGS.activation, FLAGS.pre_lr, FLAGS.pre_lr_dropstep) + FLAGS.pretrain_label)
        if FLAGS.pre_lr_stop:
            weights_save_dir = weights_save_dir+'_prelrstop'
        weights = np.load(os.path.join(weights_save_dir, "weights_{}.npy".format(FLAGS.pretrain_resnet_iterations))).tolist()
        bais_list = [bais_item for bais_item in weights.keys() if '_bias' in bais_item]
        for bais_key in bais_list:
            sess.run(tf.assign(model.ss_weights[bais_key], weights[bais_key]))
        for key in weights.keys():
            sess.run(tf.assign(model.weights[key], weights[key]))
        print('Weights loaded')
    else:
        weights = np.load(FLAGS.logdir + '/' + exp_string +  '/weights_' + str(FLAGS.test_iter) + '.npy').tolist()
        ss_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/ss_weights_' + str(FLAGS.test_iter) + '.npy').tolist()
        fc_weights = np.load(FLAGS.logdir + '/' + exp_string +  '/fc_weights_' + str(FLAGS.test_iter) + '.npy').tolist()
        for key in weights.keys():
            sess.run(tf.assign(model.weights[key], weights[key]))
        for key in ss_weights.keys():
            sess.run(tf.assign(model.ss_weights[key], ss_weights[key]))
        for key in fc_weights.keys():
            sess.run(tf.assign(model.fc_weights[key], fc_weights[key]))
        print('Weights loaded')
        print('Info: ' + exp_string)
        print('Test iter: ' + str(FLAGS.test_iter))
        

    if FLAGS.train:
        train(model, sess, exp_string)
    else:
        test(model, sess, exp_string)

if __name__ == "__main__":

    if FLAGS.pretrain == True:
        pretrain_main()
    else:
        main()
