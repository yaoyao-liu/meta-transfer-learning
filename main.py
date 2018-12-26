import os
from tensorflow.python.platform import flags
from trainer.meta import MetaTrainer
from trainer.pre import PreTrainer

FLAGS = flags.FLAGS

### Basic Options
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_string('phase', 'meta', '-')
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir_label', 'experiment_1', 'directory for summaries and checkpoints.')
flags.DEFINE_string('logdir_base', './logs/', '-')
flags.DEFINE_bool('full_gpu_memory_mode', False, '-')
flags.DEFINE_float('gpu_rate', 0.8, '-')

### Pretrain Phase Options
flags.DEFINE_float('pre_lr', 0.001, 'the pretrain learning rate')
flags.DEFINE_integer('pre_lr_dropstep', 5000, '-')
flags.DEFINE_integer('pretrain_class_num', 64, '-')
flags.DEFINE_integer('pretrain_resnet_iterations', 30000, 'number of pretraining iterations.')
flags.DEFINE_float('pretrain_dropout', 0.9, 'step size alpha for inner gradient update.')
flags.DEFINE_string('pretrain_folders', './minia/mini_nh/train', '-')
flags.DEFINE_string('pretrain_label', 'mini_nh', '-')
flags.DEFINE_bool('pre_lr_stop', False, '-')

### Metatrain Phase Options
flags.DEFINE_bool('metatrain', True, '-')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') 
flags.DEFINE_integer('meta_batch_size', 2, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') 
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('lr_drop_step', 5000, '-')
flags.DEFINE_integer('save_step', 500, '-')
flags.DEFINE_string('activation', 'leaky_relu', 'leaky_relu, relu, or None')
flags.DEFINE_integer('test_num_updates', 100, 'number of inner gradient updates during training.')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('test_iter', 1000, 'iteration to load model')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')


exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.update_lr) + '.pretrainstep' + str(FLAGS.pretrain_resnet_iterations) + '.dropout' + str(FLAGS.pretrain_dropout) +'.actifunc' + FLAGS.activation +'.lrdrop' + str(FLAGS.lr_drop_step) + '.pretr' + FLAGS.pretrain_label

if FLAGS.norm == 'batch_norm':
    exp_string += '.batchnorm'
elif FLAGS.norm == 'layer_norm':
    exp_string += '.layernorm'
elif FLAGS.norm == 'None':
    exp_string += '.nonorm'
else:
    print('Norm setting not recognized')

FLAGS.exp_string = exp_string
FLAGS.logdir = FLAGS.logdir_base + FLAGS.logdir_label
FLAGS.pretrain_dir = FLAGS.logdir_base + 'pretrain_weights'

if not os.path.exists(FLAGS.logdir_base):
    os.mkdir(FLAGS.logdir_base)
if not os.path.exists(FLAGS.logdir):
    os.mkdir(FLAGS.logdir)
if not os.path.exists(FLAGS.pretrain_dir):
    os.mkdir(FLAGS.pretrain_dir)


def main():
    if FLAGS.phase=='pre':
        trainer = PreTrainer()
        trainer.pre_train()
    elif FLAGS.phase=='meta':   
        trainer = MetaTrainer()
        if FLAGS.metatrain:
            trainer.train()
        else:
            trainer.test()
    else:
        print('Please set correct phase')                

if __name__ == "__main__":
    main()
