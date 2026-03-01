import os
import pprint
import sys

import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from lib.fast_rcnn.train import get_training_roidb, train_net
from lib.fast_rcnn.config import cfg_from_file, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from ctpn.tf_runtime import configure_tensorflow_runtime

if __name__ == '__main__':
    # Enables TensorFlow GPU memory growth via ctpn.tf_runtime.configure_tensorflow_runtime().
    gpus = configure_tensorflow_runtime()

    cfg_from_file('ctpn/text.yml')
    cfg.TRAIN.IMS_PER_BATCH = 1 
    print('Using config:')
    pprint.pprint(cfg)
    imdb = get_imdb('voc_2007_trainval')
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    log_dir = get_log_dir(imdb)
    print('Output will be saved to `{:s}`'.format(output_dir))
    print('Logs will be saved to `{:s}`'.format(log_dir))

    device_name = '/gpu:0' if gpus else '/cpu:0'
    print(device_name)

    with tf.device(device_name):
        network = get_network('VGGnet_train')

    with tf.device(device_name):
        train_net(network, imdb, roidb,
                  output_dir=output_dir,
                  log_dir=log_dir,
                  pretrained_model=None,
                  max_iters=int(cfg.TRAIN.max_steps),
                  restore=bool(int(cfg.TRAIN.restore)))
