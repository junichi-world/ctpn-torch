import os
import pprint
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from lib.fast_rcnn.train import get_training_roidb, train_net
from lib.fast_rcnn.config import cfg_from_file, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg

if __name__ == '__main__':
    cfg_from_file('ctpn/text.yml')
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is not available. GPU学習にはCUDA対応版PyTorchが必要です。'
            ' torch={}, torch.version.cuda={}'.format(torch.__version__, torch.version.cuda)
        )
    torch.cuda.set_device(cfg.GPU_ID)
    torch.backends.cudnn.benchmark = True

    print('Using config:')
    pprint.pprint(cfg)
    imdb = get_imdb('voc_2007_trainval')
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    log_dir = get_log_dir(imdb)
    print('Output will be saved to `{:s}`'.format(output_dir))
    print('Logs will be saved to `{:s}`'.format(log_dir))

    device = torch.device(f'cuda:{cfg.GPU_ID}')
    print('Using device: {}'.format(device))
    print('GPU: {}'.format(torch.cuda.get_device_name(cfg.GPU_ID)))

    network = get_network('VGGnet_train')

    train_net(network, imdb, roidb,
              output_dir=output_dir,
              log_dir=log_dir,
              pretrained_model=None,
              max_iters=int(cfg.TRAIN.max_steps),
              restore=bool(int(cfg.TRAIN.restore)),
              require_cuda=True)
