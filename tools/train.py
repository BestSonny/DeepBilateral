import argparse
import logging
import numpy as np
import os
import tqdm
import setproctitle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import _init_paths
import models as models
import data_loader as dp
import train_net as train
from bilateral.modules.bilateral_slice import BilateralSlice, BilateralSliceApply

logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)

class Parameters(object):

    def set_parameters(self, args):
        for key, value in args.items():
            setattr(self, key, value)

    def __str__(self):
        string = '{'
        for key, value in self.__dict__.items():
            string = string + '{0}  :  {1},\r\n'.format(key, value)
        string = string + '}'
        return string
def main(args, model_params, data_params):
    procname = os.path.basename(args.checkpoint_dir)
    setproctitle.setproctitle('hdrnet_{}'.format(procname))

    log.info('Preparing summary and checkpoint directory {}'.format(args.checkpoint_dir))
    if not os.path.exists(args.checkpoint_dir):
      os.makedirs(args.checkpoint_dir)

    model = getattr(models, args.model_name)(num_classes=2, pretrained=True)
    data_loader = getattr(dp, 'DataLoader')

    train_loader_parameters = Parameters()
    train_loader_parameters.set_parameters({'path': args.data_dir,
                                            'is_train': args.data_dir,
                                            'shuffle': True,
                                            'data_params': data_params,
                                            'model_params': model_params,
                                            'data_pipeline': args.data_pipeline})
    train_loader = data_loader()
    train_loader.initialize(train_loader_parameters)
    train_dataset = train_loader.load_dataset()
    val_dataset = train_dataset
    train.train_net(model, train_dataset, val_dataset, args)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # pylint: disable=line-too-long
  # ----------------------------------------------------------------------------
  req_grp = parser.add_argument_group('required')
  req_grp.add_argument('checkpoint_dir', default=None, help='directory to save checkpoints to.')
  req_grp.add_argument('data_dir', default=None, help='input directory containing the training .tfrecords or images.')
  req_grp.add_argument('eval_data_dir', default=None, type=str, help='directory with the validation data.')

  # Training, logging and checkpointing parameters
  train_grp = parser.add_argument_group('training')
  train_grp.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate for the stochastic gradient update.')
  train_grp.add_argument('--log_interval', type=int, default=1, help='interval between log messages (in s).')
  train_grp.add_argument('--summary_interval', type=int, default=120, help='interval between tensorboard summaries (in s)')
  train_grp.add_argument('--checkpoint_interval', type=int, default=600, help='interval between model checkpoints (in s)')
  train_grp.add_argument('--eval_interval', type=int, default=3600, help='interval between evaluations (in s)')

  # Debug and perf profiling
  debug_grp = parser.add_argument_group('debug and profiling')
  debug_grp.add_argument('--profiling', dest='profiling', action='store_true', help='outputs a profiling trace.')
  debug_grp.add_argument('--noprofiling', dest='profiling', action='store_false')

  # Data pipeline and data augmentation
  data_grp = parser.add_argument_group('data pipeline')
  data_grp.add_argument('--batch_size', default=16, type=int, help='size of a batch for each gradient update.')
  data_grp.add_argument('--shuffle', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
  data_grp.add_argument('--data_threads', default=2, help='number of threads to load and enqueue samples.')
  data_grp.add_argument('--rotate', dest="rotate", action="store_true", help='rotate data augmentation.')
  data_grp.add_argument('--norotate', dest="rotate", action="store_false")
  data_grp.add_argument('--flipud', dest="flipud", action="store_true", help='flip up/down data augmentation.')
  data_grp.add_argument('--noflipud', dest="flipud", action="store_false")
  data_grp.add_argument('--fliplr', dest="fliplr", action="store_true", help='flip left/right data augmentation.')
  data_grp.add_argument('--nofliplr', dest="fliplr", action="store_false")
  data_grp.add_argument('--random_crop', dest="random_crop", action="store_true", help='random crop data augmentation.')
  data_grp.add_argument('--norandom_crop', dest="random_crop", action="store_false")

  # Model parameters
  model_grp = parser.add_argument_group('model_params')
  model_grp.add_argument('--model_name', default=models.__all__[0], type=str, help='classname of the model to use.', choices=models.__all__)
  model_grp.add_argument('--data_pipeline', default=dp.__all__[0], help='classname of the data pipeline to use.', choices=dp.__all__)
  model_grp.add_argument('--net_input_size', default=256, type=int, help="size of the network's lowres image input.")
  model_grp.add_argument('--output_resolution', default=[512, 512], type=int, nargs=2, help='resolution of the output image.')
  model_grp.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='normalize batches. If False, uses the moving averages.')
  model_grp.add_argument('--nobatch_norm', dest='batch_norm', action='store_false')
  model_grp.add_argument('--channel_multiplier', default=1, type=int,  help='Factor to control net throughput (number of intermediate channels).')
  model_grp.add_argument('--guide_complexity', default=16, type=int,  help='Control complexity of the guide network.')
  model_grp.add_argument('--max_iter', default=100000, type=int, help='max iteration of training')
  model_grp.add_argument('--lr', dest='learning_rate', default=0.1, type=float, help='initial learning rate')
  model_grp.add_argument('--momentum', default=0.9, type=float,help='momentum')
  model_grp.add_argument('--wd', dest='weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
  model_grp.add_argument('--seed', default=1, type=int, help='random seed (default: 1)')
  model_grp.add_argument('--print_freq', default=20, type=int, help='print frequency (default: 20)')
  # Bilateral grid parameters
  model_grp.add_argument('--luma_bins', default=8, type=int,  help='Number of BGU bins for the luminance.')
  model_grp.add_argument('--spatial_bin', default=16, type=int,  help='Size of the spatial BGU bins (pixels).')

  parser.set_defaults(
      profiling=False,
      flipud=False,
      fliplr=False,
      rotate=False,
      random_crop=True,
      batch_norm=False)
  # ----------------------------------------------------------------------------
  # pylint: enable=line-too-long

  args = parser.parse_args()

  model_params = {}
  for a in model_grp._group_actions:
    model_params[a.dest] = getattr(args, a.dest, None)

  data_params = {}
  for a in data_grp._group_actions:
    data_params[a.dest] = getattr(args, a.dest, None)

  main(args, model_params, data_params)
