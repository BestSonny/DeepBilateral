import argparse
import logging
import numpy as np
import os
import re
import tqdm
import setproctitle
from PIL import Image
import cv2
import visdom
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as utils
import torchvision.transforms as transforms
import _init_paths
import models as models
from bilateral.modules.bilateral_slice import BilateralSlice, BilateralSliceApply

logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)

def get_transformer(args):
    normalize = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    image_scale = transforms.Scale(size=[args.output_resolution[1], args.output_resolution[0]], interpolation=Image.BILINEAR)
    transform = transforms.Compose([
        image_scale,
        transforms.ToTensor(),
        normalize
        ])
    return transform

def get_input_list(path):
    regex = re.compile(".*.(png|jpeg|jpg|tif|tiff)")
    if os.path.isdir(path):
        inputs = os.listdir(path)
        inputs = [os.path.join(path, f) for f in inputs if regex.match(f)]
        log.info("Directory input {}, with {} images".format(path, len(inputs)))

    elif os.path.splitext(path)[-1] == ".txt":
        dirname = os.path.dirname(path)
        with open(path, 'r') as fid:
            inputs = [l.strip() for l in fid.readlines()]
            inputs = [os.path.join(dirname, 'input', im) for im in inputs]
            log.info("Filelist input {}, with {} images".format(path, len(inputs)))
    elif regex.match(path):
        inputs = [path]
        log.info("Single input {}".format(path))
    return inputs

def main(args):
    setproctitle.setproctitle('test_run')

    inputs = get_input_list(args.input)
    model = getattr(models, args.model_name)(pretrained=True)
    checkpoint = torch.load(args.checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    transform = get_transformer(args)
    cuda = torch.cuda.is_available()
    torch.manual_seed(111)
    vis = visdom.Visdom()
    if cuda:
        torch.cuda.manual_seed(111)
        model = model.cuda()

    for idx, input_path in enumerate(inputs):
        if args.limit is not None and idx >= args.limit:
            log.info("Stopping at limit {}".format(args.limit))
            break
        image = Image.open(input_path).convert('RGB')
        tensor_image = transform(image)
        if cuda:
            tensor_image = tensor_image.cuda()
        data_var = Variable(tensor_image, volatile=True)
        data_var = data_var.view(-1,data_var.size()[0],data_var.size()[1],data_var.size()[2]).contiguous()
        print data_var.size()
        output = model(data_var)
        output = output.mul(0.5).add(0.5).data.cpu()
        grid = utils.make_grid(output,nrow=1,padding=0)
        print grid.max(), grid.min()
        print data_var.max(), data_var.min()
        visual = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        vis.image(visual.transpose([2,0,1]), opts=dict(title='data'),
                               win=1)
        raw_input()

if __name__ == '__main__':
  # -----------------------------------------------------------------------------
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_dir', default=None, help='path to the saved model variables')
  parser.add_argument('input', default=None, help='path to the validation data')
  parser.add_argument('output', default=None, help='path to save the processed images')
  parser.add_argument('--model_name', default=models.__all__[0], type=str, help='classname of the model to use.', choices=models.__all__)
  parser.add_argument('--output_resolution', default=[512, 512], type=int, nargs=2, help='resolution of the output image.')

  # Optional
  parser.add_argument('--lowres_input', default=None, help='path to the lowres, TF inputs')
  parser.add_argument('--hdrp', dest="hdrp", action="store_true", help='special flag for HDR+ to set proper range')
  parser.add_argument('--nohdrp', dest="hdrp", action="store_false")
  parser.add_argument('--debug', dest="debug", action="store_true", help='If true, dumps debug data on guide and coefficients.')
  parser.add_argument('--limit', type=int, help="limit the number of images processed.")
  parser.set_defaults(hdrp=False, debug=False)
  # pylint: enable=line-too-long
  # -----------------------------------------------------------------------------

  args = parser.parse_args()
  main(args)
