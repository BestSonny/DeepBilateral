import random
from PIL import Image
from builtins import object
import torchsample
import torch.utils.data as data
import torchvision.transforms as transforms


__all__ = [
  'ImageFolderDataSet',
  ]

class BaseDataLoader(object):
    def __init__(self):
        pass

    def initialize(self, args):
        self.args = args
        pass

    def load_data(self):
        return None

def CreateDataset(args):
    dataset = None
    if args.data_pipeline == 'ImageFolderDataSet':
        from datasets import ImageFolderDataSet
        dataset = ImageFolderDataSet()
    else:
        raise ValueError("Dataset [%s] not recognized." % args.dataset_mode)
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(args)
    return dataset

class DataLoader(BaseDataLoader):
    def initialize(self, args, **kwargs):
        BaseDataLoader.initialize(self, args)

        self.dataset = CreateDataset(args)
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=args.data_params['batch_size'],
            shuffle=args.data_params['shuffle'],
            num_workers=args.data_params['data_threads'],
            **kwargs)

    def name(self):
        return 'DataLoader'

    def load_dataset(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.args.max_dataset_size)
