import os
import os.path as osp
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

IMAGE_EXTENSION = [
    '.jpg',
]

def default_loader(path):
    return Image.open(path).convert('RGB')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def filter_image(filename):
    return any(filename.endswith(extension) for extension in IMAGE_EXTENSION)

def make_dataset(dirs):
    images = []
    for dir in dirs.split(':'):
        assert osp.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname) and filter_image(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, args):
        pass


class ImageFolderDataSet(BaseDataset):

    def __init__(self):
        super(ImageFolderDataSet, self).__init__()

    def initialize(self, args):
        self.args = args
        imgs = make_dataset(self.args.path)

        normalize = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        image_scale = transforms.Scale(size=[self.args.model_params['output_resolution'][1], self.args.model_params['output_resolution'][0]], interpolation=Image.BILINEAR)
        target_scale = transforms.Scale(size=[self.args.model_params['output_resolution'][1], self.args.model_params['output_resolution'][0]], interpolation=Image.NEAREST)
        binary_transform = transforms.Lambda(lambda x: x.gt(0.5))
        transform = transforms.Compose([
            image_scale,
            transforms.ToTensor(),
            ])
        target_transform = transforms.Compose([
            target_scale,
            transforms.ToTensor(),
            binary_transform,
            ])
        if self.args.is_train == False:
            transform = transforms.Compose([
                image_scale,
                transforms.ToTensor(),
                normalize,
                ])
            target_transform = transforms.Compose([
                target_scale,
                transforms.ToTensor(),
                binary_transform,
                ])

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

    def __len__(self):
        return len(self.imgs)

    def name(self):
        return 'ImageFolderDataSet'

    def __getitem__(self, index):
        img_file = self.imgs[index]
        img = self.loader(img_file)

        label_file = img_file.replace('.jpg','.mask.png')
        label = Image.open(label_file).convert('P')
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        random.seed(seed) # apply this seed to img tranfsorms
        if self.transform is not None:
            img = self.transform(img)
        random.seed(seed) # apply this seed to label tranfsorms
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label
