import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet, model_urls
from torch.autograd import Variable
import numpy as np
from layers import *
from modules.bilateral_slice import BilateralSlice, BilateralSliceApply

__all__ = [
  'PSPNet',
]

class PSPDec(nn.Module):
    def __init__(self, in_features, out_features, output_size):
        super(PSPDec, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, 1, bias=False)
        self.bn =  nn.BatchNorm2d(out_features, momentum=.95)
        self.output_size = output_size

    def forward(self, x):
        size = x.size()
        x = F.adaptive_avg_pool2d(x, output_size=self.output_size)
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x,inplace=True)
        x = F.upsample(x, size[2:], mode='bilinear')
        return x

class Identity(nn.Module):
    def __init__(self, in_features, out_features):
        super(Identity, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, 1, bias=False)
        filter_var = Variable(torch.zeros(out_features, in_features, 1, 1))
        init.dirac(filter_var)
        self.conv.weight.data.copy_(filter_var.data)

    def forward(self, x):
        return self.conv(x)

class CurveChannel(nn.Module):
    def __init__(self, npts, in_features):
        super(CurveChannel, self).__init__()
        shifts_ = np.linspace(0, 1, npts, endpoint=False, dtype=np.float32)
        shifts_ = shifts_[:, np.newaxis]
        shifts_ = np.tile(shifts_, (1, in_features))
        slopes_ = np.zeros([npts, in_features], dtype=np.float32)
        slopes_[0, :] = 1.0

        self.shift = Parameter(torch.from_numpy(shifts_))
        self.slopes = Parameter(torch.from_numpy(slopes_))
        self.npts = npts

        self.conv = nn.Conv2d(in_features, 1, 1, bias=True)
        self.conv.weight.data.fill_(1.0/in_features)
        self.conv.bias.data.fill_(0)
        self.hardtanh = nn.Hardtanh(0,1)

    def forward(self, x):
        batch, channel, height, width = x.size()
        expand_x = x.expand(self.npts, batch, channel, height, width).permute(1,0,2,3,4).contiguous()
        expand_shift = self.shift.expand(batch, height, width, self.npts, channel)
        shift = expand_shift.permute(0, 3, 4, 1, 2).contiguous()
        expand_slopes = self.slopes.expand(batch, height, width, self.npts, channel)
        slopes = expand_slopes.permute(0, 3, 4, 1, 2).contiguous()
        x = slopes * F.relu(expand_x-shift)
        x = self.conv(x.sum(1))
        return self.hardtanh(x)


class PSPNet(nn.Module):
    def __init__(self, depth=8, channel=12, pretrained=False, **kwargs):
        super(PSPNet, self).__init__()
        self.model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            self.model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        del self.model.avgpool
        del self.model.fc
        self.depth = depth
        self.channel = channel

        for m in self.model.layer2.modules():
            if isinstance(m, nn.Conv2d):
                    m.stride = 1
        for m in self.model.layer3.modules():
            if isinstance(m, nn.Conv2d):
                    m.stride = 1
        for m in self.model.layer4.modules():
            if isinstance(m, nn.Conv2d):
                    m.stride = 1

        self.layer5a = PSPDec(512, 128, 1)
        self.layer5b = PSPDec(512, 128, 2)
        self.layer5c = PSPDec(512, 128, 4)
        self.layer5d = PSPDec(256, 128, 8)

        self.upscale_factor = 4

        self.grid_feature = nn.Sequential(
            nn.BatchNorm2d(512+128*4, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Conv2d(512+128*4, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Conv2d(512, self.depth * self.channel, 3, padding=1),
        )

        self.guide = nn.Sequential(
            Identity(3,3),
            CurveChannel(16,3),
        )

        self.bilateral_slice = BilateralSlice()
        self.bilateral_slice_apply = BilateralSliceApply(True)
        self.clip = nn.Hardtanh(0,1)

    def forward(self, input):
        batch_size, channel, height, width = input.size()
        lower_input = F.upsample(input, (height/2, width/2), mode='bilinear')

        #guide
        # print('input', input.size())
        guide = self.guide(input)


        # print('x', x.size())
        x = self.model.conv1(lower_input)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        # print('conv1', x.size())
        x = self.model.layer1(x)
        # print('layer1', x.size())
        x = self.model.layer2(x)
        # print('layer2', x.size())
        x = self.model.layer3(x)
        # print('layer3', x.size())
        x = self.model.layer4(x)
        # print('layer4', x.size())

        feature5a =  self.layer5a(x)
        feature5b =  self.layer5b(x)
        feature5c =  self.layer5c(x)
        feature5d =  self.layer5c(x)

        feature = torch.cat([x,
            feature5a,
            feature5b,
            feature5c,
            feature5d,
        ], 1)

        grid = self.grid_feature(feature)
        # grid =  F.pixel_shuffle(feature, self.upscale_factor)
        grid = grid.resize(batch_size, self.depth, self.channel, height/8, width/8)

        trans_grid = grid.permute(0,3,4,1,2).contiguous()

        # print('guide', guide.size())
        guide = guide.resize(batch_size, height, width)
        slice_coeff = self.bilateral_slice(trans_grid, guide)

        # print('slice_coeff', slice_coeff.size())
        transpose_input = input.permute(0,2,3,1).contiguous()

        output = self.bilateral_slice_apply(trans_grid, slice_coeff, transpose_input)

        final_output = output.permute(0,3,1,2).contiguous()

        return final_output
