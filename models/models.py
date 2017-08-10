import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet, model_urls

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

class PSPNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, **kwargs):
        super(PSPNet, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            parameters =  model_zoo.load_url(model_urls['resnet18'])
            self.model.load_state_dict(parameters)
        del self.model.avgpool
        del self.model.fc
        self.num_classes = num_classes

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

        self.upscale_factor = 8

        self.upsample_feature = nn.Sequential(
            nn.BatchNorm2d(512+128*4, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Conv2d(512+128*4, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Conv2d(512, self.upscale_factor * self.upscale_factor * num_classes, 3, padding=1),
        )

    def forward(self, input):
        size = input.size()
        # print('x', x.size())
        x = self.model.conv1(input)
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

        feature = self.upsample_feature(feature)
        pixel =  F.pixel_shuffle(feature, self.upscale_factor)

        return F.log_softmax(pixel)
