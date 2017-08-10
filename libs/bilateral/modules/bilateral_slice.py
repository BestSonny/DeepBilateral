import torch
from torch.nn.modules.module import Module
from bilateral.functions.bilateral_slice_func import BilateralSliceFunc, BilateralSliceApplyFunc
from torch.autograd import Variable, Function

class BilateralSlice(Module):
    def __init__(self):
        super(BilateralSlice, self).__init__()

    def forward(self, grid, guide):
        return BilateralSliceFunc()(grid, guide)


class BilateralSliceApply(Module):
    def __init__(self, has_offset=True):
        super(BilateralSliceApply, self).__init__()
        self.has_offset = has_offset

    def forward(self, grid, guide, input):
        if self.has_offset:
            return BilateralSliceApplyFunc()(grid, guide, input, Variable(input.data.new(1)))
        else:
            return BilateralSliceApplyFunc()(grid, guide, input, Variable(input.data.new(2)))
