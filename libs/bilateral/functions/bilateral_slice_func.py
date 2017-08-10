import torch
from torch.autograd import Function
from _ext import bilateral
from torch.autograd import Variable

class BilateralSliceFunc(Function):

    def __init__(self):
        super(BilateralSliceFunc, self).__init__()

    def forward(self, grid, guide):

        assert(grid.is_contiguous() == True)
        assert(guide.is_contiguous() == True)

        self.save_for_backward(grid, guide)

        batch_size, height, width, depth, channels = grid.size()
        batch_size, guide_height, guide_width = guide.size()

        output = grid.new(batch_size, guide_height, guide_width, channels).zero_()

        assert output.is_cuda == True, "current only support gpu version"
        bilateral.bilateral_slice_forward_cuda(grid, guide, output)

        return output

    def backward(self, grad_output):

        assert(grad_output.is_contiguous() == True)

        grid, guide = self.saved_tensors
        batch_size, height, width, depth, channels = grid.size()
        batch_size, guide_height, guide_width = guide.size()

        grad_grid = grid.new().resize_as_(grid).zero_()
        grad_guide = guide.new().resize_as_(guide).zero_()

        assert grid.is_cuda == True, "current only support gpu version"
        bilateral.bilateral_slice_backward_cuda(grid, guide, grad_output,
                                                grad_grid, grad_guide)
        return grad_grid, grad_guide



class BilateralSliceApplyFunc(Function):

    def __init__(self):
        super(BilateralSliceApplyFunc, self).__init__()

    def forward(self, grid, guide, input, has_offset):

        assert(grid.is_contiguous() == True)
        assert(guide.is_contiguous() == True)
        assert(input.is_contiguous() == True)

        self.save_for_backward(grid, guide, input)

        batch_size, guide_height, guide_width, input_channels = input.size()
        batch_size, height, width, depth, nchannels = grid.size()

        output = grid.new()

        # if has_offset.size(0) == 1:
        #     output_channel = nchannels / (input_channels + 1)
        # else:
        #     output_channel = nchannels / input_channels
        #
        # output = grid.new(batch_size, guide_height, guide_width, output_channel).zero_()

        assert output.is_cuda == True, "current only support gpu version"
        bilateral.bilateral_slice_apply_forward_cuda(grid, guide, input, output, has_offset)

        return output

    def backward(self, grad_output):

        # assert(grad_output.is_contiguous() == True)
        #
        # grid, guide = self.saved_tensors
        # batch_size, height, width, depth, channels = grid.size()
        # batch_size, guide_height, guide_width = guide.size()
        #
        # grad_grid = grid.new().resize_as_(grid).zero_()
        # grad_guide = guide.new().resize_as_(guide).zero_()
        #
        # assert grid.is_cuda == True, "current only support gpu version"
        # bilateral.bilateral_slice_backward_cuda(grid, guide, grad_output,
        #                                         grad_grid, grad_guide)
        return None, None, None, None
