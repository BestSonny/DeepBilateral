import os
import sys
import os.path
import unittest
import tempfile
import torch
import warnings
import random
import numpy as np
import skimage.color as skcolor
from gradcheck import *
from torch.autograd.gradcheck import gradgradcheck
from torch.autograd._functions import *
from torch.autograd import Variable, Function
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.bilateral_slice import BilateralSlice, BilateralSliceApply

SEED = 0
SEED_SET = 0

class TestCase(unittest.TestCase):
    precision = 1e-5

    def setUp(self):
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    def assertTensorsSlowEqual(self, x, y, prec=None, message=''):
        max_err = 0
        self.assertEqual(x.size(), y.size())
        for index in iter_indices(x):
            max_err = max(max_err, abs(x[index] - y[index]))
        self.assertLessEqual(max_err, prec, message)

    def safeCoalesce(self, t):
        tc = t.coalesce()

        value_map = {}
        for idx, val in zip(t._indices().t(), t._values()):
            idx_tup = tuple(idx)
            if idx_tup in value_map:
                value_map[idx_tup] += val
            else:
                value_map[idx_tup] = val.clone() if torch.is_tensor(val) else val

        new_indices = sorted(list(value_map.keys()))
        new_values = [value_map[idx] for idx in new_indices]
        if t._values().ndimension() < 2:
            new_values = t._values().new(new_values)
        else:
            new_values = torch.stack(new_values)

        new_indices = t._indices().new(new_indices).t()
        tg = t.new(new_indices, new_values, t.size())

        self.assertEqual(tc._indices(), tg._indices())
        self.assertEqual(tc._values(), tg._values())

        return tg

    def unwrapVariables(self, x, y):
        if isinstance(x, Variable) and isinstance(y, Variable):
            return x.data, y.data
        elif isinstance(x, Variable) or isinstance(y, Variable):
            raise AssertionError("cannot compare {} and {}".format(type(x), type(y)))
        return x, y

    def assertEqual(self, x, y, prec=None, message=''):
        if prec is None:
            prec = self.precision

        x, y = self.unwrapVariables(x, y)

        if torch.is_tensor(x) and torch.is_tensor(y):
            def assertTensorsEqual(a, b):
                super(TestCase, self).assertEqual(a.size(), b.size())
                if a.numel() > 0:
                    b = b.type_as(a)
                    b = b.cuda(device=a.get_device()) if a.is_cuda else b.cpu()
                    # check that NaNs are in the same locations
                    nan_mask = a != a
                    self.assertTrue(torch.equal(nan_mask, b != b))
                    diff = a - b
                    diff[nan_mask] = 0
                    if diff.is_signed():
                        diff = diff.abs()
                    max_err = diff.max()
                    self.assertLessEqual(max_err, prec, message)
            self.assertEqual(x.is_sparse, y.is_sparse, message)
            if x.is_sparse:
                x = self.safeCoalesce(x)
                y = self.safeCoalesce(y)
                assertTensorsEqual(x._indices(), y._indices())
                assertTensorsEqual(x._values(), y._values())
            else:
                assertTensorsEqual(x, y)
        elif type(x) == str and type(y) == str:
            super(TestCase, self).assertEqual(x, y)
        elif type(x) == set and type(y) == set:
            super(TestCase, self).assertEqual(x, y)
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertEqual(len(x), len(y))
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec, message)
        else:
            try:
                self.assertLessEqual(abs(x - y), prec, message)
                return
            except:
                pass
            super(TestCase, self).assertEqual(x, y, message)

    def assertNotEqual(self, x, y, prec=None, message=''):
        if prec is None:
            prec = self.precision

        x, y = self.unwrapVariables(x, y)

        if torch.is_tensor(x) and torch.is_tensor(y):
            if x.size() != y.size():
                super(TestCase, self).assertNotEqual(x.size(), y.size())
            self.assertGreater(x.numel(), 0)
            y = y.type_as(x)
            y = y.cuda(device=x.get_device()) if x.is_cuda else y.cpu()
            nan_mask = x != x
            if torch.equal(nan_mask, y != y):
                diff = x - y
                if diff.is_signed():
                    diff = diff.abs()
                diff[nan_mask] = 0
                max_err = diff.max()
                self.assertGreaterEqual(max_err, prec, message)
        elif type(x) == str and type(y) == str:
            super(TestCase, self).assertNotEqual(x, y)
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertNotEqual(x, y)
        else:
            try:
                self.assertGreaterEqual(abs(x - y), prec, message)
                return
            except:
                pass
            super(TestCase, self).assertNotEqual(x, y, message)

    def assertObjectIn(self, obj, iterable):
        for elem in iterable:
            if id(obj) == id(elem):
                return
        raise AssertionError("object not found in iterable")

    if sys.version_info < (3, 2):
        # assertRaisesRegexp renamed assertRaisesRegex in 3.2
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

# class BilateralSliceTest(TestCase):
#
#     def setUp(self):
#         self.net = BilateralSlice().cuda()
#
#     def run_bilateral_slice(self, grid_data, guide_data):
#         self.net.zero_grad()
#         return self.net(grid_data, guide_data)
#
#     def test_shape_is_correct(self):
#         N = random.randint(1, 8)
#         H = random.randint(1, 8)
#         W = random.randint(1, 8)
#         C = random.randint(1, 8)
#         D = random.randint(1, 8)
#         GH = random.randint(1, 20)
#         GW = random.randint(1, 20)
#         grid_shape = torch.Size([N, H, W, D, C])
#         guide_shape = torch.Size([N, GH, GW])
#         grid_data = Variable(torch.randn(*grid_shape), requires_grad=True).cuda()
#         guide_data = Variable(torch.randn(*guide_shape), requires_grad=True).cuda()
#
#         output_data = self.run_bilateral_slice(grid_data, guide_data)
#         self.assertTrue(output_data.size() == torch.Size([N, GH, GW, C]))
#
#     def test_interpolate(self):
#         batch_size = 3
#         h = 3
#         w = 4
#         d = 3
#         grid_shape = [batch_size, h, w, d, 1]
#         grid_data = np.zeros(grid_shape).astype(np.float32)
#         grid_data[:, :, :, 1 :] = 1.0
#         grid_data[:, :, :, 2 :] = 2.0
#         grid_data_variable = Variable(torch.from_numpy(grid_data), requires_grad=True).cuda()
#
#         guide_shape = [batch_size, 10, 9]
#         target_shape = [batch_size, 10, 9, 1]
#
#         for val in range(d):
#             target_data = val*np.ones(target_shape)
#             target_data = target_data.astype(np.float32)
#             target_data_variable = Variable(torch.from_numpy(target_data), requires_grad=True).cuda()
#
#             guide_data = (val/(1.0*d))*np.ones(guide_shape).astype(np.float32)
#             guide_data_variable = Variable(torch.from_numpy(guide_data), requires_grad=True).cuda()
#
#             output_data = BilateralSlice()(grid_data_variable, guide_data_variable)
#
#             diff = torch.max((target_data_variable-output_data).abs()).cpu().data.numpy()[0]
#             self.assertLess(diff, 5e-4)
#
#     def test_grid_gradient(self):
#         batch_size = 3
#         h = 8
#         w = 5
#         gh = 6
#         gw = 3
#         d = 7
#         nchans = 4
#         grid_shape = [batch_size, gh, gw, d, nchans]
#         guide_shape = [batch_size, h, w]
#         output_shape = [batch_size, h, w, nchans]
#         grid_data_variable = Variable(torch.rand(batch_size, gh, gw, d, nchans).cuda(), requires_grad=True)
#         guide_data_variable = Variable(torch.rand(batch_size, h, w).cuda(), requires_grad=True)
#
#         def _as_tuple(x):
#             if isinstance(x, tuple):
#                 return x
#             elif isinstance(x, list):
#                 return tuple(x)
#             else:
#                 return x
#
#         inputs = (grid_data_variable, guide_data_variable)
#         func = BilateralSlice().cuda()
#         output = func(*inputs)
#         output = _as_tuple(output)
#
#         for i, o in enumerate(output):
#             if not o.requires_grad:
#                 continue
#
#             def fn(input):
#                 return _as_tuple(func(*input))[i].data
#
#             analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(_as_tuple(inputs), o)
#             numerical = get_numerical_jacobian(fn, inputs, inputs)
#
#             grid_numerical, guide_numerical = numerical
#             grid_analytical, guide_analytical = analytical
#             # print grid_numerical, guide_numerical, grid_analytical, guide_analytical
#             for a, n in zip(grid_analytical, grid_numerical):
#                 self.assertLess((a - n).abs().max(), 1e-4)
#
#     def test_guide_gradient(self):
#         batch_size = 2
#         h = 7
#         w = 8
#         d = 5
#         gh = 3
#         gw = 4
#         nchans = 2
#         grid_shape = [batch_size, gh, gw, d, nchans]
#         guide_shape = [batch_size, h, w]
#         output_shape = [batch_size, h, w, nchans]
#         grid_data_variable = Variable(torch.rand(batch_size, gh, gw, d, nchans).cuda(), requires_grad=True)
#         guide_data_variable = Variable(torch.rand(batch_size, h, w).cuda(), requires_grad=True)
#
#         def _as_tuple(x):
#             if isinstance(x, tuple):
#                 return x
#             elif isinstance(x, list):
#                 return tuple(x)
#             else:
#                 return x
#
#         inputs = (grid_data_variable, guide_data_variable)
#         func = BilateralSlice().cuda()
#         output = func(*inputs)
#         output = _as_tuple(output)
#
#         for i, o in enumerate(output):
#             if not o.requires_grad:
#                 continue
#
#             def fn(input):
#                 return _as_tuple(func(*input))[i].data
#
#             analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(_as_tuple(inputs), o)
#             numerical = get_numerical_jacobian(fn, inputs, inputs, eps=1e-4)
#
#             grid_numerical, guide_numerical = numerical
#             grid_analytical, guide_analytical = analytical
#             # print grid_numerical, guide_numerical, grid_analytical, guide_analytical
#             thresh = 5e-3
#             diff = (guide_analytical - guide_numerical).abs().numpy()
#             x, y = np.where(diff>thresh)
#             for i in range(len(x)):
#               in_x = x[i] % w
#               in_y = x[i] / w
#               out_c = y[i] % nchans
#               out_x = (y[i]/nchans) % w
#               out_y = (y[i]/nchans) / w
#               print "output ({},{},{}) - input ({},{})\n  guide: {:f}\n  theoretical: {:f}\n  numerical: {:f}".format(
#                   out_y, out_x, out_c, in_y, in_x, np.ravel(guide_data_variable.data.cpu().numpy())[x[i]], guide_analytical[x[i], y[i]], guide_numerical[x[i],y[i]])
#
#             print len(x), 'of', len(np.ravel(diff)), 'errors'
#
#             self.assertLess(np.amax(diff), thresh)



class BilateralSliceApplyTest(TestCase):

    def setUp(self):
        self.net = BilateralSliceApply().cuda()
        self.net_no_offset = BilateralSliceApply(has_offset=False).cuda()

    def run_bilateral_slice_apply_no_offset(self, grid_data, guide_data, input):
        self.net_no_offset.zero_grad()
        return self.net_no_offset(grid_data, guide_data, input)

    def run_bilateral_slice_apply(self, grid_data, guide_data, input):
        self.net.zero_grad()
        return self.net(grid_data, guide_data, input)

    def test_shape_is_correct(self):

        N = random.randint(1, 8)
        H = random.randint(1, 8)
        W = random.randint(1, 8)
        D = random.randint(1, 8)
        GH = random.randint(1, 20)
        GW = random.randint(1, 20)
        grid_shape = torch.Size([N, H, W, D, 12])
        guide_shape = torch.Size([N, GH, GW])
        input_shape = torch.Size([N, GH, GW, 3])

        grid_data = Variable(torch.randn(*grid_shape).cuda(), requires_grad=True)
        guide_data = Variable(torch.randn(*guide_shape).cuda(), requires_grad=True)
        input_data = Variable(torch.randn(*input_shape).cuda(), requires_grad=True)
        has_offset_true = Variable(torch.ones(1).cuda(), requires_grad=True)
        has_offset_false = Variable(torch.zeros(1).cuda(), requires_grad=True)


        output_data = self.run_bilateral_slice_apply(grid_data, guide_data, input_data)
        output_data_no_offset = self.run_bilateral_slice_apply_no_offset(grid_data, guide_data, input_data)
        print output_data_no_offset
        self.assertTrue(output_data.size() == torch.Size([N, GH, GW, 3]))
        self.assertTrue(output_data_no_offset.size() == torch.Size([N, GH, GW, 4]))

    def test_interpolate(self):
        pass

    # def test_grid_gradient(self):
    #     batch_size = 3
    #     h = 8
    #     w = 5
    #     gh = 6
    #     gw = 3
    #     d = 7
    #     nchans = 4
    #     grid_shape = [batch_size, gh, gw, d, nchans]
    #     guide_shape = [batch_size, h, w]
    #     output_shape = [batch_size, h, w, nchans]
    #     grid_data_variable = Variable(torch.rand(batch_size, gh, gw, d, nchans).cuda(), requires_grad=True)
    #     guide_data_variable = Variable(torch.rand(batch_size, h, w).cuda(), requires_grad=True)
    #
    #     def _as_tuple(x):
    #         if isinstance(x, tuple):
    #             return x
    #         elif isinstance(x, list):
    #             return tuple(x)
    #         else:
    #             return x
    #
    #     inputs = (grid_data_variable, guide_data_variable)
    #     func = BilateralSlice().cuda()
    #     output = func(*inputs)
    #     output = _as_tuple(output)
    #
    #     for i, o in enumerate(output):
    #         if not o.requires_grad:
    #             continue
    #
    #         def fn(input):
    #             return _as_tuple(func(*input))[i].data
    #
    #         analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(_as_tuple(inputs), o)
    #         numerical = get_numerical_jacobian(fn, inputs, inputs)
    #
    #         grid_numerical, guide_numerical = numerical
    #         grid_analytical, guide_analytical = analytical
    #         # print grid_numerical, guide_numerical, grid_analytical, guide_analytical
    #         for a, n in zip(grid_analytical, grid_numerical):
    #             self.assertLess((a - n).abs().max(), 1e-4)
    #
    # def test_guide_gradient(self):
    #     batch_size = 2
    #     h = 7
    #     w = 8
    #     d = 5
    #     gh = 3
    #     gw = 4
    #     nchans = 2
    #     grid_shape = [batch_size, gh, gw, d, nchans]
    #     guide_shape = [batch_size, h, w]
    #     output_shape = [batch_size, h, w, nchans]
    #     grid_data_variable = Variable(torch.rand(batch_size, gh, gw, d, nchans).cuda(), requires_grad=True)
    #     guide_data_variable = Variable(torch.rand(batch_size, h, w).cuda(), requires_grad=True)
    #
    #     def _as_tuple(x):
    #         if isinstance(x, tuple):
    #             return x
    #         elif isinstance(x, list):
    #             return tuple(x)
    #         else:
    #             return x
    #
    #     inputs = (grid_data_variable, guide_data_variable)
    #     func = BilateralSlice().cuda()
    #     output = func(*inputs)
    #     output = _as_tuple(output)
    #
    #     for i, o in enumerate(output):
    #         if not o.requires_grad:
    #             continue
    #
    #         def fn(input):
    #             return _as_tuple(func(*input))[i].data
    #
    #         analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(_as_tuple(inputs), o)
    #         numerical = get_numerical_jacobian(fn, inputs, inputs, eps=1e-4)
    #
    #         grid_numerical, guide_numerical = numerical
    #         grid_analytical, guide_analytical = analytical
    #         # print grid_numerical, guide_numerical, grid_analytical, guide_analytical
    #         thresh = 5e-3
    #         diff = (guide_analytical - guide_numerical).abs().numpy()
    #         x, y = np.where(diff>thresh)
    #         for i in range(len(x)):
    #           in_x = x[i] % w
    #           in_y = x[i] / w
    #           out_c = y[i] % nchans
    #           out_x = (y[i]/nchans) % w
    #           out_y = (y[i]/nchans) / w
    #           print "output ({},{},{}) - input ({},{})\n  guide: {:f}\n  theoretical: {:f}\n  numerical: {:f}".format(
    #               out_y, out_x, out_c, in_y, in_x, np.ravel(guide_data_variable.data.cpu().numpy())[x[i]], guide_analytical[x[i], y[i]], guide_numerical[x[i],y[i]])
    #
    #         print len(x), 'of', len(np.ravel(diff)), 'errors'
    #
    #         self.assertLess(np.amax(diff), thresh)

if __name__ == '__main__':
    unittest.main()
