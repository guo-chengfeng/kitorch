from ..functional.conv_torch import conv_transpose2d
from ..utils import rand, rand_like
import unittest
import torch
import time
from . import check, to_torch, Timer
import numpy as np


def main_conv(input, weight, bias=None, stride=None, padding=None, output_padding=None):
    start = time.time()
    out = conv_transpose2d(input, weight, stride=stride, padding=padding, output_padding=output_padding)
    g = rand_like(out)
    out.backward(g)
    Timer.show_time((time.time() - start), "Numpy conv2d")

    t_input, t_weight, v = to_torch([input, weight, g])
    start = time.time()
    t_out = torch.conv_transpose2d(t_input, t_weight, bias=None, stride=stride,
                                   padding=padding, output_padding=output_padding)
    t_out.backward(v)

    Timer.show_time((time.time() - start), "torch conv2d")

    check(out, t_out, grad=False, prefix="out", print_max=True)
    check(input, t_input, prefix="input grad", print_max=True)
    check(weight, t_weight, prefix="weight grad", print_max=True)


class Conv2dTansposedTest(unittest.TestCase):
    def test_1(self):
        print('\n')
        batch_size = 128
        in_channel = 10
        iH, iW = (32, 32)
        out_channel = 20
        kH, kW = (5, 5)
        padding = (2, 1)
        stride = (2, 3)
        output_padding = (1, 2)
        input = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(in_channel, out_channel, kH, kW, requires_grad=True)
        main_conv(input, weight, padding=padding, stride=stride, output_padding=output_padding)


if __name__ == '__main__':
    unittest.main()
