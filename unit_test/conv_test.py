from ..functional.conv import conv2d
from ..utils import rand, rand_like
import unittest
import torch
import time
from . import check, to_torch, Timer


def main_conv(input, weight, bias=None, stride=None, padding=None):
    start = time.time()
    out = conv2d(input, weight, bias=bias, stride=stride, padding=padding)
    g = rand_like(out)

    out.backward(g)
    Timer.show_time((time.time() - start), "Numpy conv2d")

    if bias:
        t_input, t_weight, t_bias, v = to_torch([input, weight, bias, g])
        start = time.time()
        t_out = torch.conv2d(t_input, t_weight, bias=t_bias, stride=stride, padding=padding)
        t_out.backward(v)
    else:
        t_input, t_weight, v = to_torch([input, weight, g])
        start = time.time()
        t_out = torch.conv2d(t_input, t_weight, stride=stride, padding=padding)
        t_out.backward(v)

    Timer.show_time((time.time() - start), "torch conv2d")
    check(out, t_out, grad=False, prefix="out", print_max=True)

    check(input, t_input)
    check(weight, t_weight)
    if bias:
        check(bias, t_bias)


class Conv2dTest(unittest.TestCase):
    def test_1(self):
        print("\n test_1: 卷积核5X5")
        batch_size = 512
        in_channel = 10
        iH, iW = (32, 32)
        out_channel = 20
        kH, kW = (5, 5)
        padding = (0, 0)
        stride = (3,3)
        input = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
        bias = rand(out_channel, requires_grad=True)
        main_conv(input, weight, bias=bias, padding=padding, stride=stride)

    def test_2(self):
        print("\n test_2: 卷积核5X5")
        batch_size = 128
        in_channel = 10
        iH, iW = (28, 28)
        out_channel = 20
        kH, kW = (4, 4)
        padding = (1, 2)
        stride = (1, 1)
        inputs = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
        bias = rand(out_channel, requires_grad=True)
        main_conv(inputs, weight, bias=bias, padding=padding, stride=stride)

    def test_3(self):
        print("\n test_3: 卷积核5X5")
        batch_size = 128
        in_channel = 10
        iH, iW = (28, 28)
        out_channel = 20
        kH, kW = (4, 4)
        padding = (0, 0)
        stride = (2, 2)
        inputs = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
        bias = rand(out_channel, requires_grad=True)
        main_conv(inputs, weight, bias=bias, padding=padding, stride=stride)

    def test_4(self):
        print("\n test_4: 卷积核5X5")
        batch_size = 128
        in_channel = 10
        iH, iW = (28, 28)
        out_channel = 20
        kH, kW = (4, 4)
        padding = (1, 2)
        stride = (2, 2)
        inputs = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
        bias = rand(out_channel, requires_grad=True)
        main_conv(inputs, weight, bias=bias, padding=padding, stride=stride)


if __name__ == '__main__':
    unittest.main()
