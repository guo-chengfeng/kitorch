from ..functional.conv_by_torch import conv2d
from ..utils import rand,rand_like
import unittest
import torch
import time

def show_time(prefix,elapsed_time):
    print()
    if elapsed_time < 0.001:
        print(prefix," %.2f"%(elapsed_time*1000*1000),' us')
    elif elapsed_time < 1:
        print(prefix, " %.2f"%(elapsed_time * 1000), ' ms')
    else:
        print(prefix, " %.2f"%elapsed_time, ' s')

def main_conv(inputs,weight,bias=None,padding=None,elapsed_time=False):
    start = time.time()

    out = conv2d(inputs, weight,bias=bias,padding=padding)
    out_grad = rand_like(out)
    out.backward(out_grad)

    if elapsed_time:
        show_time("Numpy conv2d elapsed time:",(time.time()-start))

    t_inputs = torch.from_numpy(inputs.numpy())
    t_weight = torch.from_numpy(weight.numpy())
    t_out_grad = torch.from_numpy(out_grad.numpy())
    t_inputs.requires_grad = True
    t_weight.requires_grad = True

    if bias:
        t_bias = torch.from_numpy(bias.numpy())
        t_bias.requires_grad = True
    else:
        t_bias = None

    start = time.time()
    if padding:
        t_out = torch.conv2d(t_inputs, t_weight, bias=t_bias,padding=padding)
    else:
        t_out = torch.conv2d(t_inputs, t_weight,bias=t_bias)
    t_out.backward(t_out_grad)
    if elapsed_time:
        show_time("Torch conv2d elapsed time:",(time.time()-start))


    inputs_grad_cmp = abs(inputs.grad.data - t_inputs.grad.numpy()) < 1e-8
    weight_grad_cmp = abs(weight.grad.data - t_weight.grad.numpy()) < 1e-8
    assert inputs_grad_cmp.prod() == 1, "测试通过"
    assert weight_grad_cmp.prod() == 1, "测试通过"

    out_cmp = abs(out.data - t_out.data.numpy()) < 1e-8
    assert out_cmp.prod() == 1, "测试通过"


    if bias:
        bias_grad_cmp = abs(bias.grad.data - t_bias.grad.numpy()) < 1e-8
        assert bias_grad_cmp.prod() == 1, "测试通过"


class Conv2dTest(unittest.TestCase):
    def test_1(self):
        batch_size = 10
        in_channel = 2
        iH,iW = (17,19)
        out_channel = 4
        kH,kW = (5,6)
        inputs = rand(batch_size,in_channel,iH,iW,requires_grad=True)
        weight = rand(out_channel,in_channel,kH,kW,requires_grad=True)
        main_conv(inputs,weight)

    def test_2(self):
        batch_size = 10
        in_channel = 2
        iH, iW = (17, 19)
        out_channel = 4
        kH, kW = (4, 5)
        inputs = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
        bias = rand(out_channel, requires_grad=True)
        main_conv(inputs, weight,bias=bias)

    def test_3(self):
        batch_size = 10
        in_channel = 2
        iH, iW = (17, 19)
        out_channel = 4
        kH, kW = (4, 5)
        padding = (2, 2)
        inputs = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
        bias = rand(out_channel, requires_grad=True)
        main_conv(inputs, weight,bias=bias,padding=padding)

    def test_4(self):
        batch_size = 10
        in_channel = 2
        iH, iW = (17, 19)
        out_channel = 14
        kH, kW = (4, 5)
        padding = (2, 2)
        inputs = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
        bias = rand(out_channel, requires_grad=True)
        main_conv(inputs, weight,bias=bias,padding=padding)

    def test_5(self):
        print("\n测试效率，卷积核5X5")
        batch_size = 128
        in_channel = 2
        iH, iW = (28, 28)
        out_channel = 10
        kH, kW = (5, 5)
        inputs = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
        bias = rand(out_channel, requires_grad=True)
        main_conv(inputs, weight,bias=bias,elapsed_time=True)

    def test_6(self):
        print("\n测试效率，卷积核3X3")
        batch_size = 128
        in_channel = 2
        iH, iW = (28, 28)
        out_channel = 10
        kH, kW = (3, 3)
        inputs = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
        bias = rand(out_channel, requires_grad=True)
        main_conv(inputs, weight,bias=bias,elapsed_time=True)

    def test_7(self):
        print("\n测试效率，卷积核3X3")
        batch_size = 10
        in_channel = 2
        iH, iW = (28, 28)
        out_channel = 128
        kH, kW = (3, 3)
        inputs = rand(batch_size, in_channel, iH, iW, requires_grad=True)
        weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
        bias = rand(out_channel, requires_grad=True)
        main_conv(inputs, weight,bias=bias,elapsed_time=True)


if __name__ == '__main__':
    unittest.main()

# batch_size = 128
# in_channel = 3
# iH, iW = (28, 28)
# out_channel = 10
# kH, kW = (5, 5)
# inputs = rand(batch_size, in_channel, iH, iW, requires_grad=True)
# weight = rand(out_channel, in_channel, kH, kW, requires_grad=True)
# bias = rand(out_channel, requires_grad=True)
# out = conv2d(inputs, weight,bias=bias)
# out_grad = rand_like(out)
# out.backward(out_grad)