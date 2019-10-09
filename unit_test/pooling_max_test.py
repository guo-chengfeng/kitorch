from ..functional.pooling import maxpool2d
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

def main_pooling(inputs,kernel_size,padding=None,elapsed_time=False):
    start = time.time()

    out = maxpool2d(inputs,kernel_size,padding=padding)
    out_grad = rand_like(out)
    out.backward(out_grad)

    if elapsed_time:
        show_time("Numpy maxpool2d elapsed time:",(time.time()-start))

    t_inputs = torch.from_numpy(inputs.numpy())
    t_out_grad = torch.from_numpy(out_grad.numpy())
    t_inputs.requires_grad = True
    if padding:
        MaxPool2d = torch.nn.MaxPool2d(kernel_size,padding=padding)
    else:
        MaxPool2d = torch.nn.MaxPool2d(kernel_size)

    start = time.time()
    t_out = MaxPool2d(t_inputs)
    t_out.backward(t_out_grad)

    if elapsed_time:
        show_time("Torch maxpool2d elapsed time:",(time.time()-start))

    inputs_grad_cmp = abs(inputs.grad.data - t_inputs.grad.numpy()) < 1e-8
    assert inputs_grad_cmp.prod() == 1, "测试通过"

class Maxpool2dTest(unittest.TestCase):
    def test_1(self):
        batch_size = 10
        in_channel = 2
        iH,iW = (20,20)
        kernel_size = (2,2)
        inputs = rand(batch_size,in_channel,iH,iW,requires_grad=True)
        main_pooling(inputs,kernel_size)


    def test_2(self):
        batch_size = 10
        in_channel = 2
        iH,iW = (20,21)
        kernel_size = (2,3)
        inputs = rand(batch_size,in_channel,iH,iW,requires_grad=True)
        main_pooling(inputs,kernel_size)


    def test_3(self):
        batch_size = 10
        in_channel = 2
        iH,iW = (20,20)
        kernel_size = (4,2)
        inputs = rand(batch_size,in_channel,iH,iW,requires_grad=True)
        main_pooling(inputs,kernel_size)

    def test_4(self):
        batch_size = 10
        in_channel = 2
        iH,iW = (18,18)
        kernel_size = (4,2)
        padding = (1,1)
        inputs = rand(batch_size,in_channel,iH,iW,requires_grad=True)
        main_pooling(inputs,kernel_size,padding=padding)

    def test_5(self):
        batch_size = 10
        in_channel = 2
        iH,iW = (20,19)
        kernel_size = (2,3)
        padding = (0,1)
        inputs = rand(batch_size,in_channel,iH,iW,requires_grad=True)
        main_pooling(inputs,kernel_size,padding=padding)


    def test_6(self):
        batch_size = 128
        in_channel = 2
        iH,iW = (34,34)
        kernel_size = (2,3)
        padding = (0,1)
        inputs = rand(batch_size,in_channel,iH,iW,requires_grad=True)
        main_pooling(inputs,kernel_size,padding=padding,elapsed_time=True)


if __name__ == '__main__':
    unittest.main()

