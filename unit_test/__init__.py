#!/usr/bin/python

# encoding: utf-8
"""
@author:  guo_chengfeng
@contact: chf_guo@163.com
@file: __init__.py
@time: 2019/10/8 13:11
@version: 0.1
@desc:
"""
import torch
import numpy as np
import time


def to_torch(tensor_list):
    torch_tensor = []
    for tensor in tensor_list:
        x = torch.from_numpy(tensor.numpy())
        x.requires_grad = True
        torch_tensor.append(x)
    return torch_tensor


def check(a, b, eps=1e-8, grad=True,prefix = None, print_max=False):
    if prefix:
        print(prefix,end=' ')

    if grad:
        grad_0 = a.grad.numpy()
        grad_1 = b.grad.numpy()

    else:
        if isinstance(a, torch.Tensor):
            grad_0 = a.data.numpy()
            grad_1 = b.numpy()
        else:
            grad_0 = a.numpy()
            grad_1 = b.data.numpy()

    diff = np.abs(grad_1 - grad_0) / grad_0
    result = diff < eps
    if print_max:
        print('max relative diff ',diff.max())

    assert result.prod(), ("TEST FAILED","eps=",eps,"diff=",diff.max())



class Timer:
    def __init__(self):
        self._tic = time.time()
        self._toc = time.time()

    @staticmethod
    def show_time(elapsed_time, prefix=None):
        if prefix:
            print(prefix, end=' ')
        if elapsed_time < 0.001:
            print("elapsed time: %.2f" % (elapsed_time * 1000 * 1000), ' us')
        elif elapsed_time < 1:
            print("elapsed time: %.2f" % (elapsed_time * 1000), ' ms')
        else:
            print("elapsed time: %.2f" % elapsed_time, ' s')

    @property
    def tic(self):
        self._tic = time.time()

    @property
    def toc(self):
        self._toc = time.time()
        self.show_time(elapsed_time=self._toc - self._tic)


timer = Timer()
