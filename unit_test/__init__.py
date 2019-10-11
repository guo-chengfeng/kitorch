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


def check(a, b, eps=1e-8):
    grad_0 = a.grad.numpy()
    grad_1 = b.grad.numpy()
    result = np.abs(grad_1 - grad_0) < eps
    assert result.prod(), ("TEST FAILED",
                           grad_0,
                           grad_1)


class Timer:
    def __init__(self):
        self._tic = time.time()
        self._toc = time.time()

    @staticmethod
    def show_time(elapsed_time):
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
