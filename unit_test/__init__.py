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
from ..utils import Timer


def to_torch(tensor_list):
    torch_tensor = []
    for tensor in tensor_list:
        x = torch.from_numpy(tensor.numpy())
        x.requires_grad = True
        torch_tensor.append(x)
    return torch_tensor


def check(a, b, eps=1e-8, grad=True, prefix=None, print_max=False):
    if prefix:
        print(prefix, end=' ')

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

    diff = np.abs(grad_1 - grad_0) / (grad_0+eps)
    result = diff < eps
    if print_max:
        print('max relative diff ', diff.max())

    assert result.prod(), ("TEST FAILED", "eps=", eps, "diff=", diff.max(),grad_0)


timer = Timer()
