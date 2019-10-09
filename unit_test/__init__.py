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
def to_torch(tensor_list):
    torch_tensor = []
    for tensor in tensor_list:
        x = torch.from_numpy(tensor.numpy())
        x.requires_grad = True
        torch_tensor.append(x)
    return torch_tensor


def check(a, b,eps=1e-8):
    grad_0 = a.grad.numpy()
    grad_1 = b.grad.numpy()
    result = np.abs(grad_1 - grad_0) < eps
    assert result.prod(), ("TEST FAILED",
                           grad_0,
                           grad_1)
