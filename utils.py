#!/usr/bin/python

# encoding: utf-8
"""
@author:  guo_chengfeng
@contact: chf_guo@163.com
@file: utils.py
@time: 2019/10/8 10:24
@version: 0.1
@desc:
"""
import numpy as np
from typing import List
from .tensor import Edge,Tensor


def CatBackward(grad: 'Tensor', t: 'Tensor', args: List) -> 'Tensor':
    axis, index = args
    length = t.shape[axis]
    indices = [i for i in range(index * length, (index + 1) * length)]
    return Tensor(grad.data.take(indices, axis) * np.ones_like(t.data))


def cat(ts, axis=0):
    assert isinstance(ts, (list, tuple)), \
        "concat(): argument 'tensors' must be tuple or list of Tensors, not Tensor"

    _data = [t.data for t in ts]
    data = np.concatenate(_data, axis=axis)
    requires_grad = False
    depends_on = []
    grad_on = CatBackward

    for index, t in enumerate(ts):
        requires_grad = requires_grad or t.requires_grad
        if t.requires_grad:
            depends_on.append(Edge(t, [axis, index]))

    return Tensor(data, requires_grad, depends_on, grad_on)


def StackBackward(grad: 'Tensor', t: 'Tensor', args: List) -> 'Tensor':
    axis, index = args
    return Tensor(grad.data.take(index, axis))


def stack(ts, axis=0):
    assert isinstance(ts, (list, tuple)), \
        "stack(): argument 'tensors' must be tuple or list of Tensors, not Tensor"

    _data = [t.data for t in ts]
    data = np.stack(_data, axis=axis)
    requires_grad = False
    depends_on = []
    grad_on = StackBackward

    for index, t in enumerate(ts):
        requires_grad = requires_grad or t.requires_grad
        if t.requires_grad:
            depends_on.append(Edge(t, [axis, index]))

    return Tensor(data, requires_grad, depends_on, grad_on)


def tensor_factory(*args,
                   method=None,
                   dtype=np.float,
                   requires_grad: bool = False,
                   shift=None,
                   scale=None,
                   trimmean=False,
                   name: str = None) -> Tensor:
    assert method, "No method is specified"
    data = method(args[0], dtype=dtype)
    if shift:
        data += shift

    if trimmean:
        data -= data.mean()

    if scale:
        data *= scale

    return Tensor(data, requires_grad=requires_grad, name=name, is_leaf=True)


def empty(*args, requires_grad: bool = False, trimmean=False,
          dtype=np.float, shift=None, scale=None, name: str = None) -> Tensor:
    return tensor_factory(args,
                          method=np.empty, trimmean=trimmean,
                          dtype=dtype, shift=shift, scale=scale, name=name,
                          requires_grad=requires_grad)


def empty_like(t: Tensor, requires_grad: bool = False, trimmean=False,
               dtype=np.float, shift=None, scale=None, name: str = None) -> Tensor:
    return tensor_factory(t.shape,
                          method=np.empty, trimmean=trimmean,
                          dtype=dtype, shift=shift, scale=scale, name=name,
                          requires_grad=requires_grad)


def ones(*args, requires_grad: bool = False, trimmean=False,
         dtype=np.float, shift=None, scale=None, name: str = None) -> Tensor:
    return tensor_factory(args,
                          method=np.ones, trimmean=trimmean,
                          dtype=dtype, shift=shift, scale=scale,
                          requires_grad=requires_grad,
                          name=name)


def ones_like(t: Tensor, requires_grad: bool = False, trimmean=False,
              dtype=np.float, shift=None, scale=None, name: str = None) -> Tensor:
    return tensor_factory(t.shape,
                          method=np.ones, trimmean=trimmean,
                          dtype=dtype, shift=shift, scale=scale,
                          requires_grad=requires_grad,
                          name=name)


def zeros(*args, requires_grad: bool = False, trimmean=False,
          dtype=np.float, shift=None, scale=None, name: str = None) -> Tensor:
    return tensor_factory(args,
                          method=np.zeros, trimmean=trimmean,
                          dtype=dtype, shift=shift, scale=scale, name=name,
                          requires_grad=requires_grad)


def zeros_like(t: Tensor, requires_grad: bool = False, trimmean=False,
               dtype=np.float, shift=None, scale=None, name: str = None) -> Tensor:
    return tensor_factory(t.shape,
                          method=np.zeros, trimmean=trimmean,
                          dtype=dtype, shift=shift, scale=scale, name=name,
                          requires_grad=requires_grad)


def rand(*args, requires_grad: bool = False,
         shift=None, scale=None, name: str = None,
         trimmean=False) -> Tensor:
    data = np.random.rand(*args)

    if shift:
        data += shift

    if trimmean:
        data -= data.mean()

    if scale:
        data *= scale

    return Tensor(data, requires_grad=requires_grad, name=name)


def randn(*args, requires_grad: bool = False,
          shift=None, scale=None, name: str = None,
          trimmean=False) -> Tensor:
    data = np.random.randn(*args)
    if shift:
        data += shift

    if trimmean:
        data -= data.mean()

    if scale:
        data *= scale

    return Tensor(data, requires_grad=requires_grad, name=name)


def rand_like(t: Tensor, requires_grad: bool = False,
              shift=None, scale=None, name: str = None,
              trimmean=False) -> Tensor:
    data = np.random.rand(*t.shape)
    if shift:
        data += shift
    if scale:
        data *= scale
    if trimmean:
        data -= data.mean()
    return Tensor(data, requires_grad=requires_grad, name=name)


def randn_like(t: Tensor, requires_grad: bool = False,
               shift=None, scale=None, name: str = None,
               trimmean=False) -> Tensor:
    data = np.random.randn(*t.shape)
    if shift:
        data += shift
    if scale:
        data *= scale

    if trimmean:
        data -= data.mean()
    return Tensor(data, requires_grad=requires_grad, name=name)


def from_numpy(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)
