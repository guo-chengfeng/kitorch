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
import time
import numpy as np
from typing import List
from .tensor import Edge, Tensor


def to_pair(num):
    """convert a int to tuple of (int,int)"""

    if isinstance(num, int):
        return num, num
    else:
        return num


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

    def print_time(self, prefix, update_tic=False):
        self._toc = time.time()
        self.show_time(elapsed_time=self._toc - self._tic, prefix=prefix)
        if update_tic:
            self._tic = time.time()


timer = Timer()


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
         trimmean=False, dtype=None) -> Tensor:
    data = np.random.rand(*args)
    if dtype:
        data = data.astype(dtype=dtype)

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


# save  and load model
import pickle


def save_model(model, file_path):
    with open(file_path, 'wb') as fin:
        pickle.dump(model, fin)
    print("model saved at %s " % file_path)


def load_model(file_path):
    with open(file_path, 'rb') as fout:
        return pickle.load(fout)
