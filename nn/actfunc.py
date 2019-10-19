#!/usr/bin/python

# encoding: utf-8
"""
@author:  guo_chengfeng
@contact: chf_guo@163.com
@file: activation.py
@time: 2019/10/12 14:55
@version: 0.1
@desc:
"""
from ..tensor import Tensor
from ..functional import softmax, sigmoid, log_softmax, tanh, relu
from .module import Layer


class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return relu(input)

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return sigmoid(input)

    def __repr__(self):
        return "Sigmoid()"


class Softmax(Layer):
    def __init__(self, dim, deoverflow=True):
        super(Softmax, self).__init__()
        self.dim = dim
        self.deoverflow = deoverflow

    def forward(self, input: Tensor) -> Tensor:
        return softmax(input, self.dim, self.deoverflow)

    def __repr__(self):
        string = "LogSoftmax(dim=%s" % (self.dim)
        if self.deoverflow:
            string += ",deoverflow=True"
        string += ")"
        return string


class LogSoftmax(Layer):
    def __init__(self, dim, deoverflow=True):
        super(LogSoftmax, self).__init__()
        self.dim = dim
        self.deoverflow = deoverflow

    def forward(self, input: Tensor) -> Tensor:
        return log_softmax(input, self.dim, self.deoverflow)

    def __repr__(self):
        string = "LogSoftmax(dim=%s" % (self.dim)
        if self.deoverflow:
            string += ",deoverflow=True"
        string += ")"
        return string


class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return tanh(input)

    def __repr__(self):
        return "Tanh()"

