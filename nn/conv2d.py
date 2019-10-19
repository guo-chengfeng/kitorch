import numpy as np
from .. import functional as F
from ..tensor import Tensor
from ..utils import rand
from .module import Layer


def to_pair(num):
    """convert a int to tuple of (int,int)"""
    if isinstance(num, int):
        return num, num
    else:
        return num


class Conv2d(Layer):
    """
    卷积层
    """

    def __init__(self, in_channel, out_channel, kernel_size, bias=True, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = to_pair(kernel_size)
        self.need_bias = bias

        self._parameters = []
        kH, kW = to_pair(kernel_size)
        scale = 1.0 / np.sqrt(out_channel)
        self.weight = rand(out_channel, in_channel, kH, kW, requires_grad=True,
                           shift=-0.5, scale=scale)
        self._parameters.append(self.weight)

        if bias:
            self.bias = rand(out_channel, requires_grad=True, shift=-0.5, scale=scale)
            self._parameters.append(self.bias)
        else:
            self.bias = None

        self.stride = (1, 1) if stride is None else to_pair(stride)
        self.padding = (0, 0) if padding is None else to_pair(padding)

    def forward(self, x: Tensor):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def __repr__(self):
        string = "Conv2d(in_channel=%s, out_channel=%s, kernel_size=%s" % (self.in_channel,
                                                                           self.out_channel, self.kernel_size)
        if self.need_bias:
            string += ",bias=True"
        if self.stride != (1, 1):
            string += ",stride=%s" % str(self.stride)
        if self.padding != (0, 0):
            string += ",padding=%s" % str(self.padding)
        string += ")"
        return string


class ConvTranspose2d(Layer):
    """
    反卷积层
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0):
        super(ConvTranspose2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = to_pair(kernel_size)

        self._parameters = []

        kH, kW = to_pair(kernel_size)
        scale = 1.0 / np.sqrt(out_channel)
        self.weight = rand(out_channel, in_channel, kH, kW, requires_grad=True,
                           shift=-0.5, scale=scale)
        self._parameters.append(self.weight)

        self.stride = (1, 1) if stride is None else to_pair(stride)
        self.bais = (0, 0) if padding is None else to_pair(padding)

        self.output_padding = output_padding

    def forward(self, x: Tensor):
        return F.conv_transpose2d(x, self.weight, stride=self.stride,
                                  padding=self.padding, output_padding=self.output_padding)

    def __repr__(self):
        string = "Conv2dTransposed(in_channel=%s, out_channel=%s, kernel_size=%s" % (self.in_channel,
                                                                                     self.out_channel, self.kernel_size)
        if self.stride != (1, 1):
            string += ",stride=%s" % str(self.stride)
        if self.padding != (0, 0):
            string += ",padding=%s" % str(self.padding)
        if self.output_padding > 0:
            string += ",output_padding=%s" % str(self.output_padding)
        string += ")"
        return string
