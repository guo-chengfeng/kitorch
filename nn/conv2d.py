import numpy as np
from .. import functional as F
from ..tensor import Tensor
from ..utils import rand
from .layer import Layer


class Conv2d(Layer):
    """
    卷积层
    """

    def __init__(self, in_channel, out_channel, kernel_size, bias=True, padding=None):
        super(Conv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.need_bias = bias

        self.parameters = []
        if isinstance(kernel_size, int):
            kH = kernel_size
            kW = kernel_size
        else:
            kH, kW = kernel_size
        scale = 1.0 / np.sqrt(out_channel)
        self.weight = rand(out_channel, in_channel, kH, kW, requires_grad=True,
                           shift=-0.5, scale=scale)
        self.parameters.append(self.weight)

        if bias:
            self.bias = rand(out_channel, requires_grad=True, shift=-0.5, scale=scale)
            self.parameters.append(self.bias)
        else:
            self.bias = None
        if padding:
            if isinstance(padding, int):
                self.padding = (padding, padding)
            else:
                self.padding = padding
        else:
            self.padding = None

    def forward(self, x: Tensor):
        return F.conv2d(x, self.weight, self.bias, self.padding)

    def __repr__(self):
        string = "Conv2d(in_channel=%s, out_channel=%s, kernel_size=%s" % (self.in_channel,
                                                                         self.out_channel, self.kernel_size)
        if self.need_bias:
            string += ",bias=True"
        if self.padding:
            string += ",padding=%s" % self.padding
        string += ")"
        return string
