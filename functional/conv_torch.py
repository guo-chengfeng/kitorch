"""
@author:  guo_chengfeng
@contact: chf_guo@163.com
@file: __init__.py
@time: 2019/10/15 13:11
@version: 0.1
@desc: An implementation of the conv2d forward and backward by torch.conv2d

The core code of conv2d_forward is torch.conv2d.
Then I use conv2d_forward to calculate the output, input's grad and weight's grad
It is fastest, but low precision, especially weight's grad.
It need lots and lots of sum to calculate weight's grad, error accumulate in the process
"""

import numpy as np
from ..tensor import Tensor, Edge
from ..utils import zeros
import torch

step_in = zeros(1, requires_grad=True)


def bi_tuple(num):
    """convert a int to tuple of (int,int)"""
    if isinstance(num, int):
        return num, num
    else:
        return num


def conv2d_forward(x, w, b=None, stride=(1, 1), padding=(0, 0)):
    """
    Forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.
    Input:
    - x: Input data of shape (N, C_in, H, W)
    - w: Filter weights of shape (C_out, C_in, kH, kW)
    - b: Biases, of shape (C_out,)
    - out: Output data, of shape (N, C_out, H', W') where H' and W' are given by
      H' = 1 + (H + 2*pad[0] - kH) / stride[0]
      W' = 1 + (W + 2*pad[1] - kW) / stride[1]

    The shapes of X and W satisfy:
        (H + 2 * padding[0] - kH) % stride[0] == 0
        (W + 2 * padding[1] - kW) % stride[1] == 0
    """

    """
    Faster implementation [[[NOT MINE]]]
    """
    N, C, H, W = x.shape
    _, _, kH, kW = w.shape
    assert (H + 2 * padding[0] - kH) % stride[0] == 0
    assert (W + 2 * padding[1] - kW) % stride[1] == 0

    if padding[0] > 0 or padding[1] > 0:
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')
    else:
        x_padded = x

    if b is not None:
        bias = torch.Tensor(b)
    else:
        bias = None
    try:
        out = torch.conv2d(torch.Tensor(x_padded),
                           torch.Tensor(w),
                           bias=bias, stride=stride)
    except Exception:
        out = torch.conv2d(torch.Tensor(x_padded),
                           torch.Tensor(w.copy()),
                           bias=bias, stride=stride)

    return out


def conv2d_backward(dout, input, input_requires_grad,
                    weight_requires_grad, weight,
                    stride, padding):
    """
    Backward pass for a convolutional layer.
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, bi, conv_param) as in conv_forward
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw = None, None
    _, _, kH, kW = weight.shape
    N, C, H, W = dout.shape

    if stride[0] > 1 or stride[1] > 1:
        _dout = np.zeros((N, C, H * stride[0] - 1, W * stride[1] - 1))
        index_i = np.repeat(np.arange(0, H) * stride[0], W)
        index_j = np.tile(np.arange(0, W) * stride[1], H)
        _dout[:, :, index_i, index_j] = dout.reshape(N, C, -1)
    else:
        _dout = dout

    if weight_requires_grad:
        if padding[0] > 0 or padding[1] > 0:
            p = padding
            x_padded = np.pad(input, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')
        else:
            x_padded = input

        x_padded = np.swapaxes(x_padded, axis1=0, axis2=1)
        __dout = np.swapaxes(_dout, axis1=0, axis2=1)
        dw = conv2d_forward(x_padded, __dout, stride=(1, 1), padding=(0, 0))
        dw = np.swapaxes(dw, axis1=0, axis2=1)

    if input_requires_grad:
        _weight = np.swapaxes(weight, axis1=0, axis2=1)
        _weight = np.flip(_weight, (2, 3))
        dx = conv2d_forward(_dout, _weight, stride=(1, 1), padding=(kH - 1, kW - 1))
        if padding[0] > 0 and padding[1] > 0:
            dx = dx[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

        elif padding[0] == 0 and padding[1] > 0:
            dx = dx[:, :, :, padding[1]:-padding[1]]

        elif padding[0] > 0 and padding[1] == 0:
            dx = dx[:, :, padding[0]:-padding[0], :]
    return dx, dw


def Conv2dBackward(grad: 'Tensor', depends_on) -> 'Tensor':
    edge = depends_on[0]
    input, weight, bias, stride, padding = edge.args
    tensor_grad = []
    if bias is not None:
        tensor_grad.append((bias,
                            Tensor(np.sum(grad.data, axis=(0, 2, 3)))))

    input_requires_grad = input.requires_grad
    weight_requires_grad = weight.requires_grad
    if input_requires_grad or weight_requires_grad:
        dx, dw = conv2d_backward(grad.data, input.data,
                                 input_requires_grad,
                                 weight_requires_grad, weight.data,
                                 stride, padding)
        if weight_requires_grad:
            tensor_grad.append((weight, Tensor(dw)))
        if input_requires_grad:
            tensor_grad.append((input, Tensor(dx)))
    return tensor_grad


def conv2d(input: Tensor, weight: Tensor, bias=None, stride=None, padding=None) -> Tensor:
    requires_grad = input.requires_grad or weight.requires_grad
    _stride = (1, 1) if stride is None else bi_tuple(stride)
    _padding = (0, 0) if padding is None else bi_tuple(padding)

    if bias is not None:
        data = conv2d_forward(input.data, weight.data, bias.data, _stride, _padding)
        requires_grad = requires_grad or bias.requires_grad
    else:
        data = conv2d_forward(input.data, weight.data, None, _padding, _stride)

    grad_fn = Conv2dBackward
    depends_on = []
    if requires_grad:
        depends_on.append(Edge(step_in, [input, weight, bias, stride, _padding]))

    return Tensor(data, requires_grad, depends_on, grad_fn, is_simple=False)

#################################
# ##TODO conv2d_transposed
# def conv2d_transposed(input: Tensor, weight: Tensor, bias=None, stride=None, padding=None):
#    pass
