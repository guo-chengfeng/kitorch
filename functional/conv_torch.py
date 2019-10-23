"""
@author:  guo_chengfeng
@contact: chf_guo@163.com
@file: __init__.py
@time: 2019/10/15 13:11
@version: 0.1
@desc: An implementation of the conv2d forward and backward by torch.conv2d

The core code of conv2d_forward is torch.conv2d.
Then I use conv2d_forward to calculate the output, input's grad and weight's grad
It is faster than conv_by_torch
"""

import numpy as np
from ..tensor import Tensor, Edge
from ..utils import zeros, to_pair
import torch

step_in = zeros(1, requires_grad=True)


# 开端一定padding
# 末端根据stride确定是否padding
def repadding(features, pad, k_length, stride):
    out_f = (features + 2 * pad - k_length) // stride + 1
    dilation_f = (out_f - 1) * stride + k_length - pad
    pad0 = pad
    pad1 = int(dilation_f - features)
    pad1, unused = (pad1, 0) if pad1 > 0 else (0, -pad1)
    return pad0, pad1, unused


def conv2d(input: Tensor, weight: Tensor, bias=None, stride=None, padding=None) -> Tensor:
    requires_grad = input.requires_grad or weight.requires_grad
    stride = (1, 1) if stride is None else to_pair(stride)
    padding = (0, 0) if padding is None else to_pair(padding)
    N, C, H, W = input.shape
    _, _, kH, kW = weight.shape
    padding = repadding(H, padding[0], kH, stride[0]), \
              repadding(H, padding[1], kH, stride[1])

    if bias is not None:
        data = conv2d_forward(input.data, weight.data, bias.data, stride, padding)
        requires_grad = requires_grad or bias.requires_grad
    else:
        data = conv2d_forward(input.data, weight.data, None, stride, padding, )

    grad_fn = Conv2dBackward
    depends_on = []
    if input.requires_grad:
        depends_on.append(Edge(input, ['input', weight.data, stride, padding]))

    if weight.requires_grad:
        depends_on.append(Edge(weight, ['weight', input.data, stride, padding]))

    if bias is not None:
        if bias.requires_grad:
            depends_on.append(Edge(bias, ['bias']))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def conv2d_forward(x, w, b=None, stride=(1, 1), padding=((0, 0, 0), (0, 0, 0)), forward_type=True):
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

    # This is not necessary
    # assert (H + 2 * padding[0] - kH) % stride[0] == 0
    # assert (W + 2 * padding[1] - kW) % stride[1] == 0
    x_padded = x
    if forward_type:
        pad00, pad01, _ = padding[0]
        pad10, pad11, _ = padding[1]
        if pad00 or pad01 or pad10 or pad11:
            x_padded = np.pad(x, ((0, 0), (0, 0), (pad00, pad01), (pad10, pad11)), mode='constant')
    else:
        pad0, pad1 = padding
        if pad0 or pad1:
            x_padded = np.pad(x, ((0, 0), (0, 0), (pad0, pad0), (pad1, pad1)), mode='constant')

    if b is not None:
        bias = torch.tensor(b)

    else:
        bias = None

    out = torch.conv2d(torch.tensor(x_padded),
                       torch.tensor(w),
                       bias=bias, stride=stride)
    return out.data.numpy()


def Conv2dBackward(grad: 'Tensor', t: 'Tensor', cache) -> 'Tensor':
    btype = cache[0]
    if btype == 'bias':
        return Tensor(np.sum(grad.data, axis=(0, 2, 3)))

    grad_data = grad.data
    N, C, H, W = grad_data.shape
    stride = cache[2]
    padding = cache[3]
    pad00, pad01, unused0 = padding[0]
    pad10, pad11, unused1 = padding[1]

    if stride[0] or stride[1]:
        _grad_data = np.zeros((N, C, (H - 1) * stride[0] + 1, (W - 1) * stride[1] + 1))
        index_i = np.repeat(np.arange(0, H) * stride[0], W)
        index_j = np.tile(np.arange(0, W) * stride[1], H)
        _grad_data[:, :, index_i, index_j] = grad_data.reshape(N, C, -1)
        grad_data = _grad_data

    if btype == 'weight':
        input = cache[1]
        _, _, kH, kW = t.shape
        x_padded = input
        if pad00 or pad01 or pad10 or pad11:
            x_padded = np.pad(input, ((0, 0), (0, 0), (pad00, pad01), (pad10, pad11)), mode='constant')

        x_padded = np.swapaxes(x_padded, axis1=0, axis2=1)
        grad_data = np.swapaxes(grad_data, axis1=0, axis2=1)
        dw = conv2d_forward(x_padded, grad_data, stride=(1, 1), padding=(0, 0), forward_type=False)
        dw = np.swapaxes(dw, axis1=0, axis2=1)
        if unused0 or unused1:
            unused0 = -unused0 if unused0 else None
            unused1 = -unused1 if unused1 else None
            return Tensor(dw[:, :, 0:unused0, 0:unused1])

        return Tensor(dw)

    # btype == 'input'
    weight = cache[1]
    _, _, kH, kW = weight.shape
    _weight = np.swapaxes(weight, axis1=0, axis2=1)
    _weight = np.flip(_weight, (2, 3)).copy()
    # _weight必须copy,不然可能报：
    # ValueError: some of the strides of a given numpy array are negative
    # 的错误

    dx = conv2d_forward(grad_data, _weight, stride=(1, 1), padding=(kH - 1, kW - 1), forward_type=False)

    if unused0 or unused1:
        dx = np.pad(dx, ((0, 0), (0, 0), (0, unused0), (0, unused1)), mode='constant')

    if pad00 or pad01:
        pad01 = -pad01 if pad01 else None
        pad11 = -pad11 if pad11 else None
        dx = dx[:, :, pad00:pad01, pad10:pad11]

    return Tensor(dx)


#################################
def conv_transpose2d(input: Tensor, weight: Tensor, stride=1, padding=0, output_padding=0):
    """
    :param input: minibatch , in_channels , iH , iW
    :param weight: in_channels,out_channels,kH , kW
    :param stride:
    :param padding:
    :param output_padding:
    :return:
    """

    stride = (1, 1) if stride is None else to_pair(stride)
    padding = (0, 0) if padding is None else to_pair(padding)
    output_padding = (0, 0) if output_padding is None else to_pair(output_padding)
    _, _, kH, kW = weight.shape
    assert output_padding[0] < stride[0] and output_padding[1] < stride[1], \
        "output padding must be smaller than stride,but got output_padding=%s and stride=%s" % (output_padding, stride)
    assert kH > padding[1] and kW > padding[1], \
        "padding must be smaller than kernel size, but got kernel_size=%s and padding=%s" % ((kH, kW), padding)

    N, C, H, W = input.shape
    if stride[0] > 1 or stride[1] > 1:
        inf_input = np.zeros((N, C, (H - 1) * stride[0] + 1, (W - 1) * stride[1] + 1))
        index_i = np.repeat(np.arange(0, H) * stride[0], W)
        index_j = np.tile(np.arange(0, W) * stride[1], H)
        inf_input[:, :, index_i, index_j] = input.data.reshape(N, C, -1)
    else:
        inf_input = input.data

    _, _, kH, kW = weight.shape

    pad00 = kH - 1 - padding[0]
    pad01 = pad00 + output_padding[0]
    pad10 = kW - 1 - padding[1]
    pad11 = pad10 + output_padding[1]

    if pad00 or pad01 or pad10 or pad11:
        x_padded = np.pad(inf_input, ((0, 0), (0, 0), (pad00, pad01), (pad10, pad11)),
                          mode='constant')
    else:
        x_padded = inf_input

    _weight = np.swapaxes(weight.data, axis1=0, axis2=1)
    _weight = np.flip(_weight, (2, 3)).copy()

    out = torch.conv2d(torch.tensor(x_padded),
                       torch.tensor(_weight),
                       bias=None)

    data = out.numpy()
    requires_grad = input.requires_grad or weight.requires_grad
    grad_fn = Conv2dTransposedBackward
    depends_on = []

    if input.requires_grad:
        depends_on.append(Edge(input, ['input', weight.data, stride, padding]))

    if weight.requires_grad:
        depends_on.append(Edge(weight, ['weight', x_padded, stride, padding]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def Conv2dTransposedBackward(grad: 'Tensor', t: 'Tensor', cache) -> 'Tensor':
    btype, x_or_w, stride, padding = cache
    if btype == 'input':
        x_grad = torch.conv1d(torch.tensor(grad.data),
                              torch.tensor(x_or_w), bias=None,
                              stride=stride, padding=padding)

        return Tensor(x_grad.numpy())

    x_padded = np.swapaxes(x_or_w, axis1=0, axis2=1)
    grad_data = np.swapaxes(grad.data, axis1=0, axis2=1)

    w_grad = torch.conv1d(torch.tensor(x_padded),
                          torch.tensor(grad_data), bias=None)
    w_grad = w_grad.numpy()
    w_grad = np.flip(w_grad, (2, 3))
    return Tensor(w_grad)
