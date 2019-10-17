"""
@author:  guo_chengfeng
@contact: chf_guo@163.com
@file: __init__.py
@time: 2019/10/15 13:11
@version: 0.1
@desc: An implementation of the conv2d forward and backward

The core code of conv2d_forward is from https://github.com/cthorey/CS231/blob/master/assignment2/
It is faster than conv_naive

"""

import numpy as np
from ..tensor import Tensor, Edge
from ..utils import zeros

step_in = zeros(1, requires_grad=True)


def bi_tuple(num):
    """convert a int to tuple of (int,int)"""
    if isinstance(num, int):
        return num, num
    else:
        return num


def get_im2col_indices(x_shape, kernel_height, kernel_width, padding=(0, 0), stride=(1, 1)):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding[0] - kernel_height) % stride[0] == 0
    assert (W + 2 * padding[1] - kernel_height) % stride[1] == 0
    out_height = int((H + 2 * padding[0] - kernel_height) / stride[0] + 1)
    out_width = int((W + 2 * padding[1] - kernel_width) / stride[1] + 1)

    i0 = np.repeat(np.arange(kernel_height), kernel_width)
    i0 = np.tile(i0, C)
    i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(kernel_width), kernel_height * C)
    j1 = stride[1] * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), kernel_height * kernel_width).reshape(-1, 1)

    return k, i, j


def im2col_indices(x, kernel_height, kernel_width, padding=(0, 0), stride=(1, 1)):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    if sum(padding) > 0:
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')
    else:
        x_padded = x

    k, i, j = get_im2col_indices(x.shape, kernel_height, kernel_width, padding,
                                 stride)

    # most time consuming
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(kernel_height * kernel_width * C, -1)

    return cols


# SLOW
def col2im_indices(cols, x_shape, kernel_height=3, kernel_width=3, padding=(0, 0),
                   stride=(1, 1)):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = int(H + 2 * padding[0]), int(W + 2 * padding[1])
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, kernel_height, kernel_width, padding,
                                 stride)

    cols_reshaped = cols.reshape(C * kernel_height * kernel_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == (0, 0):
        return x_padded
    if padding[0] > 0 and padding[1] > 0:
        return x_padded[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

    if padding[0] == 0 and padding[1] > 0:
        return x_padded[:, :, :, padding[1]:-padding[1]]

    return x_padded[:, :, padding[0]:-padding[0], :]


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
    num_filters, _, kernel_height, kernel_width = w.shape

    # Create output
    out_height = int(1 + (H + 2 * padding[0] - kernel_height) / stride[0])
    out_width = int(1 + (W + 2 * padding[1] - kernel_width) / stride[1])
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    x_cols = im2col_indices(x, w.shape[2], w.shape[3], padding, stride)

    if b is not None:
        res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)
    else:
        res = w.reshape((w.shape[0], -1)).dot(x_cols)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    return out, x_cols


def conv2d_backward(dout, input_requires_grad, input_shape,
                    weight_requires_grad, weight,
                    x_cols, stride, padding):
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
    """
    """

    num_filters, _, kH, kW = weight.shape
    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)

    if weight_requires_grad:
        dw = dout_reshaped.dot(x_cols.T).reshape(weight.shape)

    if input_requires_grad:
        # 1.32  s
        # dx_cols = weight.reshape(num_filters, -1).T.dot(dout_reshaped)
        # dx = col2im_indices(dx_cols, input_shape, kH, kW, padding, stride)

        # 460ms
        if stride[0] > 1 or stride[1] > 1:
            _dout = np.zeros((N, C, (H - 1) * stride[0] + 1, (W - 1) * stride[1] + 1))
            index_i = np.repeat(np.arange(0, H) * stride[0], W)
            index_j = np.tile(np.arange(0, W) * stride[1], H)
            _dout[:, :, index_i, index_j] = dout.reshape(N, C, -1)
        else:
            _dout = dout

        _weight = np.swapaxes(weight, axis1=0, axis2=1)
        _weight = np.flip(_weight, (2, 3))

        dx, _ = conv2d_forward(_dout, _weight, stride=(1, 1), padding=(kH - 1, kW - 1))
        if padding[0] > 0 and padding[1] > 0:
            dx = dx[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

        elif padding[0] == 0 and padding[1] > 0:
            dx = dx[:, :, :, padding[1]:-padding[1]]

        elif padding[0] > 0 and padding[1] == 0:
            dx = dx[:, :, padding[0]:-padding[0], :]

    return dx, dw


def Conv2dBackward(grad: 'Tensor', depends_on) -> 'Tensor':
    edge = depends_on[0]
    input, weight, bias, x_cols, stride, padding = edge.args
    tensor_grad = []
    if bias is not None:
        tensor_grad.append((bias,
                            Tensor(np.sum(grad.data, axis=(0, 2, 3)))))

    input_requires_grad = input.requires_grad
    weight_requires_grad = weight.requires_grad
    if input_requires_grad or weight_requires_grad:
        dx, dw = conv2d_backward(grad.data,
                                 input_requires_grad, input.shape,
                                 weight_requires_grad, weight.data,
                                 x_cols, stride, padding)
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
        data, x_cols = conv2d_forward(input.data, weight.data, bias.data, _stride, _padding)
        requires_grad = requires_grad or bias.requires_grad
    else:
        data, x_cols = conv2d_forward(input.data, weight.data, None, _padding, _stride)

    grad_fn = Conv2dBackward
    depends_on = []
    if requires_grad:
        depends_on.append(Edge(step_in, [input, weight, bias, x_cols, _stride, _padding]))

    return Tensor(data, requires_grad, depends_on, grad_fn, is_simple=False)


#################################
# ##TODO conv2d_transposed
def conv_transpose2d(input: Tensor, weight: Tensor, bias=None, stride=None, padding=None):
    raise NotImplementedError
