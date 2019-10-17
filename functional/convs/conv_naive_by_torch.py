import numpy as np
from tensor import Tensor, Edge
import torch


def Conv2dBackward(output_grad: 'Tensor', t: 'Tensor', other_args) -> 'Tensor':
    backward_type = other_args[0]

    if backward_type == 'inputs':
        weight = other_args[1]
        _data = _Conv2dBackward0(output_grad.data, weight.data)
        padding = other_args[2]
        if padding:
            pH = padding[0]
            pW = padding[1]
            if pH > 0 and pW > 0:
                data = _data[:, :, pH:-pH, pW:-pW]
            elif pH > 0 and pW == 0:
                data = _data[:, :, pH:-pH, :]
            elif pW > 0 and pH == 0:
                data = _data[:, :, :, pW:-pW]
            else:
                return Tensor(_data)
            return Tensor(data)
        else:
            return Tensor(_data)
    elif backward_type == 'weight':
        inputs = other_args[1]
        padding = other_args[2]
        if padding:
            pad_width = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
            data = np.lib.pad(inputs.data, pad_width, mode='constant', constant_values=0)
        else:
            data = inputs.data
        return Tensor(_Conv2dBackward1(data, output_grad.data))
    else:
        bias_grad = output_grad.data.sum(axis=(3, 2, 0))
        return Tensor(bias_grad)


def conv2d(inputs: Tensor, weight: Tensor, bias=None, stride = None, padding=None) -> Tensor:
    """
    :param inputs: input tensor of shape (batch_size X in_channel X iH X iW)
    :param weight: filters of shape (out_channel X in_channel X kH X kW)
    :param bias: optional bias tensor of shape (out_channel)
    :param padding: padding on both sides of the input.A tuple (padH,padW) or None, default: None
    :return: output tensor of shape (batch_size X out_channel X iH+2*padH-kH+1 X iW+2*padW-kW+1)
    """

    if padding:
        weight_shape = weight.shape
        assert padding[0] <= weight_shape[2] / 2 and padding[1] <= weight_shape[
            3] / 2, "pad should be smaller than half of kernel size"

    need_bias = bias is not None
    if need_bias:
        data = ndarray_conv2d(inputs.data, weight.data, bias.data, padding)
        requires_grad = inputs.requires_grad or weight.requires_grad or bias.requires_grad
    else:
        data = ndarray_conv2d(inputs.data, weight.data, padding)
        requires_grad = inputs.requires_grad or weight.requires_grad

    grad_fn = Conv2dBackward
    depends_on = []
    if inputs.requires_grad:
        depends_on.append(Edge(inputs, ['inputs', weight.copy(), padding]))

    if weight.requires_grad:
        depends_on.append(Edge(weight, ['weight', inputs.copy(), padding]))

    if need_bias:
        if bias.requires_grad:
            depends_on.append(Edge(bias, ['bias']))

    return Tensor(data, requires_grad, depends_on, grad_fn)


# input: mini_batch X in_channel X iH X iW
# weight: out_channel X in_channel X kH X kW
# output: mini_batch X out_channel X iH X iW
def ndarray_conv2d(inputs: np.ndarray, weight: np.ndarray, bias=None, padding=None):
    if padding:
        pad_width = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        inputs = np.lib.pad(inputs, pad_width, mode='constant', constant_values=0)

    mini_batch, in_channel_1, iH, iW = inputs.shape
    out_channel, in_channel_2, kH, kW = weight.shape
    need_bias = bias is not None
    assert in_channel_1 == in_channel_2, "in_channel of input and weight must be equal"
    if need_bias:
        assert bias.size == out_channel, "out_channel of bias and weight must be equal"

    # oH = iH - kH + 1
    # oW = iW - kW + 1
    # out_puts = np.empty((mini_batch, out_channel, oH, oW))
    # 为了提高效率，我们需要控制最外层的循序条件
    # 在第3和第4维度是必须要进行循环操作的，能优化的就只有第1和第2维度
    # 最外层的循环次数应当最小
    if need_bias:
        torch_out_puts = torch.conv2d(input=torch.tensor(inputs),
                                      weight=torch.tensor(weight),
                                      bias=torch.tensor(bias))
    else:
        torch_out_puts = torch.conv2d(input=torch.tensor(inputs),
                                      weight=torch.tensor(weight.copy()))

    return torch_out_puts.numpy()


# inputs的梯度
def _Conv2dBackward0(output_grad, weight):
    weight_shape = weight.shape
    padding = weight_shape[2] - 1, weight_shape[3] - 1
    weight_0 = np.swapaxes(weight, axis1=0, axis2=1)
    weight_1 = np.flip(weight_0, (2, 3))
    return ndarray_conv2d(output_grad, weight_1, padding=padding)


# Weight的导数
# 不论 nopython 模式还是其他模式，效率都下降了
# @jit(float64[:, :, :, :]
#          (float64[:, :, :, :],
#           float64[:, :, :, :],
#          ), nopython=True)
def _Conv2dBackward1(inputs: np.ndarray, outputs: np.ndarray):
    mini_batch, in_channel, iH, iW = inputs.shape
    mini_batch, out_channel, oH, oW = outputs.shape
    assert iH >= oH and iW >= oW, "the input'size must >= output'size "
    inputs_1 = np.swapaxes(inputs, axis1=0, axis2=1)
    outputs_1 = np.swapaxes(outputs, axis1=0, axis2=1)
    weight_grad = torch.conv2d(torch.tensor(inputs_1), torch.tensor(outputs_1)).numpy()
    return np.swapaxes(weight_grad, axis1=0, axis2=1)
