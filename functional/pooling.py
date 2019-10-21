from ..tensor import Tensor, Edge, np
from ..utils import to_pair


# 类似于卷积
def avgpool2d(inputs, kernel_size, padding=None):
    kernel_size = to_pair(kernel_size)
    padding = (0, 0) if padding is None else to_pair(padding)

    if padding:
        assert padding[0] <= kernel_size[0] / 2 and padding[1] <= kernel_size[
            1] / 2, "pad should be smaller than half of kernel size"

    data = ndarray_avgpool2d(inputs.data, kernel_size, padding=padding)
    requires_grad = inputs.requires_grad
    depends_on = []
    grad_fn = Avgpool2dBackward
    if requires_grad:
        depends_on.append(Edge(inputs, [kernel_size, padding]))
    return Tensor(data, requires_grad, depends_on, grad_fn)


def Avgpool2dBackward(output_grad: 'Tensor', t: 'Tensor', other_args) -> 'Tensor':
    kernel_size = other_args[0]
    padding = other_args[1]
    _data = ndarray_avgpool2d_backward(output_grad.data, kernel_size)
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


def maxpool2d(inputs, kernel_size, padding=None):
    kernel_size = to_pair(kernel_size)
    padding = (0, 0) if padding is None else to_pair(padding)

    if padding:
        assert padding[0] <= kernel_size[0] / 2 and padding[1] <= kernel_size[
            1] / 2, "pad should be smaller than half of kernel size"

    data, max_idxs = ndarray_maxpool2d(inputs.data, kernel_size, padding=padding)
    requires_grad = inputs.requires_grad
    depends_on = []
    grad_fn = Maxpool2dBackward
    if requires_grad:
        depends_on.append(Edge(inputs, [max_idxs, kernel_size, padding]))
    return Tensor(data, requires_grad, depends_on, grad_fn)


def Maxpool2dBackward(output_grad: 'Tensor', t: 'Tensor', other_args) -> 'Tensor':
    max_idxs = other_args[0]
    kernel_size = other_args[1]
    padding = other_args[2]
    _data = ndarray_maxpool2d_backward(output_grad.data, max_idxs, kernel_size)
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


# input: mini_batch X in_channel X iH X iW
# weight: out_channel X in_channel X kH X kW
# output: mini_batch X out_channel X iH X iW
def ndarray_avgpool2d(inputs: np.ndarray, kernel_size, padding=None):
    if padding:
        pad_width = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        inputs = np.lib.pad(inputs, pad_width, mode='constant', constant_values=0)

    mini_batch, in_channel, iH, iW = inputs.shape
    kH, kW = kernel_size
    assert iH / kH == iH // kH and iW / kW == iW // kW, "The size of inputs must be divide exactly by kernel_size"
    scale = kH * kW

    oH = iH // kH
    oW = iW // kW
    out_puts = np.empty((mini_batch, in_channel, oH, oW))
    for h in range(oH):
        for w in range(oW):
            inputs[:, :, h * kH:(h + 1) * kH, w * kW:(w + 1) * kW].sum(axis=(3, 2), out=out_puts[:, :, h, w])
    out_puts /= scale
    return out_puts


def ndarray_avgpool2d_backward(output_grad, kernel_size):
    kH, kW = kernel_size
    scale = kH * kW
    mini_batch, out_channel, oH, oW = output_grad.shape
    iH = kH * oH
    iW = kW * oW
    inputs_grad = np.empty((mini_batch, out_channel, iH, iW))
    for h in range(oH):
        for w in range(oW):
            inputs_grad[:, :, h * kH:(h + 1) * kH, w * kW:(w + 1) * kW] = \
                np.expand_dims(np.expand_dims(output_grad[:, :, h, w], axis=2), axis=3)
    inputs_grad /= scale
    return inputs_grad


from numba import jit, float64, void, int32


@jit(void(float64[:, :, :, :],
          float64[:, :, :, :],
          int32, int32, int32, int32, int32, int32), nopython=True)
def ndarray_maxpool2d_forward_core(inputs, max_idxs, mini_batch, in_channel, oH, oW, kH, kW):
    for batch in range(mini_batch):
        for in_ch in range(in_channel):
            for h in range(oH):
                for w in range(oW):
                    max_idxs[batch, in_ch, h, w] = \
                        inputs[batch, in_ch, h * kH:(h + 1) * kH, w * kW:(w + 1) * kW].argmax()


@jit(void(float64[:, :, :, :],
          float64[:, :, :, :],
          float64[:, :, :, :],
          int32, int32, int32, int32, int32, int32), nopython=True)
def ndarray_maxpool2d_backward_core(inputs_grad, output_grad, max_idxs, mini_batch, out_channel, oH, oW, kH, kW):
    for batch in range(mini_batch):
        for out_ch in range(out_channel):
            for h in range(oH):
                for w in range(oW):
                    ind = max_idxs[batch, out_ch, h, w]
                    ind_2 = int(ind // kW)
                    ind_3 = int(ind % kW)
                    inputs_grad[batch, out_ch, h * kH + ind_2, w * kW + ind_3] = output_grad[batch, out_ch, h, w]


def ndarray_maxpool2d(inputs: np.ndarray, kernel_size, padding=None):
    if padding:
        pad_width = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        inputs = np.lib.pad(inputs, pad_width, mode='constant', constant_values=0)

    mini_batch, in_channel, iH, iW = inputs.shape
    kH, kW = kernel_size
    assert iH / kH == iH // kH and iW / kW == iW // kW, "The size of inputs must be divide exactly by kernel_size"

    oH = iH // kH
    oW = iW // kW
    outputs = np.empty((mini_batch, in_channel, oH, oW))
    max_idxs = np.empty((mini_batch, in_channel, oH, oW))
    ndarray_maxpool2d_forward_core(inputs, max_idxs, mini_batch, in_channel, oH, oW, kH, kW)
    for h in range(oH):
        for w in range(oW):
            outputs[:, :, h, w] = \
                inputs[:, :, h * kH:(h + 1) * kH, w * kW:(w + 1) * kW].max(axis=3).max(axis=2)

    return outputs, max_idxs


def ndarray_maxpool2d_backward(output_grad, max_idxs, kernel_size):
    kH, kW = kernel_size
    mini_batch, out_channel, oH, oW = output_grad.shape
    iH = kH * oH
    iW = kW * oW
    inputs_grad = np.zeros((mini_batch, out_channel, iH, iW))
    ndarray_maxpool2d_backward_core(inputs_grad, output_grad, max_idxs, mini_batch, out_channel, oH, oW, kH, kW)

    return inputs_grad
