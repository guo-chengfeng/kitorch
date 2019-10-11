from typing import Optional, List, Tuple
from ..tensor import Edge, Tensor
from ..utils import rand, zeros, zeros_like
from . import Layer
import numpy as np

step_in = zeros(1, requires_grad=True)


def dropout_mask_factory(p, seq_len, batch_size, hidden_size):
    scale = 1 / (1 - p)
    return scale * np.random.binomial(1, 1 - p, (seq_len, batch_size, hidden_size))


def sigmoid(data):
    return 1.0 / (1.0 + np.exp(-data))


class RNNBase(Layer):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0, bidirectional=False, mode="RNN"):
        super(RNNBase, self).__init__()
        self.parameters = []
        self.hidden_size = hidden_size
        self.need_bias = bias
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if dropout > 0 and num_layers == 1:
            raise RuntimeError("dropout option adds dropout after all but last recurrent layer, so non-zero dropout "
                               "expects num_layers greater than 1, but got dropout=%s and num_layers=%s" % (
                                   dropout, num_layers))

        self.dropout = dropout

        self.need_dropout = self.dropout > 0 and self.training

        if mode == "LSTM":
            gate_size = 4 * hidden_size
        elif mode == "GRU":
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self.weight_h = []
        self.weight_i = []
        self.bias = []
        scale = 2.0 / np.sqrt(self.hidden_size)

        for layer in range(num_layers):

            layer_input_size = input_size if layer == 0 else hidden_size
            weight = rand(layer_input_size, gate_size, requires_grad=True, scale=scale, trimmean=True)
            self.weight_i.append(weight)
            self.parameters.append(weight)

            weight = rand(hidden_size, gate_size, requires_grad=True, scale=scale, trimmean=True)
            self.weight_h.append(weight)
            self.parameters.append(weight)

            if self.need_bias:
                bias = rand(gate_size, requires_grad=True, scale=scale, trimmean=True)
                self.parameters.append(bias)
            else:
                bias = None
            self.bias.append(bias)

        for para in self.parameters:
            para.grad = zeros_like(para)

        if mode == 'RNN':
            self.rnn_layer = BasicRNNLayerForward
        elif mode == 'LSTM':
            self.rnn_layer = LSTMLayerForward


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0, bidirectional=False):
        """
        input_size: 输入x的特性维度， the number of features
        hidden_size: 隐层神经元数量
        num_layers: 隐层数量，每层的神经元数量为hidden_size, 默认为1
        bias: 是否需要偏执，默认True
        dropout: 如果大于0，则在除了最后一层外添加一个dropout层
        bidirectional: 是否为双向RNN,默认False
        """
        super(RNN, self).__init__(input_size, hidden_size, num_layers, bias, dropout, bidirectional, "RNN")

    def forward(self, input: Tensor, h0: Tensor):
        h_state = [None for _ in range(self.num_layers)]
        hn = [None for _ in range(self.num_layers + 1)]
        hn2 = [None for _ in range(self.num_layers + 1)]
        hn[0] = input.data

        requires_grad = input.requires_grad or self.weight_h[0].requires_grad or h0.requires_grad

        seq_len, batch_size, _ = input.shape

        dropout_mask = [dropout_mask_factory(self.dropout, seq_len, batch_size, self.hidden_size)
                        for _ in range(self.num_layers - 1)] if self.need_dropout else None

        for layer in range(self.num_layers):
            if self.need_dropout and layer > 0:
                _hn, _hn2 = self.rnn_layer(requires_grad, self.need_bias,
                                           self.weight_i[layer], self.weight_h[layer], self.bias[layer],
                                           hn[layer] * dropout_mask[layer - 1], h0.data[layer])

            else:
                _hn, _hn2 = self.rnn_layer(requires_grad, self.need_bias,
                                           self.weight_i[layer], self.weight_h[layer], self.bias[layer],
                                           hn[layer], h0.data[layer])
            h_state[layer] = _hn[-1]
            hn[layer + 1] = np.stack(_hn[1::])
            hn2[layer + 1] = _hn2

        requires_grads = (input.requires_grad, h0.requires_grad, self.weight_h[0].requires_grad)
        grad_fn = RNNBackward
        depends_on = []
        if requires_grad:
            depends_on.append(Edge(step_in, ['output', requires_grads, input, h0, hn, hn2, dropout_mask, self]))

        output = Tensor(hn[-1], requires_grad=requires_grad,
                        grad_fn=grad_fn, depends_on=depends_on, is_simple=False)

        depends_on = []
        if requires_grad:
            depends_on.append(Edge(step_in, ['h_state', requires_grads, input, h0, hn, hn2, dropout_mask, self]))

        h_state = Tensor(np.stack(h_state), requires_grad=requires_grad,
                         grad_fn=grad_fn, depends_on=depends_on, is_simple=False)

        return output, h_state


def BasicRNNLayerForward(requires_grad: bool, need_bias: bool, w_i: Tensor, w_h: Tensor, bias: Optional[Tensor],
                         input: np.ndarray, h0: np.ndarray) -> Tuple:
    seq_len, batch_size, _ = input.shape
    _w_i = w_i.data
    _w_h = w_h.data
    _bias = bias.data if need_bias else None

    hn = [0 for _ in range(seq_len + 1)]
    hn2 = [0 for _ in range(seq_len)] if requires_grad else None
    hn[0] = h0

    for seq in range(seq_len):
        if need_bias:
            output_h = input[seq] @ _w_i + hn[seq] @ _w_h + _bias
        else:
            output_h = input[seq] @ _w_i + hn[seq] @ _w_h

        hn[seq + 1] = np.tanh(output_h)
        if requires_grad:
            hn2[seq] = 1 - hn[seq + 1] ** 2

    return hn, hn2


def RNNBackward(grad: Tensor, depends_on):
    edge = depends_on[0]
    args = edge.args
    backward_type, requires_grads, input, h0, hn, hn2, dropout_mask, self = args
    seq_len = len(hn[0])
    input_requires_grad, h0_requires_grad, w_requires_grad = requires_grads
    h0_grad = [None for _ in range(self.num_layers)]

    if backward_type == 'output':
        output_grad = grad
        h_grad = [None for _ in range(self.num_layers)]
    else:
        output_grad = None
        h_grad = grad

    for layer in reversed(range(self.num_layers)):
        i_rg = input_requires_grad if layer == 0 else True
        output_grad, _h0_grad = BasicRNNLayerBackward(output_grad, h_grad[layer], hn[layer + 1], hn2[layer + 1],
                                                      self.weight_i[layer], self.weight_h[layer], self.bias[layer],
                                                      self.need_bias, hn[layer], h0.data[layer],
                                                      i_rg, h0_requires_grad, w_requires_grad, seq_len)

        h0_grad[layer] = _h0_grad
        if output_grad is not None:
            if layer > 0 and self.need_dropout:
                output_grad = Tensor(np.stack(output_grad) * dropout_mask[layer - 1])
            else:
                output_grad = Tensor(np.stack(output_grad))

    tensor_grad = []
    if input_requires_grad:
        tensor_grad.append((input, output_grad))

    if h0_requires_grad:
        tensor_grad.append((h0, Tensor(np.stack(h0_grad))))

    return tensor_grad


# output的反向传播
def BasicRNNLayerBackward(output_grad: Tensor, h_grad: Tensor,
                          hn, hn2, w_i: Tensor, w_h: Tensor, bias: Optional[Tensor], need_bias: bool,
                          input: np.ndarray, h0: np.ndarray, input_requires_grad, h0_requires_grad,
                          w_requires_grad, seq_len: int) -> List:
    delta = [0 for _ in range(seq_len)]
    delta[-1] = 0
    if output_grad:
        delta[-1] += output_grad.data[-1]
    if h_grad:
        delta[-1] += h_grad.data

    delta[-1] *= hn2[-1]

    if output_grad:
        for seq in reversed(range(0, seq_len - 1)):
            delta[seq] = output_grad.data[seq] + delta[seq + 1] @ w_h.data.T
            delta[seq] *= hn2[seq]
    else:
        for seq in reversed(range(0, seq_len - 1)):
            delta[seq] = (delta[seq + 1] @ w_h.data.T) * hn2[seq]

    grads = []

    # 计算x
    if input_requires_grad:
        grad = [delta[seq] @ w_i.data.T for seq in range(0, seq_len)]
    else:
        grad = None
    grads.append(grad)

    if h0_requires_grad:
        grad = delta[0] @ w_h.data.T
    else:
        grad = None

    grads.append(grad)
    if not w_requires_grad:
        return grads

    for seq in range(0, seq_len):
        w_i.grad.data += input[seq].T @ delta[seq]

    for seq in range(1, seq_len):
        w_h.grad.data += hn[seq - 1].T @ delta[seq]

    w_h.grad.data += h0.T @ delta[0]

    if need_bias:
        for seq in range(0, seq_len):
            bias.grad.data += delta[seq].sum(axis=0)

    return grads


class LSTM(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0, bidirectional=False):
        """
        input_size: 输入x的特性维度， the number of features
        hidden_size: 隐层神经元数量
        num_layers: 隐层数量，每层的神经元数量为hidden_size, 默认为1
        bias: 是否需要偏执，默认True
        dropout: 如果大于0，则在除了最后一层外添加一个dropout层
        bidirectional: 是否为双向RNN,默认False
        """
        super(LSTM, self).__init__(input_size, hidden_size, num_layers, bias, dropout, bidirectional, "LSTM")

    def forward(self, input: Tensor, h0: Tensor, c0: Tensor):
        h_state = [None for _ in range(self.num_layers)]
        c_state = [None for _ in range(self.num_layers)]
        hn = [None for _ in range(self.num_layers + 1)]
        d = [None for _ in range(self.num_layers + 1)]
        hn[0] = input.data

        requires_grad = input.requires_grad or self.weight_h[0].requires_grad or h0.requires_grad or c0.requires_grad

        seq_len, batch_size, _ = input.shape

        dropout_mask = [dropout_mask_factory(self.dropout, seq_len, batch_size, self.hidden_size)
                        for _ in range(self.num_layers - 1)] if self.need_dropout else None

        for layer in range(self.num_layers):
            if self.need_dropout and layer > 0:
                _hn, _cn, _d = self.rnn_layer(requires_grad, self.need_bias,
                                              self.weight_i[layer], self.weight_h[layer], self.bias[layer],
                                              hn[layer] * dropout_mask[layer - 1], h0.data[layer], c0.data[layer],
                                              self.hidden_size)

            else:
                _hn, _cn, _d = self.rnn_layer(requires_grad, self.need_bias,
                                              self.weight_i[layer], self.weight_h[layer], self.bias[layer],
                                              hn[layer], h0.data[layer], c0.data[layer], self.hidden_size)

            h_state[layer] = _hn[-1]
            c_state[layer] = _cn[-1]
            hn[layer + 1] = np.stack(_hn[1::])
            d[layer + 1] = _d

        requires_grads = (input.requires_grad, h0.requires_grad, c0.requires_grad, self.weight_h[0].requires_grad)
        grad_fn = LSTMBackward
        depends_on = []
        if requires_grad:
            depends_on.append(Edge(step_in, ['output', requires_grads, input, h0, c0, hn, d, dropout_mask, self]))

        output = Tensor(hn[-1], requires_grad=requires_grad,
                        grad_fn=grad_fn, depends_on=depends_on, is_simple=False)

        depends_on = []
        if requires_grad:
            depends_on.append(Edge(step_in, ['h_state', requires_grads, input, h0, c0, hn, d, dropout_mask, self]))

        h_state = Tensor(np.stack(h_state), requires_grad=requires_grad,
                         grad_fn=grad_fn, depends_on=depends_on, is_simple=False)

        depends_on = []
        if requires_grad:
            depends_on.append(Edge(step_in, ['c_state', requires_grads, input, h0, c0, hn, d, dropout_mask, self]))

        c_state = Tensor(np.stack(c_state), requires_grad=requires_grad,
                         grad_fn=grad_fn, depends_on=depends_on, is_simple=False)

        return output, h_state, c_state


def LSTMLayerForward(requires_grad: bool, need_bias: bool, w_i: Tensor, w_h: Tensor, bias: Optional[Tensor],
                     input: np.ndarray, h0: np.ndarray, c0: np.ndarray, hidden_size):
    seq_len, batch_size, _ = input.shape
    _w_i = w_i.data
    _w_h = w_h.data
    _bias = bias.data if need_bias else None
    sz = hidden_size
    sz2 = hidden_size * 2
    sz3 = hidden_size * 3

    hn = [0 for _ in range(seq_len + 1)]
    cn = [0 for _ in range(seq_len + 1)]

    b_at = [0 for _ in range(seq_len)] if requires_grad else None
    b_kt = [0 for _ in range(seq_len)] if requires_grad else None
    b_da = [0 for _ in range(seq_len)] if requires_grad else None
    b_dk = [0 for _ in range(seq_len)] if requires_grad else None
    b_do = [0 for _ in range(seq_len)] if requires_grad else None
    b_ot = [0 for _ in range(seq_len)] if requires_grad else None

    hn[0] = h0
    cn[0] = c0

    for seq in range(seq_len):
        if need_bias:
            output_h = input[seq] @ _w_i + hn[seq] @ _w_h + _bias
        else:
            output_h = input[seq] @ _w_i + hn[seq] @ _w_h

        ft = sigmoid(output_h[:, 0:sz])
        it = sigmoid(output_h[:, sz:sz2])
        gt = np.tanh(output_h[:, sz2:sz3])
        ot = sigmoid(output_h[:, sz3:])

        cn[seq + 1] = ft * cn[seq] + it * gt
        kt = np.tanh(cn[seq + 1])
        hn[seq + 1] = ot * kt

        if requires_grad:
            df = ft * (1 - ft) * cn[seq]
            di = it * (1 - it) * gt
            dg = (1 - gt ** 2) * it
            do = ot * (1 - ot) * kt
            dk = (1 - kt ** 2) * ot

            at = np.concatenate([ft, it, gt], axis=1)
            da = np.concatenate([df, di, dg], axis=1)

            b_at[seq] = at
            b_kt[seq] = kt
            b_da[seq] = da
            b_dk[seq] = dk
            b_ot[seq] = ot
            b_do[seq] = do

    return hn, cn, (b_at, b_kt, b_da, b_dk, b_ot, b_do)


def LSTMBackward(grad: Tensor, depends_on):
    edge = depends_on[0]
    args = edge.args
    backward_type, requires_grads, input, h0, c0, hn, d, dropout_mask, self = args
    seq_len = len(hn[0])
    input_requires_grad, h0_requires_grad, c0_requires_grad, w_requires_grad = requires_grads
    h0_grad = [None for _ in range(self.num_layers)]
    c0_grad = [None for _ in range(self.num_layers)]

    if backward_type == 'output':
        output_grad = grad
        h_grad = [None for _ in range(self.num_layers)]
        c_grad = [None for _ in range(self.num_layers)]
    elif backward_type == 'h_state':
        output_grad = None
        h_grad = grad
        c_grad = [None for _ in range(self.num_layers)]
    else:
        output_grad = None
        c_grad = grad
        h_grad = [None for _ in range(self.num_layers)]

    for layer in reversed(range(self.num_layers)):
        i_rg = input_requires_grad if layer == 0 else True

        output_grad, _h0_grad, _c0_grad = LSTMLayerBackward(output_grad, h_grad[layer], c_grad[layer], hn[layer + 1],
                                                            d[layer + 1], self.weight_i[layer], self.weight_h[layer],
                                                            self.bias[layer], self.need_bias, hn[layer], h0.data[layer],
                                                            i_rg, h0_requires_grad, c0_requires_grad,
                                                            w_requires_grad, seq_len, self.hidden_size)

        h0_grad[layer] = _h0_grad
        c0_grad[layer] = _c0_grad
        if output_grad is not None:
            if layer > 0 and self.need_dropout:
                output_grad = Tensor(np.stack(output_grad) * dropout_mask[layer - 1])
            else:
                output_grad = Tensor(np.stack(output_grad))

    tensor_grad = []
    if input_requires_grad:
        tensor_grad.append((input, output_grad))

    if h0_requires_grad:
        tensor_grad.append((h0, Tensor(np.stack(h0_grad))))

    if c0_requires_grad:
        tensor_grad.append((c0, Tensor(np.stack(c0_grad))))

    return tensor_grad


# output的反向传播
def LSTMLayerBackward(output_grad: Tensor, h_grad: Tensor, c_grad: Tensor,
                      hn: np.ndarray, d: Tuple, w_i: Tensor, w_h: Tensor, bias: Optional[Tensor], need_bias: bool,
                      input: np.ndarray, h0: np.ndarray, input_requires_grad, h0_requires_grad, c0_requires_grad,
                      w_requires_grad, seq_len: int, hidden_size: int) -> List:

    sz = hidden_size
    sz2 = hidden_size * 2
    sz3 = hidden_size * 3

    b_at, b_kt, b_da, b_dk, b_ot, b_do = d
    is_output_backward = True if output_grad else False
    is_hn_backward = True if h_grad else False
    is_cn_backward = True if c_grad else False

    delta_h = [0 for _ in range(seq_len)]
    delta_c = [0 for _ in range(seq_len)]
    delta_h[-1] = 0
    if is_output_backward:
        delta_h[-1] += output_grad.data[-1]

    if is_hn_backward:
        delta_h[-1] += h_grad.data

    if is_cn_backward:
        delta_c[-1] = c_grad.data

    # 再计算上面的delta，没有上一层的梯度
    w_h_o = w_h.data[:, sz3:].T
    w_h_a = w_h.data[:, 0:sz3].T
    for seq in reversed(range(0, seq_len - 1)):
        # delta_h
        grad_a = b_da[seq + 1].copy()
        whs = delta_c[seq + 1] + delta_h[seq + 1] * b_dk[seq + 1]
        grad_a[:, 0:sz] *= whs
        grad_a[:, sz:sz2] *= whs
        grad_a[:, sz2:sz3] *= whs

        grad_o = delta_h[seq + 1] * b_do[seq + 1]
        delta_h[seq] = grad_o @ w_h_o + grad_a @ w_h_a

        if is_output_backward:
            delta_h[seq] += output_grad.data[seq]

        # delta_c
        delta_c[seq] = b_at[seq + 1][0:, 0:sz] * whs

    grads = []

    # 计算input的梯度
    if input_requires_grad:
        grad = [0 for _ in range(seq_len)]
        w_h_o = w_i.data[:, sz3:].T
        w_h_a = w_i.data[:, 0:sz3].T
        for seq in range(0, seq_len):
            grad_a = b_da[seq].copy()
            wis = delta_c[seq] + delta_h[seq] * b_dk[seq]
            grad_a[:, 0:sz] *= wis
            grad_a[:, sz:sz2] *= wis
            grad_a[:, sz2:sz3] *= wis

            grad[seq] = delta_h[seq] * b_do[seq] @ w_h_o + grad_a @ w_h_a
    else:
        grad = None
    grads.append(grad)

    if h0_requires_grad:
        grad_a = b_da[0].copy()
        whs = delta_c[0] + delta_h[0] * b_dk[0]
        grad_a[:, 0:sz] *= whs
        grad_a[:, sz:sz2] *= whs
        grad_a[:, sz2:sz3] *= whs
        grad = delta_h[0] * b_do[0] @ w_h.data[:, sz3:].T + grad_a @ w_h.data[:, 0:sz3].T

    else:
        grad = None
    grads.append(grad)

    if c0_requires_grad:
        grad = delta_c[0] * b_at[0][0:, 0:sz] + delta_h[0] * b_at[0][0:, 0:sz] * b_dk[0]
    else:
        grad = None
    grads.append(grad)

    if not w_requires_grad:
        return grads

    for seq in range(0, seq_len):
        # 首先计算 f,i,g
        grad_a = b_da[seq].copy()
        grad_o = b_do[seq].copy() * delta_h[seq]
        wis = delta_c[seq] + delta_h[seq] * b_dk[seq]
        grad_a[:, 0:sz] *= wis
        grad_a[:, sz:sz2] *= wis
        grad_a[:, sz2:sz3] *= wis
        if seq == 0:
            w_h.grad.data[:, 0:sz3] += h0.T @ grad_a
            w_h.grad.data[:, sz3:] += h0.T @ grad_o
        else:
            w_h.grad.data[:, 0:sz3] += hn[seq - 1].T @ grad_a
            w_h.grad.data[:, sz3:] += hn[seq - 1].T @ grad_o

        w_i.grad.data[:, 0:sz3] += input[seq].T @ grad_a
        w_i.grad.data[:, sz3:] += input[seq].T @ grad_o

        if need_bias:
            bias.grad.data[0:sz3] += grad_a.sum(axis=0)
            bias.grad.data[sz3:] += grad_o.sum(axis=0)

    return grads