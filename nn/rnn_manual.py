# https://blog.csdn.net/qiusuoxiaozi/article/details/54286706
# https://blog.csdn.net/gangyin5071/article/details/79762352

from typing import List
from ..tensor import Edge, Tensor
from ..utils import rand, zeros
from . import Layer
import numpy as np

# 引导进入反向传播的辅助变量
step_in = zeros(1, requires_grad=True)
step_out = zeros(1)


class RNNBase(Layer):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0, mode="RNN"):
        super(RNNBase, self).__init__()
        self.parameters = []
        self.hidden_size = hidden_size
        self.need_bias = bias
        self.num_layers = num_layers

        if dropout > 0 and num_layers == 1:
            raise RuntimeError("dropout option adds dropout after all but last recurrent layer, so non-zero dropout "
                               "expects num_layers greater than 1, but got dropout=%s and num_layers=%s" % (
                                   dropout, num_layers))

        self.dropout = dropout

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
            # 输入门参数
            layer_input_size = input_size if layer == 0 else hidden_size
            weight = rand(layer_input_size, gate_size, requires_grad=True, scale=scale, trimmean=True)
            weight.grad = Tensor(np.zeros_like(weight.data))

            self.weight_i.append(weight)
            self.parameters.append(weight)

            weight = rand(hidden_size, gate_size, requires_grad=True, scale=scale, trimmean=True)
            weight.grad = Tensor(np.zeros_like(weight.data))
            self.weight_h.append(weight)
            self.parameters.append(weight)

        if self.need_bias:
            for i in range(num_layers):
                bias = rand(gate_size, requires_grad=True, scale=scale, trimmean=True)
                bias.grad = Tensor(np.zeros_like(bias.data))
                self.bias.append(bias)
                self.parameters.append(bias)


class LSTM(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0):
        """
        input_size: 输入x的特性维度， the number of features
        hidden_size: 隐层神经元数量
        num_layers: 隐层数量，每层的神经元数量为hidden_size, 默认为1
        bias: 是否需要偏执，默认True
        dropout: 如果大于0，则在除了最后一层外添加一个dropout层
        """
        super(LSTM, self).__init__(input_size, hidden_size, num_layers, bias, dropout, "LSTM")

    def sigmoid(self, data):
        return 1.0 / (1.0 + np.exp(-data))

    def __call__(self, input: Tensor, h0: Tensor, c0: Tensor):
        seq_len, batch_size, _ = input.shape  # (seq, batch, feature)
        grad_fn = LSTMBackward
        sz = self.hidden_size
        # 构建H,C
        # 所有节点的输出
        # h03
        # h02
        # h01
        #  *  x1   x2   x3
        h = []
        cn = [0]
        b_at = []
        b_kt = []
        b_da = []
        b_dk = []
        b_do = []
        b_ot = []

        for i in range(seq_len + 1):
            tmp = [None for _ in range(self.num_layers + 1)]
            h.append(tmp)
            b_at.append(tmp[::])
            b_kt.append(tmp[::])
            b_da.append(tmp[::])
            b_dk.append(tmp[::])
            b_ot.append(tmp[::])
            b_do.append(tmp[::])

        if h0:
            tmp = [h0.data[i] for i in range(self.num_layers)]
        else:
            tmp = [np.zeros((batch_size, self.hidden_size)) for i in range(self.num_layers)]
        h[0][1::] = tmp

        if c0:
            cn += [c0.data[i] for i in range(self.num_layers)]
        else:
            cn += [np.zeros((batch_size, self.hidden_size)) for i in range(self.num_layers)]

        dropout_mask = []
        requires_grad = self.weight_i[0].requires_grad or input.requires_grad or h0.requires_grad or c0.requires_grad
        self.need_dropout = self.training and requires_grad and self.dropout > 0

        # 此时我们需要记录dropout的mask
        if self.need_dropout:
            p = self.dropout
            scale = 1 / (1 - p)
            for seq in range(seq_len):
                masks = []
                for layer in range(self.num_layers - 1):
                    masks.append(scale * np.random.binomial(1, 1 - p, (batch_size, self.hidden_size)))
                dropout_mask.append(masks)

        output = []
        for seq in range(seq_len):
            h[seq + 1][0] = input[seq, :, :].data
            for i in range(self.num_layers):
                if i > 0 and self.need_dropout:
                    if self.need_bias:
                        output_h = h[seq + 1][i] * dropout_mask[seq][i - 1] @ self.weight_i[i].data + h[seq][i + 1] @ \
                                   self.weight_h[i].data + self.bias[i].data
                    else:
                        output_h = h[seq + 1][i] * dropout_mask[seq][i - 1] @ self.weight_i[i].data + h[seq][i + 1] @ \
                                   self.weight_h[i].data
                else:
                    if self.need_bias:
                        output_h = h[seq + 1][i] @ self.weight_i[i].data + h[seq][i + 1] @ self.weight_h[i].data + \
                                   self.bias[i].data
                    else:
                        output_h = h[seq + 1][i] @ self.weight_i[i].data + h[seq][i + 1] @ self.weight_h[i].data

                ft = self.sigmoid(output_h[:, 0:sz])
                it = self.sigmoid(output_h[:, sz:2 * sz])
                gt = np.tanh(output_h[:, 2 * sz:3 * sz])
                ot = self.sigmoid(output_h[:, 3 * sz:])

                ct0 = cn[i + 1]

                ct = ft * ct0 + it * gt
                kt = np.tanh(ct)
                h[seq + 1][i + 1] = ot * kt
                cn[i + 1] = ct

                if requires_grad:
                    df = ft * (1 - ft) * ct0
                    di = it * (1 - it) * gt
                    dg = (1 - gt ** 2) * it
                    do = ot * (1 - ot) * kt
                    dk = (1 - kt ** 2) * ot

                    at = np.concatenate([ft, it, gt], axis=1)
                    da = np.concatenate([df, di, dg], axis=1)

                    b_at[seq + 1][i + 1] = at
                    b_kt[seq + 1][i + 1] = kt
                    b_da[seq + 1][i + 1] = da
                    b_dk[seq + 1][i + 1] = dk
                    b_ot[seq + 1][i + 1] = ot
                    b_do[seq + 1][i + 1] = do

            output.append(h[seq + 1][-1])

        depends_on = []
        if requires_grad:
            depends_on.append(
                Edge(step_in,
                     ['output', self, h, [b_at, b_ot, b_kt, b_da, b_do, b_dk], dropout_mask, input, h0, c0]))

        output = Tensor(np.stack(output),
                        requires_grad=requires_grad,
                        depends_on=depends_on,
                        grad_fn=grad_fn,
                        is_simple=False)

        depends_on = []
        if requires_grad:
            depends_on = [
                Edge(step_in,
                     ['hn', self, h, [b_at, b_ot, b_kt, b_da, b_do, b_dk], dropout_mask, input, h0, c0])]

        hn = Tensor(np.stack(h[-1][1::]),
                    requires_grad=requires_grad,
                    depends_on=depends_on,
                    grad_fn=grad_fn,
                    is_simple=False)

        depends_on = []
        if requires_grad:
            depends_on.append(
                Edge(step_in,
                     ['cn', self, h, [b_at, b_ot, b_kt, b_da, b_do, b_dk], dropout_mask, input, h0, c0]))

        cn = Tensor(np.stack(cn[1::]),
                    requires_grad=requires_grad,
                    depends_on=depends_on,
                    grad_fn=grad_fn,
                    is_simple=False)

        return output, hn, cn


def LSTMBackward(output_grad: Tensor, t: 'Tensor', args: List) -> 'Tensor':
    backward_type, self, h, bridge, dropout_mask, input, h0, c0 = args
    b_at, b_ot, b_kt, b_da, b_do, b_dk = bridge
    # h = other_args[2]  # shape: seq + 1,num_layers+1,batch_size, hidden_size
    seq_len = len(h) - 1
    num_layers = len(h[0]) - 1
    batch_size, sz = h[1][1].shape  # sz = hidden_size
    sz3 = sz * 3
    sz2 = sz * 2

    is_output_backward = False
    is_hn_backward = False
    is_cn_backward = False
    if backward_type == 'output':
        is_output_backward = True
    elif backward_type == 'hn':
        is_hn_backward = True
    else:
        is_cn_backward = True

    # 反向传播梯度
    # 与 h,c保持一致
    delta_h = []
    delta_c = []
    for seq in range(seq_len + 1):
        tmp = [0 for _ in range(num_layers + 1)]
        delta_h.append(tmp)
        delta_c.append(tmp[::])

    # 先计算右上角处的delta，没有后一时刻的梯度
    # delta_h[-1][-1] = o_grad[-1] + h_grad[-1]
    # delta_c[-1][-1] = c_grad[-1]
    # ==>
    grad = output_grad.data
    if is_cn_backward:
        delta_c[-1][-1] = grad[-1]
    else:
        delta_h[-1][-1] = grad[-1]

    # 再计算右边的delta，没有后一时刻的梯度
    for layer in range(self.num_layers - 1, 0, -1):

        grad_a = b_da[-1][layer + 1].copy()
        delta = delta_h[-1][layer + 1]
        wis = delta_c[-1][layer + 1] + delta * b_dk[-1][layer + 1]

        grad_a[:, 0:sz] *= wis
        grad_a[:, sz:sz2] *= wis
        grad_a[:, sz2:sz3] *= wis

        grad_o = delta * b_do[-1][layer + 1]
        data = grad_o @ self.weight_i[layer].data[:, sz3:].T + \
               grad_a @ self.weight_i[layer].data[:, 0:sz3].T

        if self.need_dropout:
            data *= dropout_mask[-1][layer - 1]

        # delta_h[-1][layer] = h_grad[layer-1] + data
        # # delta_c 不会沿着layer层传播
        # delta_c[-1][layer] = c_grad[layer-1]

        # ===>
        if is_hn_backward:
            # 隐藏状态hn的梯度也要加上
            delta_h[-1][layer] = grad[layer - 1] + data
        else:
            delta_h[-1][layer] = data

        # delta_c 不会沿着layer层传播
        if is_cn_backward:
            # 隐藏状态cn的梯度也要加上
            delta_c[-1][layer] = grad[layer - 1]

    # 再计算上面的delta，没有上一层的梯度
    w_h_o = self.weight_h[-1].data[:, sz3:].T
    w_h_a = self.weight_h[-1].data[:, 0:sz3].T
    for seq in range(seq_len - 1, 0, -1):

        # delta_h
        grad_a = b_da[seq + 1][-1].copy()
        whs = delta_c[seq + 1][-1] + delta_h[seq + 1][-1] * b_dk[seq + 1][-1]
        grad_a[:, 0:sz] *= whs
        grad_a[:, sz:sz2] *= whs
        grad_a[:, sz2:sz3] *= whs

        grad_o = delta_h[seq + 1][-1] * b_do[seq + 1][-1]
        data = grad_o @ w_h_o + grad_a @ w_h_a

        # delta_h[seq][-1] = o_grad[seq - 1] + data
        # ===>
        if is_output_backward:
            # 输出的梯度也是加上
            delta_h[seq][-1] = grad[seq - 1] + data
        else:
            delta_h[seq][-1] = data

        # delta_c->ok
        delta_c[seq][-1] = b_at[seq + 1][-1][0:, 0:sz] * whs

    for layer in range(self.num_layers - 1, 0, -1):

        w_i_a = self.weight_i[layer].data[:, 0:sz3].T
        w_i_o = self.weight_i[layer].data[:, sz3:].T
        w_h_a = self.weight_h[layer - 1].data[:, 0:sz3].T
        w_h_o = self.weight_h[layer - 1].data[:, sz3:].T

        for seq in range(seq_len - 1, 0, -1):
            # ####从上层传递到该层#########
            # ##{begin}
            grad_a = b_da[seq][layer + 1].copy()
            delta = delta_h[seq][layer + 1]
            wis = delta_c[seq][layer + 1] + delta * b_dk[seq][layer + 1]

            grad_a[:, 0:sz] *= wis
            grad_a[:, sz:sz2] *= wis
            grad_a[:, sz2:sz3] *= wis
            grad_o = delta * b_do[seq][layer + 1]

            data_1 = grad_o @ w_i_o + grad_a @ w_i_a

            if self.need_dropout:
                data_1 *= dropout_mask[seq - 1][layer - 1]

            # ##{end}

            # ####从上一时刻传递到该时刻#########
            # ##{begin}
            grad_a = b_da[seq + 1][layer].copy()
            whs = delta_c[seq + 1][layer] + delta_h[seq + 1][layer] * b_dk[seq + 1][layer]
            grad_a[:, 0:sz] *= whs
            grad_a[:, sz:sz2] *= whs
            grad_a[:, sz2:sz3] *= whs
            grad_o = delta_h[seq + 1][layer] * b_do[seq + 1][layer]

            data_2 = grad_o @ w_h_o + grad_a @ w_h_a

            delta_h[seq][layer] = data_1 + data_2
            # ##{end}

            # delta_c->ok
            delta_c[seq][layer] = b_at[seq + 1][layer][0:, 0:sz] * whs

    # ###############################反向传播梯度#################################
    # RNN参数梯度计算
    if self.weight_h[0].requires_grad:
        for layer in range(num_layers):
            for seq in range(1, seq_len + 1):
                # 首先计算 f,i,g
                grad_a = b_da[seq][layer + 1].copy()
                wis = delta_c[seq][layer + 1] + delta_h[seq][layer + 1] * b_dk[seq][layer + 1]
                grad_a[:, 0:sz] *= wis
                grad_a[:, sz:sz2] *= wis
                grad_a[:, sz2:sz3] *= wis
                self.weight_h[layer].grad.data[:, 0:sz3] += h[seq - 1][layer + 1].T @ grad_a
                # 计算o
                grad_o = b_do[seq][layer + 1].copy() * delta_h[seq][layer + 1]
                self.weight_h[layer].grad.data[:, sz3:] += h[seq - 1][layer + 1].T @ grad_o

                self.weight_i[layer].grad.data[:, 0:sz3] += h[seq][layer].T @ grad_a
                self.weight_i[layer].grad.data[:, sz3:] += h[seq][layer].T @ grad_o

                if self.need_bias:
                    self.bias[layer].grad.data[0:sz3] += grad_a.sum(axis=0)
                    self.bias[layer].grad.data[sz3:] += grad_o.sum(axis=0)

    tensor_grad = []
    if input.requires_grad:
        grad_data = [0 for _ in range(seq_len)]
        w_h_o = self.weight_i[0].data[:, sz3:].T
        w_h_a = self.weight_i[0].data[:, 0:sz3].T
        for seq in range(1, seq_len + 1):
            grad_a = b_da[seq][1].copy()
            wis = delta_c[seq][1] + delta_h[seq][1] * b_dk[seq][1]
            grad_a[:, 0:sz] *= wis
            grad_a[:, sz:sz2] *= wis
            grad_a[:, sz2:sz3] *= wis

            grad_data[seq - 1] = delta_h[seq][1] * b_do[seq][1] @ w_h_o + grad_a @ w_h_a

        tensor_grad.append(input, Tensor(np.stack(grad_data)))

    if h0 is not None:
        if h0.requires_grad:
            grad_data = [0 for _ in range(num_layers)]
            for layer in range(1, num_layers + 1):
                grad_a = b_da[1][layer].copy()
                whs = delta_c[1][layer] + delta_h[1][layer] * b_dk[1][layer]
                grad_a[:, 0:sz] *= whs
                grad_a[:, sz:sz2] *= whs
                grad_a[:, sz2:sz3] *= whs

                grad_data[layer - 1] = delta_h[1][layer] * b_do[1][layer] @ \
                                       self.weight_h[layer - 1].data[:, sz3:].T + \
                                       grad_a @ self.weight_h[layer - 1].data[:, 0:sz3].T

        tensor_grad.append(h0, Tensor(np.stack(grad_data)))

    if c0 is not None:
        if c0.requires_grad:
            grad_data = [0 for _ in range(num_layers)]
            for layer in range(1, num_layers + 1):
                grad_data[layer - 1] = delta_c[1][layer] * b_at[1][layer][0:, 0:sz] + \
                                       delta_h[1][layer] * b_at[1][layer][0:, 0:sz] * b_dk[1][layer]

        tensor_grad.append(c0, Tensor(np.stack(grad_data)))

    return tensor_grad


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0):
        """
        input_size: 输入x的特性维度， the number of features
        hidden_size: 隐层神经元数量
        num_layers: 隐层数量，每层的神经元数量为hidden_size, 默认为1
        bias: 是否需要偏执，默认True
        dropout: 如果大于0，则在除了最后一层外添加一个dropout层
        """
        super(RNN, self).__init__(input_size, hidden_size, num_layers, bias, dropout, "RNN")

    def __call__(self, input: Tensor, h0: Tensor):
        seq_len, batch_size, _ = input.shape  # (seq, batch, feature)
        requires_grad = True
        depends_on = []
        grad_fn = RNNBackward

        # 构建H
        # 所有节点的输出
        h = []
        for i in range(seq_len + 1):
            tmp = [None for _ in range(self.num_layers + 1)]
            h.append(tmp)
        # h03
        # h02
        # h01
        #  *  x1   x2   x3
        if h0:
            tmp = [h0.data[i] for i in range(self.num_layers)]
        else:
            tmp = [np.zeros((batch_size, self.hidden_size)) for i in range(self.num_layers)]

        h[0][1::] = tmp
        for seq in range(seq_len):
            h[seq + 1][0] = input[seq, :, :].data

        self.need_dropout = self.dropout > 0 and self.training and requires_grad
        dropout_mask = []
        # 此时我们需要记录dropout的mask
        if self.need_dropout:
            p = self.dropout
            scale = 1 / (1 - p)
            for seq in range(seq_len):
                masks = []
                for layer in range(self.num_layers - 1):
                    masks.append(scale * np.random.binomial(1, 1 - p, (batch_size, self.hidden_size)))
                dropout_mask.append(masks)

        output = []
        for seq in range(seq_len):
            for i in range(0, self.num_layers):
                if i > 0 and self.need_dropout:
                    if self.need_bias:
                        output_h = h[seq + 1][i] @ self.weight_i[i].data + h[seq][i + 1] @ \
                                   self.weight_h[i].data + self.bias[i].data
                    else:
                        output_h = h[seq + 1][i] @ self.weight_i[i].data + h[seq][i + 1] @ \
                                   self.weight_h[i].data
                else:
                    if self.need_bias:
                        output_h = h[seq + 1][i] @ self.weight_i[i].data + h[seq][i + 1] @ self.weight_h[i].data + \
                                   self.bias[i].data
                    else:
                        output_h = h[seq + 1][i] @ self.weight_i[i].data + h[seq][i + 1] @ self.weight_h[i].data

                h[seq + 1][i + 1] = np.tanh(output_h)

            output.append(h[seq + 1][-1])

        depends_on = []
        if requires_grad:
            depends_on.append(Edge(step_in, ['output', self, h, dropout_mask, input, h0]))
            output = Tensor(np.stack(output),
                            requires_grad=requires_grad,
                            depends_on=depends_on,
                            grad_fn=grad_fn,
                            is_simple=False)

        depends_on = []
        if requires_grad:
            depends_on.append(Edge(step_in, ['hn', self, h, dropout_mask, input, h0]))

        hn = Tensor(np.stack(h[-1][1::]),
                    requires_grad=requires_grad,
                    depends_on=depends_on,
                    grad_fn=grad_fn,
                    is_simple=False)

        return output, hn


def RNNBackward(output_grad: 'Tensor',depends_on:List) -> 'Tensor':
    edge = depends_on[0]
    backward_type, self, h, dropout_mask, input, h0 = edge.args
    seq_len = len(h) - 1
    num_layers = len(h[0]) - 1
    batch_size, hidden_size = h[1][1].shape

    # 计算从输出传入的梯度的计算
    # output_grad的shape和output的shape是一致的，(seq, batch, hidden_size)
    # hn_grad的shape和hn的shape是一致的，(num_layers, batch, hidden_size)
    if 'output' in backward_type:
        o_grad = output_grad.data
        h_grad = np.zeros((num_layers, batch_size, hidden_size))
    else:
        o_grad = np.zeros((seq_len, batch_size, hidden_size))
        h_grad = output_grad.data

    # ###############################反向传播梯度#################################
    delta = []
    for seq in range(seq_len + 1):
        tmp = [0 for _ in range(num_layers + 1)]
        delta.append(tmp)

    # 先计算右上角处的delta，没有后一时刻的梯度
    delta[-1][-1] = o_grad[-1] + h_grad[-1]

    # 再计算右边的delta，没有后一时刻的梯度

    for layer in range(self.num_layers - 1, 0, -1):
        delta[-1][layer] = h_grad[layer] + (1 - h[-1][layer + 1] ** 2) * delta[-1][layer + 1] @ \
                           self.weight_i[layer].data.T

        if self.dropout > 0:
            delta[-1][layer] *= dropout_mask[-1][layer - 1]

    # 再计算上面的delta，没有上一层的梯度
    for seq in range(seq_len - 1, 0, -1):
        delta[seq][-1] = o_grad[seq] + (1 - h[seq + 1][-1] ** 2) * delta[seq + 1][-1] @ self.weight_h[-1].data.T

    # 内部
    for layer in range(self.num_layers - 1, 0, -1):
        for seq in range(seq_len - 1, 0, -1):
            data_1 = (1 - h[seq][layer + 1] ** 2) * delta[seq][layer + 1] @ self.weight_i[layer].data.T
            if self.dropout > 0:
                data_1 *= dropout_mask[seq - 1][layer - 1]

            delta[seq][layer] = data_1 + \
                                (1 - h[seq + 1][layer] ** 2) * delta[seq + 1][layer] @ self.weight_h[
                                    layer - 1].data.T

    # ###############################反向传播梯度#################################

    # RNN参数梯度计算
    if self.weight_i[0].requires_grad:
        for layer in range(num_layers):
            for seq in range(1, seq_len + 1):
                self.weight_i[layer].grad.data += h[seq][layer].T @ (
                        delta[seq][layer + 1] * (1 - h[seq][layer + 1] ** 2))

                self.weight_h[layer].grad.data += h[seq - 1][layer + 1].T @ (
                        delta[seq][layer + 1] * (1 - h[seq][layer + 1] ** 2))

            if self.need_bias:
                for seq in range(1, seq_len + 1):
                    self.bias[layer].grad.data += (delta[seq][layer + 1] * (1 - h[seq][layer + 1] ** 2)).sum(axis=0)

    tensor_grad = []
    if input.requires_grad:
        grad_data = [0 for _ in range(seq_len)]
        for seq in range(1, seq_len + 1):
            grad_data[seq - 1] = (delta[seq][1] * (1 - h[seq][1] ** 2)) @ self.weight_i[0].data.T
        tensor_grad.append((input, Tensor(np.stack(grad_data))))

    if h0 is None:
        return tensor_grad

    if h0.requires_grad:
        grad_data = [0 for _ in range(num_layers)]
        for layer in range(1, num_layers + 1):
            grad_data[layer - 1] = (delta[1][layer] * (1 - h[1][layer] ** 2)) @ self.weight_h[layer - 1].data.T
        tensor_grad.append((h0, Tensor(np.stack(grad_data))))
    return tensor_grad
