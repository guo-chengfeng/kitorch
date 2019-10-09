from .. import tensor as mt, Dependency, Tensor
from .. import functional as F
from . import Layer
import numpy as np


class Cell:
    def __init__(self, w_i=None, w_h=None, bias=None, hidden_size=0):
        self.w_i = w_i
        self.w_h = w_h
        self.bias = bias
        self.need_bias = False if bias is None else True
        self.hidden_size = hidden_size
        self.num = 0

class RNNBase(Layer):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, mode="RNN", ):
        super(RNNBase, self).__init__()
        self.parameters = []
        self.hidden_size = hidden_size
        self.need_bias = bias
        self.num_layers = num_layers
        self.batch_first = batch_first

        assert 1 > dropout >= 0, "dropout should >=0 and < 1"
        self.dropout = dropout

        if mode == "LSTM":
            gate_size = 4 * hidden_size
        elif mode == "GRU":
            gate_size = 2 * hidden_size
        else:
            gate_size = hidden_size

        self.weight_h = []
        self.weight_i = []
        self.bias = []

        scale = 1.0 / np.sqrt(self.hidden_size)

        for layer in range(num_layers):
            # 输入门参数
            layer_input_size = input_size if layer == 0 else hidden_size
            weight = mt.rand(layer_input_size, gate_size, requires_grad=True,trimmean=True, scale=scale)

            self.weight_i.append(weight)
            self.parameters.append(weight)

            weight = mt.rand(hidden_size, gate_size, requires_grad=True, trimmean=True, scale=scale)
            self.weight_h.append(weight)
            self.parameters.append(weight)

        if self.need_bias:
            for i in range(num_layers):
                bias = mt.rand(gate_size, requires_grad=True,trimmean=True, scale=scale)
                self.bias.append(bias)
                self.parameters.append(bias)
        else:
            self.bias = None

        if mode == "LSTM":
            self.cell = self.cell = LSTMCell(self.weight_i, self.weight_h, self.bias, self.hidden_size)
        elif mode == "GRU":
            self.cell = None
        else:
            self.cell = BasicRNNCell(self.weight_i, self.weight_h, self.bias, self.hidden_size)



class LSTMCell(Cell):
    def __init__(self, w_i=None, w_h=None, bias=None, hidden_size=0):
        super(LSTMCell, self).__init__(w_i, w_h, bias, hidden_size)

    def sigmoid(self, data):
        return 1.0 / (1.0 + np.exp(-data))

    def __call__(self, layer, x: Tensor, h: Tensor, c: Tensor):
        sz = self.hidden_size
        if self.need_bias:
            output_h = x.data @ self.w_i[layer].data + h.data @ self.w_h[layer].data + self.bias[layer].data
        else:
            output_h = x.data @ self.w_i[layer].data + h.data @ self.w_h[layer].data

        ft = self.sigmoid(output_h[:, 0:sz])
        it = self.sigmoid(output_h[:, sz:2 * sz])
        gt = np.tanh(output_h[:, 2 * sz:3 * sz])
        ot = self.sigmoid(output_h[:, 3 * sz:])

        ct = ft * c.data + it * gt
        kt = np.tanh(ct)
        ht = ot * kt

        df = ft * (1 - ft) * c.data
        di = it * (1 - it) * gt
        dg = (1 - gt ** 2) * it
        do = ot * (1 - ot) * kt
        dk = (1 - kt ** 2) * ot

        at = np.concatenate([ft, it, gt], axis=1)
        da = np.concatenate([df, di, dg], axis=1)

        bridge_data = [at, ot, kt, da, do, dk]

        requires_grad = True
        grad_fn = self.backward

        depends_on = [Dependency(self.w_i[0], [layer, bridge_data, 'hw', x, h, c])]
        if x.requires_grad:
            depends_on.append(Dependency(x, [layer, bridge_data, 'hx']))
        if h.requires_grad:
            depends_on.append(Dependency(h, [layer, bridge_data, 'hh']))
        if c.requires_grad:
            depends_on.append(Dependency(c, [layer, bridge_data, 'hc']))

        h_tensor = Tensor(ht, requires_grad, depends_on, grad_fn)

        depends_on = [Dependency(self.w_i[0], [layer, bridge_data, 'cw', x, h, c])]
        if x.requires_grad:
            depends_on.append(Dependency(x, [layer, bridge_data, 'cx']))
        if h.requires_grad:
            depends_on.append(Dependency(h, [layer, bridge_data, 'ch']))
        if c.requires_grad:
            depends_on.append(Dependency(c, [layer, bridge_data, 'cc']))

        c_tensor = Tensor(ct, requires_grad, depends_on, grad_fn)

        return h_tensor, c_tensor

    def backward(self, grad: Tensor, t: Tensor, other_args) -> Tensor:
        self.num += 1

        sz = self.hidden_size
        layer = other_args[0]
        at, ot, kt, da, do, dk = other_args[1]

        backward_type = other_args[2]
        # print("step into backward %s times, %s layer it's type is '%s'" % (self.num,layer, backward_type))
        # ok
        if backward_type == 'cw':
            # grad,shape = batch_size,hidden_zie
            # da.shape = batch_size,hidden_zie*3
            delta = da.copy()
            for i in range(3):
                delta[:, sz * i:sz * (i + 1)] *= grad.data

            x = other_args[3]
            h = other_args[4]

            self.w_i[layer].grad.data[:, 0:3 * sz] += x.data.T @ delta
            self.w_h[layer].grad.data[:, 0:3 * sz] += h.data.T @ delta
            if self.need_bias:
                self.bias[layer].grad.data[0:3 * sz] += delta.sum(axis=0)

            return mt.zeros_like(t)

        if backward_type == 'cx':
            delta = da.copy()
            for i in range(3):
                delta[:, sz * i:sz * (i + 1)] *= grad.data

            data = delta @ self.w_i[layer].data[:, 0:3 * sz].T

            return Tensor(data)

        if backward_type == 'ch':
            delta = da.copy()
            for i in range(3):
                delta[:, sz * i:sz * (i + 1)] *= grad.data
            data = delta @ self.w_h[layer].data[:, 0:3 * sz].T
            return Tensor(data)

        if backward_type == 'cc':
            # dc/dc_{t-1} = f
            return Tensor(grad.data * at[:, 0:sz])

        if backward_type == 'hw':
            # grad,shape = batch_size,hidden_zie
            # da.shape = batch_size,hidden_zie*3
            delta = da.copy()
            for i in range(3):
                delta[:, sz * i:sz * (i + 1)] *= grad.data * dk

            x = other_args[3]
            h = other_args[4]

            self.w_i[layer].grad.data[:, 0:3 * sz] += x.data.T @ delta
            self.w_h[layer].grad.data[:, 0:3 * sz] += h.data.T @ delta
            if self.need_bias:
                self.bias[layer].grad.data[0:3 * sz] += delta.sum(axis=0)

            delta = grad.data * do
            self.w_i[layer].grad.data[:, 3 * sz:] += x.data.T @ delta
            self.w_h[layer].grad.data[:, 3 * sz:] += h.data.T @ delta
            if self.need_bias:
                self.bias[layer].grad.data[3 * sz:] += delta.sum(axis=0)

            return mt.zeros_like(t)

        if backward_type == 'hx':
            delta = da.copy()
            for i in range(3):
                delta[:, sz * i:sz * (i + 1)] *= grad.data*dk

            data = delta @ self.w_i[layer].data[:, 0:3 * sz].T
            data += grad.data * do @ self.w_i[layer].data[:, 3 * sz::].T
            return Tensor(data)

        if backward_type == 'hh':
            delta = da.copy()
            for i in range(3):
                delta[:, sz * i:sz * (i + 1)] *= grad.data*dk

            data = delta @ self.w_h[layer].data[:, 0:3 * sz].T
            data += grad.data * do @ self.w_h[layer].data[:, 3 * sz::].T
            return Tensor(data)

        if backward_type == 'hc':
            # dc/dc_{t-1} = f
            return Tensor(grad.data * dk * at[:, 0:sz])






class LSTM(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0):
        """
        input_size: 输入x的特性维度， the number of features
        hidden_size: 隐层神经元数量
        num_layers: 隐层数量，每层的神经元数量为hidden_size, 默认为1
        nonlinearity: tanh或者relu激活函数，默认tanh
        bias: 是否需要偏执，默认True
        batch_first: 如果是True,输入x的形状应该为(batch, seq, feature)
                     如果是False,(seq, batch, feature)，默认是False

        dropout: 如果大于0，则在除了最后一层外添加一个dropout层
        """
        super(LSTM, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, "LSTM")

    def __call__(self, input: mt.Tensor, h0: mt.Tensor, c0: mt.Tensor):
        seq_len = input.shape[0]  # (seq, batch, feature)
        if self.batch_first:
            seq_len = input.shape[1]  # (batch, seq, feature)

        # (num_layers, batch_size, hidden_size)
        batch_size = input.shape[0] if self.batch_first else input.shape[1]
        output = []
        if h0 is not None:
            hn = [h0[i] for i in range(self.num_layers)]
        else:
            hn = [mt.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]


        if c0 is not None:
            cn = [c0[i] for i in range(self.num_layers)]
        else:
            cn = [mt.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

        for seq in range(seq_len):
            x = input[:, seq, :] if self.batch_first else input[seq, :, :]
            for layer in range(self.num_layers):

                x, c = self.cell(layer, x, hn[layer], cn[layer])

                hn[layer] = x
                cn[layer] = c

                if self.dropout > 0:
                    if layer < self.num_layers - 1:
                        x = F.dropout(x, self.dropout)

            output.append(x)


        hn = mt.stack(hn)
        cn = mt.stack(cn)
        output = mt.stack(output)
        if self.batch_first:
            output = mt.swapaxes(output, 0, 1)

        return output, hn,cn







































class BasicRNNCell(Cell):
    def __init__(self, w_i=None, w_h=None, bias=None, hidden_size=0):
        super(BasicRNNCell, self).__init__(w_i, w_h, bias, hidden_size)

    def __call__(self, layer, x: Tensor, h: Tensor):

        if self.need_bias:
            output_h = x.data @ self.w_i[layer].data + h.data @ self.w_h[layer].data + self.bias[layer].data
        else:
            output_h = x.data @ self.w_i[layer].data + h.data @ self.w_h[layer].data

        data = np.tanh(output_h)
        requires_grad = True
        grad_fn = self.backward

        tanh2 = 1 - data ** 2
        depends_on = [Dependency(self.w_i[0], [layer, tanh2, 'w', x, h])]
        if x.requires_grad:
            depends_on.append(Dependency(x, [layer, tanh2, 'x']))
        if h.requires_grad:
            depends_on.append(Dependency(h, [layer, tanh2, 'h']))

        return Tensor(data, requires_grad, depends_on, grad_fn)

    def backward(self, grad: Tensor, t: Tensor, other_args) -> Tensor:
        self.num += 1
        layer = other_args[0]
        tanh2 = other_args[1]
        backward_type = other_args[2]
        delta = grad.data * tanh2

        if backward_type == 'w':
            x = other_args[3]
            h = other_args[4]
            # 计算bias的梯度
            if self.need_bias:
                # delta的第一个维度是batch
                self.bias[layer].grad.data += delta.sum(axis=0)

            self.w_i[layer].grad.data += x.data.T @ delta
            self.w_h[layer].grad.data += h.data.T @ delta

            return mt.zeros_like(t)

        elif backward_type == 'x':
            # 计算x的梯度
            return Tensor(delta @ self.w_i[layer].data.T)
        else:
            # 计算h的梯度
            return Tensor(delta @ self.w_h[layer].data.T)


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0):
        """
        input_size: 输入x的特性维度， the number of features
        hidden_size: 隐层神经元数量
        num_layers: 隐层数量，每层的神经元数量为hidden_size, 默认为1
        nonlinearity: tanh或者relu激活函数，默认tanh
        bias: 是否需要偏执，默认True
        batch_first: 如果是True,输入x的形状应该为(batch, seq, feature)
                     如果是False,(seq, batch, feature)，默认是False

        dropout: 如果大于0，则在除了最后一层外添加一个dropout层
        """
        super(RNN, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, "RNN")

    def __call__(self, input: mt.Tensor, h0: mt.Tensor):
        seq_len = input.shape[0]  # (seq, batch, feature)
        if self.batch_first:
            seq_len = input.shape[1]  # (batch, seq, feature)

        # (num_layers, batch_size, hidden_size)
        batch_size = input.shape[0] if self.batch_first else input.shape[1]
        output = []
        if h0 is not None:
            hn = [h0[i] for i in range(self.num_layers)]
        else:
            hn = [mt.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

        for seq in range(seq_len):
            x = input[:, seq, :] if self.batch_first else input[seq, :, :]
            for layer in range(self.num_layers):
                x = self.cell(layer, x, hn[layer])
                hn[layer] = x

                if self.dropout > 0:
                    if layer < self.num_layers - 1:
                        x = F.dropout(x, self.dropout)

            output.append(x)

        hn = mt.stack(hn)
        output = mt.stack(output)
        if self.batch_first:
            output = mt.swapaxes(output, 0, 1)

        return output, hn
