from ..tensor import Tensor,swapaxes
from ..utils import rand, zeros,stack
from .. import functional as F
from .module import Layer
import numpy as np


class RNNBase(Layer):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 nonlinearity='tanh', batch_first=False, dropout=0, mode="RNN", ):
        super(RNNBase, self).__init__()
        self._parameters= []
        self.hidden_size = hidden_size
        self.need_bias = bias
        self.num_layers = num_layers
        if nonlinearity.lower() == 'tanh':
            self.act_func = F.tanh
        elif nonlinearity.lower() == 'relu':
            self.act_func = F.relu
        else:
            raise RuntimeError('Unsupported activation function')

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
            weight = rand(layer_input_size, gate_size, requires_grad=True, trimmean=True, scale=scale)

            self.weight_i.append(weight)
            self.parameters.append(weight)

            weight = rand(hidden_size, gate_size, requires_grad=True, trimmean=True, scale=scale)
            self.weight_h.append(weight)
            self.parameters.append(weight)

        if self.need_bias:
            for i in range(num_layers):
                bias = rand(gate_size, requires_grad=True, trimmean=True, scale=scale)
                self.bias.append(bias)
                self.parameters.append(bias)


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 nonlinearity='tanh', batch_first=False, dropout=0):
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
        super(RNN, self).__init__(input_size, hidden_size, num_layers, bias,
                                  nonlinearity, batch_first, dropout, "RNN")

    def __call__(self, input: Tensor, h0: Tensor):
        seq_len = input.shape[0]  # (seq, batch, feature)
        if self.batch_first:
            seq_len = input.shape[1]  # (batch, seq, feature)

        # (num_layers, batch_size, hidden_size)

        output = []
        if h0 is not None:
            hn = [h0[i] for i in range(self.num_layers)]
        else:
            batch_size = input.shape[0] if self.batch_first else input.shape[1]
            hn = [zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

        for seq in range(seq_len):
            x = input[:, seq, :] if self.batch_first else input[seq, :, :]
            for i in range(self.num_layers):
                if self.need_bias:
                    output_h = x @ self.weight_i[i] + hn[i] @ self.weight_h[i] + self.bias[i]
                else:
                    output_h = x @ self.weight_i[i] + hn[i] @ self.weight_h[i]
                x = self.act_func(output_h)
                hn[i] = x

                if self.dropout > 0:
                    if i < self.num_layers - 1:
                        x = F.dropout(x, self.dropout)

            output.append(x)

        hn = stack(hn)
        output = stack(output)
        if self.batch_first:
            output = swapaxes(output, 0, 1)

        return output, hn


class LSTM(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 nonlinearity='tanh', batch_first=False, dropout=0):
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
        super(LSTM, self).__init__(input_size, hidden_size, num_layers, bias,
                                   nonlinearity, batch_first, dropout, "LSTM")

    def __call__(self, input: Tensor, h0: Tensor, c0: Tensor):
        # # ii,if,ig,io
        # self.weight_i = [[], [], [], []]
        # self.bias_i = [[], [], [], []]
        #
        # # hi,hf,hg,ho
        # self.weight_h = [[], [], [], []]
        # self.bias_h = [[], [], [], []]

        seq_len = input.shape[0]  # (seq, batch, feature)
        if self.batch_first:
            seq_len = input.shape[1]  # (batch, seq, feature)

        batch_size = input.shape[0] if self.batch_first else input.shape[1]
        # (num_layers, batch_size, hidden_size)
        sz = self.hidden_size
        output = []
        if h0 is not None:
            hn = [h0[i] for i in range(self.num_layers)]
        else:
            hn = [zeros(batch_size, sz) for _ in range(self.num_layers)]

        if c0 is not None:
            cn = [c0[i] for i in range(self.num_layers)]
        else:
            cn = [zeros(batch_size, sz) for _ in range(self.num_layers)]




        for seq in range(seq_len):
            x = input[:, seq, :] if self.batch_first else input[seq, :, :]

            for i in range(self.num_layers):
                if self.need_bias:
                    output_h = x @ self.weight_i[i] + hn[i] @ self.weight_h[i] + self.bias[i]
                else:
                    output_h = x @ self.weight_i[i] + hn[i] @ self.weight_h[i]

                ft = F.sigmoid(output_h[:, 0:sz])
                it = F.sigmoid(output_h[:, sz:2 * sz])
                gt = F.tanh(output_h[:, 2 * sz:3 * sz])
                ot = F.sigmoid(output_h[:, 3 * sz:])


                cn[i] = ft * cn[i] + it * gt
                x = ot * F.tanh(cn[i])
                hn[i] = x

                if self.dropout > 0:
                    if i < self.num_layers - 1:
                        x = F.dropout(x, self.dropout)

            output.append(x)

        hn = stack(hn)
        cn = stack(cn)
        output = stack(output)
        if self.batch_first:
            output = swapaxes(output, 0, 1)

        return output, hn, cn
