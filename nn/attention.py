from typing import Optional, List, Tuple
from ..tensor import Edge, Tensor
from ..utils import rand, zeros, zeros_like
from . import Layer
import numpy as np


class SelfAttention(Layer):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_q = rand(input_size, hidden_size)
        self.w_k = rand(input_size, hidden_size)
        self.w_v = rand(input_size, hidden_size)
        self.parameters = []
        self.parameters.append(self.w_q)
        self.parameters.append(self.w_k)
        self.parameters.append(self.w_v)

    def forward(self, input: Tensor):
        # input.shape: batch_size, seq_len, features
        Q = input @ self.w_q
        K = input @ self.w_k
        V = input @ self.w_v

        A = (K.transpose(1, 2) @ Q) / np.sqrt(self.input_size)
        B = V @ A.softmax(dim=1)

        return B
