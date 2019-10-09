import numpy as np
from ..tensor import Tensor
from ..utils import rand
from .layer import Layer


class Linear(Layer):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.need_bias = bias

        self.parameters = []
        scale = 1.0 / np.sqrt(out_features)
        self.weight = rand(in_features, out_features, requires_grad=True,
                           shift=-0.5, scale=scale)

        self.parameters.append(self.weight)
        if bias:
            self.bias = rand(out_features, requires_grad=True, shift=-0.5, scale=scale)
            self.parameters.append(self.bias)

    def forward(self, x: Tensor):
        if self.bias:
            return x @ self.weight + self.bias
        else:
            return x @ self.weight

    def __repr__(self):
        string = "Linear(in_features=%s, out_features=%s" % (self.in_features, self.out_features)
        if self.need_bias:
            string += ",bias=True"
        string += ")"
        return string
