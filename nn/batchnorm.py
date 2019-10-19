from ..utils import rand, zeros
from ..tensor import Edge, Tensor
from .module import Layer
import numpy as np


class BatchNorm(Layer):
    def __init__(self, num_features, dim=2, affine=True, momentum=0.1, eps=1e-5, track_running_stats=True):
        super(BatchNorm, self).__init__()
        assert dim == 2 or dim == 3 or dim == 4, 'expected 2D,3D or 4D input (got {}D input)'.format(dim)
        self.num_features = num_features
        self.dim = dim
        shape = [1 for _ in range(dim)]
        shape[1] = num_features

        # do sum at which axes
        reduce_axis = [i for i in range(dim)]
        reduce_axis.pop(1)
        self.reduce_axis = tuple(reduce_axis)

        self.affine = affine
        self.eps = eps
        self._parameters = []
        if self.affine:
            self.weight = rand(*shape, requires_grad=True)
            self.bias = zeros(*shape, requires_grad=True)

            self._parameters.append(self.weight)
            self._parameters.append(self.bias)

        else:
            self.weight = None
            self.bias = None

        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.running_data = [None, None]

        if self.track_running_stats:
            # running_mean,running_var
            self.running_data = [np.zeros(shape), np.ones(shape)]

    @property
    def running_mean(self):
        return self.running_data[0]

    @property
    def running_var(self):
        return self.running_data[1]

    def forward(self, input):
        if self.training:
            if self.affine:
                return l2normalizationTrain(input, self) * self.weight + self.bias
            else:
                return l2normalizationTrain(input, self)
        else:
            if self.affine:
                return l2normalizationEval(input, self) * self.weight + self.bias
            else:
                return l2normalizationEval(input, self)

    def __repr__(self):
        string = "BatchNorm(num_features=%s, dim=%s, eps=%s" % (self.num_features, self.dim, self.eps)
        if self.affine:
            string += ",affine=True"
        if self.track_running_stats:
            string += ",momentum=%s,track_running_stats=True" % self.momentum
        string += ")"
        return string


def L2NormalizationBackward(out_grad: 'Tensor', t: 'Tensor', other_args) -> 'Tensor':
    reduce_axis, num_unit, scale, m, y_hat = other_args
    delta = scale * out_grad.data
    delta_1 = m * (delta - y_hat * ((delta * y_hat).sum(axis=reduce_axis, keepdims=True)))
    grad = delta_1 - delta_1.sum(axis=reduce_axis, keepdims=True) / num_unit
    return Tensor(grad)


def l2normalizationEval(input: Tensor, bn) -> Tensor:
    # 正向计算
    x = input.data
    shape = x.shape
    assert bn.dim == len(shape), ValueError('mismatch dims: excepted %sD input, but got %sD' % (bn.dim, len(shape)))

    num_unit = 1
    reduce_axis = bn.reduce_axis
    for axis in reduce_axis:
        num_unit *= shape[axis]
    if bn.track_running_stats:
        mean = bn.running_mean
    else:
        mean = x.mean(axis=reduce_axis, keepdims=True)

    x_hat = x - mean
    if bn.track_running_stats:
        var = bn.running_var
    else:
        var = (x_hat ** 2 + bn.eps).sum(axis=reduce_axis, keepdims=True) / num_unit
    return Tensor(x_hat / (np.sqrt(var)))


def l2normalizationTrain(input: Tensor, bn) -> Tensor:
    # 正向计算
    x = input.data
    shape = x.shape
    assert bn.dim == len(shape), ValueError('mismatch dims: excepted %sD input, but got %sD' % (bn.dim, len(shape)))

    num_unit = 1
    reduce_axis = bn.reduce_axis
    for axis in reduce_axis:
        num_unit *= shape[axis]
    mean = x.mean(axis=reduce_axis, keepdims=True)
    x_hat = x - mean
    scale = np.sqrt(num_unit)
    var_0 = (x_hat ** 2 + bn.eps).sum(axis=reduce_axis, keepdims=True)
    m = 1 / (np.sqrt(var_0))
    y_hat = m * x_hat
    data = scale * m * x_hat

    if bn.training and bn.track_running_stats:
        # 当利用样本估计全体数据的方差时，需要进行Bessel’s correction贝塞尔校正
        bn.running_data[0] = (1 - bn.momentum) * bn.running_mean + bn.momentum * mean
        bn.running_data[1] = (1 - bn.momentum) * bn.running_var + bn.momentum * var_0 / (num_unit - 1)

    requires_grad = input.requires_grad
    depends_on = []
    grad_fn = L2NormalizationBackward
    if requires_grad:
        depends_on.append(Edge(input, [reduce_axis, num_unit, scale, m, y_hat]))

    output = Tensor(data,
                    requires_grad=requires_grad,
                    depends_on=depends_on,
                    grad_fn=grad_fn)

    return output


class BatchNorm1d(BatchNorm):
    def __init__(self, num_features, affine=True, momentum=0.1, eps=1e-5, track_running_stats=True):
        super(BatchNorm1d, self).__init__(num_features, 2, affine, momentum, eps, track_running_stats)

    def __repr__(self):
        string = "BatchNorm1d(num_features=%s, eps=%s" % (self.num_features, self.eps)
        if self.affine:
            string += ",affine=True"
        if self.track_running_stats:
            string += ",momentum=%s,track_running_stats=True" % self.momentum
        string += ")"
        return string


class BatchNorm2d(BatchNorm):
    def __init__(self, num_features, affine=True, momentum=0.1, eps=1e-5, track_running_stats=True):
        super(BatchNorm2d, self).__init__(num_features, 4, affine, momentum, eps, track_running_stats)

    def __repr__(self):
        string = "BatchNorm2d(num_features=%s, eps=%s" % (self.num_features, self.eps)
        if self.affine:
            string += ",affine=True"
        if self.track_running_stats:
            string += ",momentum=%s,track_running_stats=True" % self.momentum
        string += ")"
        return string
