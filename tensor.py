from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np


class Edge(NamedTuple):
    tensor: 'Tensor'
    cache: List


Array = Union[float, int, np.ndarray]
array_type = (float, int, np.ndarray)

Scalar = Union[float, int]
scalar_type = (float, int)

FIAT = Union[float, int, np.ndarray, 'Tensor']


def ensure_array(array: Union[float, int, list, np.ndarray]) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    else:
        return np.array(array)


class TensorBase(object):
    _requires_grad = True
    __array_priority__ = 10000.0

    def __init__(self, requires_grad=False):
        self._requires_grad = requires_grad

    @classmethod
    def set_grad_enabled(cls, is_enabled):
        cls._requires_grad = is_enabled

    @classmethod
    def get_grad_enabled(cls):
        return cls._requires_grad

    @property
    def requires_grad(self):
        return self._requires_grad and TensorBase._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        self._requires_grad = requires_grad


class Tensor(TensorBase):
    def __init__(self,
                 data: Union[float, int, list, np.ndarray],
                 requires_grad: bool = False,
                 depends_on: List[Edge] = None,
                 grad_fn: Callable = None,
                 name: Optional[str] = None,
                 is_leaf: bool = False,
                 is_simple: bool = True
                 ) -> None:
        super().__init__(requires_grad)
        self.data = ensure_array(data)
        self.depends_on = depends_on or []
        self.grad: Optional['Tensor'] = None
        self.grad_fn: Optional[Callable] = grad_fn
        self.name: Optional[str] = name
        self.is_leaf = is_leaf
        self.is_simple = is_simple

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dim(self):
        return len(self.data.shape)

    def zero_grad(self) -> None:
        if self.grad:
            self.grad.data[:] = 0

    def __repr__(self) -> str:
        data = self.data.__repr__()[6:-1]
        if self.name:
            string = f"{self.name}:Tensor({data}"
        else:
            string = f"Tensor({data}"
        if self.requires_grad:
            string += f",requires_grad={self.requires_grad}"
            if self.grad_fn:
                string += f",grad_fn=<{self.grad_fn.__repr__().split(' ')[1]}>"
        return string + ")"

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            grad = Tensor(1)

        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data))

        self.grad.data += grad.data

        if self.is_simple:
            for edge in self.depends_on:
                tensor = edge.tensor
                _grad_ = self.grad_fn(grad, tensor, edge.cache)
                tensor.backward(_grad_)
        else:
            if self.depends_on:
                tensor_grad = self.grad_fn(grad, self.depends_on)
                for tensor, grad in tensor_grad:
                    tensor.backward(grad)

    def __bool__(self):
        return bool(self.data.size)

    def __len__(self):
        return self.shape[0]

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __neg__(self):
        return _sub_array_tensor(0, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __pow__(self, pow):
        return power(self, pow)

    def __getitem__(self, idxs) -> 'Tensor':
        return slice(self, idxs)

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self):
        return self.data.item()

    # 两个拷贝函数，一个只拷贝数据，一个全部拷贝
    def copy(self):
        return Tensor(self.data.copy())

    def deepcopy(self):
        other = Tensor(data=self.data.copy(),
                       requires_grad=self.requires_grad,
                       depends_on=self.depends_on,
                       grad_fn=self.grad_fn,
                       name=self.name,
                       is_leaf=self.is_leaf,
                       is_simple=self.is_simple)
        if self.grad:
            other.grad = self.grad.copy()

        return other

    # 两个约并函数
    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        return sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        return mean(self, axis=axis, keepdims=keepdims)

    def norm(self, order=2, reduction="sum"):
        return norm(self, order=order, reduction=reduction)

    # 常用数学操作
    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def __abs__(self):
        return abs(self)

    # 5个激活函数
    def sigmoid(self):
        return sigmoid(self)

    def relu(self):
        return relu(self)

    def tanh(self):
        return tanh(self)

    def softmax(self, dim=1, deoverflow=True):
        return softmax(self, dim, deoverflow)

    def log_softmax(self, dim=1, deoverflow=True):
        return log_softmax(self, dim, deoverflow)

    def reshape(self, newshape):
        return reshape(self, newshape)

    def swapaxes(self, axis1, axis2):
        return swapaxes(self, axis1, axis2)

    def transpose(self, axis1, axis2):
        return transpose(self, axis1, axis2)


def SilceBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    grad_data = np.zeros_like(t.data)
    grad_data[cache[0]] = grad.data
    return Tensor(grad_data)


def slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad
    grad_fn = SilceBackward
    depends_on = []
    if requires_grad:
        depends_on = [Edge(t, [idxs])]

    return Tensor(data, requires_grad, depends_on, grad_fn)


def LogSoftmaxBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    dim, data = cache
    _sum = np.sum(grad.data, axis=dim, keepdims=True)
    grad_data = grad.data - data * _sum
    return Tensor(grad_data)


def log_softmax(t: 'Tensor', dim=1, deoverflow=True) -> 'Tensor':
    assert len(t.shape) <= 2, "Except N-D Tensor(N<=2), but get %s-D" % len(t.shape)
    assert dim < len(t.shape), "The argument `dim` should not bigger than Tensor's dimension"
    if deoverflow:
        _data = t.data - np.max(t.data, axis=dim, keepdims=True)
    else:
        _data = t.data
    a = np.exp(_data)
    _sum = a.sum(axis=dim, keepdims=True)
    data = _data - np.log(_sum)
    requires_grad = t.requires_grad
    depends_on = []
    grad_fn = LogSoftmaxBackward
    if t.requires_grad:
        depends_on.append(Edge(t, [dim, a / _sum]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def SoftmaxBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    dim, data = cache
    _sum = (grad.data * data).sum(axis=dim, keepdims=True)
    grad_data = data * (grad.data - _sum)
    return Tensor(grad_data)


def softmax(t: 'Tensor', dim=1, deoverflow=True) -> 'Tensor':
    assert dim < len(t.shape), "The argument `dim` should not bigger than Tensor's dimension"
    if deoverflow:
        a = np.exp(t.data - np.max(t.data, axis=dim, keepdims=True))
    else:
        a = np.exp(t.data)
    _sum = a.sum(axis=dim, keepdims=True)

    data = a / _sum
    requires_grad = t.requires_grad
    depends_on = []
    grad_fn = SoftmaxBackward
    if t.requires_grad:
        depends_on.append(Edge(t, [dim, data]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def TanhBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    data = cache[0]
    grad_data = grad.data * (1 - data * data)
    return Tensor(grad_data)


def tanh(t: 'Tensor') -> 'Tensor':
    data = np.tanh(t.data)
    requires_grad = t.requires_grad
    depends_on = []
    grad_fn = TanhBackward
    if t.requires_grad:
        depends_on.append(Edge(t, [data]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def SigmoidBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    data = cache[0]
    grad_data = grad.data * data * (1 - data)
    return Tensor(grad_data)


def sigmoid(t: 'Tensor') -> 'Tensor':
    data = 1 / (1 + np.exp(-t.data))
    requires_grad = t.requires_grad
    depends_on = []
    grad_fn = SigmoidBackward
    if t.requires_grad:
        depends_on.append(Edge(t, [data]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def ReluBackward(output_grad: 'Tensor', t: 'Tensor', other_cache: []) -> 'Tensor':
    data = other_cache[0]
    grad_data = np.zeros_like(data)
    grad_data[data > 0] = 1
    grad_data = output_grad.data * grad_data
    return Tensor(grad_data)


def relu(t: 'Tensor') -> 'Tensor':
    data = t.data.copy()
    data[t.data < 0] = 0
    requires_grad = t.requires_grad
    depends_on = []
    grad_fn = ReluBackward
    if t.requires_grad:
        depends_on.append(Edge(t, [data]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def SumBackWard(grad: 'Tensor', var: 'Tensor', cache: List) -> 'Tensor':
    axis, keepdims = cache

    shape = grad.shape
    if shape == () or shape == (1,):
        return Tensor(grad.data * np.ones_like(var.data))

    if keepdims:
        return Tensor(grad.data * np.ones_like(var.data))

    if isinstance(axis, int):
        return Tensor(np.expand_dims(grad.data.copy(), axis) * np.ones_like(var.data))
    else:
        grad_data = grad.data
        for i in axis:
            np.expand_dims(grad_data, axis[i])
        return Tensor(grad_data * np.ones_like(var.data))


def sum(var: Tensor, axis=None, keepdims=False) -> Tensor:
    data = var.data.sum(axis=axis, keepdims=keepdims)
    requires_grad = var.requires_grad
    grad_fn = SumBackWard
    depends_on = []
    if var.requires_grad:
        depends_on = [Edge(var, [axis, keepdims])]

    return Tensor(data, requires_grad, depends_on, grad_fn)


def MeanBackWard(grad: 'Tensor', var: 'Tensor', cache: List) -> 'Tensor':
    axis, keepdims = cache
    shape = grad.shape
    if shape == () or shape == (1,):
        return Tensor(grad.data / var.size * np.ones_like(var.data))

    num_unit = 1
    if isinstance(axis, int):
        num_unit = var.shape[axis]
    elif axis is None:
        num_unit = var.size
    else:
        for i in axis:
            num_unit *= var.shape[i]

    if keepdims:
        return Tensor(grad.data / num_unit * np.ones_like(var.data))

    if isinstance(axis, int):
        return Tensor(np.expand_dims(grad.data / num_unit, axis) * np.ones_like(var.data))
    else:
        grad_data = grad.data / var.size
        for i in axis:
            np.expand_dims(grad_data, axis[i])
        return Tensor(grad_data * np.ones_like(var.data))


def mean(var: Tensor, axis=None, keepdims=False) -> Tensor:
    data = var.data.mean(axis=axis, keepdims=keepdims)
    requires_grad = var.requires_grad
    grad_fn = MeanBackWard
    depends_on = []
    if requires_grad:
        depends_on = [Edge(var, [axis, keepdims])]

    return Tensor(data, requires_grad, depends_on, grad_fn)


def reduce_grad(grad_data: np.ndarray, res_shape: tuple, var_shape: tuple) -> np.ndarray:
    # reduce broadcasting
    # if var.data is a scalar or array with shape of (1,)
    if var_shape == () or var_shape == (1,):
        return grad_data.sum()

    # len(res_shape) >= len(var_shape)
    dim = len(res_shape) - len(var_shape)
    if dim > 0:
        axis = tuple([i for i in range(dim)])
        grad_data = grad_data.sum(axis=axis)

    axis = tuple([i for i in range(len(var_shape)) if (var_shape[i] == 1 and res_shape[dim + i] != 1)])
    if axis:
        grad_data = grad_data.sum(axis=axis, keepdims=True)

    return grad_data


def AddBackward(grad: 'Tensor', var: 'Tensor', cache: List) -> 'Tensor':
    grad_data = grad.data * np.ones(grad.shape)
    res_shape = grad.shape
    var_shape = var.shape

    if res_shape == var_shape:
        return Tensor(grad_data)
    else:
        return Tensor(reduce_grad(grad_data, res_shape, var_shape))


def _add_array_tensor(lhs: Array, rhs: 'Tensor') -> 'Tensor':
    data = lhs + rhs.data
    requires_grad = rhs.requires_grad
    depends_on = []
    grad_fn = AddBackward
    if rhs.requires_grad:
        depends_on.append(Edge(rhs, []))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def _add_tensor_tensor(lhs: 'Tensor', rhs: 'Tensor') -> 'Tensor':
    data = lhs.data + rhs.data
    requires_grad = lhs.requires_grad or rhs.requires_grad
    depends_on = []
    grad_fn = AddBackward
    if requires_grad:
        if lhs.requires_grad:
            depends_on.append(Edge(lhs, []))
        if rhs.requires_grad:
            depends_on.append(Edge(rhs, []))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def add(lhs: FIAT, rhs: FIAT):
    if isinstance(lhs, Tensor):
        if isinstance(rhs, Tensor):
            return _add_tensor_tensor(lhs, rhs)
        elif isinstance(rhs, array_type):
            return _add_array_tensor(rhs, lhs)
        else:
            raise TypeError("add: rhs must be a Tensor, float, int or ndarray, but got %s" % type(rhs))

    elif isinstance(lhs, array_type):
        if isinstance(rhs, Tensor):
            return _add_array_tensor(lhs, rhs)
        else:
            raise TypeError("add: if lhs is a float, int or ndarray, rhs must be a Tensor, but got %s" % type(rhs))
    else:
        raise TypeError("add: unsupported operand type(s) for %s + %s " % (type(lhs), type(rhs)))


def SubBackward(grad: 'Tensor', var: 'Tensor', cache: List) -> 'Tensor':
    is_right = True if cache[0] == 'right' else False
    grad_data = grad.data * np.ones(grad.shape)
    if is_right:
        grad_data *= -1

    res_shape = grad.shape
    var_shape = var.shape

    if res_shape == var_shape:
        return Tensor(grad_data)
    else:
        return Tensor(reduce_grad(grad_data, res_shape, var_shape))


def _sub_array_tensor(lhs: Array, rhs: 'Tensor', ) -> 'Tensor':
    data = lhs - rhs.data
    requires_grad = rhs.requires_grad
    depends_on = []
    grad_fn = SubBackward
    if rhs.requires_grad:
        depends_on.append(Edge(rhs, ['right']))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def _sub_tensor_tensor(lhs: 'Tensor', rhs: 'Tensor') -> 'Tensor':
    data = lhs.data - rhs.data
    requires_grad = lhs.requires_grad or rhs.requires_grad
    depends_on = []
    grad_fn = SubBackward
    if requires_grad:
        if lhs.requires_grad:
            depends_on.append(Edge(lhs, ['left']))
        if rhs.requires_grad:
            depends_on.append(Edge(rhs, ['right']))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def sub(lhs: FIAT, rhs: FIAT):
    if isinstance(lhs, Tensor):
        if isinstance(rhs, Tensor):
            return _sub_tensor_tensor(lhs, rhs)
        elif isinstance(rhs, array_type):
            return _add_array_tensor(-rhs, lhs)
        else:
            raise TypeError("sub: rhs must be a Tensor, float, int or ndarray, but got %s" % type(rhs))

    elif isinstance(lhs, array_type):
        if isinstance(rhs, Tensor):
            return _sub_array_tensor(lhs, rhs)
        else:
            raise TypeError("sub: if lhs is a float, int or ndarray, rhs must be a Tensor, but got %s" % type(rhs))
    else:
        raise TypeError("sub: unsupported operand type(s) for %s - %s " % (type(lhs), type(rhs)))


def _mul_tensor_tensor(lhs: 'Tensor', rhs: 'Tensor') -> 'Tensor':
    data = lhs.data * rhs.data
    requires_grad = lhs.requires_grad or rhs.requires_grad
    depends_on = []
    grad_fn = MulBackward
    if lhs.requires_grad:
        depends_on.append(Edge(lhs, [rhs.data]))

    if rhs.requires_grad:
        depends_on.append(Edge(rhs, [lhs.data]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def _mul_array_tensor(lhs: Scalar, rhs: 'Tensor') -> 'Tensor':
    data = lhs * rhs.data
    requires_grad = rhs.requires_grad
    depends_on = []
    grad_fn = MulBackward
    if rhs.requires_grad:
        depends_on.append(Edge(rhs, [lhs]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def mul(lhs: FIAT, rhs: FIAT):
    if isinstance(lhs, Tensor):
        if isinstance(rhs, Tensor):
            return _mul_tensor_tensor(lhs, rhs)
        elif isinstance(rhs, array_type):
            return _mul_array_tensor(rhs, lhs)
        else:
            raise TypeError("mul: rhs must be a Tensor, float, int or ndarray, but got %s" % type(rhs))

    elif isinstance(lhs, array_type):
        if isinstance(rhs, Tensor):
            return _mul_array_tensor(lhs, rhs)
        else:
            raise TypeError("mul: if lhs is a float, int or ndarray, rhs must be a Tensor, but got %s" % type(rhs))
    else:
        raise TypeError("mul: unsupported operand type(s) for %s * %s " % (type(lhs), type(rhs)))


# alias of mul func
multiply = mul


def MulBackward(grad: 'Tensor', var: 'Tensor', cache: List) -> 'Tensor':
    data = cache[0]
    grad_data = grad.data * data

    res_shape = grad.shape
    var_shape = var.shape

    if res_shape == var_shape:
        return Tensor(grad_data)
    else:
        return Tensor(reduce_grad(grad_data, res_shape, var_shape))


def _divide_tensor_tensor(lhs: 'Tensor', rhs: 'Tensor') -> 'Tensor':
    data = lhs.data / rhs.data
    requires_grad = lhs.requires_grad or rhs.requires_grad
    depends_on = []
    grad_fn = DivideBackward
    if lhs.requires_grad:
        depends_on.append(Edge(lhs, ['left', rhs.data]))

    if rhs.requires_grad:
        depends_on.append(Edge(rhs, ['right', lhs.data]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def _divide_array_tensor(lhs: Scalar, rhs: 'Tensor', ) -> 'Tensor':
    data = lhs / rhs.data
    requires_grad = rhs.requires_grad
    depends_on = []
    grad_fn = DivideBackward
    if rhs.requires_grad:
        depends_on.append(Edge(rhs, ['right', lhs]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def divide(lhs: FIAT, rhs: FIAT):
    if isinstance(lhs, Tensor):
        if isinstance(rhs, Tensor):
            return _divide_tensor_tensor(lhs, rhs)
        elif isinstance(rhs, array_type):
            return _mul_array_tensor(1 / rhs, lhs)
        else:
            raise TypeError("divide: rhs must be a Tensor, float, int or ndarray, but got %s" % type(rhs))

    elif isinstance(lhs, array_type):
        if isinstance(rhs, Tensor):
            return _divide_array_tensor(lhs, rhs)
        else:
            raise TypeError("divide: if lhs is a float, int or ndarray, rhs must be a Tensor, but got %s" % type(rhs))
    else:
        raise TypeError("divide: unsupported operand type(s) for %s / %s " % (type(lhs), type(rhs)))


def DivideBackward(grad: 'Tensor', var: 'Tensor', cache: List) -> 'Tensor':
    is_right = True if cache[0] == 'right' else False
    if is_right:
        #  - 1/x^2
        grad_data = -grad.data * cache[1] / (var.data ** 2)
    else:
        grad_data = grad.data / cache[1]

    res_shape = grad.shape
    var_shape = var.shape

    if res_shape == var_shape:
        return Tensor(grad_data)
    else:
        return Tensor(reduce_grad(grad_data, res_shape, var_shape))


def _matmul_tensor_tensor(lhs: 'Tensor', rhs: 'Tensor') -> 'Tensor':
    assert len(lhs.shape) > 1 and len(rhs.shape) > 1, "Expected N-D(N>1) matrix, but got 1-D array"
    data = lhs.data @ rhs.data
    requires_grad = lhs.requires_grad or rhs.requires_grad
    depends_on = []
    grad_fn = MatmulBackward
    if lhs.requires_grad:
        depends_on.append(Edge(lhs, ['left', rhs.data]))

    if rhs.requires_grad:
        depends_on.append(Edge(rhs, ['right', lhs.data]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def _matmul_array_tensor(lhs: np.ndarray, rhs: 'Tensor') -> 'Tensor':
    assert len(lhs.shape) > 1 and len(rhs.shape) > 1, "Expected N-D(N>1) matrix, but got 1-D array"
    data = lhs @ rhs.data
    requires_grad = rhs.requires_grad
    depends_on = []
    grad_fn = MatmulBackward

    if rhs.requires_grad:
        depends_on.append(Edge(rhs, ['right', lhs]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def matmul(lhs: Union['Tensor', np.ndarray], rhs: Union['Tensor', np.ndarray]) -> 'Tensor':
    if isinstance(lhs, Tensor):
        if isinstance(rhs, Tensor):
            return _matmul_tensor_tensor(lhs, rhs)
        elif isinstance(rhs, np.ndarray):
            return _matmul_array_tensor(rhs, lhs)
        else:
            raise TypeError("matmul: rhs must be a Tensor or ndarray, but got %s" % type(rhs))

    elif isinstance(lhs, np.ndarray):
        if isinstance(rhs, Tensor):
            return _matmul_array_tensor(lhs, rhs)
        else:
            raise TypeError("matmul: if lhs is a ndarray, rhs must be a Tensor, but got %s" % type(rhs))
    else:
        raise TypeError("matmul: unsupported operand type(s) for %s @ %s " % (type(lhs), type(rhs)))


def MatmulBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    is_right = True if cache[0] == 'right' else False
    data = cache[1]
    dims = len(data.shape)
    if dims > 2:
        axes = [i for i in range(dims)]
        axes[-2:] = [dims - 1, dims - 2]

        if is_right:
            grad_data = np.transpose(data, axes=axes) @ grad.data
        else:
            grad_data = grad.data @ np.transpose(data, axes=axes)
    else:
        if is_right:
            grad_data = data.T @ grad.data
        else:
            grad_data = grad.data @ data.T

    dim = len(grad_data.shape) - len(t.data.shape)
    if dim > 0:
        axis = tuple([i for i in range(dim)])
        grad_data = grad_data.sum(axis=axis)

    return Tensor(grad_data)


def AbsBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    data = np.ones_like(t.data)
    data[t.data < 0] = -1
    return Tensor(grad.data * data)


def abs(t: Tensor) -> Tensor:
    data = np.abs(t.data)
    requires_grad = t.requires_grad
    grad_fn = AbsBackward
    depends_on = []
    if requires_grad:
        depends_on.append(Edge(t, []))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def NormBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    order, reduction = cache
    if order == 1:
        data = np.ones_like(t.data)
        data[t.data < 0] = -1
    else:
        data = 2 * t.data

    if reduction == 'mean':
        data /= t.data.size

    return Tensor(grad.data * data)


def norm(t: Tensor, order=2, reduction='sum'):
    if order == 1:
        if reduction == "mean":
            data = np.abs(t.data).mean()
        else:
            data = np.abs(t.data).sum()
    elif order == 2:
        if reduction == "mean":
            data = (t.data ** 2).mean()
        else:
            data = (t.data ** 2).sum()
    else:
        raise RuntimeError('Unsupported order number')

    requires_grad = t.requires_grad
    grad_fn = NormBackward
    depends_on = []
    if requires_grad:
        depends_on.append(Edge(t, [order, reduction]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def PowerBackward(grad: 'Tensor', t: 'Tensor', cache: []) -> 'Tensor':
    pow = cache[0]
    grad_data = grad.data * pow * np.power(t.data, pow - 1)
    return Tensor(grad_data)


def power(t: 'Tensor', pow: Scalar) -> 'Tensor':
    data = np.power(t.data, pow)
    requires_grad = t.requires_grad
    depends_on = []
    grad_fn = PowerBackward
    if t.requires_grad:
        depends_on.append(Edge(t, [pow]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def LogBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    return Tensor(grad.data / (t.data + 1e-16))


def log(t: Tensor) -> Tensor:
    data = np.log(t.data)
    requires_grad = t.requires_grad
    grad_fn = LogBackward
    depends_on = []
    if requires_grad:
        depends_on.append(Edge(t, []))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def ExpBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    return Tensor(grad.data * cache[0])


def exp(t: 'Tensor') -> 'Tensor':
    data = np.exp(t.data)
    requires_grad = t.requires_grad
    depends_on = []
    grad_fn = ExpBackward
    if t.requires_grad:
        depends_on.append(Edge(t, [data]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def ReshapeBackward(grad: 'Tensor', t: 'Tensor', cache: list) -> 'Tensor':
    original_shape = t.shape
    return Tensor(grad.data.reshape(original_shape))


def reshape(t: 'Tensor', newshape):
    data = t.data.reshape(newshape)
    requires_grad = t.requires_grad
    grad_fn = ReshapeBackward
    depends_on = []
    if requires_grad:
        depends_on.append(Edge(t, []))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def SwapaxesBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    axis1, axis2 = cache
    return Tensor(np.swapaxes(grad.data, axis2, axis1))


def swapaxes(t: 'Tensor', axis1, axis2):
    data = np.swapaxes(t.data, axis1, axis2)
    requires_grad = t.requires_grad
    grad_fn = SwapaxesBackward
    depends_on = []
    if requires_grad:
        depends_on.append(Edge(t, [axis1, axis2]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def TransposeBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    return Tensor(np.transpose(grad.data, cache[0]))


def transpose(t: Tensor, axis1, axis2) -> 'Tensor':
    axes = [i for i in range(t.dim)]
    axes[axis1] = axis2
    axes[axis2] = axis1
    data = np.transpose(t.data, axes)
    requires_grad = t.requires_grad
    grad_fn = TransposeBackward
    depends_on = []
    if requires_grad:
        depends_on.append(Edge(t, [axes]))

    return Tensor(data, requires_grad, depends_on, grad_fn)


def CatBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    axis, index = cache
    length = t.shape[axis]
    indices = [i for i in range(index * length, (index + 1) * length)]
    return Tensor(grad.data.take(indices, axis) * np.ones_like(t.data))


def cat(ts, axis=0):
    assert isinstance(ts, (list, tuple)), \
        "concat(): argument 'tensors' must be tuple or list of Tensors, not Tensor"

    _data = [t.data for t in ts]
    data = np.concatenate(_data, axis=axis)
    requires_grad = False
    depends_on = []
    grad_on = CatBackward

    for index, t in enumerate(ts):
        requires_grad = requires_grad or t.requires_grad
        if t.requires_grad:
            depends_on.append(Edge(t, [axis, index]))

    return Tensor(data, requires_grad, depends_on, grad_on)


def StackBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    axis, index = cache
    return Tensor(grad.data.take(index, axis))


def stack(ts, axis=0):
    assert isinstance(ts, (list, tuple)), \
        "stack(): argument 'tensors' must be tuple or list of Tensors, not Tensor"

    _data = [t.data for t in ts]
    data = np.stack(_data, axis=axis)
    requires_grad = False
    depends_on = []
    grad_on = StackBackward

    for index, t in enumerate(ts):
        requires_grad = requires_grad or t.requires_grad
        if t.requires_grad:
            depends_on.append(Edge(t, [axis, index]))

    return Tensor(data, requires_grad, depends_on, grad_on)
