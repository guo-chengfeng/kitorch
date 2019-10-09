#
# 控制是否记录反向传播

import functools
from .tensor import TensorBase


class no_grad(object):
    def __enter__(self):
        self.grad = TensorBase.get_grad_enabled()
        TensorBase.set_grad_enabled(False)

    def __exit__(self, *args):
        TensorBase.set_grad_enabled(self.grad)

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad
