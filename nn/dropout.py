from ..tensor import Tensor
from .layer import Layer
from .. import functional as F

class Dropout(Layer):
    def __init__(self, p):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x: Tensor):
        return F.dropout(x, self.p, training=self.training)

    def __repr__(self):
        return "Dropout(p=%s)" % self.p


class Dropout2d(Layer):
    def __init__(self, p):
        super(Dropout2d, self).__init__()
        self.p = p

    def forward(self, x: Tensor):
        return F.dropout(x, self.p, training=self.training)

    def __repr__(self):
        return "Dropout2d(p=%s)" % self.p
