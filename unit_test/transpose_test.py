from . import to_torch, check
from ..utils import rand, rand_like, stack, cat
import unittest
import torch


def transpose_main(axis1, axis2):
    a = rand(2, 3, 4, requires_grad=True)
    d = a.transpose(axis1, axis2)
    g = rand_like(d)
    d.backward(g)

    x, v = to_torch([a, g])
    w = torch.transpose(x, axis1, axis2)
    w.backward(v)

    check(a, x)


class TransposeTest(unittest.TestCase):
    def test_1(self):
        transpose_main(0, 1)
        transpose_main(0, 2)
        transpose_main(1, 2)


if __name__ == '__main__':
    unittest.main()
