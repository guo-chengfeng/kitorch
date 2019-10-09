
from . import to_torch,check
from ..utils import rand, rand_like,stack,cat
import unittest
import torch

def stack_main(axis=0):
    a = rand(2, 3, 4, requires_grad=True)
    b = rand(2, 3, 4, requires_grad=True)
    c = rand(2, 3, 4, requires_grad=True)
    d = stack([a, b, c],axis=axis)
    g = rand_like(d)
    d.backward(g)

    x, y, z, v = to_torch([a, b, c, g])
    w = torch.stack([x, y, z],dim=axis)
    w.backward(v)

    check(a, x)
    check(b, y)
    check(c, z)


def cat_main(axis=0):
    a = rand(2, 3, 4, requires_grad=True)
    b = rand(2, 3, 4, requires_grad=True)
    c = rand(2, 3, 4, requires_grad=True)
    d = cat([a, b, c],axis=axis)
    g = rand_like(d)
    d.backward(g)

    x, y, z, v = to_torch([a, b, c, g])
    w = torch.cat([x, y, z],dim=axis)
    w.backward(v)

    check(a, x)
    check(b, y)
    check(c, z)

class StackTest(unittest.TestCase):
    def test_1(self):
        stack_main(axis=0)
        stack_main(axis=1)
        stack_main(axis=2)


    def test_2(self):
        cat_main(axis=0)
        cat_main(axis=1)
        cat_main(axis=2)


if __name__ == '__main__':
    unittest.main()
