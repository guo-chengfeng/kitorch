from . import to_torch,check
from ..utils import rand, rand_like
import unittest

class SumTest(unittest.TestCase):
    def test_1(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = a.sum()
        b.backward()

        x, = to_torch([a])
        z = x.sum()
        z.backward()

        check(a,x)

    def test_2(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = a.sum(axis = 1, keepdims=True)
        f = rand_like(b)
        b.backward(f)

        x,v = to_torch([a,f])
        z = x.sum(dim=1, keepdim=True)
        z.backward(v)

        check(a, x)

    def test_3(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = a.sum(axis = (1,2), keepdims=True)
        f = rand_like(b)
        b.backward(f)

        x, v = to_torch([a, f])
        z = x.sum(dim = (1,2), keepdim=True)
        z.backward(v)

        check(a,x)



class MeanTest(unittest.TestCase):
    def test_1(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = a.mean()
        b.backward()

        x, = to_torch([a])
        z = x.mean()
        z.backward()

        check(a,x)

    def test_2(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = a.mean(axis = 1, keepdims=True)
        f = rand_like(b)
        b.backward(f)

        x,v = to_torch([a,f])
        z = x.mean(dim=1, keepdim=True)
        z.backward(v)

        check(a, x)

    def test_3(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = a.mean(axis = (1,2), keepdims=True)
        f = rand_like(b)
        b.backward(f)

        x, v = to_torch([a, f])
        z = x.mean(dim = (1,2), keepdim=True)
        z.backward(v)

        check(a,x)

    def test_4(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = a.mean(axis = (0,1,2))
        f = rand_like(b)
        b.backward(f)

        x, v = to_torch([a, f])
        z = x.mean(dim = (0,1,2))
        z.backward(v)

        check(a,x)

if __name__ == '__main__':
    unittest.main()
