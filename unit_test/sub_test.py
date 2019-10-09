
from . import to_torch,check
from ..utils import rand, rand_like
import unittest

class SubTest(unittest.TestCase):
    def test_1(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = rand(1, 4, requires_grad=True)
        c = a - b
        g = rand_like(c)
        c.backward(g)

        x,y,v = to_torch([a,b,g])
        z = x - y
        z.backward(v)

        check(a,x)
        check(b,y)

    def test_2(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = rand(3, 1, requires_grad=True)
        c = a - b
        g = rand_like(c)
        c.backward(g)

        x,y,v = to_torch([a,b,g])
        z = x - y
        z.backward(v)

        check(a,x)
        check(b,y)

    def test_3(self):
        a = rand(2, 3, requires_grad=True)
        b = rand(1, 3, requires_grad=True)
        c = (a - b).sum()
        c.backward()

        x,y = to_torch([a,b])
        z = (x - y).sum()
        z.backward()

        check(a,x)
        check(b,y)


    def test_4(self):
        a = rand(1, 3, requires_grad=True)
        b = rand(3, 1, requires_grad=True)
        c = a - b
        g = rand(3,3)
        c.backward(g)


        x,y,v = to_torch([a,b,g])
        z = x - y
        z.backward(v)

        check(a,x)
        check(b,y)

    def test_5(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = rand(2,1,1, requires_grad=True)
        c = a - b
        g = rand_like(c)
        c.backward(g)

        x,y,v = to_torch([a,b,g])
        z = x - y
        z.backward(v)

        check(a,x)
        check(b,y)

    def test_6(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = rand(1, requires_grad=True)
        c = a - b
        g = rand_like(c)
        c.backward(g)

        x,y,v = to_torch([a,b,g])
        z = x - y
        z.backward(v)

        check(a,x)
        check(b,y)

    def test_7(self):
        a = rand(2, 3, 4, requires_grad=True)
        c = a - 1
        g = rand_like(c)
        c.backward(g)

        x,v = to_torch([a,g])
        z = x - 1
        z.backward(v)

        check(a,x)

    def test_8(self):
        a = rand(2, 3, 4, requires_grad=True)
        c = 1 - a
        g = rand_like(c)
        c.backward(g)

        x,v = to_torch([a,g])
        z = 1-x
        z.backward(v)

        check(a,x)

if __name__ == '__main__':
    unittest.main()
