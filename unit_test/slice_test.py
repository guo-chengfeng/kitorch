from . import to_torch,check
from ..utils import rand, rand_like,from_numpy
import unittest

class SliceTest(unittest.TestCase):
    def test_1(self):
        a = rand(30,30,requires_grad=True)
        b = a.sigmoid()
        c = b[:,1].sum()
        c.backward()

        x, = to_torch([a])
        y = x.sigmoid()
        z = y[:,1].sum()
        z.backward()

        check(a,x)

    def test_2(self):
        a = rand(30,30,requires_grad=True)
        b = a.sigmoid()
        c = b[:,1::].sum()
        c.backward()

        x, = to_torch([a])
        y = x.sigmoid()
        z = y[:,1::].sum()
        z.backward()

        check(a,x)

    def test_3(self):
        a = rand(30,30,requires_grad=True)
        b = a.sigmoid()
        c = b[1:10,5:20].sum()
        c.backward()

        x, = to_torch([a])
        y = x.sigmoid()
        z = y[1:10,5:20].sum()
        z.backward()

        check(a,x)

if __name__ == '__main__':
    unittest.main()
