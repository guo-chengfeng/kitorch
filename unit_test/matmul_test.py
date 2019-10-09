
from . import to_torch,check
from ..utils import rand, rand_like
import unittest

def main(a,b):
    c = a @ b
    g = rand_like(c)
    c.backward(g)

    x, y,v = to_torch([a, b,g])
    z = x @ y
    z.backward(v)

    check(a, x)
    check(b, y)

class MatMulTest(unittest.TestCase):
    def test_1(self):
        a = rand(3,4,requires_grad=True)
        b = rand(4,2,requires_grad=True)
        main(a, b)

    def test_2(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = rand(2, 4, 1, requires_grad=True)
        main(a,b)

    def test_3(self):
        a = rand(3, 4, requires_grad=True)
        b = rand(2, 4, 1, requires_grad=True)
        main(a,b)


    def test_4(self):
        a = rand(2, 3, 4, requires_grad=True)
        b = rand(4, 1, requires_grad=True)
        main(a,b)



if __name__ == '__main__':
    unittest.main()
