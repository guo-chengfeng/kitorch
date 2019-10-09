from . import to_torch,check
from ..utils import rand, rand_like,from_numpy
import unittest
import numpy as np
import time
import torch


def show_time(prefix,elapsed_time):
    print()
    if elapsed_time < 0.001:
        print(prefix," %.2f"%(elapsed_time*1000*1000),' us')
    elif elapsed_time < 1:
        print(prefix, " %.2f"%(elapsed_time * 1000), ' ms')
    else:
        print(prefix, " %.2f"%elapsed_time, ' s')

class ActFuncTest(unittest.TestCase):
    def test_sigmoid_1(self):
        a = rand(3,3,requires_grad=True)
        b = a.sigmoid()
        c = b.sum()
        c.backward()

        x, = to_torch([a])
        y = x.sigmoid()
        z = y.sum()
        z.backward()

        check(a,x)

    def test_sigmoid_2(self):
        a = rand(3, 3, requires_grad=True)
        b = a.sigmoid()
        g = rand_like(b)
        b.backward(g)

        x,v = to_torch([a,g])
        y = x.sigmoid()
        y.backward(v)

        check(a, x)

    def test_relu(self):
        np_array = np.random.rand(512,10,28, 28)
        a = from_numpy(np_array, requires_grad=True)
        st = time.time()
        b = a.relu()
        c = b.sum()
        c.backward()
        show_time('Numpy Relu',time.time()-st)


        x = torch.from_numpy(np_array)
        x.requires_grad = True
        st = time.time()
        y = x.relu()
        z = y.sum()
        z.backward()
        show_time('Torch Relu', time.time() - st)

        check(a,x)

    def test_tanh(self):
        a = rand(3, 3, requires_grad=True)
        b = a.tanh()
        g = rand_like(b)
        b.backward(g)

        x, v = to_torch([a, g])
        y = x.tanh()
        y.backward(v)

        check(a, x)

    def test_softmax_0(self):
        a = rand(40, 10, requires_grad=True)
        b = a.softmax(dim=1)
        g = rand_like(b)
        b.backward(g)

        x, v = to_torch([a, g])
        y = x.softmax(dim=1)
        y.backward(v)

        check(a, x)


    def test_softmax_1(self):
        a = rand(40, 10, requires_grad=True)
        b = a.softmax(dim=0)
        g = rand_like(b)
        b.backward(g)

        x, v = to_torch([a, g])
        y = x.softmax(dim=0)
        y.backward(v)

        check(a, x)


    def test_softmax_2(self):
        a = rand(40, 10, requires_grad=True)
        b = a.softmax(dim=1,deoverflow=True)
        g = rand_like(b)
        b.backward(g)

        x, v = to_torch([a, g])
        y = x.softmax(dim=1)
        y.backward(v)

        check(a, x)


    def test_softmax_3(self):
        a = rand(40, 10, requires_grad=True)
        b = a.softmax(dim=0,deoverflow=True)
        g = rand_like(b)
        b.backward(g)

        x, v = to_torch([a, g])
        y = x.softmax(dim=0)
        y.backward(v)

        check(a, x)

    def test_log_softmax_0(self):
        a = rand(10, 5, requires_grad=True)
        b = a.log_softmax(dim=1)
        g = rand_like(b)
        b.backward(g)

        x, v = to_torch([a, g])
        y = x.log_softmax(dim=1)
        y.backward(v)

        check(a, x)


    def test_log_softmax_1(self):
        a = rand(10, 5, requires_grad=True)
        b = a.log_softmax(dim=0)
        g = rand_like(b)
        b.backward(g)

        x, v = to_torch([a, g])
        y = x.log_softmax(dim=0)
        y.backward(v)

        check(a, x)


    def test_log_softmax_2(self):
        a = rand(10, 5, requires_grad=True)
        b = a.log_softmax(dim=1,deoverflow=True)
        g = rand_like(b)
        b.backward(g)

        x, v = to_torch([a, g])
        y = x.log_softmax(dim=1)
        y.backward(v)

        check(a, x)


    def test_log_softmax_3(self):
        a = rand(10, 5,  requires_grad=True)
        b = a.log_softmax(dim=0,deoverflow=True)
        g = rand_like(b)
        b.backward(g)

        x, v = to_torch([a, g])
        y = x.log_softmax(dim=0)
        y.backward(v)

        check(a, x)


if __name__ == '__main__':
    unittest.main()
