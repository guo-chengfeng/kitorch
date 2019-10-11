from ..utils import rand, rand_like, ones_like
from ..nn.rnn import RNN as RNN1
from ..nn.rnn_naive import RNN as RNN2
import time
import unittest
bias = False
input_size = 3
hidden_size = 4
num_layers = 2
seq_len = 20
batch_size = 2

input01 = rand(seq_len, batch_size, input_size, requires_grad=True)
input02 = rand(seq_len, batch_size, input_size, requires_grad=True)
input03 = input01.deepcopy()
input04 = input02.deepcopy()

input1 = input01 + input02 * 5
input2 = input03 + input04 * 5

h001 = rand(num_layers, batch_size, hidden_size, requires_grad=True)
h002 = rand(num_layers, batch_size, hidden_size, requires_grad=True)
h003 = h001.deepcopy()
h004 = h002.deepcopy()

h01 = h001 * h002
h02 = h003 * h004


def show0(output1, output2, hn1, hn2):
    print('output')
    print((output1 - output2).sum().data)

    print('hn')
    print((hn1 - hn2).sum().data)


def show1(rnn1, rnn2):
    print('weight_h-grad')
    for layer in range(num_layers):
        print('layer-%s:' % layer, (rnn1.weight_h[layer].grad - rnn2.weight_h[layer].grad).sum().data)

    print('weight_i-grad')
    for layer in range(num_layers):
        print('layer-%s:' % layer, (rnn1.weight_i[layer].grad - rnn2.weight_i[layer].grad).sum().data)

    if bias:
        print('bias-grad')
        for layer in range(num_layers):
            print('layer-%s:' % layer, (rnn1.bias[layer].grad - rnn2.bias[layer].grad).sum().data)


def show2(input1, input2, h01, h02):
    print('input-grad')
    print((input1.grad - input2.grad).sum().data)

    print('h0-grad')
    print((h01.grad - h02.grad).sum().data)


def show3(a, b, prefix):
    print(prefix)
    print((a.grad - b.grad).sum().data)


def test_1():
    rnn1 = RNN1(input_size, hidden_size, num_layers, bias=bias)
    rnn2 = RNN2(input_size, hidden_size, num_layers, bias=bias)
    for i in range(num_layers):
        rnn2.weight_h[i] = rnn1.weight_h[i].deepcopy()
        rnn2.weight_i[i] = rnn1.weight_i[i].deepcopy()
        if bias:
            rnn2.bias[i] = rnn1.bias[i].deepcopy()

    output1, hn1 = rnn1(input1, h01)
    output2, hn2 = rnn2(input2, h02)
    show0(output1, output2, hn1, hn2)

    out_grad = rand_like(output1)
    output1.backward(out_grad)
    output2.backward(out_grad)

    show1(rnn1, rnn2)
    show2(input1, input2, h01, h02)

    print('\n\n=======================')
    h_grad = rand_like(hn1)
    hn1.backward(h_grad)
    hn2.backward(h_grad)
    show1(rnn1, rnn2)
    show2(input1, input2, h01, h02)

    print('\n\n=======================')
    show3(h001, h003, 'h001-03')
    show3(h002, h004, 'h002-04')
