from ..utils import rand, rand_like, ones_like
from ..nn.rnn import LSTM as LSTM1
from ..nn.rnn_naive import LSTM as LSTM2
from . import timer
import time
import unittest

bias = True
input_size = 3
hidden_size = 4
num_layers = 2
seq_len = 4
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

c001 = rand(num_layers, batch_size, hidden_size, requires_grad=True)
c002 = rand(num_layers, batch_size, hidden_size, requires_grad=True)
c003 = c001.deepcopy()
c004 = c002.deepcopy()

c01 = c001 * c002
c02 = c003 * c004


def show0(output1, output2, hn1, hn2, cn1, cn2):
    print('output')
    print((output1 - output2).sum().data)

    print('hn')
    print((hn1 - hn2).sum().data)

    print('cn')
    print((cn1 - cn2).sum().data)


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


def show2(input1, input2, h01, h02, c01, c02):
    print('input-grad')
    print((input1.grad - input2.grad).sum().data)

    print('h0-grad')
    print((h01.grad - h02.grad).sum().data)

    print('c0-grad')
    print((c01.grad - c02.grad).sum().data)


def show3(a, b, prefix):
    print(prefix)
    print((a.grad - b.grad).sum().data)


def test_1():
    rnn1 = LSTM1(input_size, hidden_size, num_layers, bias=bias)
    rnn2 = LSTM2(input_size, hidden_size, num_layers, bias=bias)
    for i in range(num_layers):
        rnn2.weight_h[i] = rnn1.weight_h[i].deepcopy()
        rnn2.weight_i[i] = rnn1.weight_i[i].deepcopy()
        if bias:
            rnn2.bias[i] = rnn1.bias[i].deepcopy()

    output1, hn1, cn1 = rnn1(input1, h01, c01)
    output2, hn2, cn2 = rnn2(input2, h02, c02)
    show0(output1, output2, hn1, hn2, cn1, cn2)

    out_grad = rand_like(output1)
    timer.tic
    output1.backward(out_grad)
    timer.toc

    timer.tic
    output2.backward(out_grad)
    timer.toc

    show1(rnn1, rnn2)
    show2(input1, input2, h01, h02, c01, c02)

    print('\n\n=======================\nhn_backward')
    h_grad = rand_like(hn1)
    hn1.backward(h_grad)
    hn2.backward(h_grad)
    show1(rnn1, rnn2)
    show2(input1, input2, h01, h02, c01, c02)

    print('\n\n=======================\ncn_backward')
    c_grad = rand_like(cn1)
    cn1.backward(c_grad)
    cn2.backward(c_grad)
    show1(rnn1, rnn2)
    show2(input1, input2, h01, h02, c01, c02)

    print('\n\n=======================')
    show3(h001, h003, 'h001-03')
    show3(h002, h004, 'h002-04')

    show3(c001, c003, 'c001-03')
    show3(c002, c004, 'c002-04')
