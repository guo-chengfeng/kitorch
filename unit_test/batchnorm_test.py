from ..nn import BatchNorm
from ..utils import from_numpy, rand, rand_like
import numpy as np
import unittest
import torch
from torch import nn as torch_nn




class BatchNormTest(unittest.TestCase):
    def test_1(self):
        eps = 1e-5
        affine = True
        track_running_stats = True
        dim = 3
        num_features = 4
        batch_size = 100

        torch_bn = torch_nn.BatchNorm1d(num_features, affine=affine, eps=eps,
                                        track_running_stats=track_running_stats)
        input = torch.rand(batch_size, num_features, 5, requires_grad=True)
        out = torch_bn(input)
        bf = torch.rand_like(out)
        out.backward(bf)
        input_grad = input.grad.data.numpy()

        bn = BatchNorm(num_features, dim=dim, affine=affine, eps=eps,
                       track_running_stats=track_running_stats)
        if affine:
            if dim == 2:
                bn.weight.data[0, :] = torch_bn.weight.data.numpy()
            else:
                bn.weight.data[0, :, 0] = torch_bn.weight.data.numpy()

        x = from_numpy(input.data.numpy(), requires_grad=True)
        y = bn(x)
        y.backward(bf.data.numpy())

        result = np.abs(out.data.numpy() - y.numpy())
        print('out', result.mean()/np.abs(out.data.numpy()).mean())


        result = np.abs(x.grad.data - input_grad)
        print('x_grad', result.mean()/np.abs(x.grad.data).mean())


        if affine:
            if dim == 2:
                w_grad = bn.weight.grad.numpy()[0, :]
                b_grad = bn.bias.grad.numpy()[0, :]
            else:
                w_grad = bn.weight.grad.numpy()[0, :, 0]
                b_grad = bn.bias.grad.numpy()[0, :, 0]

            result = np.abs(w_grad - torch_bn.weight.grad.numpy())
            print('weight_grad', result.mean()/np.abs(torch_bn.weight.grad.numpy()).mean())


            result = np.abs(b_grad - torch_bn.bias.grad.numpy())
            print('bias_grad', result.mean()/np.abs(torch_bn.bias.grad.numpy()).mean())


        if track_running_stats:
            if dim == 2:
                r_mean = bn.running_mean[0, :]
                r_var = bn.running_var[0, :]
            else:
                r_mean = bn.running_mean[0, :, 0]
                r_var = bn.running_var[0, :, 0]

            result = np.abs(r_mean - torch_bn.running_mean.numpy())
            print('running_mean', result.mean()/np.abs(torch_bn.running_mean.numpy()).mean())


            result = np.abs(r_var - torch_bn.running_var.numpy())
            print('running_var', result.mean()/np.abs(torch_bn.running_var.numpy()).mean())


        # 开启测试模式
        torch_bn.eval()
        bn.switch_eval_mode()

        input_test = torch.rand_like(input)
        x_test = from_numpy(input_test.data.numpy())

        out_test = torch_bn(input_test)
        y_test = bn(x_test)

        result = np.abs(out_test.data.numpy() - y_test.data)
        print('test', result.mean()/np.abs(out_test.data.numpy()).mean())
        print('\n=========================')


    def  test_2(self):
        eps = 1e-5
        affine = True
        track_running_stats = True
        dim = 4
        num_features = 10
        batch_size = 256

        torch_bn = torch_nn.BatchNorm2d(num_features, affine=affine, eps=eps,
                                        track_running_stats=track_running_stats)
        input = torch.rand(batch_size, num_features, 28, 28, requires_grad=True)
        out = torch_bn(input)
        bf = torch.rand_like(out)
        out.backward(bf)
        input_grad = input.grad.data.numpy()

        bn = BatchNorm(num_features, dim=dim, affine=affine, eps=eps,
                       track_running_stats=track_running_stats)
        if affine:
            bn.weight.data[0, :, 0, 0] = torch_bn.weight.data.numpy()

        x = from_numpy(input.data.numpy(), requires_grad=True)
        y = bn(x)
        y.backward(bf.data.numpy())

        result = np.abs(out.data.numpy() - y.numpy())
        print('out', result.mean()/np.abs(out.data.numpy()).mean())

        result = np.abs(x.grad.data - input.grad.data.numpy())
        print('x_grad', result.mean()/np.abs(input.grad.data.numpy()).mean())

        if affine:
            w_grad = np.squeeze(bn.weight.grad.numpy())
            b_grad = np.squeeze(bn.bias.grad.numpy())

            result = np.abs(w_grad - torch_bn.weight.grad.numpy())
            print('weight_grad', result.mean()/np.abs(torch_bn.weight.grad.numpy()).mean())

            result = np.abs(b_grad - torch_bn.bias.grad.numpy())
            print('bias_grad', result.mean()/np.abs(torch_bn.bias.grad.numpy()).mean())

        if track_running_stats:
            r_mean = np.squeeze(bn.running_mean)
            r_var = np.squeeze(bn.running_var)

            result = np.abs(r_mean - torch_bn.running_mean.numpy())
            print('running_mean', result.mean()/np.abs(torch_bn.running_mean.numpy()).mean())

            result = np.abs(r_var - torch_bn.running_var.numpy())
            print('running_var', result.mean()/np.abs(torch_bn.running_var.numpy()).mean())

        # 开启测试模式
        torch_bn.eval()
        bn.switch_eval_mode()

        input_test = torch.rand_like(input)
        x_test = from_numpy(input_test.data.numpy())

        out_test = torch_bn(input_test)
        y_test = bn(x_test)

        result = np.abs(out_test.data.numpy() - y_test.data)
        print('test', result.mean()/np.abs(out_test.data.numpy()).mean())



if __name__ == '__main__':
    unittest.main()
