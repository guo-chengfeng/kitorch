from .optimizer import Optimizer, learn_rate
import numpy as np


class SGD(Optimizer):
    """
    带有动量的SGD
    parameters： 模型参数
    lr: 学习率， float 或者分段参数 [(5,0.02),(10,0.01),(15,0.005),(20,0.001)]
                如果使用分段参数， 第一个代表 epoch， 第二个代表学习率
                比如(5,0.02) 代表 1-5 epoch 学习率为0.02
                (10,0.01) 代表 6-10 epoch 学习率为0.01，...以此类推
                默认 0.01
    momentum： 动量矫正系数， 如采用动量算法建议使用0.9，默认不适用动量算法

    动量的计算有三种方式：
    1 原始论文 On the importance of initialization and momentum in deep learning
    v = beta * v - lr*g
    p = p + v

    2
    v = beta * v + g
    p = p - lr * v

    3
    v = beta * v + (1-beta) * g
    p = p - lr * v
    这里采用第一种方式
    """

    def __init__(self, parameters, lr=0.01, momentum=0):
        super().__init__(parameters)

        self.lr = learn_rate(lr)
        self.is_group_lr = False
        self.lr_length = 1
        if len(self.lr) > 1:
            self.is_group_lr = True
            self.lr_length = len(self.lr)

        self.need_momentum = False
        if momentum > 0:
            # 动量的系数，所有系数都采用 beta 符号
            self.beta = momentum
            self.need_momentum = True
            self.para_momentum = []
            for para in self.parameters:
                self.para_momentum.append(np.zeros(para.shape))

    def step(self, epoch=None):

        if self.is_group_lr:
            lr = self.lr[min(epoch, self.lr_length - 1)]
        else:
            lr = self.lr[0]

        # 动量SGD
        if self.need_momentum:
            for para, m in zip(self.parameters, self.para_momentum):
                m *= self.beta
                m -= lr * para.grad.data
                para.data += m
        else:
            # 标准mini-batch SGD
            for para in self.parameters:
                # 也可以用 para -= lr * para.grad
                # 但是这样会产生一些中间Tensor
                # 另外，这样也不用考虑requires_grad的值了
                para.data -= lr * para.grad.data
