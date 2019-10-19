from .optimizer import Optimizer,learn_rate
import numpy as np


class Adagrad(Optimizer):
    """
    Adagrad 优化算法
    parameters： 模型参数
    lr: 学习率， float 或者分段参数 [(5,0.02),(10,0.01),(15,0.005),(20,0.001)]
                如果使用分段参数， 第一个代表 epoch， 第二个代表学习率
                比如(5,0.02) 代表 1-5 epoch 学习率为0.02
                (10,0.01) 代表 6-10 epoch 学习率为0.01，...以此类推
                默认0.01
    eps： 提高稳定性的小值 1e-8
    """
    def __init__(self, parameters, lr=0.01, eps=1e-8):
        super().__init__(parameters)
        self.lr = learn_rate(lr)
        self.is_group_lr = False
        self.lr_length = 1
        if len(self.lr) > 1:
            self.is_group_lr = True
            self.lr_length = len(self.lr)

        self.eps = eps
        self.grad_square = []
        for para in self.parameters:
            self.grad_square.append(np.zeros(para.shape))

    def step(self, epoch=None):
        if self.is_group_lr:
            lr = self.lr[min(epoch, self.lr_length - 1)]
        else:
            lr = self.lr[0]

        for para, r in zip(self.parameters,self.grad_square):
            grad = para.grad.data
            r += grad * grad
            para.data -= lr*grad/(np.sqrt(r)+self.eps)
