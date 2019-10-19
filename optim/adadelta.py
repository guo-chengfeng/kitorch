from .optimizer import Optimizer,learn_rate
import numpy as np


class Adadelta(Optimizer):
    """
    Adadelta 优化算法

    parameters： 模型参数
    lr: 学习率，
        理论上Adadelta是不需要学习率的，但是仍然可以加入一个学习率，提供灵活性，默认学习率为1.0
        float 或者分段参数 [(5,0.02),(10,0.01),(15,0.005),(20,0.001)]
        如果使用分段参数， 第一个代表 epoch， 第二个代表学习率
        比如(5,0.02) 代表 1-5 epoch 学习率为0.02
        (10,0.01) 代表 6-10 epoch 学习率为0.01，...以此类推
        默认1.0
    beta: 用于计算梯度平方的平滑系数，默认0.9
    eps： 提高稳定性的小值 1e-8
    """
    def __init__(self, parameters, lr=1.0, beta=0.9, eps=1e-6):
        super().__init__(parameters)
        self.lr = learn_rate(lr)
        self.is_group_lr = False
        self.lr_length = 1
        if len(self.lr) > 1:
            self.is_group_lr = True
            self.lr_length = len(self.lr)

        self.beta = beta
        self.eps = eps
        self.grad_square = []
        self.acc_delta = []
        for para in self.parameters:
            self.grad_square.append(np.zeros(para.shape))
            self.acc_delta.append(np.zeros(para.shape))

    def step(self, epoch=None):
        if self.is_group_lr:
            lr = self.lr[min(epoch, self.lr_length - 1)]
        else:
            lr = self.lr[0]

        for para,acc_delta,grad_sq in zip(self.parameters,self.acc_delta,self.grad_square):
            grad = para.grad.data
            grad_sq *= self.beta
            grad_sq += (1 - self.beta) * grad * grad
            delta =  np.sqrt(acc_delta+self.eps)/np.sqrt( grad_sq+self.eps) * grad
            para.data -= lr * delta
            acc_delta *= self.beta
            acc_delta += (1 - self.beta) * delta * delta


