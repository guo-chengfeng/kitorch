from .optimizer import Optimizer,learn_rate
import numpy as np


class RMSprop(Optimizer):
    """
    parameters： 模型参数
    lr: 学习率， float 或者分段参数 [(5,0.02),(10,0.01),(15,0.005),(20,0.001)]
                如果使用分段参数， 第一个代表 epoch， 第二个代表学习率
                比如(5,0.02) 代表 1-5 epoch 学习率为0.02
                (10,0.01) 代表 6-10 epoch 学习率为0.01，...以此类推
                默认0.01
    beta: 用于计算梯度平方的平滑系数，默认0.9
          pytorch的默认值是0.99，但是效果很差，几乎不能使用
    eps： 提高稳定性的小值 1e-8

    在实际使用过程中，RMSprop已被证明是一种有效且实用的深度神经网络优化算法。
    目前它是深度学习人员经常采用的优化算法之一。
    keras文档中关于RMSprop写到：
    This optimizer is usually a good choice for recurrent neural networks.

    在卷积神经网路中RMSprop 效果并不好，并且会出现严重的震荡
    """

    def __init__(self, parameters, lr=0.01, beta=0.9,eps=1e-8):
        super().__init__(parameters)
        self.lr = learn_rate(lr)
        self.is_group_lr = False
        self.lr_length = 1
        if len(self.lr) > 1:
            self.is_group_lr = True
            self.lr_length = len(self.lr)

        self.eps = eps
        self.beta = beta
        self.grad_square = []
        for layer_parameters in self.parameters:
            layer_grad_square = []
            for para in layer_parameters:
                layer_grad_square.append(np.zeros(para.shape))
            self.grad_square.append(layer_grad_square)

    def step(self,epoch=None):
        if self.is_group_lr:
            lr = self.lr[min(epoch, self.lr_length - 1)]
        else:
            lr = self.lr[0]

        for layer_para, layer_r in zip(self.parameters,self.grad_square):
            for para, r in zip(layer_para,layer_r):
                grad = para.grad.data
                r *= self.beta
                r += (1 - self.beta) * grad * grad
                para.data -= lr*grad/(self.eps+np.sqrt(r))