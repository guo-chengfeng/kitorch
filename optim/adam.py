from .optimizer import Optimizer, learn_rate
import numpy as np

class Adam(Optimizer):
    """
    Adam实际上是把momentum和RMSprop结合起来的一种算法,在大多数任务中效果良好
    parameters： 模型参数
    lr: 学习率， float 或者分段参数 [(5,0.02),(10,0.01),(15,0.005),(20,0.001)]
                如果使用分段参数， 第一个代表 epoch， 第二个代表学习率
                比如(5,0.02) 代表 1-5 epoch 学习率为0.02
                (10,0.01) 代表 6-10 epoch 学习率为0.01，...以此类推
                默认0.001
    betas: 用于计算梯度和梯度平方的平滑系数，默认(0.9, 0.999)
    eps： 提高稳定性的小值 1e-8
    """

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(parameters)
        self.lr = learn_rate(lr)
        self.is_group_lr = False
        self.lr_length = 1
        if len(self.lr) > 1:
            self.is_group_lr = True
            self.lr_length = len(self.lr)

        self.times = 0
        self.eps = eps
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.exp_avg = []
        self.exp_avg_sq = []
        for layer_parameters in self.parameters:
            exp_avg = []
            exp_avg_sq = []
            for para in layer_parameters:
                exp_avg.append(np.zeros(para.shape))
                exp_avg_sq.append(np.zeros(para.shape))
            self.exp_avg.append(exp_avg)
            self.exp_avg_sq.append(exp_avg_sq)

    def step(self, epoch=None):
        if self.is_group_lr:
            lr = self.lr[min(epoch, self.lr_length - 1)]
        else:
            lr = self.lr[0]

        self.times += 1
        bias = np.sqrt(1 - self.beta2 ** self.times) / (1 - self.beta1 ** self.times)
        lr = bias * lr
        for _para, _exp_avg, _exp_avg_sq, in zip(self.parameters, self.exp_avg, self.exp_avg_sq):
            for para, exp_avg, exp_avg_sq in zip(_para, _exp_avg, _exp_avg_sq):
                grad = para.grad.data
                exp_avg *= self.beta1
                exp_avg += (1 - self.beta1) * grad
                exp_avg_sq *= self.beta2
                exp_avg_sq += (1 - self.beta2) * grad * grad

                para.data -= lr * exp_avg / (self.eps + np.sqrt(exp_avg_sq))




