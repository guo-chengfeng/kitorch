import numpy as np


# 这个参数比较全 https://blog.csdn.net/u012328159/article/details/80311892
#
# https://www.jianshu.com/p/33eed2e1d357
# https://www.cnblogs.com/callyblog/p/8299074.html

class Optimizer():
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self, *args):
        raise NotImplementedError

    def zero_grad(self):
        for para in self.parameters:
            para.zero_grad()


def learn_rate(lr):
    """
    返回一个学习率
    lr 可以是一个浮点数值，也可以是[(10,0.05),(20,0.01),(30,0.005)]分段数值
    """
    if isinstance(lr, (list, tuple)):
        group_lr = np.ones(lr[-1][0]) * lr[0][1]
        for i in range(1, len(lr)):
            group_lr[lr[i - 1][0]:lr[i][0]] = lr[i][1]
        return group_lr
    else:
        return np.array([lr])
