from collections import OrderedDict
from .layer import Layer
from .. import tensor as mt


class NeuralNetwork(object):
    def __init__(self):
        self.training = True
        self._layers = OrderedDict()

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if isinstance(value, Layer):
            self._layers[key] = value

    def __repr__(self):
        string = self.__class__.__name__ + "(\n"
        for name, layer in self._layers.items():
            string += "   (%s): " % name + layer.__repr__() + "\n"
        string += ")"
        return string

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def parameters(self):
        parameters = []
        for layer in self._layers.values():
            parameters.append(layer.parameters)
        return parameters

    def zero_grad(self):
        for layer in self._layers.values():
            layer.zero_grad()

    def switch_mode(self, mode=True):
        """
        switch the module in training or evaluating mode.
        :param mode:  bool, True for trainging mode, False for evaluating mode
        """
        self.training = mode
        for layer in self._layers.values():
            layer.switch_mode(mode)

    def switch_train_mode(self):
        self.switch_mode(True)

    def switch_eval_mode(self):
        self.switch_mode(False)

    # keep API same with Torch
    def train(self):
        self.switch_train_mode()

    def eval(self):
        self.switch_eval_mode()


Module = NeuralNetwork
