from collections import OrderedDict
from ..grad_mode import no_grad


class NeuralNetwork(object):
    def __init__(self):
        self.training = True
        self._parameters = []
        self._layers = OrderedDict()

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if isinstance(value, NeuralNetwork):
            self._layers[key] = value
            self._parameters += value.parameters()

    def __repr__(self):
        string = self.__class__.__name__ + "(\n"
        for name, layer in self._layers.items():
            string += "   (%s): " % name + layer.__repr__() + "\n"
        string += ")"
        return string

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        if self.training:
            return self.forward(*input)
        else:
            with no_grad():
                return self.forward(*input)

    def parameters(self):
        return self._parameters

    def zero_grad(self):
        for para in self._parameters:
            para.zero_grad()

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
Layer = NeuralNetwork


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for id, layer in enumerate(args):
            self.__setattr__('layer_%s' % id, layer)
            self.num_layers = id

    def forward(self, input):
        output = input
        for layer in self._layers.values():
            output = layer.forward(output)

        return output
