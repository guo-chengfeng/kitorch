from .. import no_grad


class Layer(object):
    def __init__(self):
        self.parameters = []
        self.training = True

    def __call__(self, *input):
        if self.training:
            return self.forward(*input)

        else:
            with no_grad():
                return self.forward(*input)

    def forward(self, *input):
        raise NotImplementedError

    def zero_grad(self):
        for para in self.parameters:
            para.zero_grad()

    def freeze(self):
        # No more to train this layer
        for para in self.parameters:
            para.requires_grad = False

    def switch_mode(self, mode):
        # switch the module in training or evaluating mode.
        self.training = mode

    def switch_train_mode(self):
        self.switch_mode(True)

    def switch_eval_mode(self):
        self.switch_mode(False)
