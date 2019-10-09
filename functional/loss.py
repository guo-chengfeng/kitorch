def nll_loss(input, target, dim=1):
    """
    The negative log likelihood loss.
    """
    assert len(input.shape) == 2, "Except 2D Tensor"
    d0, d1 = input.shape
    if dim == 0:
        idxs = [i for i in range(d1)]
        return -input[target, idxs].mean()
    else:
        idxs = [i for i in range(d0)]
        return -input[idxs, target].mean()


def cross_entropy(input, target, dim=1):
    """
    This criterion combines `log_softmax` and `nll_loss` in a single function.
    """
    assert len(input.shape) == 2, "Except 2D Tensor"
    d0, d1 = input.shape
    if dim == 0:
        idxs = [i for i in range(d1)]
        return -input[target, idxs].log().mean()
    else:
        idxs = [i for i in range(d0)]
        return -input[idxs, target].log().mean()


def mse_loss(y_hat, y):
    """
    Measures the element-wise mean squared error.
    """
    return ((y_hat - y) ** 2).mean()
