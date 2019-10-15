from ..tensor import Tensor, np


def dropout(inputs, p=0.5, training=True):
    """
    每个元素按照努利分布概率分布置为零
    p: probability of an element to be zeroed. Default: 0.5
    training: apply dropout if is ``True``. Default: ``True``
    """

    if training:
        assert p < 1, "probability must < 1"
        scale = 1 / (1 - p)
        mask = Tensor(scale * np.random.binomial(1, 1 - p, inputs.shape))
        return inputs * mask
    else:
        return inputs


def dropout2d(inputs, p=0.5, training=True):
    """
    对4维(N, C, H, W)矩阵进行操作，每个样本的channel有一定概率被置为零
    p: probability of an element to be zeroed. Default: 0.5
    training: apply dropout if is ``True``. Default: ``True``
    Usually the input comes from conv2d modules.
    Shape:
    - Inputs: :math:`(N, C, H, W)`
    - Output: :math:`(N, C, H, W)` (same shape as inputs)
    """

    if training:
        assert p < 1, "probability must < 1"
        assert len(inputs.shape) == 4, "the shape of input should be (N, C, H, W)"

        (N, C, H, W) = inputs.shape
        scale = 1 / (1 - p)
        ones_block = np.ones((H, W)) * scale
        _mask = scale * np.random.binomial(1, 1 - p, (N, C))
        mask = np.zeros((N, C, H, W))
        for i in range(N):
            for j in range(C):
                if _mask[i, j] > 1:
                    mask[i, j] = ones_block

        return inputs * Tensor(mask)
    else:
        return inputs
