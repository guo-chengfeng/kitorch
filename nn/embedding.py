import numpy as np
from typing import List
from ..tensor import Tensor, Edge
from ..utils import rand
from .module import Layer


class Embedding(Layer):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: : [batch_size,max_seq_len]
        - Output: :  [batch_size,max_seq_len,embedding_dim]

    """

    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._parameters = []
        self.weight = rand(num_embeddings, embedding_dim, requires_grad=True)
        self._parameters.append(self.weight)

    def from_pretrained(self, embeddings, freeze=True):
        if freeze:
            self.weight.requires_grad = False

        self.weight.data[:] = embeddings.data[:]
        print('from_pretrained', self.weight.data.dtype)

    def forward(self, x: Tensor):
        return embedding(x, self.weight, self.embedding_dim)

    def __repr__(self):
        string = "Embedding(num_embeddings=%s, embedding_dim=%s" % (self.num_embeddings, self.embedding_dim)
        string += ")"
        return string


def EmbeddingBackward(grad: 'Tensor', t: 'Tensor', cache: List) -> 'Tensor':
    grad_data = np.zeros_like(t, dtype=grad.data.dtype)
    index, embedding_dim = cache
    out_grad = grad.data.reshape(-1, embedding_dim)
    for i, idx in enumerate(index):
        grad_data[idx] += out_grad[i]

    return Tensor(grad_data)


def embedding(x: 'Tensor', weight: 'Tensor', embedding_dim):
    requires_grad = weight.requires_grad
    shape = list(x.shape)
    shape.append(embedding_dim)
    index = np.reshape(x.data, -1)
    data = weight.data[index]

    grad_fn = EmbeddingBackward
    depends_on = []
    if requires_grad:
        depends_on.append(Edge(weight, [index, embedding_dim]))

    return Tensor(np.reshape(data, shape),
                  requires_grad, depends_on, grad_fn)
