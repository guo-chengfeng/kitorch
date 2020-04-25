from ..nn.embedding import Embedding
from ..tensor import Tensor
import unittest
import torch
from torch.nn import Embedding as torchEmbedding
from . import check


class EmbeddingTest(unittest.TestCase):
    def test_1(self):
        emb1 = torchEmbedding(10, 3)
        ipt = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        out = emb1(ipt)

        g = out.sigmoid()
        f = g.sum()
        f.backward()

        emb2 = Embedding(10, 3)
        emb2.from_pretrained(Tensor(emb1.weight.detach().numpy()), freeze=False)
        ipt = Tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        out = emb2(ipt)
        g = out.sigmoid()
        f = g.sum()
        f.backward()

        check(emb1.weight,emb2.weight,eps=1e-6)


if __name__ == '__main__':
    unittest.main()
