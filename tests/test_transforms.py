import torch
import unittest

from transforms import InvertZero

class TransformsUT(unittest.TestCase):

    def test_invert(self):
        tr = InvertZero()
        data = torch.ones((8, 10, 1))
        data = tr(data)
        print(data)
        self.assertTrue(torch.any(data < 0))
