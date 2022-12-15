import torch
import unittest
from model import MyModel, RNNState

class MyModelUT(unittest.TestCase):

    def testForward31(self):
        m = MyModel(
            input_sz=3,
            output_sz=1
        )
        _input_t = torch.tensor([1.0, 2.0, 3.0]).reshape((1,1,3))
        out = m.forward(_input_t, extend_output_size_to_input=False)
        self.assertEqual(out.shape, (1, 1, 1))

    def testForward11(self):
        m = MyModel(
            input_sz=1,
            output_sz=1
        )
        _input_t = torch.tensor([1.0]).reshape((1,1,1))
        out = m.forward(_input_t)
        self.assertEqual(out.shape, (1, 1, 1))
