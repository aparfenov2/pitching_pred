import torch
import unittest
from models.model import MyModel, RNNState
from models.linear_model import LinearModel

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


class LinearModelUT(unittest.TestCase):
    def test1(self):
        m = LinearModel(
            freq=4,
            history_len_s=20
        )
        y = torch.zeros((1,1000,1))
        preds = m.forward(y, future=10)
        self.assertEqual(preds.shape, (1, 10, 1))
