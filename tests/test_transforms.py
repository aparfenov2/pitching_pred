import torch
import unittest
from dataset import MyDataModule
from transforms import InvertZero

class TransformsUT(unittest.TestCase):

    def test_invert(self):
        tr = InvertZero()
        data = torch.ones((8, 10, 1))
        data = tr(data)
        # print("test_invert", data)
        self.assertTrue(torch.any(data < 0))

    def test_invert2(self):
        m = MyDataModule(
            fn_train = "tests/data/sin_train.dat",
            fn_test = "tests/data/sin_train.dat",
            test_L = 100,
            test_augs = ["transforms.InvertZero"]
        )
        dl = m.test_dataloader()
        for y,t in dl[0]:
            pass
