import torch
import unittest
from dataset import MyDataModule
from transforms import InvertZero
import yaml
import io

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

    def test_ChannelsToFeatures(self):
        train_config = """
L: 112
stride: 0.2
multiply: 20
transforms:
- transforms.InvertZero
- transforms.InvertTime
- transforms.InvertMean
- transforms.BiasAndScale
- transforms.AddSpeed:
    distance: 1
- transforms.ChannelsToFeatures:
    channels:
        - KK_v
"""
        with io.StringIO(train_config) as stream:
            train_config = yaml.safe_load(stream)

        m = MyDataModule(
            fn_train = "tests/data/sin_train.dat",
            fn_test = "tests/data/sin_train.dat",
            freq=1,
            base_freq=1,
            train_config=train_config
        )
        dl = m.train_dataloader()
        for batch in dl:
            assert "KK_v" in batch
            y = batch["y"]
            assert y.shape[-1] == 2
