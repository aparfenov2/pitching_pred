import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from pytorch_lightning import LightningDataModule
import unittest

class MyDataset(Dataset):
    def __init__(self, data, t):
        self.data = data
        self.t = t

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.t[idx]


class MyDataModule(LightningDataModule):

    def __init__(self,
        fn_train = "data/NPN_1155_part2.dat",
        fn_test =  "data/NPN_1155_part1.dat",
        cols=['KK'],
        batch_size: int = 32,
        test_batch_size: int = 8,
        test_L=30000,
        L=1000,
        freq=50/4,
        base_freq=50,
    ):
        super().__init__()

        self.freq = freq
        self.base_freq = base_freq
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.cols = cols
        self.test_L = test_L
        self.L = L
        self.train_set = self.read_data_and_make_dataset(fn_train, cols, L=L, set_name="train")
        self.val_set = self.read_data_and_make_dataset(fn_test, cols, L=L, set_name="val")
        self.test_set = self.read_data_and_make_dataset(fn_test, cols, L=test_L, set_name="test")

    @staticmethod
    def add_speed_to_data(_data):
        for col in _data.columns:
            _data[f"{col}_v"] = _data[col].shift(20, fill_value=0) - _data[col]

    def read_data_and_make_dataset(self, fn, cols, L, set_name):

        data = pd.read_csv(fn, sep=" ")
        gaps = self._find_gaps(data)
        gaps = [0, *gaps, len(data)]
        print(set_name, gaps)

        datas = []
        ts    = []
        divider = int(self.base_freq / self.freq)

        for g0,g1 in zip(gaps, gaps[1:] ):
            _data = data[g0+1: g1-1].copy()
            self.add_speed_to_data(_data)
            t     = _data["sec"]
            if "msec" in _data.columns:
                t     += _data["msec"]/1000
            _data = _data[cols].values
            t     = t.values
            print(f"{set_name}: slice with gaps removed len={len(_data)}")
            _data = _data[::divider]
            t     = t[::divider]
            print(f"{set_name}: slice after divider {divider} len={len(_data)}")
            _data = _data[:(len(_data)//L)*L]
            t     =     t[:(len(_data)//L)*L]
            print(f"{set_name}: slice after L {L} len={len(_data)}")
            if len(_data.shape) == 1:
                _data = _data.reshape(-1, L, 1).astype('float32')
            else:
                _data = _data.reshape(-1, L, _data.shape[-1]).astype('float32')
            t     = t.reshape(-1, L)
            datas += [_data]
            ts    += [t]
        data = np.concatenate(datas, axis=0)
        t    = np.concatenate(ts,    axis=0)
        assert len(t) == len(data), str(len(t)) + ' ' + str(len(data))
        print(f"{set_name}: data.shape {data.shape}")
        return MyDataset(data, t)

    def train_val_dataset(dataset, val_split=0.25):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        datasets['val'] = Subset(dataset, val_idx)
        return datasets

    @staticmethod
    def _find_gaps(data):
        a = data["sec"].values
        threshold = 2
        return np.where(abs(np.diff(a))>threshold)[0] + 1

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, num_workers=2)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, num_workers=2)


class MyDataModuleUT(unittest.TestCase):

    @unittest.skip("disabled")
    def test_speed(self):
        m = MyDataModule()
        dl = m.train_dataloader()
        for x, y in dl:
            self.assertTrue('KK' in m.cols)
            self.assertTrue('KK_v' in m.cols)

    def test_test_shape(self):
        m = MyDataModule()
        dl = m.test_dataloader()
        print("test_dl.length", len(dl))
        self.assertGreater(len(dl), 0)
        for x, y in dl:
            self.assertEqual(y.shape, (m.test_batch_size, m.test_L, len(m.cols)))
            break
