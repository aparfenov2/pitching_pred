import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from pytorch_lightning import LightningDataModule
from utils import resolve_classpath

class MyDataset(Dataset):
    def __init__(self, data, t, name):
        self.data = data
        self.t = t
        self.name = name

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
        train_multiply=1,
        test_multiply=1,
        L=1000,
        freq=50/4,
        base_freq=50,
        test_only=False,
        train_augs=[],
        test_augs=[]
    ):
        super().__init__()

        self.freq = freq
        self.base_freq = base_freq
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.cols = cols
        self.test_L = test_L
        self.L = L
        self.train_augs = self.make_augs(train_augs)
        self.test_augs = self.make_augs(test_augs)
        if not test_only:
            self.train_set = self.read_data_and_make_dataset(
                fn_train, cols, L=L,
                set_name="train:"+fn_train,
                multiply=train_multiply,
                transforms=self.train_augs
                )
            self.val_set = self.read_data_and_make_dataset(fn_train, cols, L=L,
                set_name="val:"+fn_train,
                multiply=1,
                transforms=self.test_augs
                )
        if isinstance(fn_test, str):
            fn_test = [fn_test]
        self.test_set = [self.read_data_and_make_dataset(
                fn, cols, L=test_L,
                set_name="test:"+fn,
                multiply=test_multiply,
                transforms=self.test_augs
            ) for fn in fn_test]

    @staticmethod
    def add_speed_to_data(_data):
        for col in _data.columns:
            _data[f"{col}_v"] = _data[col].shift(20, fill_value=0) - _data[col]

    def make_augs(augs):
        ret = []
        for aug in augs:
            if isinstance(aug, dict):
                aug_classpath = list(aug.keys())[0]
                aug_init_args = aug[aug_classpath]
                aug = resolve_classpath(aug_classpath)(**aug_init_args)
            else:
                assert isinstance(aug, list)
                aug = resolve_classpath(aug)()
            ret += [aug]
        return torch.nn.Sequential(ret)

    def read_data_and_make_dataset(self, fn, cols, L, set_name, multiply, transforms):

        data = pd.read_csv(fn, sep=" ")
        gaps = self._find_gaps(data)
        gaps = [0, *gaps, len(data)] * multiply
        print(set_name, gaps)

        datas = []
        ts    = []
        divider = int(self.base_freq / self.freq)

        for g0,g1 in zip(gaps, gaps[1:] ):
            if g0 > g1:
                continue
            _data = data[g0+1: g1-1].copy()
            self.add_speed_to_data(_data)
            t     = _data["sec"]
            if "msec" in _data.columns:
                t     += _data["msec"]/1000
            _data = _data[cols].values
            t     = t.values
            t -= t[0]
            # print(f"{set_name}: slice with gaps removed len={len(_data)}")
            _data = _data[::divider]
            t     = t[::divider]
            # print(f"{set_name}: slice after divider {divider} len={len(_data)}")
            _data = _data[:(len(_data)//L)*L]
            t     =     t[:(len(_data)//L)*L]
            # print(f"{set_name}: slice after L {L} len={len(_data)}")
            if len(_data.shape) == 1:
                _data = _data.reshape(-1, L, 1).astype('float32')
            else:
                _data = _data.reshape(-1, L, _data.shape[-1]).astype('float32')
            t     = t.reshape(-1, L, 1)
            datas += [_data]
            ts    += [t]
        data = np.concatenate(datas, axis=0)
        t    = np.concatenate(ts,    axis=0)

        print("apply transforms")
        data = transforms(data)
        print("transforms applied")

        assert len(t) == len(data), str(len(t)) + ' ' + str(len(data))
        print(f"{set_name}: data.shape {data.shape}")
        return MyDataset(data, t, set_name)

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
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        if not isinstance(self.test_set, list):
            return [DataLoader(self.test_set, batch_size=self.test_batch_size, num_workers=1)]
        return [DataLoader(ts, batch_size=self.test_batch_size, num_workers=1) for ts in self.test_set]

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, num_workers=1)
