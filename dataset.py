import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from pytorch_lightning import LightningDataModule


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data[:, :-1]
        self.data_shifted = data[:, 1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data_shifted[idx]


class MyDataModule(LightningDataModule):

    def __init__(self,
        fn_train = "NPN_1155_part2.dat",
        fn_test =  "NPN_1155_part1.dat",
        cols=['KK'],
        batch_size: int = 32,
        freq=50/4
    ):
        super().__init__()

        self.freq = freq
        self.base_freq = 50
        self.batch_size = batch_size
        L = 1000

        data = pd.read_csv(fn_train, sep=" ")
        gaps = self._find_gaps(data)
        gaps = [0, *gaps, len(data)]
        data = data[cols].values
        data = self._make_sequences(data, gaps, L)
        self.train_set = MyDataset(data)

        data = pd.read_csv(fn_test, sep=" ")
        gaps = self._find_gaps(data)
        gaps = [0, *gaps, len(data)]
        data = data[cols].values
        data = self._make_sequences(data, gaps, L)
        self.test_set = MyDataset(data)

        # self.mnist_train, self.mnist_test, self.mnist_val = random_split(dataset, [0.8, 0.1, 0.1])
        # self.train_set, self.test_set = self.train_val_dataset(dataset)

    def train_val_dataset(dataset, val_split=0.25):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        datasets['val'] = Subset(dataset, val_idx)
        return datasets

    def _make_sequences(self, data, gaps, L):
        datas = []
        divider = int(self.base_freq / self.freq)
        for g0,g1 in zip(gaps, gaps[1:] ):
            _data = data[g0+1: g1-1]
            _data = _data[::divider]
            _data = _data[:(len(_data)//L)*L]
            _data = _data.reshape(-1, L).astype('float32')
            datas += [_data]
        return np.concatenate(datas, axis=0)

    def _find_gaps(self, data):
        a = data["sec"].values
        threshold = 2
        return np.where(abs(np.diff(a))>threshold)[0] + 1

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=1)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=1)
