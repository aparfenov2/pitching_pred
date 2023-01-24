import torch
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Subset
from pytorch_lightning import LightningDataModule
from utils import make_augs
from typing import Dict, List

class MyDataset(Dataset):
    def __init__(self, data, t, name):
        self.data = data
        self.t = t
        self.name = name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.t[idx]

class MyDataset2(Dataset):
    def __init__(self, data:Dict[str, np.ndarray], name:str):
        self.data = data
        self.name = name

    def __len__(self):
        return len(self.data["t"])

    def __getitem__(self, idx):
        return {
            k : v[idx] for k,v in self.data.items()
        }

    def __str__(self) -> str:
        ret = self.name + ":\n"
        for k,v in self.data.items():
            ret += f"\t{k}\t{v.shape}\n"
        return ret

DEFAULT_TRAIN_CONFIG = {
    "L":500,
    "stride": 0.5,
    "multiply": 10,
    "transforms": [
        "transforms.InvertZero"
    ]
}

DEFAULT_VAL_CONFIG = {
    "L":500,
    "transforms": [
        "transforms.InvertZero"
    ]
}

DEFAULT_TEST_CONFIG = {
    "L":1000,
}

class MyDataModule(LightningDataModule):

    def __init__(self,
        fn_train = "data/NPN_1155_part2.dat",
        fn_test =  "data/NPN_1155_part1.dat",
        cols=['KK'],
        batch_size: int = 32,
        test_batch_size: int = 8,
        freq=4,
        base_freq=50,
        test_only=False,
        train_pipeline=None,
        val_pipeline=None,
        test_pipeline=None,
        train_config=DEFAULT_TRAIN_CONFIG,
        val_config=DEFAULT_VAL_CONFIG,
        test_config=DEFAULT_TEST_CONFIG,
    ):
        super().__init__()

        self.freq = freq
        self.base_freq = base_freq
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.cols = cols

        def mixin_common_args(config):
            config["base_freq"] = base_freq
            config["freq"] = freq
            config["cols"] = cols

        train_config = copy.deepcopy(train_config)
        val_config = copy.deepcopy(val_config)
        test_config = copy.deepcopy(test_config)

        mixin_common_args(train_config)
        mixin_common_args(val_config)
        val_config["L"] = train_config["L"]
        mixin_common_args(test_config)

        if train_pipeline is None:
            train_pipeline = self.make_legacy_pipeline(**train_config)

        if val_pipeline is None:
            val_pipeline = self.make_legacy_pipeline(**val_config)

        if test_pipeline is None:
            test_pipeline = self.make_legacy_pipeline(**test_config)

        train_pipeline = make_augs(train_pipeline)
        val_pipeline = make_augs(val_pipeline)
        test_pipeline = make_augs(test_pipeline)

        if not test_only:
            self.train_set = MyDataset2(
                data=train_pipeline(fn_train),
                name="train:"+fn_train
            )
            self.val_set = MyDataset2(
                data=val_pipeline(fn_train),
                name="val:"+fn_train
            )
            print(self.train_set)
            print(self.val_set)

        if not isinstance(fn_test, list):
            fn_test = [fn_test]

        self.test_set = [
            MyDataset2(
                data=test_pipeline(fn),
                name="test:"+fn
                ) for fn in fn_test
            ]
        for ts in self.test_set:
            print(ts)

    @staticmethod
    def make_legacy_pipeline(
        cols:List=["KK"],
        base_freq:int=50,
        freq:int=4,
        L:int=1000,
        stride:float=1.0,
        multiply:int=1,
        transforms:List=[]
        ):
        return [
            "transforms.ReadCSV",
            "transforms.FixTime",
            "transforms.RelativeTime",
            "transforms.AsFloat32",
            {
                "transforms.PandasToDictOfNpArrays": {
                    "mapping": {
                        "sec": "t",
                        **{col: "y" for col in cols}
                    }
                }
            },
            {
                "transforms.Downsample": {
                    "base_freq": base_freq,
                    "freq": freq
                }
            },
            {
                "transforms.FindGapsAndTransform": {
                    "for_each_contiguous_block": [
                        {
                            "transforms.StrideAndMakeBatches": {
                                "L": L,
                                "stride": stride
                            }
                        }
                    ]
                }
            },
            {
                "transforms.ConcatBatches": {
                    "multiply": multiply
                }
            },
            *transforms
        ]

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
            _data = _data[::divider]
            t     = t[::divider]
            _data = _data[:(len(_data)//L)*L]
            t     =     t[:(len(_data)//L)*L]
            if len(_data.shape) == 1:
                _data = _data.reshape(-1, L, 1).astype('float32')
            else:
                _data = _data.reshape(-1, L, _data.shape[-1]).astype('float32')
            t     = t.reshape(-1, L, 1).astype('float32')
            datas += [_data]
            ts    += [t]
        data = np.concatenate(datas, axis=0)
        t    = np.concatenate(ts,    axis=0)
        data = torch.tensor(data)
        t    = torch.tensor(t)

        # np.savetxt("before.csv", data.flatten().numpy())
        data = transforms(data)
        # np.savetxt("after.csv", data.flatten().numpy())


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
