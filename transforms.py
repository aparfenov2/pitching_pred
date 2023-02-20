import torch
import pandas as pd
import numpy as np
from typing import Dict, List
from utils import make_augs
from collections import defaultdict

# transforms, elements of pipeline

# --------------------------------- input: pd.DataFrame -----------------------------------------
class SplitAtGaps(torch.nn.Module):
    @staticmethod
    def _find_gaps(data):
        a = data["sec"].values
        threshold = 2
        return np.where(abs(np.diff(a))>threshold)[0] + 1

    def forward(self, data: pd.DataFrame):
        gaps = self._find_gaps(data)
        gaps = [0, *gaps, len(data)]
        return [data[g0+1: g1-1] for g0,g1 in zip(gaps, gaps[1:] )]

class ReadCSV(torch.nn.Module):
    def forward(self, fn: str) -> pd.DataFrame:
        return pd.read_csv(fn, sep=" ")

class FixTime(torch.nn.Module):
    def forward(self, _data: pd.DataFrame):
        t     = _data["sec"]
        if "msec" in _data.columns:
            t     += _data["msec"]/1000
        return _data.assign(**{"sec": t})

class RelativeTime(torch.nn.Module):
    def forward(self, _data: pd.DataFrame):
        t     = _data["sec"]
        t -= t[0]
        return _data.assign(**{"sec": t})

class AsFloat32(torch.nn.Module):
    def forward(self, _data: pd.DataFrame):
        return _data.astype('float32')

class AddSpeed(torch.nn.Module):
    def __init__(self, distance=20):
        super().__init__()
        self.delay = distance

    def add_speed_to_data(self, _data: pd.DataFrame):
        for col in _data.columns:
            _data[f"{col}_v"] = _data[col].shift(self.delay, fill_value=0) - _data[col]

    def forward(self, data: pd.DataFrame):
        _y = data.copy()
        self.add_speed_to_data(_y)
        return _y

class PandasToDictOfNpArrays(torch.nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping
        self.inv_mapping = defaultdict(list)
        for src_col, trg_col in mapping.items():
            self.inv_mapping[trg_col] += [src_col]

    def forward(self, _data: pd.DataFrame):
        ret = {}
        for trg_col, src_cols in self.inv_mapping.items():
            ret[trg_col] = _data[src_cols].values

        for col in _data.columns:
            if col not in ret:
                ret[col] = _data[[col]].values
        return ret

# --------------------------------- input: dict of ndarray -----------------------------------------

class Downsample(torch.nn.Module):
    def __init__(self, base_freq, freq):
        super().__init__()
        self.divider = int(base_freq / freq)

    def forward(self, _data: Dict[str, np.ndarray]):
        return {
            k: v[::self.divider] for k,v in _data.items()
        }

class FindGapsAndTransform(torch.nn.Module):
    def __init__(self, for_each_contiguous_block):
        super().__init__()
        self.transforms = make_augs(for_each_contiguous_block)

    @staticmethod
    def _find_gaps(a):
        threshold = 2
        return np.where(abs(np.diff(a))>threshold)[0] + 1

    def forward(self, data: Dict[str, np.ndarray]):
        gaps = self._find_gaps(data["t"])
        gaps = [0, *gaps, len(data["t"])]
        ret = defaultdict(list)
        for g0,g1 in zip(gaps, gaps[1:]):
            _ret = self.transforms(
                    {
                        k: v[g0+1: g1-1] for k, v in data.items()
                    }
                )
        for k, v in data.items():
            ret[k] += [_ret[k]]
        return ret

class StrideAndMakeBatches(torch.nn.Module):
    def __init__(self, L, stride=0.5) -> None:
        super().__init__()
        self.L = L
        self.stride = stride

    def make_slices_gen(self, data):
        assert len(data.shape) == 2, str(data.shape) # L, F
        offset = 0
        while offset + self.L < len(data):
            yield data[offset: offset + self.L]
            offset += int(self.L * self.stride)

    def forward(self, data: Dict[str, np.ndarray]):
        return {
            k: list(self.make_slices_gen(v)) for k, v in data.items()
        }

class ConcatBatches(torch.nn.Module):
    def __init__(self, multiply=1) -> None:
        super().__init__()
        self.multiply = multiply

    def forward(self, data: Dict[str, List[List[np.ndarray]]]):
        def ravel(l):
            return [torch.tensor(a) for ll in l for a in ll]
        return {
            k: torch.stack(ravel(v)).repeat(self.multiply,1,1) for k, v in data.items()
        }

# --------------------------------- input: tensor -----------------------------------------

class InvertZero(torch.nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, data: Dict[str, torch.Tensor]):
        data = dict(data)
        y = data["y"]
        prob_msk = torch.rand((y.shape[0],)) < self.prob
        _y = y.clone()
        _y[prob_msk] = -y[prob_msk]
        data["y"] = _y
        return data

class InvertMean(torch.nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, data: Dict[str, torch.Tensor]):
        data = dict(data)
        y = data["y"].clone()
        prob_msk = torch.rand((y.shape[0],)) < self.prob
        mean = torch.mean(y, dim=1)
        y[prob_msk] = y[prob_msk] - mean[prob_msk]
        y[prob_msk] = -y[prob_msk]
        y[prob_msk] = y[prob_msk] + mean[prob_msk]
        data["y"] = y
        return data

class InvertTime(torch.nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, data: Dict[str, torch.Tensor]):
        data = dict(data)
        y = data["y"]
        prob_msk = torch.rand((y.shape[0],)) < self.prob
        _y = y.clone()
        _y[prob_msk] = y[prob_msk].flip(dims=(1,))
        data["y"] = _y
        return data

class BiasAndScale(torch.nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, data: Dict[str, torch.Tensor]):
        data = dict(data)
        y = data["y"].clone()
        r1 = torch.min(y, dim=1)
        r2 = torch.max(y, dim=1)
        prob_msk = torch.rand((y.shape[0],)) < self.prob

        y[prob_msk] = y[prob_msk] - torch.mean(y[prob_msk], dim=1)

        bias = (r2 - r1) * torch.rand((y.shape[0],)) + r1
        bias = bias / 2.0
        r1 = 0.5
        r2 = 1.5
        scale = (r2 - r1) * torch.rand((y.shape[0],)) + r1
        y[prob_msk] = y[prob_msk] * scale[prob_msk] + bias[prob_msk]
        data["y"] = y
        return data

class AddNoiseChannel(torch.nn.Module):
    def __init__(self, amplitude=0.1):
        super().__init__()
        self.amplitude = amplitude

    def forward(self, data: Dict[str, torch.Tensor]):
        data = dict(data)
        y = data["y"]
        noise = (torch.rand_like(y) - 0.5) * self.amplitude
        data["noise"] = noise
        return data
