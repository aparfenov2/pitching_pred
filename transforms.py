import torch
import torch.nn as nn

# transforms, elements of pipeline
class ToSlices(nn.Module):

    def __init__(self, L, gaps, stride=0.5) -> None:
        super().__init__()
        self.L = L
        self.gaps = gaps
        self.stride = stride

    @staticmethod
    def make_slices_gen(data, L, gaps, stride=1.0):
        assert len(data.shape) == 2 # L, F
        offset = 0
        gaps_id = 0
        while offset + L < len(data):
            if gaps_id < len(gaps) and offset + L > gaps[gaps_id]:
                offset = gaps[gaps_id]
                gaps_id += 1
            yield data[offset: offset + L]
            offset += int(L * stride)

    def forward(self, data):
        return list(self.make_slices_gen(data, self.L, self.gaps, self.stride))

class InvertZero(torch.nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, data):
        prob_msk = torch.rand((data.shape[0],)) < self.prob
        _copy = data.clone()
        _copy[prob_msk] = -data[prob_msk]
        return _copy
