import torch

class InvertZero(torch.nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, data):
        prob_msk = torch.rand((data.shape[0],)) < self.prob
        _copy = data.clone()
        _copy[prob_msk] = -data[prob_msk]
        return _copy
