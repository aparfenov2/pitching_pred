import torch
import torch.nn as nn

class MeanModel(nn.Module):
    def __init__(self,
        history_len
    ):
        super().__init__()
        self.history_len = history_len
        self.stub_ff = nn.Linear(1,1)

    def forward(self, y, future:int=0, **kwargs):
        outputs = []
        for i in range(-y.shape[1] + self.history_len + future , future):
            y_slice = y[:,-self.history_len - future + i: -future + i]
            outputs += [torch.mean(y_slice, dim=1)]
        return torch.stack(outputs, dim=1)

    def make_preds_gen(self, y, future_len: int):
        assert y.dim() == 3
        preds = self.forward(y[:, :-future_len ], future=future_len)
        for i in range(preds.shape[1]):
            preds_last = [y[:, -future_len + i], preds[:,i]]
            yield y[:,future_len + i], preds[:,i], preds_last
