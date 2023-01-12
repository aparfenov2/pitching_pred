import torch
import torch.nn as nn
from .base import TimeSeries, ModelBase

class MeanModel(ModelBase):
    def __init__(self,
        history_len=None,
        history_len_s=None,
        freq=None,
        **kvargs
    ):
        super().__init__()
        if history_len is not None:
            self.history_len = history_len
        else:
            self.history_len = int(freq * history_len_s)
        self.stub_ff = nn.Linear(1,1)

    def forward(self, y, future:int=0, **kwargs):
        outputs = []
        for i in range(y.shape[1] - self.history_len):
            outputs += [torch.mean(y[:,i: i + self.history_len], dim=1)]
        return torch.stack(outputs, dim=1)

    def make_preds_gen(self, ts : TimeSeries, future_len: int):
        y,t = ts.y, ts.t
        assert y.dim() == 3, str(y.shape)
        assert t.dim() == 3, str(t.shape)

        preds = self.forward(y)

        for i in range(y.shape[1] - self.history_len - future_len):
            y_0   = y[:, i + self.history_len].unsqueeze(1)
            t_0   = t[:, i + self.history_len].unsqueeze(1)
            y_fut = y[:, i + self.history_len + future_len].unsqueeze(1)
            t_fut = t[:, i + self.history_len + future_len].unsqueeze(1)
            p_fut = preds[:,i].unsqueeze(1)
            yield t_fut, y_fut, p_fut, TimeSeries(
                torch.cat([t_0, t_fut], dim=1),
                torch.cat([y_0, p_fut], dim=1)
                )
