import torch
import torch.nn as nn
from .base import TimeSeries

class MeanModel(nn.Module):
    def __init__(self,
        history_len
    ):
        super().__init__()
        self.history_len = history_len
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

    def make_preds(self, ts: TimeSeries, future_len: int):
        en = self.make_preds_gen(ts, future_len)
        en = list(en)
        t = [e[0] for e in en]
        t = torch.cat(t, dim=1)
        y = [e[1] for e in en]
        y = torch.cat(y, dim=1)
        p = [e[2] for e in en]
        p = torch.cat(p, dim=1)
        pl = [e[3] for e in en]
        return TimeSeries(t, y), TimeSeries(t, p), pl
