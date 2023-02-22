import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .base import TimeSeries, ModelBase

# hist 50 s
# 50 v, 50 v'
# output: 1, +10s

class LinearModel(ModelBase):

    def __init__(self,
        freq,
        future_len_s,
        history_len_s=50,
        num_layers=1,
        num_feats=1,
        hidden_sz=100,
        **kwargs
    ) -> None:
        super().__init__()
        self.num_points = int(history_len_s)
        self.history_len_s = history_len_s
        self.history_len = int(history_len_s * freq)
        self.freq = freq
        self.future_len = int(freq * future_len_s)
        layers = []
        #          y                    y'
        input_sz = self.num_points * num_feats + (self.num_points - 1)
        for i in range(num_layers - 1):
            layers += [nn.Linear(input_sz, hidden_sz)]
            layers += [nn.ReLU()]
            input_sz = hidden_sz
        layers += [nn.Linear(input_sz, 1)]
        self.seq = nn.Sequential(*layers)

    @property
    def min_y_length_s(self):
        return self.history_len_s + 1

    def make_preds_gen(self, ts : TimeSeries, future_len: int):
        y,t = ts.y, ts.t
        assert y.dim() == 3, str(y.shape)
        assert t.dim() == 3, str(t.shape)

        for i in range(y.shape[1] - self.history_len - future_len + 1): # min 2 pts: 0-hist, 1-gt
            y_0   = y[:, i + self.history_len - 1].unsqueeze(1) # current pt is included in hist
            t_0   = t[:, i + self.history_len - 1].unsqueeze(1)
            y_hist   = y[:, i: i + self.history_len]
            y_fut = y[:, i + self.history_len + future_len - 1].unsqueeze(1)
            t_fut = t[:, i + self.history_len + future_len - 1].unsqueeze(1)

            p_fut = self.forward_one_step(y_hist)

            yield t_fut, y_fut, p_fut, TimeSeries(
                torch.cat([t_0, t_fut], dim=1),
                torch.cat([y_0[:,:,:1], p_fut], dim=1)
                )

    def forward_one_step(self, y:torch.Tensor):
        assert y.shape[1] == self.history_len, f"{y.shape[1]} == {self.history_len}"
        y1s = y[:,::self.freq] # 50 vals
        assert y1s.shape[1] == self.num_points, f"{y1s.shape[1]} == {self.num_points}"
        y1s_dt = y1s[:,1:] - y1s[:,:1]
        y1s_dt = y1s_dt[:,:,0].unsqueeze(-1) # only y
        y1s = y1s.reshape(y1s.shape[0], y1s.shape[1] * y1s.shape[2], 1)
        assert y1s_dt.shape[1] == self.num_points - 1, f"{y1s_dt.shape[1]} == {self.num_points} - 1"
        assert y1s.shape[-1] == y1s_dt.shape[-1] == 1, f"{y1s.shape[-1]} == {y1s_dt.shape[-1]} == 1"

        inp = torch.cat([y1s, y1s_dt], dim=1).reshape(y.shape[0], -1)
        return self.seq(inp).reshape(y.shape[0], 1, 1)

    def forward(self, y):
        preds = []
        for i in range(y.shape[1] - self.history_len + 1):
            y_hist = y[:, i: i + self.history_len]
            preds += [self.forward_one_step(y_hist)]
        return torch.cat(preds, dim=1)

    def get_loss(self, batch, criterion):
        y = batch["y"]
        src   = y[:, :-self.future_len]  # [[...................]<- future_len ->]
        y_fut = y[:, self.history_len + self.future_len - 1:]  # [<- history_len + future_len - 1 ->[......................]]
        preds = self.forward(src)
        assert preds.shape == y_fut.shape, f"{preds.shape} == {y_fut.shape}"
        return criterion(preds, y_fut)
