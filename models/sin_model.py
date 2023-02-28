import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import Tensor
from .base import TimeSeries, ModelBase

# hist 50 s
# 50 v, 50 v'
# output: 1, +10s

@torch.jit.script
def linspace(start: Tensor, stop: Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]

    return out

class SinModel(ModelBase):

    def __init__(self,
        freq,
        future_len_s,
        train_future_len_s,
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
        self.train_future_len_s = train_future_len_s
        self.train_future_len = int(train_future_len_s * freq)
        layers = []
        #                       y                    y'
        # input_sz = self.num_points * num_feats + (self.num_points - 1)
        input_sz = self.num_points * num_feats
        for i in range(num_layers - 1):
            layers += [nn.Linear(input_sz, hidden_sz)]
            layers += [nn.ReLU()]
            input_sz = hidden_sz
        layers += [nn.Linear(input_sz, 2)]
        self.seq = nn.Sequential(*layers)


    def forward(self, src, trg):
        assert src.shape[1] == self.history_len, f"{src.shape[1]} == {self.history_len}"
        y1s = src[:,::self.freq] # 50 vals
        inp = y1s.reshape(src.shape[0], -1)
        params = self.seq(inp)
        bias = torch.mean(src, dim=1) # [64, 2]
        amp = torch.max(src, dim=1).values - bias
        bias = bias[:,None,:]
        amp = amp[:,None,:]
        phase = params[:,0]
        phase = phase[..., None, None] # [64, 1, 1]
        frq = params[:,1]
        min_frq = 1/10
        max_frq = 1/2
        # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        frq = (((torch.tanh(frq) - (-1)) * (max_frq - min_frq)) / (1 - (-1))) + min_frq
        w = linspace(torch.zeros_like(frq), 2 * torch.pi * frq * self.train_future_len_s, trg.shape[1]).swapdims(0, 1)[...,None] # [64, 20, 1]
        ret = bias + amp * torch.sin(phase + w)
        return ret

    def get_loss(self, batch, criterion):
        y = batch["y"]
        noise = None
        if "noise" in batch:
            noise = batch["noise"]
        losses = []
        for i in range(y.shape[1] - self.history_len - self.train_future_len + 1):
            src = y[:, i: i + self.history_len]
            if noise is not None:
                src = src + noise[:, i: i + self.history_len]
            trg = y[:, i + self.history_len: i + self.history_len + self.train_future_len]
            output = self.forward(src, trg)
            output = torch.cat([output[:,:,:1], trg[:,:,1:]], dim=-1) # append other features to predictions
            loss = criterion(output, trg)
            losses += [loss]
        return torch.mean(torch.stack(losses, dim=0))

    def make_preds_gen(self, ts : TimeSeries, future_len: int):
        y,t = ts.y, ts.t
        assert y.dim() == 3, str(y.shape)
        assert t.dim() == 3, str(t.shape)
        assert y.shape[1] >= self.history_len + future_len, f"y.shape[1] {y.shape[1]} self.history_len {self.history_len} future_len {future_len}"
        assert not self.training

        for i in range(y.shape[1] - self.history_len - future_len + 1):

            src = y[:, i: i + self.history_len]
            trg = y[:, i + self.history_len: i + self.history_len + future_len]
            output = self.forward(src, trg)

            y_fut = y[:, i + self.history_len + future_len - 1].unsqueeze(1)
            t_fut = t[:, i + self.history_len + future_len - 1].unsqueeze(1)
            p_fut = output[:,-1].unsqueeze(1)

            yield t_fut, y_fut, p_fut, TimeSeries(
                t[:, i + self.history_len: i + self.history_len + future_len],
                output
                )


def test_model():
    model = SinModel(
        freq=1,
        future_len_s=3,
        num_feats=2
    )
    batch = {
        "y": torch.rand(1, 112, 2)
    }
    # preds = model.forward(batch)
    criterion = nn.L1Loss()
    loss = model.get_loss(batch, criterion)

if __name__ == '__main__':
    test_model()