import torch
import torch.nn as nn

class TimeSeries:
    def __init__(self, t, y) -> None:
        self.t = t
        self.y = y

class ModelBase(nn.Module):

    @property
    def min_y_length_s(self):
        raise NotImplementedError()

    def forward(self, y, future):
        # emulate behaviour of previous models
        assert future > 0
        src = y
        trg = torch.zeros((y.shape[0], future, y.shape[2]), dtype=y.dtype)
        trg[:, 0] = y[:,-1]
        pred = self.forward1(src=src, trg=trg)
        return torch.cat([src, pred], dim=1)

    def make_preds_gen(self, ts : TimeSeries, future_len: int):
        raise NotImplementedError()

    def make_preds(self, ts: TimeSeries, future_len: int):
        en = self.make_preds_gen(ts, future_len)
        en = list(en)
        t = [e[0] for e in en]
        t = torch.cat(t, dim=1)
        y = [e[1] for e in en]
        y = torch.cat(y, dim=1)
        p = [e[2] for e in en]
        p = torch.cat(p, dim=1)
        pls = [e[3] for e in en]
        return TimeSeries(t, y), TimeSeries(t, p), pls

    def base_training_step(self, batch, lit):
        future_len = int(lit.freq * lit.train_future_len_s)
        data, t = batch
        x = data[:, :-1]
        y = data[:, 1:]
        out = self(x[:,:-future_len], future=future_len, extend_output_size_to_input=False)
        # assert out.shape == y[...,:out.shape[-1]].shape, str(out.shape) + " " + str(y[...,:out.shape[-1]].shape)
        return lit.criterion(out, y)

    def training_step(self, batch, lit):
        raise NotImplementedError()

    def validation_step(self, batch, lit):
        raise NotImplementedError()

    def base_validation_step(self, batch, lit):
        data, t = batch
        x = data[:, :-1]
        y = data[:, 1:]
        future_len = int(lit.freq * lit.future_len_s)
        # print(x.shape, y.shape) # [32,999,1]
        pred = self(x[:,:-future_len], future=future_len)

        loss = lit.criterion(pred[:,:-future_len], y[:,:-future_len])
        pred_loss = lit.criterion(pred[:,-future_len:], y[:,-future_len:])
        return {
            'val_loss': loss,
            'val_pred_loss': pred_loss
        }
