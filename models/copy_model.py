import torch
import torch.nn as nn

from .base import ModelBase, TimeSeries

class CopyModel(ModelBase):
    """
    Evaluation only reference model
    """
    def __init__(self,
        offset_s,
        freq,
        future_len_s,
        **kwargs
        ):
        super().__init__()
        assert -future_len_s <= offset_s < 0, str(offset_s)
        self.offset = int(offset_s * freq)
        self.stub_ff = nn.Linear(1,1)

    def make_preds_gen(self, ts : TimeSeries, future_len: int):
        y,t = ts.y, ts.t
        assert y.dim() == 3, str(y.shape)
        assert t.dim() == 3, str(t.shape)
        for i in range(y.shape[1] - future_len):
            y_0   = y[:, i].unsqueeze(1)
            t_0   = t[:, i].unsqueeze(1)
            y_fut = y[:, i + future_len].unsqueeze(1)
            t_fut = t[:, i + future_len].unsqueeze(1)
            p_fut = y[:, i + future_len + self.offset].unsqueeze(1)

            yield t_fut, y_fut, p_fut, TimeSeries(
                torch.cat([t_0, t_fut], dim=1),
                torch.cat([y_0, p_fut], dim=1)
                )
