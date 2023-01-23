import torch
import torchcde
from .base import TimeSeries, ModelBase
import torch.nn as nn
import torch.nn.functional as F

### Neural CDE ###
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, n_layers):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.hidden_hidden_channels = hidden_hidden_channels

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(
            torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                        for _ in range(n_layers - 1)
                        )
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)

    def forward(self, t, z):
        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, n_layers, hidden_hidden_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels, hidden_hidden_channels, n_layers)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.grid_points,
                              method='rk4')

        z_T = z_T[:,-1,:]
        pred_y = self.readout(z_T)
        return pred_y

class NCDEModel(ModelBase):
    def __init__(
        self,
        ncde: NeuralCDE,
        freq,
        history_len_s=50,
        ):
        super().__init__()
        self.ncde = ncde
        self.history_len = int(history_len_s * freq)

    def forward(self, src: TimeSeries):
        assert src.t.dim() == 3 and src.t.shape[-1] == 1
        assert src.y.dim() == 3 and src.y.shape[-1] == 1

        ts_src = torch.cat([src.t, src.y], dim=2)  # include time as a channel
        src_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(ts_src)
        return self.ncde.forward(src_coeffs)

    def training_step(self, batch, lit):
        y, t = batch
        y_hist = y[:,-self.history_len:-lit.future_len]
        t_hist = t[:,-self.history_len:-lit.future_len]
        y_fut  = y[:,-1]
        y_pred = self.forward(TimeSeries(t_hist, y_hist))
        return lit.criterion(y_pred, y_fut)

    def validation_step(self, batch, lit):
        y, t = batch
        y_hist = y[:,-self.history_len:-lit.future_len]
        t_hist = t[:,-self.history_len:-lit.future_len]
        y_fut  = y[:,-1]
        y_pred = self.forward(TimeSeries(t_hist, y_hist))
        return {
            'val_pred_loss': lit.val_criterion(y_fut, y_pred)
        }

    def make_preds_gen(self, ts : TimeSeries, future_len: int):
        y,t = ts.y, ts.t
        assert y.dim() == 3, str(y.shape)
        assert t.dim() == 3, str(t.shape)

        for i in range(y.shape[1] - self.history_len - future_len):
            y_0   = y[:, i + self.history_len].unsqueeze(1)
            t_0   = t[:, i + self.history_len].unsqueeze(1)
            t_hist   = t[:, i: i + self.history_len]
            y_hist   = y[:, i: i + self.history_len]
            y_fut = y[:, i + self.history_len + future_len].unsqueeze(1)
            t_fut = t[:, i + self.history_len + future_len].unsqueeze(1)

            p_fut = self.forward(TimeSeries(t_hist, y_hist)).unsqueeze(1)

            yield t_fut, y_fut, p_fut, TimeSeries(
                torch.cat([t_0, t_fut], dim=1),
                torch.cat([y_0, p_fut], dim=1)
                )


def make_model(
    input_channels,
    hidden_channels,
    output_channels,
    n_layers,
    hidden_hidden_channels,
    history_len_s,
    future_len_s,
    freq
    ):
    ncde = NeuralCDE(
        input_channels,
        hidden_channels,
        output_channels,
        n_layers,
        hidden_hidden_channels,
    )
    model = NCDEModel(
        ncde,
        history_len_s,
        freq
        )
    return model
