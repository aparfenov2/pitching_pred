# https://www.kaggle.com/code/masatomurakawamm/n-beats-dnn-for-univariate-time-series-forecast/notebook

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TimeSeries, ModelBase

class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)
        self.fc_b = nn.Linear(width, width, bias=False)
        self.fc_f = nn.Linear(width, width, bias=False)
        self.g_b = nn.Linear(width, input_dim)
        self.g_f = nn.Linear(width, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        theta_b = self.fc_b(x)
        theta_f = self.fc_f(x)

        backcast = self.g_b(theta_b)
        forecast = self.g_f(theta_f)

        return backcast, forecast


class NBeatsStack(nn.Module):
    def __init__(self, n_blocks, input_dim, output_dim, width):
        super().__init__()
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = NBeatsBlock(input_dim, output_dim, width)
            self.blocks.append(block)

    def forward(self, x):
        stack_forecast = []
        for i in range(self.n_blocks):
            backcast, forecast = self.blocks[i](x)
            x = x - backcast
            stack_forecast.append(forecast)
        stack_forecast = torch.stack(stack_forecast, axis=-1)
        stack_forecast = torch.sum(stack_forecast, axis=-1)
        stack_residual = x
        return stack_residual, stack_forecast


class NBeatsModel(nn.Module):
    def __init__(self, n_blocks, n_stacks,
                 input_dim, output_dim, width):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width

        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            stack = NBeatsStack(n_blocks, input_dim, output_dim, width)
            self.stacks.append(stack)

    def forward(self, x):
        global_forecast = []
        for i in range(self.n_stacks):
            stack_residual, stack_forecast = self.stacks[i](x)
            x = stack_residual
            global_forecast.append(stack_forecast)
        global_forecast = torch.stack(global_forecast, axis=-1)
        global_forecast = torch.sum(global_forecast, axis=-1)
        return global_forecast


class NBeatsWrp(ModelBase):
    def __init__(self, model, device,
        history_len_s,
        future_len_s,
        freq
    ):
        super().__init__()
        self.device = device
        self.model = model
        self.history_len = int(history_len_s * freq)
        self.future_len = int(future_len_s * freq)

    def forward(self, x):
        x = x.squeeze(dim=-1)
        output = self.model.forward(x)
        return output.unsqueeze(dim=-1)

    def get_loss(self, batch, criterion):
        y = batch["y"]
        noise = None
        if "noise" in batch:
            noise = batch["noise"]
        losses = []
        for i in range(y.shape[1] - self.history_len - self.future_len + 1):
            src = y[:, i: i + self.history_len]
            if noise is not None:
                src = src + noise[:, i: i + self.history_len]
            trg = y[:, i + self.history_len: i + self.history_len + self.future_len]
            output = self.forward(src)
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
            # trg = y[:, i + self.history_len: i + self.history_len + future_len]
            output = self.forward(src)

            y_fut = y[:, i + self.history_len + future_len - 1].unsqueeze(1)
            t_fut = t[:, i + self.history_len + future_len - 1].unsqueeze(1)
            p_fut = output[:,-1].unsqueeze(1)

            yield t_fut, y_fut, p_fut, TimeSeries(
                t[:, i + self.history_len: i + self.history_len + future_len],
                output
                )

def make_model(
        history_len_s,
        future_len_s,
        freq,

        n_blocks = 2,
        n_stacks = 4,
        width = 32,

):

    history_len = int(history_len_s * freq)
    future_len = int(future_len_s * freq)

    input_dim = history_len
    output_dim = future_len

    model = NBeatsModel(
        n_blocks,
        n_stacks,
        input_dim,
        output_dim,
        width
    )
    device = 'cpu'
    wrp = NBeatsWrp(
        model,
        device,
        history_len_s,
        future_len_s,
        freq
        )
    return wrp
