import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from train_lit import LitPitchingPred

import matplotlib as mpl
mpl.use('TkAgg')

# model = LitPitchingPred()
model = LitPitchingPred.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=127-step=640.ckpt")
model.eval()


data = pd.read_csv("NPN_1155_part2.dat", sep=" ")
data = data["KK"].values
data = data[::4]
HZ = 50/4

self = model.model
_input = torch.from_numpy(data).type(torch.float32)
# print(_input.split(1, dim=0)[0].size())
# exit(0)

h_t = torch.zeros(self.hidden_sz, dtype=torch.float32)
c_t = torch.zeros(self.hidden_sz, dtype=torch.float32)
h_t2 = torch.zeros(self.hidden_sz, dtype=torch.float32)
c_t2 = torch.zeros(self.hidden_sz, dtype=torch.float32)

window_size = int(40 * HZ)
future_len = int(5 * HZ)
gts = [0]
preds = [0]

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

# xgts = np.arange(0, len(gts))
# xpreds = np.arange(0, len(preds))
# lgts, = ax.plot(xgts, gts, 'b', label = 'x', linewidth = 3)
# lpreds, = ax.plot(xpreds, preds, 'r', label = 'extrapolation')

delay_line = []
mse_max = 0

for step, _input_t in enumerate(_input.split(1, dim=0)):

    delay_line += [_input_t]
    if len(delay_line) < future_len:
        continue
    input_delayed = delay_line.pop(0)

    h_t, c_t = self.lstm1(input_delayed, (h_t, c_t))
    h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
    pred = self.linear(h_t2)

    gts += [input_delayed.item()]
    if len(gts) > window_size:
        gts.pop(0)

    _h_t, _c_t = h_t, c_t
    _h_t2, _c_t2 = h_t2, c_t2

    preds_all = []
    for i in range(future_len - 1):
        _h_t, _c_t = self.lstm1(pred, (_h_t, _c_t))
        _h_t2, _c_t2 = self.lstm2(_h_t, (_h_t2, _c_t2))
        pred = self.linear(_h_t2)
        preds_all += [pred.item()]

    preds += [pred.item()]
    if len(preds) > window_size:
        preds.pop(0)

    if step % 10 == 0:
        xgts = np.arange(0, len(gts) + len(delay_line))
        xpreds = np.arange(0, len(preds)) + future_len
        xpa = np.arange(0, len(preds_all)) + len(gts)

        ax.clear()
        ax.plot(xgts, gts + delay_line, 'b', label = 'x')
        ax.plot(xpreds, preds, 'r', label = 'pred')
        ax.plot(xpa, preds_all, 'g', label = 'pred_tmp')
        ax.axvline(len(gts), color='b', ls='dashed')


        # lgts.set_data(xgts, gts)
        # lpreds.set_data(xpreds, preds)

        # ax.relim()
        # ax.autoscale_view(True,True,True)
        fig.canvas.draw()
        fig.canvas.flush_events()

        assert len(gts) == len(preds)
        _input = torch.tensor(gts)
        target = torch.tensor(preds)
        mse = F.mse_loss(_input, target, reduction='mean')
        if mse.item() > mse_max:
            mse_max = mse.item()
        print(f"step {step} time {step/HZ:4.4f} MSE: max {mse_max:4.4f} current {mse.item():4.4f}")
