import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
import argparse

import matplotlib.pyplot as plt

from train_lit import LitPitchingPred
from model import MyModel, RNNState
from metrics import get_mse, get_mae, make_preds_gen

import matplotlib as mpl
mpl.use('TkAgg')


parser = argparse.ArgumentParser()
parser.add_argument('weights_path')
parser.add_argument('data_path')
args = parser.parse_args()

# model = LitPitchingPred()
model = LitPitchingPred.load_from_checkpoint(args.weights_path)
model.eval()


data = pd.read_csv(args.data_path, sep=" ")
data = data["KK"].values
data = data[::4]
HZ = 50/4

self = model.model
_input = torch.from_numpy(data).type(torch.float32)
# print(_input.split(1, dim=0)[0].size())
# exit(0)

window_size = int(40 * HZ)
future_len = int(5 * HZ)
gts = []
preds = []

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

mse_max = 0
mae_max = 0

en = make_preds_gen(_input, self, future_len)

for step, (gt, pred, preds_last, delay_line) in enumerate(en):

    gts += [gt]
    if len(gts) > window_size:
        gts.pop(0)

    preds += [pred]
    if len(preds) > window_size:
        preds.pop(0)

    if step % 10 == 0:
        xgts = np.arange(0, len(gts) + len(delay_line))
        xpreds = np.arange(0, len(preds)) + future_len
        xpa = np.arange(0, len(preds_last)) + len(gts)

        ax.clear()
        ax.plot(xgts, gts + delay_line, 'b', label = 'x')
        ax.plot(xpreds, preds, 'r', label = 'pred')
        ax.plot(xpa, preds_last, 'g', label = 'pred_tmp')
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
        mse_mean, _mse_max = get_mse(_input, target)
        mae_mean, _mae_max = get_mae(_input, target)
        mse_max = max(mse_max, _mse_max)
        mae_max = max(mae_max, _mae_max)

        print(f"step {step} time {step/HZ:4.4f} mse_mean {mse_mean:4.4f} mse_max {mse_max:4.4f} mae_mean {mae_mean:4.4f} mae_max {mae_max:4.4f}")
