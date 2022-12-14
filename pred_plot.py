import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
import argparse
import yaml

import matplotlib.pyplot as plt

from train_lit import LitPitchingPred
from model import MyModel, RNNState
from metrics import get_mse, get_mae, make_preds_gen
from dataset import MyDataModule
from visualization import draw_preds_plot

import matplotlib as mpl
mpl.use('TkAgg')


parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('weights_path')
parser.add_argument('--data-path')
parser.add_argument('--feature-id', type=int)

args = parser.parse_args()

with open(args.config, 'r') as stream:
    cfg = yaml.safe_load(stream)

# model = LitPitchingPred()
model = LitPitchingPred.load_from_checkpoint(
    args.weights_path,
    **cfg['model']
    )
model.eval()

data_path = cfg['data']['fn_test']
if args.data_path is not None:
    data_path = args.data_path
data = pd.read_csv(data_path, sep=" ")
MyDataModule.add_speed_to_data(data)

cols = cfg['data']['cols']
data = data[cols].values
divider = cfg['data']['base_freq'] / cfg['data']['freq']
data = data[::int(divider)]
HZ = cfg['data']['freq']
data = data.squeeze()
if len(data.shape) == 1:
    data = data[:,np.newaxis]
data = np.atleast_2d(data)
_input = torch.from_numpy(data).type(torch.float32)
_input = _input[None, ...]

window_size = int(40 * HZ)
future_len = int(5 * HZ)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

mse_max = 0
mae_max = 0
heat_metrics = max(window_size, int(50 * HZ))

en = make_preds_gen(_input, model.model, future_len)

gts = []
preds = []
for step, (gt, pred, preds_last) in enumerate(en):

    gts += [gt]
    if len(gts) > window_size:
        gts.pop(0)

    preds += [pred]
    if len(preds) > window_size:
        preds.pop(0)

    if step % 10 == 0:
        ax.clear()
        draw_preds_plot(ax, gts, preds, preds_last, future_len, args.feature_id, cols)

        fig.canvas.draw()
        fig.canvas.flush_events()

        mse_str = ""
        if args.feature_id is not None or len(cols) < 2:
            if args.feature_id is None:
                args.feature_id = 0
            assert len(gts) == len(preds)
            ngts = np.concatenate(gts, axis=0)
            ngts = ngts[:, args.feature_id]
            npreds = np.concatenate(preds, axis=0)
            npreds = npreds[:, args.feature_id]
            _input = torch.tensor(ngts)[None,...]
            target = torch.tensor(npreds)[None,...]
            mse_mean, _mse_max = get_mse(_input, target)
            mae_mean, _mae_max = get_mae(_input, target)
            if step > heat_metrics:
                mse_max = max(mse_max, _mse_max)
                mae_max = max(mae_max, _mae_max)
            # mse_str = f" mse_mean {mse_mean:4.4f} mse_max {mse_max:4.4f} mae_mean {mae_mean:4.4f} mae_max {mae_max:4.4f}"
            print("mse_mean", mse_mean, "mse_max", mse_max, "mae_mean", mae_mean, "mae_max", mae_max)
        print(f"step {step} time {step/HZ:4.4f}{mse_str}")
