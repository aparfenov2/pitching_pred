import torch
from tqdm import tqdm
from torch.nn import functional as F
import pandas as pd
import torch.nn as nn
from models import TimeSeries

def make_preds(y,t, model, future_len, batch_n=None, batch_total=None):
    # expected: tensors
    batch_n_str = ""
    if batch_n is not None:
        if batch_total is None:
            batch_n_str = " (batch " + str(batch_n + 1) + ")"
        else:
            batch_n_str = " (batch " + str(batch_n + 1) + " of " + str(batch_total) +")"
    with torch.no_grad():
        en = model.make_preds_gen(TimeSeries(t,y), future_len)
        en = tqdm(en, total=y.shape[1], desc="calculate test metrics" + batch_n_str + " bsz " + str(y.shape[0]))
        en = list(en)
    ts = [e[0] for e in en]
    gts = [e[1] for e in en]
    preds = [e[2] for e in en]
    # ts = t[:, future_len-1:].split(1, dim=1)
    # ts = ts[:len(preds)]
    assert len(gts) == len(preds) == len(ts), f"{len(gts)} == {len(preds)} == {len(ts)}"
    assert gts[0].dim() == 3 and gts[0].shape == preds[0].shape
    assert ts[0].dim() == 3 and ts[0].shape[:-1] == gts[0].shape[:-1] and ts[0].size(2) == 1, f"ts[0].shape {ts[0].shape} gts[0].shape {gts[0].shape}"
    return gts, preds, ts

def get_mse(_input, target):
    mse = F.mse_loss(_input, target, reduction='none')
    return mse.mean(axis=1), mse.max(axis=1).values

def get_mae(_input, target):
    mae = F.l1_loss(_input, target, reduction='none')
    return mae.mean(axis=1), mae.max(axis=1).values

class MaxMAELoss(nn.Module):
    def forward(self, _input, target):
        mean, max = get_mae(_input, target)
        return max.mean()

def _relative_mae_metric(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    sample_frq: float,
    window_size_s=30,
    ):
    assert y.dim() == 3, str(y.shape) # N,L,F
    assert sample_frq > 0
    window_size = int(window_size_s * sample_frq)
    if window_size > y.shape[1]:
        raise Exception(f"y.shape {y.shape} window_size {window_size} > y.shape[1] {y.shape[1]} window_size_s {window_size_s} sample_frq {sample_frq}")
    unfolded_y = y.unfold(dimension=1, size=window_size, step=window_size // 2) # 12, 15, 3, 120
    diff = y - y_hat
    unfolded_diff = diff.unfold(dimension=1, size=window_size, step=window_size // 2)
    y_ranges_min = unfolded_y.min(dim=-1).values # 4,5,2
    y_ranges_max = unfolded_y.max(dim=-1).values
    y_ranges = y_ranges_max - y_ranges_min + 1e-5
    y_ranges = y_ranges[..., None] # 4,5,2,1
    ret = unfolded_diff.abs().div(y_ranges.abs()) # 4,5,2,36
    return ret

def relative_mae_metric(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    sample_frq: float,
    window_size_s=30,
    return_average_per_feature=False,
    future_len_s: int=0
    ):
    window_size = int(window_size_s * sample_frq)
    future_len = int(future_len_s * sample_frq)
    if future_len > 0:
        if future_len > window_size:
            raise Exception(f"future_len {future_len} > window_size {window_size}")

    ret = _relative_mae_metric(y, y_hat, sample_frq, window_size_s)

    if future_len > 0:
        # apply max on only the last window
        ret = ret[:,-1:,:,-future_len:]
        if return_average_per_feature:
            return ret.max(dim=(0,1,3))
        return ret.max()

    if return_average_per_feature:
        return ret.mean(dim=(0,1,3))
    return ret.mean()

class RelativeMAELoss(nn.Module):
    def __init__(self, **kwargs,
    ) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, y, y_hat):
        return relative_mae_metric(y, y_hat, **self.kwargs)


def get_all_metrics(test_dl, model, sample_frq, future_len_s, skip_len_s=10):
    future_len = future_len_s * sample_frq
    skip_len = skip_len_s * sample_frq
    gts = []
    preds = []
    ts = []
    for i,batch in enumerate(test_dl):
        y, t = batch["y"], batch["t"]
        n_feats = y.shape[-1]
        _gts, _preds, _ts = make_preds(y,t, model, future_len, batch_n=i, batch_total=len(test_dl))
        gts += [torch.stack(_gts[skip_len:], axis=1).reshape(-1, n_feats)]
        preds += [torch.stack(_preds[skip_len:], axis=1).reshape(-1, n_feats)]
        ts += [torch.stack(_ts[skip_len:], axis=1).reshape(-1, n_feats)]

    tgts = torch.cat(gts, axis=0).unsqueeze(0) # make it (1, N*L, F)
    tpreds = torch.cat(preds, axis=0).unsqueeze(0)
    assert tgts.dim() == tpreds.dim() == 3, f"tgts.shape {tgts.shape} tpreds.shape {tpreds.shape}"

    rel_mae = _relative_mae_metric(y=tgts, y_hat=tpreds, sample_frq=sample_frq)
    mae = get_mae(tgts, tpreds)
    mse = get_mse(tgts, tpreds)
    metrics = {
        "rel_mae.mean": rel_mae.mean(dim=(0,1,3)),
        "rel_mae.max": rel_mae.max(dim=3).values.max(dim=1).values.max(dim=0).values,
        "mae.mean" : mae[0],
        "mae.max" : mae[1],
        "mse.mean" : mse[0],
        "mse.max" : mse[1],
    }

    return (
        metrics,
        gts, preds, ts,
        )

def metrics_to_pandas(gts, preds, ts, cols):
    # expects: lists of tensors
    tts = torch.cat(ts, axis=0).numpy()
    tgts = torch.cat(gts, axis=0).numpy()
    tpreds = torch.cat(preds, axis=0).numpy()
    print("metrics_to_pandas: shapes ", tgts.shape, tpreds.shape, tts.shape)

    df = pd.DataFrame()
    df["sec"] = tts[:,0]
    for i, col in enumerate(cols):
        df[col] = tgts[:,i]
        df[col + '_pred'] = tpreds[:,i]
    return df
