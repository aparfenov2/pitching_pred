import torch
from tqdm import tqdm
from torch.nn import functional as F
from model import MyModel, RNNState
import pandas as pd

def make_preds(y,t, model : MyModel, future_len, batch_n=None, batch_total=None):
    # expected: tensors

    en = model.make_preds_gen(y, future_len)
    batch_n_str = ""
    if batch_n is not None:
        if batch_total is None:
            batch_n_str = " (batch " + str(batch_n + 1) + ")"
        else:
            batch_n_str = " (batch " + str(batch_n + 1) + " of " + str(batch_total) +")"

    en = tqdm(en, total=y.shape[1], desc="calculate test metrics" + batch_n_str + " bsz " + str(y.shape[0]))
    gt_preds = list((gt, pred) for gt, pred,_ in en)
    gts = [gt for gt, pred in gt_preds]
    preds = [pred for gt, pred in gt_preds]
    ts = t[:, future_len-1:].split(1, dim=1)
    assert len(gts) == len(preds) == len(ts), f"{len(gts)} == {len(preds)} == {len(ts)}"
    return gts, preds, ts

def get_mse(_input, target):
    mse = F.mse_loss(_input, target, reduction='none')
    return mse.mean(axis=1), mse.max(axis=1).values

def get_mae(_input, target):
    mae = F.l1_loss(_input, target, reduction='none')
    return mae.mean(axis=1), mae.max(axis=1).values

def get_all_metrics(test_dl, model, future_len, skip_len=100):
    mae_means = []
    mae_maxes = []
    gts = []
    preds = []
    ts = []
    for i,(y,t) in enumerate(test_dl):
        _gts, _preds, _ts = make_preds(y,t, model, future_len, batch_n=i, batch_total=len(test_dl))

        tts = torch.stack(_ts, axis=1)[:,skip_len:]
        tgts = torch.stack(_gts, axis=1)[:,skip_len:]
        tpreds = torch.stack(_preds, axis=1)[:,skip_len:]

        ts += [_t.squeeze(dim=0) for _t in tts.split(1, dim=0)]
        gts += [_t.squeeze(dim=0) for _t in tgts.split(1, dim=0)]
        preds += [_t.squeeze(dim=0) for _t in tpreds.split(1, dim=0)]

        mae_mean, mae_max = get_mae(tgts, tpreds) # 4, 2
        mae_means += [mae_mean]
        mae_maxes += [mae_max]
        # if i>0: break

    mae_means = torch.cat(mae_means)
    mae_maxes = torch.cat(mae_maxes)
    assert len(gts) == len(preds) == len(ts), f"{len(gts)} == {len(preds)} == {len(ts)}"
    return (
        torch.mean(mae_means, axis=0),
        torch.max(mae_maxes, axis=0).values,
        gts, preds, ts,
        )

def metrics_to_pandas(gts, preds, ts, cols):
    # expects: lists of tensors
    tts = torch.cat(ts, axis=0).detach().numpy()
    tgts = torch.cat(gts, axis=0).detach().numpy()
    tpreds = torch.cat(preds, axis=0).detach().numpy()
    print("metrics_to_pandas: shapes ", tgts.shape, tpreds.shape, tts.shape)

    df = pd.DataFrame()
    df["sec"] = tts[:,0]
    for i, col in enumerate(cols):
        df[col] = tgts[:,i]
        df[col + '_pred'] = tpreds[:,i]
    return df
