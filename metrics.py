import torch
from torch.nn import functional as F
from model import MyModel, RNNState

def make_preds_gen(_input, model : MyModel, future_len):
    state = RNNState(bs=0, hidden_sz=model.hidden_sz)
    delay_line = []
    for _input_t in _input.split(1, dim=0):
        delay_line += [_input_t]
        if len(delay_line) < future_len:
            continue
        input_delayed = delay_line.pop(0)
        pred = model.forward_one_step(input_delayed, state)

        pred_state = state.clone().detach()
        preds = [pred.item()]
        for i in range(future_len - 1):
            pred = model.forward_one_step(pred, pred_state)
            preds += [pred.item()]

        yield input_delayed.item(), pred.item(), preds, delay_line

def make_preds(_input, model : MyModel, future_len):
    gt_preds = list((gt, pred) for gt, pred, preds_last in make_preds_gen(_input, model, future_len))
    gts = [gt for gt, pred in gt_preds]
    preds = [pred for gt, pred in gt_preds]

    assert len(gts) == len(preds)
    _input = torch.tensor(gts)
    target = torch.tensor(preds)
    return _input, target

def get_mse(_input, target):
    mse = F.mse_loss(_input, target, reduction='none')
    return mse.mean().item(), mse.max().item()

def get_mae(_input, target):
    mae = F.l1_loss(_input, target, reduction='none')
    return mae.mean().item(), mae.max().item()
