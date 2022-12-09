import torch
from torch.nn import functional as F
from model import MyModel, RNNState

def make_preds_gen(_input, model : MyModel, future_len, return_numpy=True):
    assert _input.dim() == 3, str(_input.dim())
    bs = _input.size(0)
    state = RNNState(bs=bs, hidden_sz=model.hidden_sz)
    delay_line = []

    def possibly_numpy(v):
        return v if not return_numpy else v.detach().numpy()

    with torch.no_grad():
        for _input_t in _input.split(1, dim=1):
            _input_t = _input_t.squeeze(dim=1)
            assert _input_t.dim() == 2, str(_input_t.dim())
            assert _input_t.size(0) == _input.size(0) # 32, 2
            delay_line += [possibly_numpy(_input_t)]
            if len(delay_line) < future_len:
                continue
            input_delayed = delay_line.pop(0)
            _input_delayed = input_delayed
            if return_numpy:
                _input_delayed = torch.tensor(input_delayed)
            pred = model.forward_one_step(_input_delayed, state)

            pred_state = state.clone().detach()
            preds = [possibly_numpy(pred)]
            for i in range(future_len - 1):
                pred = model.forward_one_step(pred, pred_state)
                preds += [possibly_numpy(pred)]

            # returns [bs,1,feats] lists
            yield input_delayed, possibly_numpy(pred), preds, delay_line

def make_preds(y, model : MyModel, future_len):
    # expected: tensors
    assert y.dim() == 2, str(y.dim())

    en = make_preds_gen(y, model, future_len, return_numpy=False)
    gt_preds = list((gt, pred) for gt, pred, _,_ in en)
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

def get_all_metrics(test_dl):
    for i,(_,y) in enumerate(test_dl):
        pass

    mse_mean, mse_max, mae_mean, mae_max = 0, 0, 0, 0
    return mse_mean, mse_max, mae_mean, mae_max