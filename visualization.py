import torch
import typing
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from models.base import TimeSeries
from utils import evaluating

def make_validation_plots(axes, model, y, freq: float, current_epoch:int = None, col_names=None):

    assert y.dim() == 2, str(y.dim())

    heat_s = 30 # heaing 6 sec
    pred_s = 5
    offset_s = 20
    window_s = 20

    heat = y[:int(max(heat_s, model.model.min_y_length_s) * freq)]
    assert len(heat) < len(y), f"{len(heat)} < {len(y)}"

    _heat = heat[None,...] # add batch dimension
    # _heat = torch.stack((heat, heat)) # min batch size is 2. Why ? Who knows ...

    pred = model(_heat, future=len(y)-len(heat))
    pred = pred[0].detach().numpy()

    y = y[int(offset_s*freq) : int((offset_s + window_s)*freq)]
    pred = pred[int(offset_s*freq) : int((offset_s + window_s)*freq)]

    for i, ax in enumerate(axes):
        # ax = fig.add_subplot(1, num_features, i + 1)
        ax.axvline((heat_s - offset_s) * freq, color='b', ls='dashed')
        ax.axvline((heat_s + pred_s - offset_s) * freq, color='b', ls='dashed')
        ax.plot(pred[:,i], color='g')
        ax.plot(y[:,i], color='b')
        if current_epoch is not None and col_names is not None:
            ax.set_title(f"Эпоха {current_epoch} частота {freq:3.2f} переменная {col_names[i]}")

def make_figure() -> mpl.figure.Figure:
    fig = mpl.figure.Figure(figsize=(12, 5), dpi=100)
    fig.tight_layout(pad=0)
    return fig

def draw_to_image(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = img.reshape(int(height), int(width), 3)
    return img

def draw_preds_plot(ax, gts: TimeSeries, preds: TimeSeries, preds_last: TimeSeries, future_len_s: float, feature_id=None, cols=None, lo=None, hi=None):

    # expected: TimeSeries of Tensors dim 3 [[bs,1,feat] ...]
    assert gts.y.dim() == 3, str(gts.y.shape)
    assert gts.t.dim() == 3, str(gts.t.shape)
    assert preds.y.dim() == 3, str(preds.y.shape)
    assert preds.t.dim() == 3, str(preds.t.shape)
    assert preds_last.y.dim() == 3, str(preds_last.y.shape)
    assert preds_last.t.dim() == 3, str(preds_last.t.shape)

    # remove batch dim
    xgts = gts.t.reshape(-1, gts.t.shape[-1])
    xpreds = preds.t.reshape(-1, preds.t.shape[-1])

    ygts = gts.y.reshape(-1, gts.y.shape[-1])
    ypreds = preds.y.reshape(-1, preds.y.shape[-1])

    assert xgts.dim() == 2, str(xgts.dim())

    def make_label(l,i):
        return f"{l}_{cols[i]}"

    def make_color(ci, i):
        palette = [('blue','red','green'), ('cyan', 'salmon', 'violet'), ('steelblue', 'chocolate', 'hotpink')]
        return palette[i][ci]

    def make_style(l, i):
        return {
            'pred': '--'
        }.get(l, '-')

    for i in range(len(cols)) if feature_id is None else [feature_id]:
        ax.plot(xgts, ygts[:,i], color=make_color(0,i), label = make_label('y',i), linestyle=make_style('y',i))

    for i in range(ypreds.shape[-1]) if feature_id is None else [feature_id]:
        ax.plot(xpreds, ypreds[:,i] , color=make_color(1,i), label = make_label('pred',i), linestyle=make_style('pred',i))
        for bi in range(preds_last.y.shape[0]):
            ax.plot(preds_last.t[bi], preds_last.y[bi,:,i], color=make_color(2,i), label = make_label('pred_tmp',i))

    ax.axvline(torch.max(xgts) - future_len_s, color='b', ls='dashed')

    x1,x2 = ax.get_xlim()
    y1,y2 = ax.get_ylim()
    if lo is not None:
        y1 = lo
    if hi is not None:
        y2 = hi
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)

    if feature_id is not None:
        ax.set_title(f"Переменная {cols[feature_id]}")
    ax.legend(loc='upper right')

def make_preds_plot(
    fig, model,
    ts: TimeSeries,
    future_len_s: float,
    freq: float,
    ax=None,
    window_len_s: float=20,
    feature_id: int=None, cols: typing.List[str]=None,
    ):
    # expected: tensors
    y, t = ts.y, ts.t
    assert y.dim() == 2, str(y.dim())
    assert t.dim() == 2, str(t.dim())

    window_len = int(window_len_s * freq)
    future_len = int(future_len_s * freq)
    if ax is None:
        ax = fig.gca()
    preds = []
    pts = []
    ts_pred = TimeSeries(t.unsqueeze(0), y.unsqueeze(0))

    with evaluating(model), torch.no_grad():
        en = model.make_preds_gen(ts_pred, future_len)

        for e in tqdm(en, total=ts_pred.y.shape[1], desc="cummulative preds plot"):
            pts += [e[0]]
            preds += [e[2]]
            preds_last = e[3]

    pts = torch.cat(pts[-window_len:], dim=1)
    preds = torch.cat(preds[-window_len:], dim=1)
    yw = y[-window_len:].unsqueeze(0)
    tw = t[-window_len:].unsqueeze(0)

    draw_preds_plot(ax, TimeSeries(tw, yw), TimeSeries(pts, preds), preds_last, future_len_s, feature_id, cols)

def live_preds_plot(
    fig, ax,
    model,
    dl,
    future_len_s: float,
    freq: float,
    window_len_s: float=20,
    feature_id: int=None, cols: typing.List[str]=None,
    draw_step_size: int = 10,
    lo=None,
    hi=None
    ):

    window_len = int(window_len_s * freq)
    future_len = int(future_len_s * freq)

    preds = []
    gts = []
    ts = []

    for batch in dl:
        y, t = batch["y"], batch["t"]
        # expected: tensors
        assert y.dim() == 3, str(y.dim())
        assert t.dim() == 3, str(t.dim())

        with evaluating(model), torch.no_grad():
            en = model.make_preds_gen(TimeSeries(t, y), future_len)

            for step, e in enumerate(en):
                ts += [e[0]]
                gts += [e[1]]
                preds += [e[2]]
                preds_last = e[3]

                while len(ts) > window_len:
                    ts.pop(0)
                while len(gts) > window_len:
                    gts.pop(0)
                while len(preds) > window_len:
                    preds.pop(0)

                if step % draw_step_size == 0:
                    ax.clear()
                    tts = torch.cat(ts, dim=1)
                    tgts = torch.cat(gts, dim=1)
                    tpreds = torch.cat(preds, dim=1)

                    draw_preds_plot(ax, TimeSeries(tts, tgts), TimeSeries(tts, tpreds), preds_last, future_len_s, feature_id, cols, lo=lo, hi=hi)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
