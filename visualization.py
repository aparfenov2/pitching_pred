import torch
import typing
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from models.base import TimeSeries

def make_validation_plots(axes, model, y, freq: float, current_epoch:int = None, col_names=None):

    assert y.dim() == 2, str(y.dim())

    heat_s = 30 # heaing 6 sec
    pred_s = 5
    offset_s = 20
    window_s = 20

    heat = y[:int(heat_s * freq)]

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

def make_figure():
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

def draw_preds_plot(ax, gts: TimeSeries, preds: TimeSeries, preds_last: TimeSeries, future_len_s: float, feature_id=None, cols=None):

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

    for i in range(len(cols)) if feature_id is None else [feature_id]:
        ax.plot(xgts, ygts[:,i], color=make_color(0,i), label = make_label('y',i))
        ax.plot(xpreds, ypreds[:,i] , color=make_color(1,i), label = make_label('pred',i))
        for bi in range(preds_last.y.shape[0]):
            ax.plot(preds_last.t[bi], preds_last.y[bi,:,i], color=make_color(2,i), label = make_label('pred_tmp',i))

    ax.axvline(torch.max(xgts) - future_len_s, color='b', ls='dashed')

    if feature_id is not None:
        ax.set_title(f"Переменная {cols[feature_id]}")
    ax.legend(loc='upper right')

def make_preds_plot(
    fig, model,
    ts: TimeSeries,
    future_len_s: float,
    freq: float,
    window_len_s: float=20,
    feature_id: int=None, cols: typing.List[str]=None
    ):
    # expected: tensors
    y, t = ts.y, ts.t
    assert y.dim() == 2, str(y.dim())
    assert t.dim() == 2, str(t.dim())

    window_len = int(window_len_s * freq)
    future_len = int(future_len_s * freq)

    ax = fig.gca()
    preds = []
    gts = []
    ts = []
    y = y[:window_len]
    y = y[None, ...]
    t = t[:window_len]
    t = t[None, ...]

    en = model.make_preds_gen(TimeSeries(t, y), future_len)
    for e in tqdm(en, total=y.shape[1], desc="cummulative preds plot"):
        ts += [e[0]]
        gts += [e[1]]
        preds += [e[2]]
        preds_last = e[3]

    ts = torch.cat(ts, dim=1)
    gts = torch.cat(gts, dim=1)
    preds = torch.cat(preds, dim=1)

    draw_preds_plot(ax, TimeSeries(ts, gts), TimeSeries(ts, preds), preds_last, future_len_s, feature_id, cols)

def live_preds_plot(
    fig, ax,
    model,
    dl,
    future_len_s: float,
    freq: float,
    window_len_s: float=20,
    feature_id: int=None, cols: typing.List[str]=None
    ):

    window_len = int(window_len_s * freq)
    future_len = int(future_len_s * freq)

    for y,t in dl:

        # expected: tensors
        assert y.dim() == 3, str(y.dim())
        assert t.dim() == 3, str(t.dim())

        preds = []
        gts = []
        ts = []

        en = model.make_preds_gen(TimeSeries(t, y), future_len)

        for step, e in enumerate(en):
            ts += [e[0]]
            gts += [e[1]]
            preds += [e[2]]
            preds_last = e[3]
            if len(ts) > window_len:
                ts.pop(0)
                gts.pop(0)
                preds.pop(0)

            if step % 10 == 0:
                ax.clear()
                tts = torch.cat(ts, dim=1)
                tgts = torch.cat(gts, dim=1)
                tpreds = torch.cat(preds, dim=1)

                draw_preds_plot(ax, TimeSeries(tts, tgts), TimeSeries(tts, tpreds), preds_last, future_len_s, feature_id, cols)
                fig.canvas.draw()
                fig.canvas.flush_events()
