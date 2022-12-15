import torch
import typing
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from model import MyModel

def make_validation_plots(axes, model: MyModel, y, freq: float, current_epoch:int = None, col_names=None):

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

def draw_preds_plot(ax, gts, preds, preds_last, future_len, feature_id=None, cols=None):
    # expected: numpy arrays [[bs,1,feat] ...]

    xgts = np.arange(0, len(gts))
    xpreds = np.arange(0, len(preds))
    xpa = np.arange(0, len(preds_last)) + len(gts) - future_len

    ygts = np.concatenate(gts, axis=0)
    ypreds = np.concatenate(preds, axis=0)
    preds_last = np.concatenate(preds_last, axis=0)

    def make_label(l,i):
        return f"{l}_{cols[i]}"

    def make_color(ci, i):
        palette = [('blue','red','green'), ('cyan', 'salmon', 'violet'), ('steelblue', 'chocolate', 'hotpink')]
        return palette[i][ci]

    for i in range(len(cols)) if feature_id is None else [feature_id]:
        ax.plot(xgts, ygts[:,i], color=make_color(0,i), label = make_label('y',i))
        ax.plot(xpreds, ypreds[:,i] , color=make_color(1,i), label = make_label('pred',i))
        ax.plot(xpa, preds_last[:,i], color=make_color(2,i), label = make_label('pred_tmp',i))
    ax.axvline(len(gts) - future_len, color='b', ls='dashed')
    if feature_id is not None:
        ax.set_title(f"Переменная {cols[feature_id]}")
    ax.legend(loc='upper right')

def make_preds_plot(fig, model : MyModel, y, freq: float, feature_id: int=None, cols: typing.List[str]=None):
    # expected: tensors
    assert y.dim() == 2, str(y.dim())

    future_len = int(5 * freq)
    ax = fig.gca()
    preds = []
    gts = []
    y = y[None, ...]

    en = model.make_preds_gen(y, future_len)
    for gt, pred, preds_last in tqdm(en, total=y.shape[1], desc="cummulative preds plot"):
        gts += [gt]
        preds += [pred]

    draw_preds_plot(ax, gts, preds, preds_last, future_len, feature_id, cols)
