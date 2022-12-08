import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from model import MyModel
from metrics import make_preds_gen

def make_validation_plot(fig, model: MyModel, y, freq: float, current_epoch:int):
    heat_s = 30 # heaing 6 sec
    pred_s = 5
    offset_s = 20
    window_s = 20

    heat = y[:int(heat_s * freq)]

    pred = model(heat[None,...], future=len(y)-len(heat))
    pred = pred[0].detach().numpy()

    y = y[int(offset_s*freq) : int((offset_s + window_s)*freq)]
    pred = pred[int(offset_s*freq) : int((offset_s + window_s)*freq)]

    ax = fig.gca()
    ax.axvline((heat_s - offset_s) * freq, color='b', ls='dashed')
    ax.axvline((heat_s + pred_s - offset_s) * freq, color='b', ls='dashed')
    ax.plot(pred, color='r')
    ax.plot(y, color='g')
    ax.set_title(f"Epoch {current_epoch} freq={freq:3.2f}")


def draw_to_image(func):
    # draw few samples
    fig = mpl.figure.Figure(figsize=(6, 4), dpi=100)
    fig.tight_layout(pad=0)
    canvas = FigureCanvas(fig)

    func(fig)

    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = img.reshape(int(height), int(width), 3)
    return img


def make_preds_plot(fig, model : MyModel, y, freq: float):
    future_len = int(5 * freq)
    ax = fig.add_subplot(111)
    preds = []
    gts = []
    for gt, pred, preds_last, delay_line in make_preds_gen(y, model, future_len):
        gts += [gt]
        preds += [pred]

    xgts = np.arange(0, len(gts) + len(delay_line))
    xpreds = np.arange(0, len(preds)) + future_len
    xpa = np.arange(0, len(preds_last)) + len(gts)

    ax.clear()
    ax.plot(xgts, gts + delay_line, 'b', label = 'x')
    ax.plot(xpreds, preds, 'r', label = 'pred')
    ax.plot(xpa, preds_last, 'g', label = 'pred_tmp')
    ax.axvline(len(gts), color='b', ls='dashed')
