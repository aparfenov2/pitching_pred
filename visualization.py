import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def make_validation_plot(freq, model, y, current_epoch):
    heat_s = 30 # heaing 6 sec
    pred_s = 5
    offset_s = 20
    window_s = 20

    heat = y[:int(heat_s * freq)]

    pred = model(heat[None,...], future=len(y)-len(heat))
    pred = pred[0].detach().numpy()

    y = y[int(offset_s*freq) : int((offset_s + window_s)*freq)]
    pred = pred[int(offset_s*freq) : int((offset_s + window_s)*freq)]

    # draw few samples
    fig = mpl.figure.Figure(figsize=(6, 4), dpi=100)
    fig.tight_layout(pad=0)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.axvline((heat_s - offset_s) * freq, color='b', ls='dashed')
    ax.axvline((heat_s + pred_s - offset_s) * freq, color='b', ls='dashed')
    ax.plot(pred, color='r')
    ax.plot(y, color='g')
    ax.set_title(f"Epoch {current_epoch} freq={freq:3.2f}")

    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = img.reshape(int(height), int(width), 3)
