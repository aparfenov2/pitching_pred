import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from dataset import MyDataModule
from train_lit import LitPitchingPred, MyLightningCLI
from visualization import make_figure, make_preds_plot, live_preds_plot

mpl.use('TkAgg')

def do_snapshot():
    y = LitPitchingPred.sample_random_y(dl)
    fig, ax = plt.subplots()
    make_preds_plot(
        fig, model, ts=y,
        window_len_s=model.plots_window_s,
        future_len_s=model.future_len_s,
        freq=model.freq,
        cols=dm.cols
        )
    plt.show()

def do_live():
    plt.ion()
    fig, ax = plt.subplots()
    live_preds_plot(
        fig, ax,
        model, dl=dl,
        window_len_s=model.plots_window_s,
        future_len_s=model.future_len_s,
        freq=model.freq,
        cols=dm.cols
        )

class CLI(MyLightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser=parser)
        parser.add_argument("--ckpt_path")
        parser.add_argument("--live", action='store_true')

cli = CLI(
    LitPitchingPred, MyDataModule,
    run=False
    )
model = cli.model
if cli.config.ckpt_path is not None:
    model = LitPitchingPred.load_from_checkpoint(cli.config.ckpt_path, **cli.config.model)
dm = cli.datamodule

dl = dm.test_dataloader()[0]

model.eval()
with torch.no_grad():
    if not cli.config.live:
        do_snapshot()
    else:
        do_live()
