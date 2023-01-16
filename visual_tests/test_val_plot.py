import sys
import os
import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
from dataset import MyDataModule
from train_lit import LitPitchingPred, MyLightningCLI
from visualization import make_figure, make_preds_plot, make_validation_plots

mpl.use('TkAgg')

class CLI(MyLightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser=parser)
        parser.add_argument("--ckpt_path")

cli = CLI(
    LitPitchingPred, MyDataModule,
    run=False
    )
model = cli.model
if cli.config.ckpt_path is not None:
    model = LitPitchingPred.load_from_checkpoint(cli.config.ckpt_path, **cli.config.model)
model.eval()
dm = cli.datamodule

num_feats = 1
fig, ax = plt.subplots()
axes = [ax]
dl = dm.test_dataloader()[0]
y = LitPitchingPred.sample_random_y(dl)
make_validation_plots(axes=axes, model=model, y=y.y, freq=dm.freq)
plt.show()

