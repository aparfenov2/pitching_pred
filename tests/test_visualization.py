import sys
import os
import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
from dataset import MyDataModule
from train_lit import LitPitchingPred, MyLightningCLI
from visualization import make_figure, make_preds_plot, make_validation_plots

mpl.use('TkAgg')

class VisualizationUT(unittest.TestCase):

    def setup(self):
        sys.argv = sys.argv[:1]
        cli = MyLightningCLI(
            LitPitchingPred, MyDataModule,
            args=["-c", "tests/configs/mean_model.yaml"],
            run=False
            )
        model = cli.model
        model.eval()
        dm = cli.datamodule
        return model, dm

    def test_preds_plot(self):
        model, dm = self.setup()

        dl = dm.test_dataloader()[0]
        y = LitPitchingPred.sample_random_y(dl)

        fig, ax = plt.subplots()
        make_preds_plot(
            fig, model, ts=y,
            future_len_s=3,
            freq=dm.freq,
            cols=dm.cols
            )
        # plt.show()

    def test_val_plot(self):
        model, dm = self.setup()
        num_feats = 1
        fig, ax = plt.subplots()
        axes = [ax]
        dl = dm.test_dataloader()[0]
        y = LitPitchingPred.sample_random_y(dl)
        make_validation_plots(axes=axes, model=model, y=y.y, freq=dm.freq)
        # plt.show()
