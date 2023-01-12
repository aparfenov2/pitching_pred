import sys
import os
import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
from dataset import MyDataModule
from train_lit import LitPitchingPred, MyLightningCLI
from visualization import make_figure, make_preds_plot

# mpl.use('TkAgg')

class VisualizationUT(unittest.TestCase):

    def test1(self):
        sys.argv = sys.argv[:1]
        cli = MyLightningCLI(
            LitPitchingPred, MyDataModule,
            args=["-c", "tests/configs/mean_model.yaml"],
            run=False
            )
        model = cli.model
        dm = cli.datamodule

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
