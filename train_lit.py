import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import yaml
import cv2
import os
import json
from typing import Any

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import cli_lightning_logo, LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from dataset import MyDataModule
from visualization import make_validation_plots, draw_to_image, make_figure, make_preds_plot, draw_preds_plot
from model import MyModel
from metrics import get_all_metrics, metrics_to_pandas, RelativeMAELoss, make_preds

class LitPitchingPred(LightningModule):
    def __init__(self,
        hidden_sz = 10,
        input_sz  = 1,
        output_sz = 1,
        num_lstm_layers = 2,
        use_skip_conns=True,
        train_future_len_s = 3,

        criterion = "MSELoss",
        # params shared with datamodule
        metrics_each = 10,
        freq = 50,
        future_len_s=3,
        cols = ['KK'],
        model_name = "MyModel"
        ):
        hparams = {
            'hidden_sz': hidden_sz,
            'input_sz' : input_sz,
            'output_sz': output_sz,
            'num_lstm_layers': num_lstm_layers,
            'use_skip_conns': use_skip_conns
            }
        super(LitPitchingPred, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = eval(model_name)(**hparams)

        self.metrics_each = metrics_each
        self.freq = freq
        self.future_len_s = future_len_s
        self.train_future_len_s = train_future_len_s
        self.cols = cols

        if criterion == 'RelativeMAELoss':
            self.criterion = RelativeMAELoss(sample_frq=self.freq)
            self.criterion_pred = RelativeMAELoss(sample_frq=self.freq, future_len_s=self.future_len_s)
        else:
            self.criterion = eval("nn." + criterion)()

    def make_preds_gen(self, _input, future_len: int):
        return self.model.make_preds_gen(_input, future_len)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        data, t = batch
        x = data[:, :-1]
        y = data[:, 1:]
        future_len = int(self.freq * self.train_future_len_s)
        out = self(x[:,:-future_len], future=future_len, extend_output_size_to_input=False)
        # assert out.shape == y[...,:out.shape[-1]].shape, str(out.shape) + " " + str(y[...,:out.shape[-1]].shape)
        loss = self.criterion(out, y)
        self.log("train_loss", loss)
        return loss

    def get_test_loss(self, batch, batch_idx):
        data, t = batch
        x = data[:, :-1]
        y = data[:, 1:]
        future_len = int(self.freq * self.future_len_s)
        # print(x.shape, y.shape) # [32,999,1]
        pred = self(x[:,:-future_len], future=future_len)

        loss = self.criterion(pred[:,:-future_len], y[:,:-future_len])
        if not isinstance(self.criterion, RelativeMAELoss):
            pred_loss = self.criterion(pred[:,-future_len:], y[:,-future_len:])
        else:
            pred_loss = self.criterion_pred(pred, y)
        return loss, pred_loss

    def validation_step(self, batch, batch_idx):
        loss, pred_loss = self.get_test_loss(batch, batch_idx)
        self.log('val_loss', loss)
        self.log('val_pred_loss', pred_loss)

    def test_step(self, *args):
        return 0

    def test_epoch_end(self, outputs) -> None:

        freq=self.freq
        col_names = self.cols

        for test_dl in self.trainer.test_dataloaders:
            ds_name = os.path.splitext(os.path.basename(test_dl.dataset.name))[0]
            print(f"--------- processing {ds_name} --------")
            # calc metrics
            metrics, gts, preds, ts = get_all_metrics(
                test_dl=test_dl,
                model=self.model,
                sample_frq=freq,
                future_len_s=self.future_len_s
                )

            for i, col in enumerate(col_names):
                for k,v in metrics.items():
                    self.log(f"{ds_name}: {col}: {k}", v[i].item(), on_step=False, on_epoch=True, logger=True)

            _json = {
                col : {k : v[i].item() for k,v in metrics.items()}
                    for i, col in enumerate(col_names)
            }
            fn = self.logger.log_dir + f'/preds_{ds_name}.json'
            with open(fn, 'w') as f:
                json.dump(_json, f, indent=4)

            # save preds to csv
            df = metrics_to_pandas(gts, preds, ts, col_names)
            fn = self.logger.log_dir + f'/preds_{ds_name}.csv'
            df.to_csv(fn, index=False, sep=' ')
            print("metrics preds saved to " + fn)


            y = self.sample_random_y(test_dl)
            fig = make_figure()
            make_preds_plot(
                fig, self.model, y,
                window_len=int(20 * freq),
                future_len=int(self.future_len_s * freq),
                cols=col_names
                )
            fig.suptitle(f"ds_name {ds_name}")
            img = draw_to_image(fig)
            img = img[...,::-1]

            fn = self.logger.log_dir + f"/test_img_pred_{ds_name}.jpg"
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            cv2.imwrite(filename=fn, img=img)

    @staticmethod
    def sample_random_y(val_dl):
        random_batch = random.randint(0, len(val_dl))
        for i,(y,t) in enumerate(val_dl):
            if i >= random_batch:
                break
        y = random.choice(y) # select random sequence from random batch
        assert y.dim() == 2 # L,F
        return y

    def training_epoch_end(self, training_step_outputs):

        val_dl = self.trainer.val_dataloaders[0]
        y = self.sample_random_y(val_dl)

        num_feats = self.model.output_sz # y.size(1)
        freq=self.freq
        col_names = self.cols

        figs = [make_figure() for i in range(num_feats)]
        axes = [fig.gca() for fig in figs]

        make_validation_plots(axes=axes, model=self, y=y, freq=freq)

        for i,fig in enumerate(figs):
            fig.suptitle(f"Эпоха {self.current_epoch} частота {freq:3.2f} переменная {col_names[i]}")
            img = draw_to_image(fig)

            fn = self.logger.log_dir + '/val_imgs/' + f"test_img_{col_names[i]}_{self.current_epoch}.jpg"
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            cv2.imwrite(filename=fn, img=img)

            img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
            self.logger.experiment.add_image(f"test_img_{col_names[i]}", img)

        # preds plot
        fig = make_figure()
        make_preds_plot(
            fig, self, y,
            window_len=int(20 * freq),
            future_len=int(self.future_len_s * freq),
            cols=col_names
            )
        fig.suptitle(f"Эпоха {self.current_epoch} частота {freq:3.2f}")
        img = draw_to_image(fig)

        fn = self.logger.log_dir + '/val_preds/' + f"test_img_pred_{self.current_epoch}.jpg"
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        cv2.imwrite(filename=fn, img=img)

        img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
        self.logger.experiment.add_image(f"test_img_pred", img)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.freq", "model.freq")
        parser.link_arguments("data.cols", "model.cols")


def cli_main():
    MyLightningCLI(
        LitPitchingPred, MyDataModule,
        seed_everything_default=1234,
        save_config_overwrite=True,
        run=True
        )

if __name__ == "__main__":
    cli_main()
