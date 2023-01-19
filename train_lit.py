import torch
import random
import cv2
import os
import json
from typing import Any

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import cli_lightning_logo, LightningModule

from dataset import MyDataModule
from visualization import make_validation_plots, draw_to_image, make_figure, make_preds_plot, draw_preds_plot
from metrics import get_all_metrics, metrics_to_pandas, RelativeMAELoss, make_preds
from models.base import TimeSeries
from utils import resolve_classpath

class LitPitchingPred(LightningModule):
    def __init__(self,
        criterion = "MSELoss",
        val_criterion = "torch.nn.L1Loss",
        # params shared with datamodule
        metrics_each = 10,
        freq = 50,
        future_len_s=3,
        plots_window_s=20,
        cols = ['KK'],
        model_class_path = "MyModel",
        model_init_args = {}
        ):
        super(LitPitchingPred, self).__init__()
        model_init_args.update({
            "freq" : freq,
            "future_len_s": future_len_s
            })
        self.save_hyperparameters(model_init_args)
        self.model = resolve_classpath(model_class_path)(**model_init_args)

        self.metrics_each = metrics_each
        self.freq = freq
        self.future_len_s = future_len_s
        self.plots_window_s = plots_window_s
        self.cols = cols
        self.criterion = resolve_classpath(criterion)()
        self.val_criterion = resolve_classpath(val_criterion)()

    def training_step(self, batch, batch_idx):
        loss = self.model.training_step(
            batch=batch,
            lit=self
            )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        losses = self.model.validation_step(
            batch=batch,
            lit=self
            )
        for k,v in losses.items():
            self.log(k, v)

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
            # one line
            metrics_order = ["rel_mae.max",	"rel_mae.mean",	"mae.max", "mae.mean", "mse.max", "mse.mean"]
            print(f"one liner for ds={ds_name}")
            print("\t".join(metrics_order))
            print("\t".join([str(metrics[k].item()) for k in metrics_order]))

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
                fig, self.model, ts=y,
                future_len_s=self.future_len_s,
                window_len_s=self.plots_window_s,
                freq=freq,
                cols=col_names
                )
            fig.suptitle(ds_name)
            img = draw_to_image(fig)

            fn = self.logger.log_dir + f"/test_img_pred_{ds_name}.jpg"
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            cv2.imwrite(filename=fn, img=img[...,::-1])

            img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
            self.logger.experiment.add_image(ds_name", img)


    @staticmethod
    def sample_random_y(val_dl):
        random_batch = random.randint(0, len(val_dl))
        for i,(y,t) in enumerate(val_dl):
            if i >= random_batch:
                break
        batch_id = random.randrange(len(y))
        y = y[batch_id]
        t = t[batch_id]
        assert y.dim() == 2 # L,F
        return TimeSeries(t, y)

    def training_epoch_end(self, training_step_outputs):
        with torch.no_grad():
            val_dl = self.trainer.val_dataloaders[0]
            y: TimeSeries = self.sample_random_y(val_dl)

            # preds plot
            fig = make_figure()
            make_preds_plot(
                fig, self.model, ts=y,
                future_len_s=self.future_len_s,
                window_len_s=self.plots_window_s,
                freq=self.freq,
                cols=self.cols
                )
            fig.suptitle(f"Эпоха {self.current_epoch} частота {self.freq:3.2f}")
            img = draw_to_image(fig)

            fn = self.logger.log_dir + '/val_preds/' + f"test_img_pred_{self.current_epoch}.jpg"
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            cv2.imwrite(filename=fn, img=img[...,::-1])

            img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
            self.logger.experiment.add_image(f"test_img_pred", img)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--experiment")
        parser.link_arguments("data.freq", "model.freq")
        parser.link_arguments("data.cols", "model.cols")


def cli_main():
    try:
        import argparse
        from clearml import Task
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", required=True)
        args, _ = parser.parse_known_args()
        Task.init(project_name="pitching", task_name=args.experiment)
    except ImportError:
        print("clearml not found")

    MyLightningCLI(
        LitPitchingPred, MyDataModule,
        seed_everything_default=1234,
        save_config_overwrite=True,
        run=True
        )

if __name__ == "__main__":
    cli_main()
