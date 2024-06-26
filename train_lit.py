import torch
import random
import cv2
import os
import json
import pickle
from typing import Any

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import LightningModule

from dataset import MyDataModule
from visualization import draw_to_image, make_figure, make_preds_plot
from metrics import get_all_metrics, metrics_to_pandas
from models.base import TimeSeries
from utils import resolve_classpath

class LitPitchingPred(LightningModule):
    def __init__(self,
        criterion = "torch.nn.MSELoss",
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
        self.future_len = int(future_len_s * freq)
        self.plots_window_s = plots_window_s
        self.cols = cols
        self.criterion = resolve_classpath(criterion)()
        self.val_criterion = resolve_classpath(val_criterion)()

    def training_step(self, batch, batch_idx):
        assert self.training
        loss = self.model.get_loss(batch, criterion=self.criterion)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        assert not self.training
        loss = self.model.get_loss(batch, criterion=self.val_criterion)
        self.log("val_pred_loss", loss)

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

            # one line
            metrics_order = ["mae.max", "mae.mean"]
            print(f"one liner for ds={ds_name}")
            print("\t".join(metrics_order))
            def tensor_to_str(t):
                return str(t.item()) if t.ndim < 1 else str(t)
            print("\t".join([tensor_to_str(metrics[k].squeeze()) for k in metrics_order]))

            _json = {
                col : {k : v[i] for k,v in metrics.items()}
                    for i, col in enumerate(col_names)
            }

            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.tolist()
                    return json.JSONEncoder.default(self, obj)

            fn = self.logger.log_dir + f'/preds_{ds_name}.json'
            with open(fn, 'w') as f:
                json.dump(_json, f, indent=4, cls=NumpyEncoder)

            # save preds to csv
            df = metrics_to_pandas(gts, preds, ts, col_names)
            fn = self.logger.log_dir + f'/preds_{ds_name}.csv'
            df.to_csv(fn, index=False, sep=' ')
            print("metrics preds saved to " + fn)

            # all test set plot
            fig = make_figure()
            df = df.drop('sec', axis=1)
            df.plot(ax=fig.gca(), style='-', grid=True)
            img = draw_to_image(fig)

            fn = self.logger.log_dir + f"/test_img_pred_all_{ds_name}.jpg"
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            cv2.imwrite(filename=fn, img=img[...,::-1])
            img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
            self.logger.experiment.add_image(ds_name+"_all", img)
            fn = self.logger.log_dir + f"/test_img_pred_all_{ds_name}.pkl"
            with open(fn, 'wb') as f:
                pickle.dump(fig, f)

            # sample y plot
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
            self.logger.experiment.add_image(ds_name, img)


    @staticmethod
    def sample_random_y(val_dl):
        random_batch = random.randint(0, len(val_dl))
        for i,batch in enumerate(val_dl):
            y, t = batch["y"], batch["t"]
            if i >= random_batch:
                break
        batch_id = random.randrange(len(y))
        y = y[batch_id]
        t = t[batch_id]
        assert y.dim() == 2 # L,F
        return TimeSeries(t, y)

    def training_epoch_end(self, training_step_outputs):
        if self.trainer.current_epoch % self.metrics_each != 0:
            return
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
        parser.add_argument("--no_clearml", action='store_true', default=False)
        parser.link_arguments("data.freq", "model.freq")
        parser.link_arguments("data.cols", "model.cols")


def cli_main():
    try:
        import argparse
        from clearml import Task
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment")
        parser.add_argument("--no_clearml", action='store_true', default=False)
        args, _ = parser.parse_known_args()
        if not args.no_clearml:
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
