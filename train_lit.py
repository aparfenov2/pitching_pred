import torch.nn as nn
import torch.optim as optim
import random

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import cli_lightning_logo, LightningModule
from pytorch_lightning.trainer import Trainer

from dataset import MyDataModule
from visualization import make_validation_plots, draw_to_image, make_figure, make_preds_plot
from model import MyModel
from metrics import get_all_metrics, metrics_to_pandas

class LitPitchingPred(LightningModule):
    def __init__(self, **kwargs):
        super(LitPitchingPred, self).__init__()
        self.model = MyModel(**kwargs)
        self.save_hyperparameters()
        self.criterion = nn.MSELoss()

    def set_datamodule(self, dm: MyDataModule):
        self.datamodule = dm

    def training_step(self, batch, batch_idx):
        data, t = batch
        x = data[:, :-1]
        y = data[:, 1:]
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss)
        return loss

    def get_test_loss(self, batch, batch_idx):
        data, t = batch
        x = data[:, :-1]
        y = data[:, 1:]
        future_len = int(self.datamodule.freq * self.datamodule.future_len_s)
        # print(x.shape, y.shape) # [32,999,1]
        pred = self.model(x[:,:-future_len], future=future_len)
        loss = self.criterion(pred[:,:-future_len], y[:,:-future_len])
        pred_loss = self.criterion(pred[:,-future_len:], y[:,-future_len:])
        return loss, pred_loss

    def validation_step(self, batch, batch_idx):
        loss, pred_loss = self.get_test_loss(batch, batch_idx)
        self.log('val_loss', loss)
        self.log('val_pred_loss', pred_loss)

    def test_step(self, batch, batch_idx):
        loss, pred_loss = self.get_test_loss(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_pred_loss', pred_loss)

    def training_epoch_end(self, training_step_outputs):

        test_dl = self.trainer.val_dataloaders[0]

        random_batch = random.randint(0, len(test_dl))
        for i,(y,t) in enumerate(test_dl):
            if i >= random_batch:
                break
        y = random.choice(y) # select random sequence from random batch
        assert y.dim() == 2 # L,F

        num_feats = y.size(1)
        freq=self.datamodule.freq
        col_names = self.datamodule.cols

        figs = [make_figure() for i in range(num_feats)]
        axes = [fig.gca() for fig in figs]

        make_validation_plots(axes=axes, model=self.model, y=y, freq=freq)

        for i,fig in enumerate(figs):
            fig.suptitle(f"Эпоха {self.current_epoch} частота {freq:3.2f} переменная {col_names[i]}")
            img = draw_to_image(fig)
            img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
            self.logger.experiment.add_image(f"test_img_{col_names[i]}", img)

        # preds plot
        if self.current_epoch % self.trainer.metrics_each == 0:
            fig = make_figure()
            make_preds_plot(fig, self.model, y, freq, cols=col_names)
            fig.suptitle(f"Эпоха {self.current_epoch} частота {freq:3.2f}")
            img = draw_to_image(fig)
            img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
            self.logger.experiment.add_image(f"test_img_pred", img)

            # calc metrics
            mae_mean, mae_max, gts, preds, ts = get_all_metrics(
                self.datamodule.test_dataloader(),
                model=self.model,
                future_len=int(self.datamodule.future_len_s * freq)
                )
            for i, col in enumerate(col_names):
                self.log('mae_mean_'+col, mae_mean[i].item(), on_step=False, on_epoch=True)
                self.log('mae_max_'+col, mae_max[i].item(), on_step=False, on_epoch=True)

            # save preds to csv
            df = metrics_to_pandas(gts, preds, ts, col_names)
            fn = self.logger.log_dir + '/preds_' + str(self.current_epoch) + '.csv'
            df.to_csv(fn, index=False)
            print("metrics preds saved to " + fn)

        # plt.savefig('predict%d.pdf'%i)
        # plt.close()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union
from datetime import timedelta
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback
from pathlib import Path
from lightning_lite.utilities.types import _PATH
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.trainer.connectors.accelerator_connector import _LITERAL_WARN
from pytorch_lightning.plugins import PLUGIN_INPUT

class MyTrainer(Trainer):
    def __init__(self,
        logger: Union[Logger, Iterable[Logger], bool] = True,
        enable_checkpointing: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[_PATH] = None,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        num_nodes: int = 1,
        num_processes: Optional[int] = None,
        devices: Optional[Union[List[int], str, int]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        ipus: Optional[int] = None,
        enable_progress_bar: bool = True,
        overfit_batches: Union[int, float] = 0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: Optional[int] = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1, min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        val_check_interval: Optional[Union[int, float]] = None,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, Strategy]] = None,
        sync_batchnorm: bool = False, precision: Union[int, str] = 32,
        enable_model_summary: bool = True, num_sanity_val_steps: int = 2,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[Profiler, str]] = None,
        benchmark: Optional[bool] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        reload_dataloaders_every_n_epochs: int = 0,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        detect_anomaly: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        amp_backend: str = "native", amp_level: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        inference_mode: bool = True,

        metrics_each: int = 10,
        ) -> None:

        super().__init__(
            logger, enable_checkpointing, callbacks, default_root_dir, gradient_clip_val, gradient_clip_algorithm,
            num_nodes, num_processes, devices, gpus, auto_select_gpus, tpu_cores, ipus, enable_progress_bar,
            overfit_batches, track_grad_norm, check_val_every_n_epoch, fast_dev_run, accumulate_grad_batches,
            max_epochs, min_epochs, max_steps, min_steps, max_time, limit_train_batches, limit_val_batches,
            limit_test_batches, limit_predict_batches, val_check_interval, log_every_n_steps, accelerator, strategy,
            sync_batchnorm, precision, enable_model_summary, num_sanity_val_steps, resume_from_checkpoint, profiler, benchmark,
            deterministic, reload_dataloaders_every_n_epochs, auto_lr_find, replace_sampler_ddp, detect_anomaly,
            auto_scale_batch_size, plugins, amp_backend, amp_level, move_metrics_to_cpu, multiple_trainloader_mode, inference_mode
            )
        self.metrics_each = metrics_each

def cli_main():
    cli = LightningCLI(
        LitPitchingPred, MyDataModule,
        trainer_class=MyTrainer,
        seed_everything_default=1234, save_config_overwrite=True, run=False
        )
    cli.model.set_datamodule(cli.datamodule)

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    # predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    # print(predictions[0])


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
