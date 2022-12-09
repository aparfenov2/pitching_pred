import torch.nn as nn
import torch.optim as optim
import random

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import cli_lightning_logo, LightningModule

from dataset import MyDataModule
from visualization import make_validation_plots, draw_to_image, make_figure, make_preds_plot
from model import MyModel
from metrics import get_all_metrics

class LitPitchingPred(LightningModule):
    def __init__(self, **kwargs):
        super(LitPitchingPred, self).__init__()
        self.model = MyModel(**kwargs)
        self.save_hyperparameters()
        self.criterion = nn.MSELoss()

    def set_datamodule(self, dm: MyDataModule):
        self.datamodule = dm

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss)
        return loss

    def get_test_loss(self, batch, batch_idx):
        x, y = batch
        future_len = 150
        # print(x.shape) # [32,999,1]
        pred = self.model(x[:,:-future_len,:], future=future_len)
        loss = self.criterion(pred[:,:-future_len,:], y[:,:-future_len,:])
        pred_loss = self.criterion(pred[:,-future_len:,:], y[:,-future_len:,:])
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
        for i,(_,y) in enumerate(test_dl):
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
        fig = make_figure()

        make_preds_plot(fig, self.model, y, freq)

        img = draw_to_image(fig)
        img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
        self.logger.experiment.add_image(f"test_img_pred", img)

        # calc metrics
        mse_mean, mse_max, mae_mean, mae_max = get_all_metrics(test_dl)

        # plt.savefig('predict%d.pdf'%i)
        # plt.close()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def cli_main():
    cli = LightningCLI(LitPitchingPred, MyDataModule, seed_everything_default=1234, save_config_overwrite=True, run=False)
    cli.model.set_datamodule(cli.datamodule)

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    # predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    # print(predictions[0])


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
