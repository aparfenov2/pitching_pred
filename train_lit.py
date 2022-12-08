import torch.nn as nn
import torch.optim as optim
import random

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import cli_lightning_logo, LightningModule

from dataset import MyDataModule
from visualization import make_validation_plot, draw_to_image
from model import MyModel
from functools import partial

class LitPitchingPred(LightningModule):
    def __init__(self):
        super(LitPitchingPred, self).__init__()
        self.model = MyModel()
        self.criterion = nn.MSELoss()

    def set_datamodule(self, dm):
        self.datamodule = dm
        self.freq = self.datamodule.freq

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss)
        return loss

    def get_test_loss(self, batch, batch_idx):
        x, y = batch
        future_len = 150
        # print(x.shape) [32,999]
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
        # test_dl = self.trainer.train_dataloader

        random_batch = random.randint(0, len(test_dl))
        for i,(_,y) in enumerate(test_dl):
            if i >= random_batch:
                break
        y = random.choice(y)
        func = partial(make_validation_plot, model=self.model, y=y, freq=self.freq, current_epoch=self.current_epoch)
        img = draw_to_image(func)
        img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
        self.logger.experiment.add_image('test_img', img)

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
