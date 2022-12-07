import torch
import torch.nn as nn
import torch.optim as optim
import random

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import cli_lightning_logo, LightningModule

from dataset import MyDataModule
from visualization import make_validation_plot

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_sz = 10
        self.lstm1 = nn.LSTMCell(1, self.hidden_sz)
        self.lstm2 = nn.LSTMCell(self.hidden_sz, self.hidden_sz)
        self.linear = nn.Linear(self.hidden_sz, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_sz, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_sz, dtype=torch.float32)
        h_t2 = torch.zeros(input.size(0), self.hidden_sz, dtype=torch.float32)
        c_t2 = torch.zeros(input.size(0), self.hidden_sz, dtype=torch.float32)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs



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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log('test_loss', loss)

    def training_epoch_end(self, training_step_outputs):

        test_dl = self.trainer.val_dataloaders[0]
        # test_dl = self.trainer.train_dataloader

        random_batch = random.randint(0, len(test_dl))
        for i,(_,y) in enumerate(test_dl):
            if i >= random_batch:
                break
        y = random.choice(y)
        img = make_validation_plot(self.freq, self.model, y, self.current_epoch)
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
