import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import cli_lightning_logo, LightningDataModule, LightningModule

from torch.utils.data import Dataset, DataLoader, random_split

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_sz = 51
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

        HZ = self.HZ

        test_dl = self.trainer.val_dataloaders[0]
        # test_dl = self.trainer.train_dataloader

        random_batch = random.randint(0, len(test_dl))
        for i,(_,y) in enumerate(test_dl):
            if i >= random_batch:
                break
        # y = y[0] # first series in batch
        y = random.choice(y)
        # ds = test_dl.dataset

        heat_s = 30 # heaing 6 sec
        pred_s = 5
        offset_s = 20
        window_s = 20

        heat = y[:int(heat_s * HZ)]

        pred = self.model(heat[None,...], future=len(y)-len(heat))
        pred = pred[0].detach().numpy()

        y = y[int(offset_s*HZ) : int((offset_s + window_s)*HZ)]
        pred = pred[int(offset_s*HZ) : int((offset_s + window_s)*HZ)]

        # draw few samples
        fig = mpl.figure.Figure(figsize=(6, 4), dpi=100)
        fig.tight_layout(pad=0)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.axvline((heat_s - offset_s) * HZ, color='b', ls='dashed')
        ax.axvline((heat_s + pred_s - offset_s) * HZ, color='b', ls='dashed')
        ax.plot(pred, color='r')
        ax.plot(y, color='g')
        ax.set_title(f"Epoch {self.current_epoch} HZ={HZ:3.2f}")

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = img.reshape(int(height), int(width), 3)
        img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
        # img = np.asarray(canvas.buffer_rgba())
        # img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        # self.logger.experiment.add_image(f'test_img_{self.current_epoch}', img)
        self.logger.experiment.add_image('test_img', img)

        # plt.savefig('predict%d.pdf'%i)
        # plt.close()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data[:, :-1]
        self.data_shifted = data[:, 1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data_shifted[idx]


HZ = 50/4

class MyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        L = 1000
        data = pd.read_csv("NPN_1155_part2.dat", sep=" ")
        data = data["KK"].values
        gaps = [140580, 177660, 520700]
        data = data[:gaps[0]-1]
        data = data[::4] # HZ !
        data = data[:(len(data)//L)*L]
        data = data.reshape(-1, L).astype('float32')
        dataset = MyDataset(data)
        self.mnist_train, self.mnist_test, self.mnist_val = random_split(dataset, [0.8, 0.1, 0.1])
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def cli_main():
    cli = LightningCLI(LitPitchingPred, MyDataModule, seed_everything_default=1234, save_config_overwrite=True, run=False)
    cli.model.set_datamodule(cli.datamodule)
    cli.model.HZ = HZ

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    # predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    # print(predictions[0])


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
