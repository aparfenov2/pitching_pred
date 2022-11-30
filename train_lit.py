import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import cli_lightning_logo, LightningDataModule, LightningModule

from torch.utils.data import Dataset, DataLoader, random_split

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.float32)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.float32)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.float32)

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # future = 1000
        # pred = self.model(x, future=future)
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # future = 1000
        # pred = self.model(x, future=future)
        pred = self.model(x)
        # loss = self.criterion(pred[:, :-future], y)
        loss = self.criterion(pred, y)
        self.log('test_loss', loss)
        # y = pred.detach().numpy()
        # # draw the result
        # plt.figure(figsize=(30,10))
        # plt.title('Predict future values for time MyModels\n(Dashlines are predicted values)', fontsize=30)
        # plt.xlabel('x', fontsize=20)
        # plt.ylabel('y', fontsize=20)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # def draw(yi, color):
        #     plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
        #     plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        # draw(y[0], 'r')
        # draw(y[1], 'g')
        # draw(y[2], 'b')
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

class MyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        L = 100
        data = pd.read_csv("NPN_1155_part2.dat", sep=" ")
        data = data["KK"].values
        data = data.reshape(-1, L).astype('float32')
        dataset = MyDataset(data)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        test_size, val_size = test_size // 2, test_size // 2
        self.mnist_train, self.mnist_test, self.mnist_val = random_split(dataset, [train_size, test_size, val_size])
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
    cli = LightningCLI(LitPitchingPred, MyDataModule, seed_everything_default=1234, save_config_overwrite=True) #, run=False)
    # cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    # predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    # print(predictions[0])


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
