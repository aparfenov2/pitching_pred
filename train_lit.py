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
    def __init__(self,
        hidden_sz = 10,
        input_sz  = 1,
        output_sz = 1,
        num_lstm_layers = 2,
        criterion = "MSELoss",
        metrics_each = 10
        ):
        super(LitPitchingPred, self).__init__()

        self.model = MyModel(
            hidden_sz=hidden_sz,
            input_sz=input_sz,
            output_sz=output_sz,
            num_lstm_layers=num_lstm_layers
            )
        self.criterion = eval("nn." + criterion)()
        self.metrics_each = metrics_each

        self.save_hyperparameters({
            'hidden_sz': hidden_sz,
            'input_sz' : input_sz,
            'output_sz': output_sz,
            'num_lstm_layers': num_lstm_layers
            })

    def set_datamodule(self, dm: MyDataModule):
        self.datamodule = dm

    def training_step(self, batch, batch_idx):
        data, t = batch
        x = data[:, :-1]
        y = data[:, 1:]
        out = self.model(x, extend_output_size_to_input=False)
        # assert out.shape == y[...,:out.shape[-1]].shape, str(out.shape) + " " + str(y[...,:out.shape[-1]].shape)
        loss = self.criterion(out, y[...,:out.shape[-1]])
        self.log("train_loss", loss, on_step=True)
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

        num_feats = self.model.output_sz # y.size(1)
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
        if self.current_epoch % self.metrics_each == 0:
            fig = make_figure()
            make_preds_plot(
                fig, self.model, y,
                window_len=int(20 * freq),
                future_len=int(self.datamodule.future_len_s * freq),
                cols=col_names
                )
            fig.suptitle(f"Эпоха {self.current_epoch} частота {freq:3.2f}")
            img = draw_to_image(fig)
            img = img.swapaxes(0, 2).swapaxes(1, 2) # CHW
            self.logger.experiment.add_image(f"test_img_pred", img)

            if self.current_epoch > 0:
                # calc metrics
                mae_mean, mae_max, gts, preds, ts = get_all_metrics(
                    self.datamodule.test_dataloader(),
                    model=self.model,
                    future_len=int(self.datamodule.future_len_s * freq)
                    )
                for i, col in enumerate(col_names):
                    self.log('mae_mean_'+col, mae_mean[i].item(), on_step=False, on_epoch=True, logger=True)
                    self.log('mae_max_'+col, mae_max[i].item(), on_step=False, on_epoch=True, logger=True)

                # save preds to csv
                df = metrics_to_pandas(gts, preds, ts, col_names)
                fn = self.logger.log_dir + '/preds_' + str(self.current_epoch) + '.csv'
                df.to_csv(fn, index=False)
                print("metrics preds saved to " + fn)

        # plt.savefig('predict%d.pdf'%i)
        # plt.close()



def cli_main():
    cli = LightningCLI(
        LitPitchingPred, MyDataModule,
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
