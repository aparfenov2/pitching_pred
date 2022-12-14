import argparse
import yaml

from train_lit import LitPitchingPred
from metrics import get_all_metrics, metrics_to_pandas
from dataset import MyDataModule

parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('weights_path')
parser.add_argument('--data_path')

args = parser.parse_args()

with open(args.config, 'r') as stream:
    cfg = yaml.safe_load(stream)

# model = LitPitchingPred()
model = LitPitchingPred.load_from_checkpoint(
    args.weights_path,
    **cfg['model']
    )
model.eval()

dm = MyDataModule(**cfg['data'])

mae_mean, mae_max, gts, preds, ts = get_all_metrics(dm.test_dataloader(), model=model.model, future_len=int(3 * dm.freq))
print("mae_mean", mae_mean, "mae_max", mae_max)
df = metrics_to_pandas(gts, preds, ts, dm.cols)
df.to_csv('preds.csv', index=False)
