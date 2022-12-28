import argparse
import yaml
import json
import os

from train_lit import LitPitchingPred, MyLightningCLI
from metrics import get_all_metrics, metrics_to_pandas
from dataset import MyDataModule

cli = MyLightningCLI(
    LitPitchingPred, MyDataModule,
    run=False
    )
model = cli.model
dm = cli.datamodule

model.eval()

for test_dl in dm.test_dataloader():
    ds_name = os.path.splitext(os.path.basename(test_dl.dataset.name))[0]

    metrics, gts, preds, ts = get_all_metrics(
        test_dl=test_dl,
        model=model,
        sample_frq=dm.freq,
        future_len_s=model.future_len_s
        )

    _json = {
        col : {k : v[i].item() for k,v in metrics.items()}
            for i, col in enumerate(dm.cols)
    }
    print(json.dumps(_json, indent=4))

    fn = f'preds_{ds_name}.json'
    with open(fn, 'w') as f:
        json.dump(_json, f, indent=4)

    # save preds to csv
    df = metrics_to_pandas(gts, preds, ts, dm.cols)
    fn = f'preds_{ds_name}.csv'
    df.to_csv(fn, index=False, sep=' ')
    print("metrics preds saved to " + fn)

