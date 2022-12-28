import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('csv')
parser.add_argument('--skip', type=int, default=0)
parser.add_argument('--to', type=int)
parser.add_argument('--x-is-sec', action='store_true')
args = parser.parse_args()

df = pd.read_csv(args.csv, sep=' ')
df = df[args.skip:]
if args.to is not None:
    df = df[:args.to]
if args.x_is_sec:
    df.plot(x='sec', style='x-', grid=True)
else:
    df = df.drop('sec', axis=1)
    df.plot(style='x-', grid=True)
plt.show()
