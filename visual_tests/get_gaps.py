import pandas as pd
import numpy as np

def _find_gaps(data):
    a = data["sec"].values
    threshold = 2
    return np.where(abs(np.diff(a))>threshold)[0] + 1

fn_test =  "data/NPN_1155_part1.dat"

data = pd.read_csv(fn_test, sep=" ")
gaps = _find_gaps(data)
gaps = [0, *gaps, len(data)]

print(gaps)
print(np.diff(gaps))
