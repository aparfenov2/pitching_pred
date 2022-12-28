import torch
import torch.nn as nn

class CopyModel(nn.Module):
    def __init__(self,
        ):
        super().__init__()

    def forward(self, y, future:int=0, **kwargs):
        outputs = [y[:,1:]]
        outputs += [y[:,:-future-1:-1]]
        return torch.stack(outputs, dim=1)
