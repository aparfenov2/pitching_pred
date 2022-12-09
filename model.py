import torch
import torch.nn as nn

class RNNState:
    def __init__(self, bs, hidden_sz) -> None:
        self.hidden_sz = hidden_sz
        self.bs = bs
        assert bs > 0
        self.h_t = torch.zeros((bs, self.hidden_sz), dtype=torch.float32)
        self.c_t = torch.zeros((bs, self.hidden_sz), dtype=torch.float32)
        self.h_t2 = torch.zeros((bs, self.hidden_sz), dtype=torch.float32)
        self.c_t2 = torch.zeros((bs, self.hidden_sz), dtype=torch.float32)

    def detach(self):
        self.h_t.detach_()
        self.c_t.detach_()
        self.h_t2.detach_()
        self.c_t2.detach_()
        return self

    def clone(self):
        new = RNNState(bs=self.bs, hidden_sz=self.hidden_sz)
        new.h_t = self.h_t.clone()
        new.h_t2 = self.h_t2.clone()
        new.c_t = self.c_t.clone()
        new.c_t2 = self.c_t2.clone()
        return new

class MyModel(nn.Module):

    def __init__(self,
        hidden_sz = 10,
        input_sz  = 1
        ):
        super(MyModel, self).__init__()
        self.hidden_sz = hidden_sz
        self.input_sz = input_sz
        self.lstm1 = nn.LSTMCell(self.input_sz, self.hidden_sz)
        self.lstm2 = nn.LSTMCell(self.hidden_sz, self.hidden_sz)
        self.linear = nn.Linear(self.hidden_sz, self.input_sz)

    def forward_one_step(self, _input_t, state: RNNState):
        state.h_t, state.c_t = self.lstm1(_input_t, (state.h_t, state.c_t))
        state.h_t2, state.c_t2 = self.lstm2(state.h_t, (state.h_t2, state.c_t2))
        return self.linear(state.h_t2)

    def forward(self, _input, future = 0):
        outputs = []
        state = RNNState(bs=_input.size(0), hidden_sz=self.hidden_sz)
        assert self.input_sz ==_input.size(-1), f"{self.input_sz} {_input.size(-1)}"

        for _input_t in _input.split(1, dim=1):
            _input_t = _input_t.squeeze(dim=1) # remove L dim
            output = self.forward_one_step(_input_t, state)
            outputs += [output]
        for i in range(future):# if we should predict the future
            output = self.forward_one_step(output, state)
            outputs += [output]
        outputs = torch.stack(outputs, dim=1) # L

        return outputs

