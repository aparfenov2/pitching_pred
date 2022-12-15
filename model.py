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
        input_sz  = 1,
        output_sz = 1
        ):
        super(MyModel, self).__init__()
        self.hidden_sz = hidden_sz
        self.input_sz = input_sz
        self.output_sz = output_sz
        self.lstm1 = nn.LSTMCell(self.input_sz, self.hidden_sz)
        self.lstm2 = nn.LSTMCell(self.hidden_sz, self.hidden_sz)
        self.linear = nn.Linear(self.hidden_sz, self.output_sz)

    def forward_one_step(self, _input_t, state: RNNState):
        assert _input_t.dim() == 2 # b,f
        state.h_t, state.c_t = self.lstm1(_input_t, (state.h_t, state.c_t))
        state.h_t2, state.c_t2 = self.lstm2(state.h_t, (state.h_t2, state.c_t2))
        return self.linear(state.h_t2)

    @staticmethod
    def extend_with_static_features(tensor, copy_from):
        if tensor.shape[-1] == copy_from.shape[-1]:
            return tensor
        rest = copy_from[..., tensor.shape[-1]:]
        return torch.cat([tensor, rest], axis=-1)

    def forward(self, _input, future:int=0, extend_output_size_to_input=True):
        outputs = []
        state = RNNState(bs=_input.size(0), hidden_sz=self.hidden_sz)
        assert self.input_sz ==_input.size(-1), f"{self.input_sz} {_input.size(-1)}"

        for _input_t in _input.split(1, dim=1):
            _input_t = _input_t.squeeze(dim=1) # remove L dim
            output = self.forward_one_step(_input_t, state)
            if extend_output_size_to_input:
                output = self.extend_with_static_features(output, _input_t)
            outputs += [output]

        for i in range(future):
            if not extend_output_size_to_input:
                output = self.extend_with_static_features(output, _input_t)
            output = self.forward_one_step(output, state)
            if extend_output_size_to_input:
                output = self.extend_with_static_features(output, _input_t)
            outputs += [output]

        return torch.stack(outputs, dim=1) # L


    def make_preds_gen(self, _input, future_len: int):
        assert _input.dim() == 3, str(_input.dim())
        state = RNNState(bs=_input.size(0), hidden_sz=self.hidden_sz)
        delay_line = []

        with torch.no_grad():
            for _input_t in _input.split(1, dim=1):
                _input_t = _input_t.squeeze(dim=1)
                assert _input_t.dim() == 2, str(_input_t.dim())
                assert _input_t.size(0) == _input.size(0) # 32, 2
                delay_line += [_input_t]
                if len(delay_line) < future_len:
                    continue
                input_delayed = delay_line.pop(0)
                output = self.forward_one_step(input_delayed, state)
                output = self.extend_with_static_features(output, input_delayed)

                pred_state = state.clone().detach()
                preds = [output]
                for _ in range(future_len):
                    output = self.forward_one_step(output, state=pred_state)
                    output = self.extend_with_static_features(output, input_delayed)
                    preds += [output]

                # returns [bs,1,feats] lists
                yield _input_t, output, preds

