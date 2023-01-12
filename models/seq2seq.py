# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .base import TimeSeries, ModelBase

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [src len, batch size]

        #embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(src)

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout = dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        #input = [1, batch size]

        #embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(input, (hidden, cell))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        #prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(ModelBase):
    def __init__(self, encoder, decoder, device,
        history_len_s,
        future_len_s,
        freq
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.history_len = int(history_len_s * freq)
        self.future_len = int(future_len_s * freq)
        assert self.history_len == 0 or self.history_len >= self.future_len

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, y, future, teacher_forcing_ratio = 0):
        # emulate behaviour of previous models
        assert future > 0
        assert teacher_forcing_ratio == 0
        src = y
        trg = torch.zeros((y.shape[0], future, y.shape[2]), dtype=y.dtype)
        trg[:, 0] = y[:,-1]
        pred = self.forward1(src=src, trg=trg, teacher_forcing_ratio=teacher_forcing_ratio)
        return torch.cat([src, pred], dim=1)

    def forward1(self, src, trg, teacher_forcing_ratio = 0.5):

        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)

        #src = [src len, batch size]
        #trg = [trg len, batch size]

        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, trg_len):

            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else output

        assert trg.shape == outputs.shape, f"trg.shape {trg.shape} outputs.shape {outputs.shape}"
        outputs = outputs.permute(1, 0, 2) # N, L, F
        return outputs

    def training_step(self, batch, lit, **kwargs):
        y, t = batch
        src = y[:,-self.history_len:-self.future_len]
        trg = y[:,-self.future_len:]
        output = self.forward1(src, trg)
        return lit.criterion(output, trg)

    def validation_step(self, batch, lit, **kwargs):
        y, t = batch
        src = y[:,-self.history_len:-self.future_len]
        trg = y[:,-self.future_len:]
        output = self.forward1(src, trg, teacher_forcing_ratio=0)
        return {
            'val_pred_loss': lit.criterion(output, trg)
        }

    def make_preds_gen(self, ts : TimeSeries, future_len: int):
        y,t = ts.y, ts.t
        assert y.dim() == 3, str(y.shape)
        assert t.dim() == 3, str(t.shape)
        assert y.shape[1] > self.history_len - future_len, f"y.shape[1] {y.shape[1]} self.history_len {self.history_len} future_len {future_len}"

        for i in range(y.shape[1] - self.history_len - future_len):

            src = y[:, i: i + self.history_len]
            trg = y[:, i + self.history_len: i + self.history_len + future_len]
            output = self.forward1(src, trg)

            y_fut = y[:, i + self.history_len + future_len].unsqueeze(1)
            t_fut = t[:, i + self.history_len + future_len].unsqueeze(1)
            p_fut = output[:,-1].unsqueeze(1)

            yield t_fut, y_fut, p_fut, TimeSeries(
                t[:, i + self.history_len: i + self.history_len + future_len],
                output
                )

def make_model(
    freq,
    future_len_s,
    history_len_s = 10,
    INPUT_DIM = 1,
    OUTPUT_DIM = 1,
    HID_DIM = 512,
    N_LAYERS = 2,
    ENC_DROPOUT = 0.5,
    DEC_DROPOUT = 0.5,
):
    enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    device = 'cpu'
    model =  Seq2Seq(
        enc, dec, device,
        history_len_s=history_len_s,
        future_len_s=future_len_s,
        freq=freq
        ).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)
    return model
