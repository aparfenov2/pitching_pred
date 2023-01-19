# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .base import TimeSeries, ModelBase

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        #src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        #embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

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

        embedded = self.dropout(self.embedding(input))

        #embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

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


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        assert src.dim() == 2, str(src.shape)
        assert trg.dim() == 2, str(trg.shape)

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        #first input to the decoder is the <sos> tokens
        input = src[-1,:]

        for t in range(trg_len):

            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


class Scaler(nn.Module):
    def __init__(self, min_v, max_v, discr_dim):
        super().__init__()
        self.min_v = min_v
        self.max_v = max_v
        self.discr_dim = discr_dim

    def scale(self, v):
        return torch.clip(torch.round((v - self.min_v)/(self.max_v - self.min_v) * (self.discr_dim - 1)), 0, self.discr_dim - 1).to(dtype=torch.long)

    def scale_back(self, v):
        return v / (self.discr_dim - 1) * (self.max_v - self.min_v) + self.min_v


class Seq2SeqWrp(ModelBase):
    def __init__(self,
        seq2seq: Seq2Seq,
        scaler: Scaler,
        history_len_s,
        train_future_len_s,
        freq
    ):
        super().__init__()

        self.seq2seq = seq2seq
        self.scaler = scaler
        self.history_len = int(history_len_s * freq)
        self.train_future_len = int(train_future_len_s * freq)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, src, trg, teacher_forcing_ratio=0.5, return_unscaled=False):

        src = src.permute(1, 0, 2).squeeze(-1)
        trg1 = trg.permute(1, 0, 2).squeeze(-1)
        src = self.scaler.scale(src)
        trg1 = self.scaler.scale(trg1)

        output = self.seq2seq.forward(src, trg1)

        assert output.dim() == 3, str(output.shape)

        if return_unscaled:
            return output, trg1

        output = output.argmax(-1)
        assert output.dim() == 2, str(output.shape)
        # output = [trg_len, batch_size]
        output = self.scaler.scale_back(output)
        output = output.permute(1, 0).unsqueeze(-1)
        assert output.dim() == 3, str(output.shape)
        assert output.shape == trg.shape, f"output.shape {output.shape} trg.shape {trg.shape}"

        return output


    def training_step(self, batch, lit, **kwargs):
        y, t = batch
        src = y[:,-self.history_len:-self.train_future_len]
        trg = y[:,-self.train_future_len:]
        # trg = [batch size, trg len]

        output, trg1 = self.forward(src, trg, return_unscaled=True)
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]
        output1 = output.reshape(-1, output_dim)
        trg1 = trg1.reshape(-1)

        ce_loss = self.criterion(output1, trg1)

        output = output.argmax(-1)
        assert output.dim() == 2, str(output.shape)
        # output = [trg_len, batch_size]
        output = self.scaler.scale_back(output)
        output = output.permute(1, 0).unsqueeze(-1)
        assert output.dim() == 3, str(output.shape)

        assert output.shape == trg.shape, f"output.shape {output.shape} trg.shape {trg.shape}"

        return ce_loss + lit.criterion(output, trg)

    def validation_step(self, batch, lit, **kwargs):
        y, t = batch
        src = y[:,-self.history_len:-self.train_future_len]
        trg = y[:,-self.train_future_len:]
        # trg = [batch size, trg len]

        output = self.forward(src, trg, teacher_forcing_ratio=0)

        return {
            'val_pred_loss': lit.val_criterion(output, trg)
        }

    def make_preds_gen(self, ts : TimeSeries, future_len: int):
        y,t = ts.y, ts.t
        assert y.dim() == 3, str(y.shape)
        assert t.dim() == 3, str(t.shape)
        assert y.shape[1] > self.history_len - future_len, f"y.shape[1] {y.shape[1]} self.history_len {self.history_len} future_len {future_len}"

        for i in range(y.shape[1] - self.history_len - future_len):

            src = y[:, i: i + self.history_len]
            trg = y[:, i + self.history_len: i + self.history_len + future_len]

            output = self.forward(src, trg, teacher_forcing_ratio=0)

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
    train_future_len_s,
    history_len_s = 10,
    discr_dim = 100,
    min_v = -2,
    max_v = +2,
    ENC_EMB_DIM = 256,
    DEC_EMB_DIM = 256,
    HID_DIM = 512,
    N_LAYERS = 2,
    ENC_DROPOUT = 0.5,
    DEC_DROPOUT = 0.5,
):
    scaler = Scaler(discr_dim=discr_dim, min_v=min_v, max_v=max_v)
    enc = Encoder(
        input_dim=discr_dim,
        emb_dim=ENC_EMB_DIM,
        hid_dim=HID_DIM,
        n_layers=N_LAYERS,
        dropout=ENC_DROPOUT
        )
    dec = Decoder(
        output_dim=discr_dim,
        n_layers=N_LAYERS,
        hid_dim=HID_DIM,
        emb_dim=DEC_EMB_DIM,
        dropout=DEC_DROPOUT
    )
    device = 'cpu'
    seq2seq =  Seq2Seq(
        encoder=enc, decoder=dec, device=device)

    model = Seq2SeqWrp(
        seq2seq=seq2seq,
        scaler=scaler,
        history_len_s=history_len_s,
        train_future_len_s=train_future_len_s,
        freq=freq
        ).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)
    return model
