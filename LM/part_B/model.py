import torch
import torch.nn as nn


class VariationalDropout(nn.Module):
    def __init__(self, dropout=.9):
        super(VariationalDropout, self).__init__()

        self.dropout = dropout

    def forward(self, x):
        # Even if Gal & Ghahramani paper describe var dropout as a
        # technique to predict uncertainty, this is not fitting our needs
        # for language modeling.
        # We need to evaluate a PPL which is a fixed value so we'll keep
        # this dropout method only for training purposes.
        # It still ensures the same dropout mask for every training step
        # (standard dropout would use a different mask for every step)
        if not self.training or self.dropout <= 0:
            return x

        # Shape (batch, 1, features): the middle dim=1 ensures the same mask
        # is applied across all time steps (sequence length dimension)
        mask = torch.bernoulli(
            torch.ones(x.size(0), 1, x.size(2), device=x.device) * (1 - self.dropout)
        )

        mask /= (1 - self.dropout)

        return x * mask


class LanguageModelLSTM(nn.Module):

    def __init__(self,
                 emb_size,
                 hidden_size,
                 output_size,
                 n_layers=1,
                 pad_index=0,
                 emb_dropout=-1,
                 out_dropout=-1,
                 tie_weights=False):

        super(LanguageModelLSTM, self).__init__()

        if n_layers < 1:
            raise ValueError(f'Invalid number of layers {n_layers}. Must be >= 1')

        if tie_weights:
            print(f'Applying weight tying. Hidden layers size set to embedding size')
            hidden_size = emb_size

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = VariationalDropout(emb_dropout) if emb_dropout > 0 else nn.Identity()

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)

        self.out_dropout = VariationalDropout(out_dropout) if out_dropout > 0 else nn.Identity()
        self.output = nn.Linear(hidden_size, output_size)

        if tie_weights:
            self.output.weight = self.embedding.weight

    def forward(self, x):
        embedded_data = self.emb_dropout(self.embedding(x))
        lstm_data, _ = self.lstm(embedded_data)
        # Permute from (batch, seq, vocab) to (batch, vocab, seq) for CrossEntropyLoss
        return self.output(self.out_dropout(lstm_data)).permute(0, 2, 1)