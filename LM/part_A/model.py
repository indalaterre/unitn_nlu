import torch.nn as nn


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
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)

        self.out_dropout = nn.Dropout(out_dropout) if out_dropout > 0 else nn.Identity()
        self.output = nn.Linear(hidden_size, output_size)

        if tie_weights:
            self.output.weight = self.embedding.weight

    def forward(self, x):
        embeddings = self.emb_dropout(self.embedding(x))
        lsmt_data, _ = self.lstm(embeddings)
        return self.output(self.out_dropout(lsmt_data)).permute(0, 2, 1)