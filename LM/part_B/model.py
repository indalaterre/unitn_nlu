import math

import torch
import torch.nn as nn
import torch.utils.data as data

from collections import Counter

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
                 tye_weights=False):

        super(LanguageModelLSTM, self).__init__()

        if n_layers < 1:
            raise ValueError(f'Invalid number of layers {n_layers}. Must be >= 1')

        if tye_weights:
            print(f'Applying weight tying. Hidden layers size set to embedding size')
            hidden_size = emb_size

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = VariationalDropout(emb_dropout) if emb_dropout > 0 else nn.Identity()

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)

        self.out_dropout = VariationalDropout(out_dropout) if out_dropout > 0 else nn.Identity()
        self.output = nn.Linear(hidden_size, output_size)

        if tye_weights:
            self.output.weight = self.embedding.weight

    def forward(self, x):
        data = self.emb_dropout(self.embedding(x))
        data, _ = self.lstm(data)
        return self.output(self.out_dropout(data)).permute(0, 2, 1)


class LanguageModelDataset(data.Dataset):

    def __init__(self, corpus, lang):

        self.source = []
        self.target = []

        for sentence in corpus:
            parts = sentence.split()
            self.source.append(parts[:-1])
            self.target.append(parts[1:])

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])

        return { 'source': src, 'target': trg, 'number_tokens': len(src) }

    @staticmethod
    def mapping_seq(data, lang):
        res = []
        unk_id = lang.word2id[lang.unk_token]

        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                elif unk_id is not None:
                    tmp_seq.append(unk_id)
                else:
                    print('OOV found! You have to deal with that')
                    break

            res.append(tmp_seq)

        return res

class Language:

    def __init__(self, corpus, min_token_freq=5, special_tokens = None):
        self.unk_token = '<unk>'
        self.min_token_freq = min_token_freq

        if special_tokens is None:
            special_tokens = []

        if self.unk_token not in special_tokens:
            special_tokens.append(self.unk_token)

        self.words_counter = Counter()
        for sentence in corpus:
            for word in sentence.split():
                self.words_counter[word] += 1

        self.word2id = self.get_vocab(special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_pad_index(self):
        return self.word2id['<pad>']

    def get_vocab(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = []

        output = {}
        i = 0
        for token in special_tokens:
            output[token] = i
            i += 1

        for word, count in self.words_counter.items():
            if count >= self.min_token_freq and word not in output:
                output[word] = i
                i += 1

        return output


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = math.inf if mode == 'min' else -math.inf
        self.early_stop = False

        if mode == 'min':
            self.best_fn = lambda x, y: x < y - self.min_delta
        else:
            self.best_fn = lambda x, y: x > y + self.min_delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        is_best = self.best_fn(score, self.best_score)
        if is_best:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop, self.counter, is_best