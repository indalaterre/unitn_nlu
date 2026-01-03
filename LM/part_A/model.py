import torch
import torch.nn as nn
import torch.utils.data as data

from collections import Counter

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
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop