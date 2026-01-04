# Add functions or classes used for data loading and preprocessing

import os
import urllib.request
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils


class LanguageModelDataset(data_utils.Dataset):

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

        return {'source': src, 'target': trg, 'number_tokens': len(src)}

    @staticmethod
    def mapping_seq(data, lang):  # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found! You have to deal with that')
                    break

            res.append(tmp_seq)
        return res


class Language:

    def __init__(self, corpus, special_tokens=None):
        if special_tokens is None:
            special_tokens = []

        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_pad_index(self):
        return self.word2id['<pad>']

    @staticmethod
    def get_vocab(corpus, special_tokens):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
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


def download_dataset_if_needed(dataset_dir='../dataset'):
    base_url = 'https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/'
    files = ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']

    os.makedirs(dataset_dir, exist_ok=True)

    for filename in files:
        filepath = os.path.join(dataset_dir, filename)
        if not os.path.exists(filepath):
            url = base_url + filename
            print(f'Downloading {filename}...')
            urllib.request.urlretrieve(url, filepath)
            print(f'{filename} downloaded successfully')


def read_raw_data(path, eos_token='<eos>'):
    with open(path, 'r') as f:
        return [f'{line.strip()} {eos_token}' for line in f.readlines()]


def get_experiment_config(name):
    configs = {
        'vanilla': {
            'hidden_size': 200,
            'embedding_size': 300,
            'n_layers': 1,
            'emb_dropout': -1,
            'out_dropout': -1,
            'optimizer_name': 'sgd',
            'lr': [0.01, 0.02, 0.03],
            'train_batch': 128,
            'eval_batch': 256,
            'epochs': 100,
            'patience': 5,
            'clip': 5
        },
        'dropout': {
            'hidden_size': 200,
            'embedding_size': 300,
            'n_layers': 1,
            'emb_dropout': 0.3,
            'out_dropout': 0.3,
            'optimizer_name': 'sgd',
            'lr': [0.01, 0.02, 0.03],
            'train_batch': 128,
            'eval_batch': 256,
            'epochs': 100,
            'patience': 5,
            'clip': 5
        },
        'adamw': {
            'hidden_size': 200,
            'embedding_size': 300,
            'n_layers': 2,
            'emb_dropout': 0.3,
            'out_dropout': 0.3,
            'optimizer_name': 'adamw',
            'lr': [0.01, 0.02, 0.03],
            'train_batch': 128,
            'eval_batch': 256,
            'epochs': 100,
            'patience': 5,
            'clip': 5
        }
    }

    if name not in configs:
        raise ValueError(f'Unknown experiment: {name}. Available: {list(configs.keys())}')

    return configs[name]


def collate_fn(data, pad_token, device='cuda'):
    def merge(sequences):
        seq_lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(seq_lengths) == 0 else max(seq_lengths)

        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = seq_lengths[i]
            padded_seqs[i, :end] = seq

        return padded_seqs.detach(), seq_lengths

    data.sort(key=lambda x: len(x['source']), reverse=True)
    new_item = {key: [d[key] for d in data] for key in data[0].keys()}

    source, _ = merge(new_item['source'])
    target, lengths = merge(new_item['target'])

    new_item['source'] = source.to(device)
    new_item['target'] = target.to(device)
    new_item['number_tokens'] = sum(lengths)

    return new_item


def build_optimizer(model, optimizer_name, lr=.001):
    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=.9)
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Invalid optimizer name {optimizer_name}. Available: sgd, adamw')

    return optimizer


def build_data_sources(config):
    download_dataset_if_needed()

    train_raw = read_raw_data('dataset/ptb.train.txt')
    val_raw = read_raw_data('dataset/ptb.valid.txt')
    test_raw = read_raw_data('dataset/ptb.test.txt')

    train_batch = config['train_batch']
    eval_batch = config['eval_batch']

    lang = Language(train_raw, special_tokens=['<pad>', '<eos>'])

    train_set = LanguageModelDataset(train_raw, lang)
    val_set = LanguageModelDataset(val_raw, lang)
    test_set = LanguageModelDataset(test_raw, lang)

    fn = partial(collate_fn, pad_token=lang.get_pad_index())
    train_ds = data_utils.DataLoader(train_set,
                                     shuffle=True,
                                     collate_fn=fn,
                                     batch_size=train_batch)
    val_ds = data_utils.DataLoader(val_set,
                                   collate_fn=fn,
                                   batch_size=eval_batch)
    test_ds = data_utils.DataLoader(test_set,
                                    collate_fn=fn,
                                    batch_size=eval_batch)

    return train_ds, val_ds, test_ds, lang


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
