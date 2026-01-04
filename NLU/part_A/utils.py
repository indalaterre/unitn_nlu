# Add functions or classes used for data loading and preprocessing
import math
import json

from functools import partial
from collections import Counter

import torch
import torch.utils.data as torch_data

from sklearn.model_selection import train_test_split


def load_json_data(path):
    with open(path, 'r') as f:
        return json.load(f)


def create_train_val_test_sets(train_raw, test_raw, factor=.1):
    intents = [x['intent'] for x in train_raw]
    counter = Counter(intents)

    additional_train_set = []

    split_inputs = []
    split_labels = []

    for idx, intent in enumerate(intents):
        if counter[intent] == 1:
            additional_train_set.append(train_raw[idx])
        else:
            split_inputs.append(train_raw[idx])
            split_labels.append(intent)

    x_train, x_val, _, __ = train_test_split(split_inputs,
                                             split_labels,
                                             shuffle=True,
                                             random_state=42,
                                             test_size=factor,
                                             stratify=split_labels)
    x_train.extend(additional_train_set)

    return x_train, x_val, test_raw


class Language:

    def __init__(self, train_raw, test_raw, pad_idx=0, cutoff=0):
        self.pad_idx = pad_idx

        words, slots, intents = self.extract_vocabularies(train_raw, test_raw)

        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, include_pad=False)
        self.word2id = self.word2id(words, pad_idx, cutoff=cutoff)

        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def get_vocab_len(self):
        return len(self.word2id)

    def get_intent_len(self):
        return len(self.intent2id)

    def get_slot_len(self):
        return len(self.slot2id)

    def get_pad_index(self):
        return self.pad_idx

    @staticmethod
    def extract_vocabularies(train_raw, test_raw):

        raw = train_raw + test_raw

        words = sum([x['utterance'].split() for x in train_raw], [])
        slots = set(sum([x['slots'].split() for x in raw], []))
        intents = set([x['intent'] for x in raw])

        return words, slots, intents

    @staticmethod
    def word2id(elements, pad_idx, cutoff=0):
        vocab = {'pad': pad_idx, 'unk':  1}

        counter = Counter(elements)
        for word, count in counter.items():
            if count > cutoff:
                vocab[word] = len(vocab)

        return vocab

    @staticmethod
    def lab2id(elements, include_pad=True):
        vocab = {}
        if include_pad:
            vocab['pad'] = 0

        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class NLUDataset(torch_data.Dataset):

    def __init__(self, dataset, lang):
        self.lang = lang

        self.utterances = []
        self.intents = []
        self.slots = []

        for item in dataset:
            self.utterances.append(item['utterance'])
            self.intents.append(item['intent'])
            self.slots.append(item['slots'])

        self.slots = self.mapping_seq(self.slots, self.lang.slot2id)
        self.utterances = self.mapping_seq(self.utterances, self.lang.word2id)
        self.intents = self.mapping_intents(self.intents, self.lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return {
            'utterance': torch.Tensor(self.utterances[idx]),
            'slots': torch.Tensor(self.slots[idx]),
            'intent': self.intents[idx]
        }

    @staticmethod
    def mapping_intents(intents, intent2id):
        return [intent2id[intent] if intent in intent2id else intent2id['unk'] for intent in intents]

    @staticmethod
    def mapping_seq(x, tkn2id_mapper):
        output = []
        for seq in x:
            tmp_seq = []
            for part in seq.split():
                tmp_seq.append(tkn2id_mapper[part if part in tkn2id_mapper else 'unk'])
            output.append(tmp_seq)

        return output


def collate_fn(x, device, pad_idx=0):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)

        padded = torch.LongTensor(len(sequences), max_len).fill_(pad_idx)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded[i, :end] = seq

        return padded.detach(), lengths

    x.sort(key=lambda k: len(k['utterance']), reverse=True)
    new_item = {key: [d[key] for d in x] for key in x[0].keys()}

    src_utt, lengths = merge(new_item['utterance'])
    src_slots, _ = merge(new_item['slots'])
    src_intents = torch.LongTensor(new_item['intent'])

    return {'utterance': src_utt.to(device),
            'slots': src_slots.to(device),
            'intents': src_intents.to(device),
            'lengths': torch.LongTensor(lengths).cpu()}


def build_data_sources(device,
                       train_batch=64,
                       eval_batch=128,
                       dataset_dir='../dataset/ATIS'):
    train_raw = load_json_data(f'{dataset_dir}/train.json')
    test_raw = load_json_data(f'{dataset_dir}/test.json')

    lang = Language(train_raw, test_raw)

    train_set, val_set, test_set = create_train_val_test_sets(train_raw, test_raw)

    coll_fn = partial(collate_fn, device=device, pad_idx=lang.get_pad_index())

    train_loader = torch_data.DataLoader(NLUDataset(train_set, lang),
                                         shuffle=True,
                                         collate_fn=coll_fn,
                                         batch_size=train_batch)
    val_loader = torch_data.DataLoader(NLUDataset(val_set, lang),
                                       collate_fn=coll_fn,
                                       batch_size=eval_batch)
    test_loader = torch_data.DataLoader(NLUDataset(test_set, lang),
                                        collate_fn=coll_fn,
                                        batch_size=eval_batch)

    return train_loader, val_loader, test_loader, lang


class EarlyStopping:
    def __init__(self, patience=3, mode='min'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = math.inf if mode == 'min' else -math.inf
        self.early_stop = False

        if mode == 'min':
            self.best_fn = lambda x, y: x < y
        else:
            self.best_fn = lambda x, y: x > y

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
