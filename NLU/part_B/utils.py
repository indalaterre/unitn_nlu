# Add functions or classes used for data loading and preprocessing
import os
import math
import json
import urllib.request

import torch
import torch.utils.data as torch_data

from functools import partial
from collections import Counter

from sklearn.model_selection import train_test_split


def download_dataset_if_needed(dataset_dir = '../dataset'):
    base_url = 'https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/'
    files = ['train.json', 'test.json']

    os.makedirs(dataset_dir, exist_ok=True)

    for filename in files:
        filepath = os.path.join(dataset_dir, filename)
        if not os.path.exists(filepath):
            url = base_url + filename
            print(f'Downloading {filename}...')
            urllib.request.urlretrieve(url, filepath)
            print(f'{filename} downloaded successfully')

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

    src_utt, _ = merge(new_item['utterance'])
    src_slots, y_lengths = merge(new_item['slots'])
    src_intents = torch.LongTensor(new_item['intent'])

    attention_mask = torch.LongTensor([[1 if i != pad_idx else 0 for i in seq] for seq in src_utt])

    return {'utterance': src_utt.to(device),
            'slots': src_slots.to(device),
            'intents': src_intents.to(device),
            'attention_mask': attention_mask.to(device),
            'lengths': torch.LongTensor(y_lengths).cpu()}

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
        vocab = {'pad': pad_idx, 'unk': 1}

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


class IntentsAndSlots(torch_data.Dataset):
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        self.tokenizer = tokenizer

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids, self.slot_ids = self.mapping_seq(self.utterances, self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, slots, mapper):
        res = []
        res_slots = []
        for seq, slot in zip(data, slots):
            tmp_seq = []
            tmp_slot = []
            for word, slot in zip(seq.split(), slot.split(' ')):
                word_tokens = self.tokenizer(word)
                # remove CLS and SEP tokens
                word_tokens['input_ids'] = word_tokens['input_ids'][1:-1]
                tmp_seq.extend(word_tokens['input_ids'])
                # extend the slot for the length of the tokenized word with the padding token
                tmp_slot.extend([mapper[slot]] + [mapper['pad']] * (len(word_tokens['input_ids']) - 1))

            # add CLS and SEP tokens
            tmp_seq = [101] + tmp_seq + [102]
            res.append(tmp_seq)
            # add 0 and 0 to the slot corresponding to CLS and SEP tokens
            tmp_slot = [mapper['pad']] + tmp_slot + [mapper['pad']]
            res_slots.append(tmp_slot)

        return res, res_slots

class NLUDataset(torch_data.Dataset):

    def __init__(self, dataset, lang, tokenizer):
        self.lang = lang
        self.tokenizer = tokenizer

        self.utterances = []
        intents = []
        slots = []

        for item in dataset:
            self.utterances.append(item['utterance'])
            intents.append(item['intent'])
            slots.append(item['slots'])

        self.slot_ids = self.mapping_seq(slots, self.lang.slot2id)
        self.intent_ids = self.mapping_intents(intents, self.lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.utterances[idx], return_tensors='pt')

        utt_tokens = tokens['input_ids'][0]
        attention_token = tokens['attention_mask'][0]

        word_ids = tokens.word_ids()
        sent2words = self.utterances[idx].split()

        words = set(word_ids)
        words.remove(None)

        # take only the first word piece of each word, keep the index of the first word piece
        words = set(word_ids)
        words.remove(None)

        words_seen = set()
        mapping_slots = []
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in words_seen:
                words_seen.add(word_id)
                mapping_slots.append(i)

        mapping_slots = torch.LongTensor(mapping_slots)

        if len(mapping_slots) != len(sent2words):
            assert f"Length mismatch: mapping_slots has {len(mapping_slots)} elements, but sent2words has {len(sent2words)}"

        slots = torch.LongTensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]

        # Create the sample
        return {'utterance': utt_tokens,
                  'attention_mask': attention_token,
                  'mapping_slots': mapping_slots,
                  'sentence': sent2words,
                  'slots': slots,
                  'intent': intent}

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


def build_data_sources(device,
                       tokenizer,
                       train_batch=64,
                       eval_batch=128,
                       dataset_dir='../dataset/ATIS'):

    download_dataset_if_needed(dataset_dir)

    train_raw = load_json_data(f'{dataset_dir}/train.json')
    test_raw = load_json_data(f'{dataset_dir}/test.json')

    lang = Language(train_raw, test_raw)

    train_set, val_set, test_set = create_train_val_test_sets(train_raw, test_raw)

    coll_fn = partial(collate_fn, device=device, pad_idx=lang.get_pad_index())

    train_loader = torch_data.DataLoader(IntentsAndSlots(train_set, lang, tokenizer),
                                         shuffle=True,
                                         collate_fn=coll_fn,
                                         batch_size=train_batch)
    val_loader = torch_data.DataLoader(IntentsAndSlots(val_set, lang, tokenizer),
                                       collate_fn=coll_fn,
                                       batch_size=eval_batch)
    test_loader = torch_data.DataLoader(IntentsAndSlots(test_set, lang, tokenizer),
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
