# Add functions or classes used for data loading and preprocessing

import os
import urllib.request

def download_dataset_if_needed(dataset_dir = '../dataset'):

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

def get_experiment_config():
    return {
        'min_token_freq': 5,
        'hidden_size': 350,
        'embedding_size': 300,
        'n_layers': 1,
        'emb_dropout': -1,
        'out_dropout': -1,
        'optimizer_name': 'sgd',
        'lr': 0.0015,
        'train_batch': 128,
        'eval_batch': 256,
        'epochs': 100,
        'avg_epochs': 50,
        'patience': 5,
        'clip': 5
    }


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