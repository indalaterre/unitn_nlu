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