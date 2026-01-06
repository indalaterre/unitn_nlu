import os
import math

from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm

from model import Language, LanguageModelDataset, LanguageModelLSTM, EarlyStopping
from utils import download_dataset_if_needed, read_raw_data, get_experiment_config


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()

    total_loss = 0
    total_tokens = 0

    for batch in data:
        optimizer.zero_grad()
        output = model(batch['source'])

        loss = criterion(output, batch['target'])

        total_loss += loss.item()
        total_tokens += batch['number_tokens']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return total_loss / total_tokens


@torch.no_grad()
def eval_loop(data, criterion, model):
    model.eval()

    total_loss = 0
    total_tokens = 0

    for batch in data:
        output = model(batch['source'])
        loss = criterion(output, batch['target'])

        total_loss += loss.item()
        total_tokens += batch['number_tokens']

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return ppl, avg_loss


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


def build_data_sources(config, train_batch=64, eval_batch=128):
    dataset_dir = '../dataset'
    download_dataset_if_needed(dataset_dir)

    train_raw = read_raw_data(f'{dataset_dir}/ptb.train.txt')
    val_raw = read_raw_data(f'{dataset_dir}/ptb.valid.txt')
    test_raw = read_raw_data(f'{dataset_dir}/ptb.test.txt')

    lang = Language(train_raw,
                    min_token_freq=config['min_token_freq'],
                    special_tokens=['<pad>', '<eos>', '<unk>'])

    train_set = LanguageModelDataset(train_raw, lang)
    val_set = LanguageModelDataset(val_raw, lang)
    test_set = LanguageModelDataset(test_raw, lang)

    fn = partial(collate_fn, pad_token=lang.get_pad_index())
    train_ds = data.DataLoader(train_set,
                               shuffle=True,
                               collate_fn=fn,
                               batch_size=train_batch)
    val_ds = data.DataLoader(val_set,
                             collate_fn=fn,
                             batch_size=eval_batch)
    test_ds = data.DataLoader(test_set,
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


def collate_fn(data, pad_token, device='cuda'):
    def merge(sequences):
        seq_lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(seq_lengths) == 0 else max(seq_lengths)

        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = seq_lengths[i]
            padded_seqs[i, :end] = seq

        padded_seqs = padded_seqs.detach()
        return padded_seqs, seq_lengths

    data.sort(key=lambda x: len(x['source']), reverse=True)
    new_item = {key: [d[key] for d in data] for key in data[0].keys()}

    source, _ = merge(new_item['source'])
    target, lengths = merge(new_item['target'])

    new_item['source'] = source.to(device)
    new_item['target'] = target.to(device)
    new_item['number_tokens'] = sum(lengths)

    return new_item

def run_experiment(experiment_name, models_dir='models'):
    os.makedirs(models_dir, exist_ok=True)

    use_avg_sgd = True if experiment_name == 'nm_avg_sgd' else False
    dropout = .3 if experiment_name == 'var_dropout' else 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    experiment = get_experiment_config()

    print(f'Running experiment {experiment_name} on device {device.upper()}')
    print(f'Experiment config: {experiment}')

    train_ds, val_ds, test_ds, lang = build_data_sources(experiment)
    pad_index = lang.get_pad_index()

    lr = experiment['lr']

    hidden_size = experiment['hidden_size']
    embedding_size = experiment['embedding_size']

    model = LanguageModelLSTM(emb_size=embedding_size,
                              hidden_size=hidden_size,
                              output_size=len(lang.word2id),
                              pad_index=lang.get_pad_index(),
                              n_layers=experiment['n_layers'],
                              emb_dropout=dropout,
                              out_dropout=dropout,
                              tye_weights=True if experiment_name == 'weight_tying' else False).to(device)
    model.apply(init_weights)

    optimizer = build_optimizer(lr=lr,
                                model=model,
                                optimizer_name=experiment['optimizer_name'])
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    num_epochs = experiment['epochs']
    early_stopping = EarlyStopping(patience=experiment['patience'], mode='min')

    avg_weights = None
    avg_weights_size = 0
    avg_trigger_epoch = -1

    pbar = tqdm(range(1, num_epochs))
    for epoch in pbar:
        loss = train_loop(train_ds, optimizer, criterion_train, model, clip=experiment['clip'])

        val_ppl, val_loss = eval_loop(val_ds, criterion_eval, model)

        avg_message = '[NM-AvgSGD], ' if avg_weights is not None else ''
        pbar.set_description(f'{avg_message}Train Loss={loss:.4f}, Val Loss={val_loss:.4f}, Val PPL={val_ppl:.4f}, LR={lr:.3f}')

        should_stop, counter, is_best = early_stopping(val_ppl)
        if is_best:
            torch.save(model.state_dict(), f'{models_dir}/{experiment_name}.pt')

        if avg_weights is not None:
            avg_weights_size +=1
            set_avg_weights(model, avg_weights, avg_weights_size)

            if epoch - avg_trigger_epoch >= experiment['avg_epochs']:
                print(f'Average weights triggered at epoch {epoch}')
                break
        elif should_stop:
            if not use_avg_sgd:
                print(f'Early stopping triggered at epoch {epoch}')
                break
            else:
                # Model is not improving anymore. We need to start collecting the weight avg
                avg_weights_size = 0
                avg_trigger_epoch = epoch
                avg_weights = extract_model_parameters(model)

                early_stopping = EarlyStopping(patience=experiment['patience'], mode='min')


    if use_avg_sgd and avg_weights:
        apply_avg_weights(model, avg_weights)
        torch.save(model.state_dict(), f'{models_dir}/{experiment_name}.pt')
    else:
        model.load_state_dict(torch.load(f'{models_dir}/{experiment_name}.pt'))

    test_ppl, test_loss = eval_loop(test_ds, criterion_eval, model)
    print(f'Test PPL: {test_ppl:.4f}, Test Loss: {test_loss:.4f}')

    return test_ppl

def extract_model_parameters(model):
    return {name:param.detach().clone() for name, param in model.named_parameters()}

def set_avg_weights(model, avg_weights, size):
    avg_factor = 1 / size
    for name, param in model.named_parameters():
        avg_weights[name] = avg_weights[name] + avg_factor * (param.data - avg_weights[name])

def apply_avg_weights(model, avg_weights):
    for name, param in model.named_parameters():
        param.data.copy_(avg_weights[name])