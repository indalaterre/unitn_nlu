# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from NLU.part_A.model import UncertaintyWeighedLoss
from conll import evaluate

from model import NLUModel
from utils import build_data_sources, EarlyStopping

from torch.utils.tensorboard import SummaryWriter


def train_loop(data_loader, model, optimizer, loss_fn, clip=5):
    model.train()

    loss_sum = 0
    for batch in data_loader:
        optimizer.zero_grad()

        slots, intents = model(batch['utterance'], batch['lengths'])

        total_loss = loss_fn((intents, batch['intents']), (slots, batch['slots']))

        loss_sum += total_loss.item()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss_sum / len(data_loader)

@torch.no_grad()
def eval_loop(data_loader, model, loss_fn, lang):
    model.eval()

    label_intents, label_slots = [], []
    predicted_intents, predicted_slots = [], []

    loss_sum = 0

    for batch in data_loader:
        slots, intents = model(batch['utterance'], batch['lengths'])

        loss_sum += loss_fn((intents, batch['intents']), (slots, batch['slots']))

        label_intents.extend([lang.id2intent[x] for x in batch['intents'].tolist()])
        predicted_intents.extend([lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()])

        output_slots = torch.argmax(slots, dim=1)
        for id_seq, seq in enumerate(output_slots):
            length = batch['lengths'].tolist()[id_seq]
            utt_ids = batch['utterance'][id_seq][:length].tolist()
            gt_ids = batch['slots'][id_seq].tolist()
            gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
            utterance = [lang.id2word[elem] for elem in utt_ids]
            to_decode = seq[:length].tolist()
            label_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
            tmp_seq = []
            for id_el, elem in enumerate(to_decode):
                tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
            predicted_slots.append(tmp_seq)

    try:
        slots_report = evaluate(label_slots, predicted_slots)
    except KeyError as _:
        slots_report = {"total": {"f": 0}}

    intents_report = classification_report(label_intents, predicted_intents, zero_division=False, output_dict=True)
    return loss_sum / len(data_loader), intents_report, slots_report


def run_experiment(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader, lang = build_data_sources(device=device,
                                                                     eval_batch=config['eval_batch'],
                                                                     train_batch=config['train_batch'])

    pad_idx = config['pad_idx']

    criterion_intents = nn.CrossEntropyLoss()
    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_idx)

    intent_accuracies, slot_f1_scores = [], []

    writer = None
    for run in range(config['runs']):
        print(f'\nStarting RUN {run + 1}/{config["runs"]}')

        if run == 0:
            run_name = 'LSTM'
            if config['use_bidirectional']:
                run_name += '_bi-directional'
            if config['dropout'] > 0:
                run_name += f"_dropout_{config['dropout']}"
            writer = SummaryWriter(log_dir=f"runs/{run_name}_experiment")

        model = NLUModel(emb_dim=config['emb_dim'],
                         hidden_dim=config['hidden_dim'],

                         vocab_size=lang.get_vocab_len(),

                         out_dims=(lang.get_intent_len(), lang.get_slot_len()),

                         dropout=config['dropout'],
                         use_bidirectional=config['use_bidirectional']).to(device)

        uncertainty_loss = UncertaintyWeighedLoss(learnable_tasks=2).to(device)

        def loss_fn(i_results, s_results):
            i_loss = criterion_intents(i_results[0], i_results[1])
            s_loss = criterion_slots(s_results[0], s_results[1])
            return uncertainty_loss([i_loss, s_loss])

        optimizer = optim.AdamW(list(model.parameters()) + list(uncertainty_loss.parameters()),
                                lr=config['lr'])

        early_stopping = EarlyStopping(patience=config['patience'], mode='max')

        pbar = tqdm(range(config['epochs']))
        for epoch in pbar:
            train_loop(train_loader,
                       model=model,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       clip=config['clip'])
            val_loss, val_i_report, val_s_report = eval_loop(val_loader,
                                                             lang=lang,
                                                             model=model,
                                                             loss_fn=loss_fn)

            slots_f1_score = val_s_report['total']['f']
            should_stop, counter, is_best = early_stopping(slots_f1_score)

            if writer:
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Metric/Slot_F1', val_s_report['total']['f'], epoch)
                writer.add_scalar('Metric/Intent_Accuracy', val_i_report['accuracy'], epoch)

                loss_weights = uncertainty_loss.get_losses_weights()
                writer.add_scalar('Uncertainty/Weight_Intent', loss_weights[0], epoch)
                writer.add_scalar('Uncertainty/Weight_Slot', loss_weights[1], epoch)

            pbar.set_description(f'Val Loss: {val_loss:.4f}, Slot F1={slots_f1_score:.4f}, Intent ACC: {val_i_report["accuracy"]:.4f}')

            if is_best:
                torch.save(model.state_dict(), f'{config['models_dir']}/nlu.pt')
            elif should_stop:
                print(f'Early stopping triggered at epoch {epoch}')
                break

        model.load_state_dict(torch.load(f'{config['models_dir']}/nlu.pt'))

        _, test_i_report, test_s_report = eval_loop(test_loader,
                                                    lang=lang,
                                                    model=model,
                                                    loss_fn=loss_fn)

        slot_f1_scores.append(test_s_report['total']['f'])
        intent_accuracies.append(test_i_report['accuracy'])

    slot_f1_scores = np.asarray(slot_f1_scores)
    intent_accuracies = np.asarray(intent_accuracies)

    print('\n')
    print(f"Slot F1: {slot_f1_scores.mean():.4f} ± {slot_f1_scores.std():.4f}")
    print(f"Intent ACC: {intent_accuracies.mean():.4f} ± {intent_accuracies.std():.4f}")
