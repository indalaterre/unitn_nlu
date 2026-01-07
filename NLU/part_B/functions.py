# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer

from utils import build_data_sources, EarlyStopping
from model import NLUBertModel, UncertaintyWeighedLoss

from conll import evaluate
from sklearn.metrics import classification_report

from torch.utils.tensorboard import SummaryWriter


def train_loop(data, model, optimizer, loss_fn, clip=5):
    model.train()

    loss_sum = 0
    for batch in data:
        optimizer.zero_grad()

        slots, intents = model(batch['utterance'], attention_mask=batch['attention_mask'])

        total_loss = loss_fn((intents, batch['intents']), (slots, batch['slots']))
        loss_sum += total_loss.item()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss_sum / len(data)


@torch.no_grad()
def eval_loop(data, model, loss_fn, tokenizer, lang):
    model.eval()

    label_intents, label_slots = [], []
    predicted_intents, predicted_slots = [], []

    loss_sum = 0
    for batch in data:
        slots, intents = model(batch['utterance'], attention_mask=batch['attention_mask'])

        total_loss = loss_fn((intents, batch['intents']), (slots, batch['slots']))
        loss_sum += total_loss.item()

        label_intents.extend([lang.id2intent[x] for x in batch['intents'].tolist()])
        predicted_intents.extend([lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()])

        output_slots = torch.argmax(slots, dim=1)
        for id_seq, seq in enumerate(output_slots):
            length = batch['lengths'].tolist()[id_seq]

            gt_ids = batch['slots'][id_seq].tolist()
            utt_ids = batch['utterance'][id_seq][:length].tolist()

            filtered_gt = []
            filtered_pred = []
            filtered_tokens = []

            pred_ids = seq.tolist()

            # WordPiece sub-token alignment: only first sub-token of each word has a real label,
            # additional sub-tokens are padded. We filter to keep only positions with real labels.
            for i, gt_id in enumerate(gt_ids):
                if gt_id != lang.get_pad_index():
                    filtered_gt.append(lang.id2slot[gt_id])
                    filtered_pred.append(lang.id2slot[pred_ids[i]])
                    filtered_tokens.append(tokenizer.decode([utt_ids[i]]))

            # Append the aligned lists for ConLL evaluation
            label_slots.append([(filtered_tokens[i], filtered_gt[i]) for i in range(len(filtered_gt))])
            predicted_slots.append([(filtered_tokens[i], filtered_pred[i]) for i in range(len(filtered_pred))])

    try:
        slots_report = evaluate(label_slots, predicted_slots)
    except KeyError as _:
        slots_report = {"total": {"f": 0}}

    intents_report = classification_report(label_intents, predicted_intents, zero_division=False, output_dict=True)
    return loss_sum / len(data), intents_report, slots_report


def run_experiment(config):
    model_path = f'{config['models_dir']}/nlu_{config['model_name']}.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    train_loader, val_loader, test_loader, lang = build_data_sources(device=device,
                                                                     tokenizer=tokenizer,
                                                                     eval_batch=config['eval_batch'],
                                                                     train_batch=config['train_batch'])

    intent_fn = nn.CrossEntropyLoss()
    slot_fn = nn.CrossEntropyLoss(ignore_index=lang.get_pad_index())

    uncertainty_loss = UncertaintyWeighedLoss(learnable_tasks=2).to(device)

    writer = None

    def loss_fn(i_results, s_results):
        i_loss = intent_fn(i_results[0], i_results[1])
        s_loss = slot_fn(s_results[0], s_results[1])
        return uncertainty_loss([i_loss, s_loss])

    intent_accuracies, slot_f1_scores = [], []

    for run in range(config['runs']):
        print(f'\nStarting RUN {run + 1}/{config["runs"]}')

        if run == 0:
            writer = SummaryWriter(log_dir=f"runs/{config['model_name']}_experiment")

        model = NLUBertModel.from_pretrained(config['model_name'],
                                             freeze_bert=True,
                                             out_dims=(lang.get_intent_len(), lang.get_slot_len())).to(device)
        optimizer = optim.AdamW(list(model.parameters()) + list(uncertainty_loss.parameters()),
                                lr=config['lr'])

        early_stopping = EarlyStopping(patience=config['patience'], mode='max')
        pbar = tqdm(range(config['epochs']))
        for epoch in pbar:
            train_loop(train_loader,
                       model=model,
                       loss_fn=loss_fn,
                       clip=config['clip'],
                       optimizer=optimizer)

            val_loss, val_i_report, val_s_report = eval_loop(val_loader,
                                                             lang=lang,
                                                             model=model,
                                                             loss_fn=loss_fn,
                                                             tokenizer=tokenizer)

            slots_f1_score = val_s_report['total']['f']
            should_stop, counter, is_best = early_stopping(slots_f1_score)
            pbar.set_description(f'Slot F1={slots_f1_score:.4f}, Intent ACC: {val_i_report["accuracy"]:.4f}')

            if writer:
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Metric/Slot_F1', val_s_report['total']['f'], epoch)
                writer.add_scalar('Metric/Intent_Accuracy', val_i_report['accuracy'], epoch)

                loss_weights = uncertainty_loss.get_losses_weights()
                writer.add_scalars('Uncertainty/Weights', {
                    'Intent': loss_weights[0],
                    'Slot': loss_weights[1]
                }, epoch)

            if is_best:
                torch.save(model.state_dict(), model_path)
            elif should_stop:
                print(f'Early stopping triggered at epoch {epoch}')
                break

        model.load_state_dict(torch.load(model_path))

        _, test_i_report, test_s_report = eval_loop(test_loader,
                                                    lang=lang,
                                                    model=model,
                                                    loss_fn=loss_fn,
                                                    tokenizer=tokenizer)

        slot_f1_scores.append(test_s_report['total']['f'])
        intent_accuracies.append(test_i_report['accuracy'])

    slot_f1_scores = np.asarray(slot_f1_scores)
    intent_accuracies = np.asarray(intent_accuracies)

    print('\n')
    print(f"Slot F1: {slot_f1_scores.mean():.4f} ± {slot_f1_scores.std():.4f}")
    print(f"Intent ACC: {intent_accuracies.mean():.4f} ± {intent_accuracies.std():.4f}")