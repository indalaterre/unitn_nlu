# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from transformers import AutoTokenizer

from model import NLUBertModel
from utils import build_data_sources, EarlyStopping

from conll import evaluate
from sklearn.metrics import classification_report

def calculate_cumulative_loss(intent_loss, slot_loss):
    # TODO: Let's evaluate Uncertainty Weighting
    return intent_loss + slot_loss

def train_loop(data, model, optimizer, intent_fn, slot_fn, clip=5):

    model.train()

    loss_sum = 0
    for batch in data:
        optimizer.zero_grad()

        slots, intents = model(batch['utterance'], batch['attention_mask'], batch['token_type_ids'])

        slots_loss = slot_fn(slots, batch['slots'])
        intent_loss = intent_fn(intents, batch['intents'])

        total_loss = calculate_cumulative_loss(intent_loss, slots_loss)
        loss_sum += total_loss.item()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss_sum / len(data)

@torch.no_grad()
def eval_loop(data, model, intent_fn, slot_fn, tokenizer, lang):
    model.eval()

    label_intents, label_slots = [], []
    predicted_intents, predicted_slots = [], []

    loss_sum = 0
    for batch in data:
        slots, intents = model(batch['utterance'], batch['attention_mask'], batch['token_type_ids'])

        slots_loss = slot_fn(slots, batch['slots'])
        intent_loss = intent_fn(intents, batch['intents'])

        total_loss = calculate_cumulative_loss(intent_loss, slots_loss)
        loss_sum += total_loss.item()

        label_intents.extend([lang.id2intent[x] for x in batch['intents'].tolist()])
        predicted_intents.extend([lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()])

        output_slots = torch.argmax(slots, dim=1)
        for id_seq, seq in enumerate(output_slots):
            length = batch['lengths'].tolist()[id_seq]

            gt_ids = batch['slots'][id_seq].tolist()
            gt_slots = [lang.id2slot[elem] for elem in gt_ids]

            utt_ids = batch['utterance'][id_seq][:length].tolist()
            utterance = tokenizer.convert_ids_to_tokens(utt_ids)

            filtered_gt = []
            filtered_pred = []
            filtered_tokens = []

            pred_ids = seq.tolist()

            for i, gt_id in enumerate(gt_ids):
                # Only keep positions where the Gold Label is NOT pad
                if gt_id != lang.get_pad_index():
                    filtered_gt.append(lang.id2slot[gt_id])
                    filtered_pred.append(lang.id2slot[pred_ids[i]])
                    # Decode the single token ID to string for reference
                    filtered_tokens.append(tokenizer.decode([utt_ids[i]]))

            # Append the aligned lists for ConLL evaluation
            label_slots.append([(filtered_tokens[i], filtered_gt[i]) for i in range(len(filtered_gt))])
            predicted_slots.append([(filtered_tokens[i], filtered_pred[i]) for i in range(len(filtered_pred))])

    try:
        slots_report = evaluate(label_slots, predicted_slots)
    except KeyError as ex:
        slots_report = {"total":{"f":0}}

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

    intent_accuracies, slot_f1_scores = [], []

    for run in range(config['runs']):
        print(f'\nStarting RUN {run + 1}/{config["runs"]}')

        model = NLUBertModel.from_pretrained(config['model_name'],
                                             out_dims=(lang.get_intent_len(), lang.get_slot_len())).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

        early_stopping = EarlyStopping(patience=config['patience'], mode='max')

        pbar = tqdm(range(config['epochs']))
        for epoch in pbar:
            train_loop(train_loader, model, optimizer, intent_fn, slot_fn, config['clip'])

            val_loss, val_i_report, val_s_report = eval_loop(val_loader,
                                                             model,
                                                             intent_fn,
                                                             slot_fn,
                                                             tokenizer,
                                                             lang)

            slots_f1_score = val_s_report['total']['f']
            should_stop, counter, is_best = early_stopping(slots_f1_score)
            pbar.set_description(f'Slot F1={slots_f1_score:.4f}, Intent ACC: {val_i_report["accuracy"]:.4f}')

            if is_best:
                torch.save(model.state_dict(), model_path)
            elif should_stop:
                print(f'Early stopping triggered at epoch {epoch}')
                break

        model.load_state_dict(torch.load(model_path))

        _, test_i_report, test_s_report = eval_loop(test_loader,
                                                    model,
                                                    intent_fn,
                                                    slot_fn,
                                                    tokenizer,
                                                    lang)

        slot_f1_scores.append(test_s_report['total']['f'])
        intent_accuracies.append(test_i_report['accuracy'])


