import math


from model import *
from utils import *

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()

    total_loss = 0
    total_tokens = 0

    for batch in data:
        optimizer.zero_grad()
        output = model(batch['source'])

        loss = criterion(output, batch['target'])
        total_loss += loss.item() * batch['number_tokens']

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

def run_experiment(experiment_name, experiment, lr, models_dir='models', runs_dir='runs'):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Experiment config: {experiment}')

    train_ds, val_ds, test_ds, lang = build_data_sources(experiment)
    pad_index = lang.get_pad_index()

    hidden_size = experiment['hidden_size']
    embedding_size = experiment['embedding_size']

    model = LanguageModelLSTM(emb_size=embedding_size,
                              hidden_size=hidden_size,
                              output_size=len(lang.word2id),
                              pad_index=lang.get_pad_index(),
                              n_layers=experiment['n_layers'],
                              emb_dropout=experiment['emb_dropout'],
                              out_dropout=experiment['out_dropout'])
    model.apply(init_weights)
    model.to(device)

    optimizer = build_optimizer(lr=lr,
                                model=model,
                                optimizer_name=experiment['optimizer_name'])
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    best_val_ppl = math.inf
    num_epochs = experiment['epochs']
    early_stopping = EarlyStopping(patience=experiment['patience'], mode='min')

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'{runs_dir}/{experiment_name}')

    pbar = tqdm(range(1, num_epochs))
    for epoch in pbar:
        loss = train_loop(train_ds, optimizer, criterion_train, model, clip=experiment['clip'])

        val_ppl, val_loss = eval_loop(val_ds, criterion_eval, model)

        # Log metrics to TensorBoard
        writer.add_scalar('PPL/val', val_ppl, epoch)

        pbar.set_description(f'Train Loss={loss:.4f}, Val Loss={val_loss:.4f}, Val PPL={val_ppl:.4f}, LR={lr:.3f}')
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), f'{models_dir}/{experiment_name}.pt')

        if early_stopping(val_ppl):
            print(f'Early stopping triggered at epoch {epoch}')
            break

    # Close TensorBoard writer
    writer.close()

    model.load_state_dict(torch.load(f'{models_dir}/{experiment_name}.pt'))

    test_ppl, test_loss = eval_loop(test_ds, criterion_eval, model)
    print(f'Test PPL: {test_ppl:.4f}, Test Loss: {test_loss:.4f}')

    return test_ppl