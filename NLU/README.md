# Natural Language Understanding (NLU) – Run Notes

The NLU assignment is split into two parts (A: BiLSTM-based intent/slot model, B: BERT-based transfer model). Both
scripts save checkpoints in their local `bins/` folders and expect the dataset under `NLU/dataset/`.

## Part A – BiLSTM Intent & Slot Filling

- **Run command**: `main.py [--bid] [--drop]`
- **Arguments**
    - `--bid`: Enable bidirectional LSTM cells (default: uni-directional).
    - `--drop`: Insert a dropout layer with rate `0.3`; omit to disable dropout entirely.
- **Notes**
    - Hyper-parameters are defined inside `main.py` (learning rate `1e-4`, 100 epochs, patience 5, etc.).
    - Toggle arguments independently to reproduce the different architectural ablations requested in the lab.

## Part B – BERT-based Intent & Slot Filling

- **Run command**: `main.py [--model base|large]`
- **Arguments**
    - `--model`: Selects the HuggingFace checkpoint (`bert-base-uncased` or `bert-large-uncased`), defaulting to `base`.
- **Notes**
    - Batch sizes/epochs differ from Part A (`train_batch=64`, `eval_batch=128`) to fit GPU memory of my laptop. However
      I also tested (`train_batch=256`, `eval_batch=512`) on BERT-Large using a Google Colab A100 machine (40GB VRAM)
      without issues.
    - To test other variants (e.g., multilingual BERT), change the `model_name` construction in `main.py` accordingly.
