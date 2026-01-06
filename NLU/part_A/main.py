# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
import os
import argparse

from functions import run_experiment

if __name__ == "__main__":
    # Write the code to load the datasets and to run your functions
    # Print the results

    models_dir = 'bins'
    os.makedirs(models_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Intent and Slot Filling Task")
    parser.add_argument('--bid', action='store_true', help='Optional flag to add bidirectionality')
    parser.add_argument('--drop', action='store_true', help='Optional flag to add dropout layer')
    args = parser.parse_args()

    config = {
        'lr': .0001,
        'runs': 5,
        'clip': 5,
        'pad_idx': 0,
        'epochs': 100,
        'patience': 5,
        'emb_dim': 300,
        'hidden_dim': 200,
        'eval_batch': 256,
        'train_batch': 128,
        'models_dir': 'bins',
        'dropout': 0.3 if args.drop else -1,
        'use_bidirectional': bool(args.bid)
    }

    if args.drop:
        print(f'Applying DROPOUT: {config['dropout']:.2f}')
    if args.bid:
        print('Applying Bi-Directional LSTM cells')

    run_experiment(config=config)