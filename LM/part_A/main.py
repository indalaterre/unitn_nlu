# This file is used to run your functions and print the results
# Please write your functions or classes in the functions.py

# Import everything from functions.py file
import os
import math

from functions import run_experiment, get_experiment_config

def main(experiments):
    models_dir = 'bins'
    os.makedirs(models_dir, exist_ok=True)

    best_test_ppl = math.inf

    for experiment in experiments:
        print('\n\n')
        experiment_config = get_experiment_config(experiment)
        lr_array = experiment_config['lr']
        for lr in lr_array:
          print(f'Running experiment {experiment} with LR={lr}')
          test_ppl = run_experiment(experiment, experiment_config, lr, models_dir)
          print(f'TestPPL for LR {lr} is {test_ppl}')
          if test_ppl < best_test_ppl:
              best_test_ppl = test_ppl
              print(f'New best test PPL: {best_test_ppl:.4f}')

main(['vanilla', 'dropout', 'adamw'])