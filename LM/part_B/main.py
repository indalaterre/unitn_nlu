# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file

import os
import math

from functions import run_experiment

if __name__ == "__main__":
    # Write the code to load the datasets and to run your functions
    # Print the results

    models_dir = 'bins'
    os.makedirs(models_dir, exist_ok=True)

    best_test_ppl = math.inf
    experiments = ['weight_tying', 'var_dropout', 'nt_avg_sgd']

    for experiment in experiments:
        test_ppl = run_experiment(experiment, models_dir)
        if test_ppl < best_test_ppl:
            best_test_ppl = test_ppl
            print(f'New best test PPL: {best_test_ppl:.4f}')
