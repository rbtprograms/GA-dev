import pandas as pd
import os
from GA import select

current_dir = os.getcwd()
data_folder_path = os.path.join(current_dir, 'examples/data')
file_path = os.path.join(data_folder_path, 'baseball.dat')
data = pd.read_csv(file_path, delimiter=' ')

print(select(data, population_size=20, chromosome_length=27, generations=100, mutation_rate=0.01, max_features=27, 
                      outcome_index=0, objective_function="AIC", log_outcome=True, print_all_generation_data=True,
                      with_elitism=True, plot_all_generation_data=True,plot_output_path='/Users/robertthompson/code/robert-thompson/GA-dev/examples',
                      with_progress_bar=True))

