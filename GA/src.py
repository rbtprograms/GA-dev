# packages needed
import pandas as pd
import os
import random
from .utils import *

def select(data,chromosome_length,outcome_index,population_size=20, generations=100, num_sets=5, mutation_rate=0.01, max_features=20, 
                      objective_function="AIC", log_outcome=True, regression_type="OLS", output_all_generation_data=True, plot_all_generation_data=False, 
                      with_elitism=False, plot_output_path='', with_progress_bar=False):
    """
    Execute a genetic algorithm for feature selection and model optimization.

    Parameters:
    - data (pd.DataFrame): Clean data set with float/int variables, no missing values, sufficient dimensions.
    - population_size (int): The size of the population. Default is 20.
    - chromosome_length (int): The length of each chromosome. 
    - generations (int): The number of generations to run the algorithm. Default is 100.
    - num_sets (int): The number of subgroups(winners).
    - mutation_rate (float): The rate at which mutations should be introduced. Should be a value between 0 and 1. Default is 0.01.
    - max_features (int): The maximum number of features. Must be a positive integer.
    - outcome_index (int): Index of the outcome column (0 index). 
    - objective_function (str): The objective function to optimize. Options are "AIC", "BIC", "Adjusted R-squared",
                               "MSE", "Mallows CP". Default is "AIC".
    - log_outcome (bool): True if the outcome variable is on the log scale, False otherwise. Default is True.
    - regression_type (str): The type of regression model. Options are "OLS", "Ridge", "Lasso". Default is "OLS".
    - output_all_generation_data (bool): Flag for if user wants generation data outputted to chromosomes.txt and scores.txt. Will print the most fit individual and their associated score from each generation.
    - plot_all_generation_data (bool): Flag for if user wants generation data plotted. Will generate a plot the scores of all individuals from each generation.
    - with_elitism (bool): Will run the algorithm implementing elitism. The highest scoring individual from each generation will be preserved into the next and will continue competing.
    - plotoutput_path (str): Absolute path for where the generational scoring charts should be generated.
    - with_progress_bar (bool): Flag for if user wants a progress bar. Will display progress for generations.

    Returns:
    - None: The function performs the genetic algorithm for feature selection and model optimization,
            and prints the highest-scoring individual from each generation. It also plots AIC values
            from each generation.

    Example:
    ```python
    select(data=my_data, population_size=20, chromosome_length=15, generations=50, mutation_rate=0.02, max_features=8, outcome_index=0)
    ```
    """
    assert isinstance(population_size, int) and population_size > 0, "population_size must be a positive integer"
    assert isinstance(chromosome_length, int) and chromosome_length > 0, "chromosome_length must be a positive integer"
    # assert data.map(lambda x: isinstance(x, (int, float))).all().all(), "Data must contain only floats or integers."
    assert population_size % num_sets == 0, "Number of subgroups must be a multiple of population size"
    assert num_sets <= 0.25*population_size, "Number of subgroups (winners) cannot exceed 0.25 * population size"
    assert isinstance(max_features, int) and max_features > 0, "max_features must be a positive integer"
    assert max_features <= chromosome_length, "max_features must not exceed number of features in dataset"
    assert isinstance(mutation_rate, float) and mutation_rate >= 0 and mutation_rate <= 1, "mutation_rate must be of type(float) and must take a value between 0 and 1 (inclusive)."
    assert data.shape[1] >= 2, "Data must have at least 2 columns."
    assert data.shape[0] >= 20, "Data must have at least 20 rows."
    assert not data.isnull().any().any(), "Data must not contain missing values."
    assert isinstance(outcome_index, int) and outcome_index >= 0, "Outcome index must be a non-negative integer"
    assert objective_function in ["AIC", "BIC", "Adjusted R-squared", "MSE", "Mallows CP"], \
    "Objective function options are limited to: 'AIC', 'BIC', 'Adjusted R-squared', 'MSE', 'Mallows CP.' If not from list, default calculations are AIC."
    assert isinstance(log_outcome, bool), "log_outcome must be either True or False"
    assert regression_type in ["OLS","Ridge","Lasso"], "Regression type options are limited to: 'OLS','Ridge','Lasso'. If not from list, default calculations are OLS."

    population = initialize_population(population_size, chromosome_length, max_features)
    generation_data = Generation_Container()
    scores_data = Generation_Scores()

    for g in range(generations): #main iteration
        
        new_population = []
        pop_subgroups = []
        set_size = int(len(population) / num_sets)
        population_copy = population[:] #make a copy to keep track of removed individuals

        #random population subgroups for tournament selection
        for _ in range(int(num_sets)):
            rand_sample = random.sample(population_copy, set_size)
            pop_subgroups.append(rand_sample)

            # remove sampled individuals from original parent population
            # population_copy = [element for element in population_copy if element not in rand_sample]

        #tournament selection- choose top n individuals from each subpopulation to 
        #become parents, partnered with a random individual
        all_winners = []
        all_losers = []
        for group in pop_subgroups:
            winner, losers = select_tournament_winners(group, 
                                                        data, 
                                                        outcome_index, 
                                                        objective_function, 
                                                        log_outcome) #selecting 1 winner per subgroup
            all_winners.append(winner)
            for loser in losers:
                all_losers.append(loser)

        if with_elitism:
            winner_data = Generation_Scores()
            winner_scores = [[individual,calculate_fitness(individual, data, outcome_index, objective_function, log_outcome, regression_type)] for individual in all_winners]
            scores_data.add_scores(winner_scores)
            gen_max_winner = max(winner_scores, key=lambda x: abs(x[1]))
            new_population.append(gen_max_winner[0])

        #parent selection and child generation
        while len(all_winners) > 1:
            parent2_winner = random.uniform(0,1) < 0.8
            if parent2_winner: #winner mate selected

                parent1_ind = random.sample(list(enumerate(all_winners)), 1)[0]
                parent1 = parent1_ind[1]
                all_winners.pop(parent1_ind[0])

                parent2_ind = random.sample(list(enumerate(all_winners)), 1)[0]
                parent2 = parent2_ind[1]
                all_winners.pop(parent2_ind[0])

                child1 = crossover(parent1, parent2,population_size)
                child2 = crossover(parent1, parent2,population_size)
                child3 = crossover(parent1, parent2,population_size)

                #mutation
                child1 = mutate(child1, mutation_rate, max_features)
                child2 = mutate(child2, mutation_rate, max_features)
                child3 = mutate(child3, mutation_rate, max_features)

                new_population.append(child1)
                new_population.append(child2)
                new_population.append(child3)


            else: #loser mate selected
                parent1_ind = random.sample(list(enumerate(all_winners)), 1)[0]
                parent1 = parent1_ind[1]
                all_winners.pop(parent1_ind[0])

                parent2_ind = random.sample(list(enumerate(all_losers)), 1)[0]
                parent2 = parent2_ind[1]
                all_losers.pop(parent2_ind[0])

                child1 = crossover(parent1, parent2,population_size)
                child2 = crossover(parent1, parent2,population_size)
                child3 = crossover(parent1, parent2,population_size)

                #mutation
                child1 = mutate(child1, mutation_rate, max_features)
                child2 = mutate(child2, mutation_rate, max_features)
                child3 = mutate(child3, mutation_rate, max_features)

                new_population.append(child1)
                new_population.append(child2)
                new_population.append(child3)

        #now all_winners has either 0 or 1 elements
        if len(all_winners) > 0: #1 winner left
            parent1 = all_winners[0]
            parent2_ind = random.sample(list(enumerate(all_losers)), 1)[0]
            parent2 = parent2_ind[1]
            all_losers.pop(parent2_ind[0])

            child1 = crossover(parent1, parent2,population_size)
            child2 = crossover(parent1, parent2,population_size)
            child3 = crossover(parent1, parent2,population_size)

            #mutation
            child1 = mutate(child1, mutation_rate, max_features)
            child2 = mutate(child2, mutation_rate, max_features)
            child3 = mutate(child3, mutation_rate, max_features)

            new_population.append(child1)
            new_population.append(child2)
            new_population.append(child3)
        
        #reproduction between randomly selected leftover individuals 
        for _ in range(population_size - len(new_population)): #how many more children we need to add from randoms to maintain pop size
            #sampled with replacement to avoid running out of new parents 
            parent1 = random.sample(all_losers, 1)[0]
            parent2 = random.sample(all_losers, 1)[0]
            child1 = crossover(parent1, parent2, population_size)
            #mutation
            child1 = mutate(child1, mutation_rate, max_features)
            new_population.append(child1)

        gen_scores = [[individual,calculate_fitness(individual, data, outcome_index, objective_function, log_outcome, regression_type)] for individual in new_population]
        scores_data.add_scores(gen_scores)
        gen_max_individual_data = max(gen_scores, key=lambda x: abs(x[1]))

        generation_data.add_generation_data(gen_max_individual_data[1], gen_max_individual_data[0])
    
        if with_progress_bar:
            progress_bar(curr_generation=g, generations=generations, generation_data=generation_data)
        
        # terminate if the max fitness score in a generation converges
        if g == 0:
            None
        if g==100:
            break
        
        population = new_population #non-overlapping generations
        
    if output_all_generation_data:
        generation_data.show_all_generations()
    
    if plot_all_generation_data:
        try:
            scores_data.plot_scores(plot_output_path, objective_function)
        except:
            print("problem creating chart, most likely an issue with the path provided. see documentation for proper usage")

    final_chrom = generation_data.get_most_recent_individual()

    if outcome_index > 0:
        predictors_1 = data.iloc[:, :outcome_index]
        predictors = pd.concat([predictors_1, data.iloc[:, outcome_index+1:]], axis=1)
    else:
        predictors = data.iloc[:, outcome_index+1:]
    
    assert len(predictors.columns) == len(final_chrom), "Predictor dataframe different dimension than final chromosome. Exiting."
    
    top_predictors=[predictors.columns[i] for i in range(len(predictors.columns)) if bool(final_chrom[i])]

    return top_predictors
