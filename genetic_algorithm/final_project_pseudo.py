# packages needed
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas as pd
import os
import random

def initialize_population(population_size, chromosome_length, max_features):
    """
    Initialize a population of random chromosomes.

    Parameters:
    - population_size (int): The size of the population. Must be a positive integer.
    - chromosome_length (int): The length of each chromosome. Must be a positive integer.
    - max_features (int): The maximum number of features. Must be a positive integer.

    Returns:
    - list: A list of random chromosomes.
    """
    assert isinstance(population_size, int) and population_size > 0, "population_size must be a positive integer"
    assert isinstance(chromosome_length, int) and chromosome_length > 0, "chromosome_length must be a positive integer"
    assert isinstance(max_features, int) and max_features > 0, "max_features must be a positive integer"

    population = [generate_random_chromosome(chromosome_length, max_features) for _ in range(population_size)]

    return population

def adjust_chromosome(chromosome, max_features):
    """
    Adjust the number of features in a chromosome to match the specified max_features.

    Parameters:
    - chromosome (list): Binary list of 0s and 1s representing features.
    - max_features (int): The desired maximum number of features. Must be a positive integer.

    Returns:
    - list: Adjusted chromosome with the specified max_features.
    """
    assert all(bit in {0, 1} for bit in chromosome), "Chromosome must be a binary list of 0s and 1s"
    assert isinstance(max_features, int) and max_features > 0, "max_features must be a positive integer"

    current_sum = sum(chromosome)

    if current_sum <= max_features:
        return chromosome

    ones_indices = [i for i, value in enumerate(chromosome) if value == 1]
    indices_to_change = random.sample(ones_indices, current_sum - max_features)
    
    for index in indices_to_change:
        chromosome[index] = 0

    return chromosome



def generate_random_chromosome(chromosome_length, max_features):
    """
    Generate a random binary chromosome

    Parameters: 
    - chromosome_length (int): The length of the chromosome.
    - max_features (int): The maximum number of features. Must be a positive integer.

    Returns:
    - list: A list representing the random chromosome with 0s and 1s.
    """
    assert isinstance(chromosome_length, int) and chromosome_length > 0, "chromosome_length must be a positive integer"
    assert isinstance(max_features, int) and max_features > 0, "max_features must be a positive integer"
    
    chromosome = np.random.randint(2, size=chromosome_length).tolist()
    #ensure that the new chromsome doesn't exceed the max allowable features
    chromosome = adjust_chromosome(chromosome, max_features)

    return chromosome

def calculate_fitness(chromosome, data, outcome_index, objective_function="AIC", log_outcome=True):
    """
    Calculate the fitness of a chromosome.

    Parameters:
    - chromosome (list of bool): Binary representation of predictors.
    - data (pd.DataFrame): Input data with predictors.
    - outcome_index (int): Index of outcome column (0 index).
    - objective_function (str): Default is AIC, other options are "BIC", "Adjusted R-squared, "Deviance", "MSE", 
    "Mallows CP". 
    - log_outcome (bool): True if the outcome variable is on the log scale, False else.

    Returns:
    - float: fitness value.
    """

    if log_outcome == True:
        outcome = pd.Series(np.log(data[outcome_index].astype(float)))
        outcome_array = np.asarray(outcome)
    else:
        outcome = pd.Series(data[outcome_index].astype(float))
        outcome_array = np.asarray(outcome)
        
    # Select the predictors according to the chromosome
    if outcome_index > 0:
        predictors_1 = data.iloc[:, :outcome_index]
        predictors = pd.concat([predictors_1, data.iloc[:, outcome_index+1:]], axis=1)
    else:
        predictors = data.iloc[:, outcome_index+1:]
        
    selected_predictors = predictors.loc[:, chromosome]
    selected_predictors = sm.add_constant(selected_predictors)

    # Convert selected predictors to a NumPy array
    predictors_array = np.asarray(selected_predictors.astype(float))

    # Fit linear regression model
    model = sm.OLS(outcome_array, predictors_array).fit()

    # Calculate objective function inputs
    rss = model.ssr
    tss = np.sum((outcome - np.mean(outcome))**2)
    s = np.sum(chromosome)  # Number of predictors
    k = s + 2  # Number of parameters including intercept
    n = len(outcome) 
    mse = np.mean((outcome - model.predict(predictors_array))**2)
    
    # Return fitness functions based on objective function input
    if objective_function == "BIC":
        return n * np.log(rss / n) + k * np.log(n)
    elif objective_function == "Adjusted R-squared":
        r_squared = 1 - (rss / tss)
        return 1 - (1 - (r_squared)) * (n - 1) / (n - s - 1)
    elif objective_function == "Deviance":
        return 2 * model.llf
    elif objective_function == "MSE":
        return mse
    elif objective_function == "Mallows CP":
        return rss + 2 * s * mse / (n - s)
    else:
        # default is AIC
        return n * np.log(rss / n) + 2 * k

# ignore this function for now
def rank(scores):
    """
    Return the numeric ranking based on the highest absolute value.
    """
    sorted_indices = sorted(range(len(scores)), key=lambda k: abs(scores[k]))
    return [i + 1 for i in sorted_indices]

# ignore this function as well
 def calculate_rank_based_fitness(data, outcome, population, population_size, objective_function="AIC"):
	"""
    Calculate rank-based fitness scores for a population based on objective function.

    Parameters:
    - data (pd.DataFrame): Input data with predictors.
    - outcome (pd.Series): Outcome variable.
    - population_size (int): The size of the population.
    - population (list): List of chromosomes

    Returns:
    - list of float: Rank-based fitness scores for each chromosome in the population.
    """
	fitness_scores = [calculate_fitness(chromosome, data, outcome, objective_function="AIC") for chromosome in population]
	fitness_ranks = rank(fitness_scores) 
	p = population_size
	return [(2*r)/(p*(p+1)) for r in fitness_ranks]

def select_tournament_winners(population, winners_per_subgroup, data, outcome_index, objective_function="AIC", log_outcome=True):
	# implementing tournament selection
	# one parent selected with probability proportional to its fitness
	# other parent selected randomly

    winners = []
    indexes_to_remove = []
    losers = []
    #tournament is a fitness evaluation; winners_per_subgroup is the 
    # number of allowable winners moving onto the next generation
    fitness_scores = [calculate_fitness(chromosome, data, outcome_index, objective_function, log_outcome) for chromosome in population]  #weights to be used for parent selection
    #stores indexes of original popualtion, in descending order of fitness
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)
    
    for i in range(winners_per_subgroup):
        winners.append(population[sorted_indices[i]])
        #keep track of winners to remove from rest of breeding pool
    for i in range(winners_per_subgroup, len(population)):
        indexes_to_remove.append(sorted_indices[i])
    

    #keeping track of losing individuals for random mating later on
    for i in indexes_to_remove:
        losers.append(population[i])

    return winners, losers

def crossover(parent1, parent2, population_size):
	#We should choose how many crossover events per offspring we want, defaulting to 1 for now
	# produces a single offspring

	recombination_points = range(population_size - 1) #can't recombine at ends of chromosomes
	recomb_location_off1 = np.random.choice(recombination_points, replace=True)

	offspring = parent1[:recomb_location_off1] + parent2[recomb_location_off1:]
    
	return offspring

def mutate(chromosome, mutation_rate, max_features):
    ##This function can only be run after crossover() to ensure max number of features is not exceeded

    # mutation == True with probability 0.01, False otherwise
    mutation = [random.random() < mutation_rate for i in range(len(chromosome))]

    # '0' to '1' mutation or '1' to '0' mutation if mutation == True
    new_chromosome = [abs(chromosome[i] - 1) if mutation[i] else chromosome[i] for i in range(len(chromosome))]

    #ensure chromosome doesn't exceed max_features
    #this checks for too many features gained from both crossover and mutation
    new_chromosome = adjust_chromosome(new_chromosome, max_features)

    return new_chromosome

def genetic_algorithm(data,population_size=20, chromosome_length=27, generations=100, mutation_rate=0.01, max_features=10, 
                      outcome_index=0, objective_function="AIC", log_outcome=True):

    population = initialize_population(population_size, chromosome_length, max_features)
    generation_data = Generation_Container()
    max_score_generation = []

    for g in range(generations): #main iteration
        
        new_population = []
        pop_subgroups = []
        set_size = 5 #arbitrary starting value 
        num_sets = len(population) / set_size
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
            winners, losers = select_tournament_winners(group, 1, data, outcome_index, objective_function, log_outcome) #selecting 1 winner per subgroup
            for winner in winners:
                all_winners.append(winner)
            for loser in losers:
                all_losers.append(loser)

        #parent selection and child generation
        for ind in all_winners:
            parent1 = ind
            parent2 = all_losers.pop(random.randrange(len(all_losers))) #removes parent2 from loser list and returns selected parent2
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
            
        if not new_population:  # Check if new_population is empty
            break
        
        #reproduction between randomly selected leftover individuals 
        for _ in range(population_size - len(new_population)): #how many more children we need to add from randoms to maintain pop size
            #sampled with replacement to avoid running out of new parents 
            parent1 = random.sample(all_losers, 1)[0]
            parent2 = random.sample(all_losers, 1)[0]
            child1 = crossover(parent1, parent2, population_size)
            #mutation
            child1 = mutate(child1, mutation_rate, max_features)
            new_population.append(child1)

        max_score_generation.append(max(calculate_fitness(individual, data, outcome_index, objective_function, log_outcome) for individual in new_population))
        
        # terminate if the max fitness score in a generation converges
        if g == 0:
            None
        elif abs(max_score_generation[g] - max_score_generation[g-1]) < 1e-15:
            break
        
        #check the difference across generations for an exit condition
        
        population = new_population #non-overlapping generations
        
    #list of lists of chromosome & fitness score in population
    individual_scores = [[individual,calculate_fitness(individual, data, outcome_index, objective_function, log_outcome)] for individual in population]
        
    #save generation data
    generation_data.add_generation_data([individual[1] for individual in individual_scores], [score[0] for score in individual_scores])
        
    #find the index of the fittest individual
    most_fit_index = max(range(len(individual_scores)), key=lambda i: individual_scores[i][0])
    most_fit_individual = individual_scores[most_fit_index]
        
    #find the corresponding fields from the fittest individual 
    chromosome = most_fit_individual[0]
    chromosome.insert(outcome_index, 0)
    column_names = data.columns.tolist()
    fields = column_names[0].split()
    selected_fields = [fields[i] for i, value in enumerate(chromosome) if value == 1]

    return chromosome, selected_fields, most_fit_individual[1]


class Generation_Container:
    def __init__(self):
        self._generation_data = {}

    def add_generation_data(self, score, individual):   
        curr_generation = len(self._generation_data.keys()) + 1
        data = {
            'score': score,
            'individual': individual
        }
        self._generation_data[curr_generation] = data

    def check_last_generations(self, distance_back):
        target = max(0, len(self._generation_data) - distance_back)
        for key in sorted(self._generation_data.keys())[target:]:
            print(f'Generation {key} yielded: {self._generation_data[key]}')


# test on baseball data

# read in baseball data 
current_dir = os.getcwd()
data_folder_path = os.path.join(current_dir, 'GA-dev/genetic_algorithm/data')
file_path = os.path.join(data_folder_path, 'baseball.dat')
data = pd.read_csv(file_path)

import time
start = time.time()
genetic_algorithm(data,population_size=20, chromosome_length=28, generations=100, mutation_rate=0.01, max_features=10, 
                      outcome_index=0, objective_function="AIC", log_outcome=True)
time.time()-start












