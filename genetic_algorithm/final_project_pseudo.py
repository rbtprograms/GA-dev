## adapted from Eleanor's pseudocode

#Pseudocode representation of a genetic algorithm for variable selection

#modules needed
import random 
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def initialize_population(population_size, chromosome_length):
    """
    Initialize a population of random chromosomes.

    Parameters:
    - population_size (int): The size of the population.
    - chromosome_length (int): The length of each chromosome.

    Returns:
    - list: A list of random chromosomes.
    """
    population = [generate_random_chromosome(chromosome_length) for _ in range(population_size)]
    return population


def generate_random_chromosome(chromosome_length):
    """
    Generate a random binary chromosome

    Parameters:
    - chromosome_length (int): The length of the chromosome.

    Returns:
    - list: A list representing the random chromosome with 0s and 1s.
    """
    chromosome = np.random.randint(2, size=chromosome_length).tolist()
    return chromosome

def calculate_fitness(chromosome, data, outcome, objective_function="AIC"):
    """
    Calculate the fitness of a chromosome.

    Parameters:
    - chromosome (list of bool): Binary representation of predictors.
    - data (pd.DataFrame): Input data with predictors.
    - outcome (pd.Series): Outcome variable.
    - objective_function (str): "AIC" or "BIC" for the type of fitness to calculate.

    Returns:
    - float: fitness value.
    """
    # Select the predictors according to the chromosome
    predictors = data.iloc[:, 1:]
    selected_predictors = predictors.loc[:, chromosome]
    selected_predictors = sm.add_constant(selected_predictors)
    
    # Convert outcome to a NumPy array
    outcome_array = np.asarray(outcome)

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

def rank(scores):
        """
        Return the numeric ranking based on the highest absolute value.
        """
        sorted_indices = sorted(range(len(scores)), key=lambda k: abs(scores[k]))
        return [i + 1 for i in sorted_indices]


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



def select_parents(population):
	# implementing tournament selection
	# one parent selected with probability proportional to its fitness
	# other parent selected randomly

	fitness_scores = [calculate_fitness(chromosome, population_size, data, outcome) for chromosome in population]  #weights to be used for parent selection
	parent_indices = range(len(population))
	parent1_index = random.choices(parent_indices, weights = fitness_scores, k=1) #choosing 1 parent with probability based on fitness values
	parent1 = population[parent1_index]
	parent2 = random.choices(population, k=1) #randomly choosing other parent 

	return parent1, parent2


def crossover(parent1, parent2, population_size):
	#We should choose how many crossover events per offspring we want, defaulting to 1 for now
	# produces a single offspring

	recombination_points = range(population_size - 1) #can't recombine at ends of chromosomes
	recomb_location_off1 = random.choices(recombination_points, replace=True)

	offspring_1 = parent1[:recomb_location_off1] + parent2[recomb_location_off1:]

	return offspring


def mutate(chromosome, mutation_rate=0.01):

	# mutation == True with probability 0.01, False otherwise
	mutation = [random.random() < mutation_rate for i in range(len(chromosome))]

	# '0' to '1' mutation or '1' to '0' mutation if mutation == True
	new_chromsome = [abs(chromosome[i] - 1) if mutation[i] else chromosome[i] for i in range(len(chromosome))]

	return chromosome


def genetic_algorithm(population_size=20, chromosome_length=27, generations=100, mutation_rate=0.01, data, outcome):

	population = initialize_population(population_size, chromosome_length)

	for g in range(generations): #main iteration
		rank_fitness_scores = calculate_rank_based_fitness(population, chromosome, population_size, data, outcome)

		new_population = []
		for i in range(population_size // 2):
			parent1, parent2 = select_parents(population) #should this be done with or without replacement?
			
			#children are generated from indepdendent crossover events
			child1 = crossover(parent1, parent2)
			child2 = crossover(parent1, parent2)

			#mutation
			child1 = mutate(child1, mutation_rate)
			child2 = mutate(child2, mutation_rate)

			new_population.add(child1, child2)

		population = new_population #non-overlapping generations

		most_fit_individual = population[max(rank_fitness_scores)]

	return most_fit_individual

























