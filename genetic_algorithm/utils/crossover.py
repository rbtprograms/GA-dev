import numpy as np

def crossover(parent1, parent2, population_size):
    """
    Generates a new offspring through a crossover event between parent chromosomes.

    Parameters: 
    - parent1, parent2 (list): chromosome of parent 1/2
    - population_size (int): The size of the population.

    Returns:
    - offspring (list): chromosome of the offspring.

    """
	#We should choose how many crossover events per offspring we want, defaulting to 1 for now
	# produces a single offspring

    recombination_points = range(population_size - 1) #can't recombine at ends of chromosomes
    recomb_location_off1 = np.random.choice(recombination_points, replace=True)

    offspring = parent1[:recomb_location_off1] + parent2[recomb_location_off1:]

    return offspring