import numpy as np
from .adjust_chromosome import adjust_chromosome

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
    assert max_features <= chromosome_length, "max_features must not exceed number of features in dataset"
    
    chromosome = np.random.randint(2, size=chromosome_length).tolist()
    #ensure that the new chromsome doesn't exceed the max allowable features
    chromosome = adjust_chromosome(chromosome, max_features)

    return chromosome

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
    assert max_features <= chromosome_length, "max_features must not exceed number of features in dataset"

    population = [generate_random_chromosome(chromosome_length, max_features) for _ in range(population_size)]

    return population