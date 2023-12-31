import random
from .adjust_chromosome import adjust_chromosome

def mutate(chromosome, mutation_rate, max_features):
    """
    Performs chromosome 'mutations' ie. changes states from 0/1 to 1/0 at a set rate. Must be 
    run AFTER crossover() to ensure max number of features is not exceeded.

    Parameters:
    - chromosome (list): Binary representation of predictors.
    - mutation_rate (float): rate at which mutations should be introduced, should have value between 0 and 1.
    - max_features (int): The maximum number of features. Must be a positive integer.

    Returns:
    - new_chromosome (list): New potentially mutated chromosome.
    
    """

    # mutation == True with probability 0.01, False otherwise
    mutation = [random.random() < mutation_rate for i in range(len(chromosome))]

    # '0' to '1' mutation or '1' to '0' mutation if mutation == True
    new_chromosome = [abs(chromosome[i] - 1) if mutation[i] else chromosome[i] for i in range(len(chromosome))]

    #this checks for too many features gained from both crossover and mutation
    new_chromosome = adjust_chromosome(new_chromosome, max_features)

    return new_chromosome