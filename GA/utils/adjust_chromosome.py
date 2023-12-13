import random

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

    current_sum = sum(chromosome)

    if current_sum <= max_features:
        return chromosome

    ones_indices = [i for i, value in enumerate(chromosome) if value == 1]
    indices_to_change = random.sample(ones_indices, current_sum - max_features)
    
    for index in indices_to_change:
        chromosome[index] = 0

    return chromosome