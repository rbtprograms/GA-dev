from .calculate_fitness import calculate_fitness

def select_tournament_winners(population, data, outcome_index, objective_function="AIC", log_outcome=True):
    """
    Implement tournament selection. 'Winner' parent selected as highest fitness score, 
    other parent selected randomly but proportional to fitness score (higher scores more heavily weighted).

    Parameters:
    - population (list): Subset of total population, list of chromosomes.
    - data (pd.DataFrame): Input data with predictors.
    - outcome_index (int): Index of outcome column (0 index).
    - objective_function (string): Default is AIC, other options are "BIC", "Adjusted R-squared, "Deviance", "MSE", 
    "Mallows CP".
    - log_outcome (bool): True if the outcome variable is on the log scale, False else.


    Returns:
    - winner (list): chromosome of top 'winner' per subgroup.
    - losers (list): list of chromosomes of non-winners from subgroup.
    """

    winners = []
    indexes_to_remove = []
    losers = []

    fitness_scores = [calculate_fitness(chromosome, data, outcome_index, objective_function, log_outcome) for chromosome in population]  #weights to be used for parent selection
   
   #stores indexes of original popualtion, in descending/ascending order of fitness 
   # (ascending for AIC/BIC, deviance, MSE, Mallows CP)
    if objective_function == "BIC": 
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=False)
    elif objective_function == "Adjusted R-squared":
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)
    elif objective_function == "Deviance":
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=False)
    elif objective_function == "MSE":
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=False)
    elif objective_function == "Mallows CP":
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=False)
    else: #objective function is AIC by default
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=False)


    winner = population[sorted_indices[0]] #one winner per subgroup
        #keep track of winners to remove from rest of breeding pool
    for i in range(1, len(population)):
        indexes_to_remove.append(sorted_indices[i])
    
    #keeping track of losing individuals for random mating later on
    for i in indexes_to_remove:
        losers.append(population[i])

    return winner, losers