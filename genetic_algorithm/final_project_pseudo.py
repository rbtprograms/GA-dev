# packages needed
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
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
    assert max_features <= chromosome_length, "max_features must not exceed number of features in dataset"

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
    assert max_features <= len(chromosome), "max_features must not exceed number of features in dataset"

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
    assert max_features <= chromosome_length, "max_features must not exceed number of features in dataset"
    
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
    assert all(bit in {0, 1} for bit in chromosome), "Chromosome must be a binary list of 0s and 1s"
    assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
    assert isinstance(outcome_index, int) and outcome_index >= 0, "Outcome index must be a non-negative integer"
    assert objective_function in ["AIC", "BIC", "Adjusted R-squared", "Deviance", "MSE", "Mallows CP"], \
    "Objective function options are limited to: 'AIC', 'BIC', 'Adjusted R-squared', 'Deviance', 'MSE', 'Mallows CP.' If not from list, default calculations are AIC."
    assert isinstance(log_outcome, bool), "log_outcome must be either True or False"

    if log_outcome == True:
        outcome = pd.Series(np.log(data.iloc[:, outcome_index].astype(float)))
        outcome_array = np.asarray(outcome)
    else:
        outcome = pd.Series(data.iloc[:, outcome_index].astype(float))
        outcome_array = np.asarray(outcome)
        
    # Select the predictors according to the chromosome
    if outcome_index > 0:
        predictors_1 = data.iloc[:, :outcome_index]
        predictors = pd.concat([predictors_1, data.iloc[:, outcome_index+1:]], axis=1)
    else:
        predictors = data.iloc[:, outcome_index+1:]
    
    selected_predictors = predictors.loc[:, [bool(x) for x in chromosome]]
    selected_predictors = sm.add_constant(selected_predictors)

    # Convert selected predictors to a NumPy array
    predictors_array = np.asarray(selected_predictors.astype(float))

    # Fit linear regression model
    model = sm.OLS(outcome_array, predictors_array).fit()
    # Lasso regression
lasso_model_sklearn = Lasso(alpha=1)
lasso_model_sklearn.fit(predictors_array, outcome_array)

# Ridge regression
ridge_model_sklearn = Ridge(alpha=1)
ridge_model_sklearn.fit(predictors_array, outcome_array)

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
    """
    Implement tournament selection. 'Winner' parent selected as highest fitness score, 
    other parent selected randomly but proportional to fitness score (higher scores more heavily weighted).

    Parameters:
    - population (list): Subset of total population, list of chromosomes.
    - winners_per_subgroup (int): Number of 'winning' parents per subgroup.
    - data (pd.DataFrame): Input data with predictors.
    - outcome_index (int): Index of outcome column (0 index).
    - objective_function (string): Default is AIC, other options are "BIC", "Adjusted R-squared, "Deviance", "MSE", 
    "Mallows CP".
    - log_outcome (bool): True if the outcome variable is on the log scale, False else.


    Returns:
    - winners (list): list of chromosomes of top 'winners' per subgroup.
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
    print(chromosome)
    assert isinstance(mutation_rate, float) and mutation_rate >= 0 and mutation_rate <= 1, \
    "mutation_rate must be of type(float) and must take a value between 0 and 1 (inclusive)."
    assert isinstance(max_features, int) and max_features > 0, "max_features must be a positive integer"
    assert max_features <= len(chromosome), "max_features must not exceed number of features in dataset"

    # mutation == True with probability 0.01, False otherwise
    mutation = [random.random() < mutation_rate for i in range(len(chromosome))]

    # '0' to '1' mutation or '1' to '0' mutation if mutation == True
    new_chromosome = [abs(chromosome[i] - 1) if mutation[i] else chromosome[i] for i in range(len(chromosome))]

    #this checks for too many features gained from both crossover and mutation
    new_chromosome = adjust_chromosome(new_chromosome, max_features)

    return new_chromosome

def genetic_algorithm(data,population_size=20, chromosome_length=27, generations=100, mutation_rate=0.01, max_features=10, 
                      outcome_index=0, objective_function="AIC", log_outcome=True):

    winners_per_subgroup = 1 #change later to be paramter passed by user
    population = initialize_population(population_size, chromosome_length, max_features)
    generation_data = Generation_Container()
    scores_data = Generation_Scores()

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
            winners, losers = select_tournament_winners(group, 
                                                        winners_per_subgroup, 
                                                        data, 
                                                        outcome_index, 
                                                        objective_function, 
                                                        log_outcome) #selecting 1 winner per subgroup
            for winner in winners:
                all_winners.append(winner)
            for loser in losers:
                all_losers.append(loser)

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

        #gen_max_score = max(calculate_fitness(individual, data, outcome_index, objective_function, log_outcome) for individual in new_population)
        gen_scores = [[individual,calculate_fitness(individual, data, outcome_index, objective_function, log_outcome)] for individual in new_population]
        scores_data.add_scores(gen_scores)
        gen_max_individual_data = max(gen_scores, key=lambda x: abs(x[1]))

        generation_data.add_generation_data(gen_max_individual_data[1], gen_max_individual_data[0])
        #max_score_generation.append(max(calculate_fitness(individual, data, outcome_index, objective_function, log_outcome) for individual in new_population))
        
        # terminate if the max fitness score in a generation converges
        if g == 0:
            None
        #elif abs(max_score_generation[g] - max_score_generation[g-1]) < 1e-15:
        #    break
        # if generation_data.check_diff_last_generations(3) < .1:
        #     break
        # if g==100:
        #     break
        #check the difference across generations for an exit condition
        
        population = new_population #non-overlapping generations
        
    #save generation data

    # COMMENT THIS CODE IN TO PRINT OUT THE HIGHEST SCORING INDIVIDUAL FROM EACH GENERATION (NOT SURE IF WORKING RIGHT)
    generation_data.show_all_generations()
    
    # COMMENT THIS CODE IN TO PLOT ALL THE AIC VALUES FROM EACH GENERATION
    scores_data.plot_scores()


class Generation_Container:
    def __init__(self):
        self._generation_individuals = []
        self._generation_scores = []

    def add_generation_data(self, score, individual):
        self._generation_individuals.append(individual)
        self._generation_scores.append(score)

    def show_all_generations(self):
        for i, _ in enumerate(self._generation_individuals):
            print(f'Generation {i+1} yielded: {self._generation_scores[i]} {self._generation_individuals[i]}')

    def check_last_generations(self, distance_back):
        target = max(0, len(self._generation_individuals) - distance_back)
        for i, _ in enumerate(self._generation_individuals[target:]):
            print(f'Generation {len(self._generation_individuals) - distance_back + i + 1} yielded: {self._generation_scores[i]} {self._generation_individuals[i]}')

    def check_diff_last_generations(self, distance_back):
        target = max(0, len(self._generation_individuals) - distance_back)
        diff = 0
        if len(self._generation_scores) < 5:
            return np.abs(self._generation_scores[0])
        for i, _ in enumerate(self._generation_individuals[target:len(self._generation_individuals) - 1]):
            diff += np.abs(self._generation_scores[i] - self._generation_scores[i+1])
        diff = diff/target
        print(f'{target} generations giving diff of: {diff}')
        return diff

class Generation_Scores:
    def __init__(self):
        self.scores = []
    def add_scores(self, generation):
        gen_scores = []
        for data in generation:
            gen_scores.append(data[1])
        self.scores.append(gen_scores)
    def print_scores(self):
        print(self.scores)
    def plot_scores(self):
        import matplotlib.pyplot as plt

        x_values = []
        y_values = []
        for index, values in enumerate(self.scores):
            x_values.extend([index] * len(values))  # Repeat the index for each value in the set
            y_values.extend(values)


        plt.scatter(x_values, y_values, s=6)
        plt.title('aic across generations')
        plt.xlabel('generation')
        plt.ylabel('aic')
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()

# test on baseball data (27 predictors)

# read in baseball data 
current_dir = os.getcwd()
# data_folder_path = os.path.join(current_dir, 'GA-dev/genetic_algorithm/data')
data_folder_path = '/Users/kaileyferger/STAT243/GA-dev/genetic_algorithm/data'

file_path = os.path.join(data_folder_path, 'baseball.dat')
data = pd.read_csv(file_path, delimiter=' ')

print(genetic_algorithm(data,population_size=20, chromosome_length=27, generations=100, mutation_rate=0.01, max_features=27, 
                      outcome_index=0, objective_function="AIC", log_outcome=True))


# test on crime data (99 predictors)

# read in crime data
current_dir = os.getcwd()
data_folder_path = os.path.join(current_dir, 'GA-dev/genetic_algorithm/data')
file_path = os.path.join(data_folder_path, 'communities.data')
data2 = pd.read_csv(file_path, delimiter=',')
crime_data = data2.iloc[:, 5:] #remove some parameters
new_header = ["population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp",
              "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome",
              "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc",
              "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap",
              "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed",
              "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce",
              "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par",
              "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg",
              "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig",
              "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell",
              "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous",
              "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup",
              "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone",
              "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian",
              "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters",
              "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85",
              "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq",
              "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite",
              "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits",
              "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars",
              "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop",
              "ViolentCrimesPerPop"] #outcome is violent crimes per pop

# replace with the new header
crime_data.columns = new_header

# remove columns that contain '?'
rows_with_question_mark = crime_data[crime_data.eq('?').any(axis=1)].shape[0]
cols_contains_question_mark = (crime_data == '?').sum()
crime_data_clean = crime_data.loc[:, ~(crime_data == '?').any()]
nfields = crime_data_clean.shape[1] - 1

print(genetic_algorithm(crime_data_clean,population_size=20, chromosome_length=nfields, generations=100, mutation_rate=0.02, max_features=50, 
                      outcome_index=nfields, objective_function="AIC", log_outcome=False))









