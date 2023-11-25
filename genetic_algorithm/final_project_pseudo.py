## adapted from Eleanor's pseudocode

#Pseudocode representation of a genetic algorithm for variable selection

def initialize_population(population_size, chromosome_length):
	
	population = []
	for i in range(population_size):
		chromosome = generate_random_chromosome(chromosome_length)
		population.append(chromosome)

	return population


def generate_random_chromosome(chromosome_length):
	
	chromosome = []
	for i in range(chromosome_length):
		gene = randomly_select_gene()
		chromosome.append(gene)

	return chromosome


def randomly_select_gene():
	gene = random.uniform(0,1) #requires 'random' package
	return gene


def calculate_fitness(chromosome, population_size, data, outcome):
	#using AIC to calculate fitness of each chromosome

	s = sum(chromosome) # number of predictors
	N = population_size
	predictors = data[chromosome]
	model = lm(outcome ~ predictors)
	RSS = sum(outcome - sum(model[coefficients].predictors))
	AIC = N * log(RSS / N) + 2 * (s + 2)

	return AIC


def calculate_rank_based_fitness(population, chromosome, population_size, data, outcome):

	fitness_scores = [calculate_fitness(chromosome, population_size, data, outcome) for chromosome in population]
	r = rank(fitness_scores) #ranking function maximizing at highest magnitude negative AIC score 
	p = population_size

	return (2*r) / (p*(p+1))


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

























