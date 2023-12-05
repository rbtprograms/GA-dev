from genetic_algorithm import generate_random_chromosome, initialize_population, rank, Generation_Container

def test_generate_random_chromosome():
  chromosome_length = 5
  res = generate_random_chromosome(chromosome_length)
  assert type(res) == list
  assert len(res) == chromosome_length
  assert all(x in [0, 1] for x in res)

def test_initialize_population():
  pop_size, chromosome_length = 10, 5
  res = initialize_population(pop_size, chromosome_length)
  assert type(res) == list
  assert len(res) == 10
  assert len(res[0]) == 5

def test_rank():
  scores = [12,65,32,10,11]
  res = rank(scores)
  assert type(res) == list
  assert type(res) == list
  assert res[0] == 4
  assert all(number in res for number in range(1, 6))

def test_rank_negatives():
  scores = [12,-2,32,10,11]
  res = rank(scores)
  assert res[0] == 2

def test_save_generation_data():
  container = Generation_Container()
  container.add_generation_data(45, [1,0,0,1,1])
  container.add_generation_data(21, [1,0,0,0,0])
  container.add_generation_data(100, [1,0,1,1,1])
  container.add_generation_data(65, [1,1,1,1,1])
  container.add_generation_data(11, [0,0,1,0,1])
  container.check_last_generations(3)
  assert type(container) == int