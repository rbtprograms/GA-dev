from ..utils import *

def test_generate_random_chromosome():
  chromosome_length = 5
  res = generate_random_chromosome(chromosome_length, 5)
  assert type(res) == list
  assert len(res) == chromosome_length
  assert all(x in [0, 1] for x in res)

def test_initialize_population():
  pop_size, chromosome_length = 10, 5
  res = initialize_population(pop_size, chromosome_length, 5)
  assert type(res) == list
  assert len(res) == 10
  assert len(res[0]) == 5

