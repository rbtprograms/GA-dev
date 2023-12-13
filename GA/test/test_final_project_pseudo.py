import numpy as np
import pandas as pd

from ..utils import *
from GA import select

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

def test_genetic_algorithm():
    np.random.seed(0)
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)

    data = pd.DataFrame({
      'x1': x1, 
      'x2': x2, 
      'x3': np.random.rand(100), 
      'x4': np.random.rand(100), 
      'y': x1 + x2
      })

    result = select(data, chromosome_length=4, outcome_index=4, population_size=10, generations=100, num_sets=1, mutation_rate=.01, 
                    max_features=2, objective_function="AIC", log_outcome=True, regression_type="OLS", exit_condition_scalar=.0000000001)

    # the goal is to see if the first two coeffs are 1 and the second 2 are 0, since that is quite literally how i have structured the data
    assert result[0] == 1
    assert result[1] == 1
    assert result[2] == 0
    assert result[3] == 0