import random
import numpy as np
import pandas as pd
from sklearn import linear_model

from genetic_algorithm import GeneticAlgorithm

data = pd.read_csv('./data/baseball.dat', sep=' ')
estimator = linear_model.LogisticRegression()

GeneticAlgorithm(data)
