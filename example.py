import random
import numpy as np
import pandas as pd
from sklearn import linear_model

from genetic_algorithm import genetic_algorithm

data = pd.read_csv('./data/baseball.dat', sep=' ')
estimator = linear_model.LogisticRegression()

#right now it just prints out which model was given to it
genetic_algorithm(data)
