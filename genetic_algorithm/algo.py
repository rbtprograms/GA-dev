from sklearn import linear_model

class GeneticAlgorithm():
  def __init__(self, data, estimation_model = linear_model.LinearRegression()):
    self.estimation_model = estimation_model
    self.data = data

    print(f'model used: {self.estimation_model}')