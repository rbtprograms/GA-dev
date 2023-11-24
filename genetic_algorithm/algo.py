from sklearn import linear_model

class GeneticAlgorithm():
  '''
  right now it doesn't do much just accepts a DataFrame and a model
  '''
  def __init__(self, data, estimation_model = linear_model.LinearRegression()):
    self.estimation_model = estimation_model
    self.data = data

    print(f'model used: {self.estimation_model}')