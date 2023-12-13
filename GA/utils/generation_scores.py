import matplotlib.pyplot as plt

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
    def plot_scores(self, path):
        x_values = []
        y_values = []
        for index, values in enumerate(self.scores):
            x_values.extend([index] * len(values))
            y_values.extend(values)


        plt.scatter(x_values, y_values, s=6)
        plt.title('aic across generations') #change this to reflect user-inputted objective function
        plt.xlabel('generation')
        plt.ylabel('aic') #change this to reflect user-inputted objective function
        if (sum(y_values) < 0):
          plt.gca().invert_yaxis()
        plt.grid(True)
        plt.savefig(f'{path}/output.png')
