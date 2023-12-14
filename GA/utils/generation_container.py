import numpy as np

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
        if len(self._generation_scores) < distance_back + 1:
            return np.abs(self._generation_scores[0])
        for i, _ in enumerate(self._generation_individuals[len(self._generation_individuals)-distance_back:len(self._generation_individuals) - 1]):
            diff += np.abs(self._generation_scores[i] - self._generation_scores[i+1])
        diff = diff/target

        return diff

    def get_average_score(self):
        return sum(self._generation_scores)/len(self._generation_scores)

    def get_most_recent_individual(self):
      return self._generation_individuals[-1]