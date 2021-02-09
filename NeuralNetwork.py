import numpy as np


class NeuralNetwork:
    def __init__(self, input_neurons, output_neurons, hidden_neurons = None, weights_matrices = None):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.neuron_values = []

        if hidden_neurons is not None:
            self.hidden_neurons = hidden_neurons.copy()
        else:
            self.hidden_neurons = []

        configuration = [self.input_neurons] + self.hidden_neurons + [self.output_neurons]

        if weights_matrices is not None:
            self.weights_matrices = weights_matrices.copy()
        else:
            self.weights_matrices = []

            for layer in range(len(configuration) - 1):
                self.weights_matrices.append(np.random.rand(configuration[layer + 1], configuration[layer] + 1) * 1 - 0.5)

                if layer != len(configuration) - 2:
                    for i in range(int(configuration[layer + 1]/2)):
                        for j in range(configuration[layer]):
                            self.weights_matrices[layer][configuration[layer + 1] - i - 1][configuration[layer] - j - 1] \
                                = self.weights_matrices[layer][i][j]

                    if configuration[layer + 1] % 2 == 1:
                        for j in range(int(configuration[layer]/2)):
                            self.weights_matrices[layer][int(configuration[layer + 1]/2)][configuration[layer] - j - 1] \
                                = self.weights_matrices[layer][int(configuration[layer + 1]/2)][j]

                    for i in range(int(configuration[layer + 1] / 2)):
                        self.weights_matrices[layer][configuration[layer + 1] - i - 1][-1] \
                            = self.weights_matrices[layer][i][-1]
                else:
                    for column in range(int(configuration[layer]/2)):
                        self.weights_matrices[layer][0][-(column + 2)] = self.weights_matrices[layer][0][column]
                        self.weights_matrices[layer][1][-(column + 2)] = -self.weights_matrices[layer][1][column]

                    if configuration[layer] % 2 == 1:
                        self.weights_matrices[layer][1][int(configuration[layer]/2)] = 0

    def calc(self, values):
        self.neuron_values = [values]

        for weights in self.weights_matrices:
            values = np.append(values, 1)
            values = np.matmul(weights, values)
            values = self.tanh(values)
            self.neuron_values.append(values)

        return values

    def clone(self):
        return NeuralNetwork(self.input_neurons, self.output_neurons, self.hidden_neurons, self.weights_matrices)

    def mutate(self, fraction, factor):
        for w in range(len(self.weights_matrices)):
            rows = self.weights_matrices[w].shape[0]
            columns = self.weights_matrices[w].shape[1]

            random_variation = np.random.rand(rows, columns) * 2 * factor - factor

            if fraction < 0.5:
                min = 0
                max = 0.5 / (1 - fraction)
            else:
                min = (fraction - 0.5) / fraction
                max = 1
            random_variation *= np.round(np.random.rand(rows, columns) * (max - min) + min)

            if w != len(self.weights_matrices) - 1:
                for i in range(int(rows / 2)):
                    for j in range(columns - 1):
                        random_variation[rows - i - 1][columns - 1 - j - 1] = random_variation[i][j]

                if rows % 2 == 1:
                    for j in range(int(columns - 1 / 2)):
                        random_variation[int(rows / 2)][columns - 1 - j - 1] = random_variation[int(rows / 2)][j]

                for i in range(int(rows / 2)):
                    random_variation[rows - i - 1][-1] = random_variation[i][-1]

            self.weights_matrices[w] = self.weights_matrices[w] + random_variation

    @staticmethod
    def tanh(values):
        return np.tanh(values)

    @staticmethod
    def relu(values):
        return [max(0, i) for i in values]
