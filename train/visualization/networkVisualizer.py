from matplotlib import pyplot
import numpy as np
from train.visualization.neuron import Neuron
from train.visualization.visualizer import Visualizer
from math import atan, cos, sin


class NetworkVisualizer(Visualizer):

    def __init__(self, layers, bias):
        self.layers = layers
        self.bias = bias
        self.layer_neurons = []
        pyplot.figure()

    def update(self, weights, biases):
        pyplot.gcf().clear()
        distance_between_layers = 40
        distance_between_neurons = 20
        neuron_radius = 5
        widest_layer = max(self.layers)

        self.layer_neurons = []
        y = 0
        for layer in self.layers:
            layer = self.__initialize_layer(y, layer, distance_between_neurons, widest_layer)
            self.layer_neurons.append(layer)
            y += distance_between_layers

        for layer in self.layer_neurons:
            for neuron in layer:
                neuron.draw(neuron_radius)

        for layer_inx, w_matrix in enumerate(weights):
            for child_inx, incoming in enumerate(w_matrix):
                for parent_inx, weight in enumerate(incoming):
                    self.__line_between_two_neurons(neuron_radius, self.layer_neurons[layer_inx + 1][child_inx],
                                                    self.layer_neurons[layer_inx][parent_inx], self.__sigmoid(weight),
                                                    weight > 0)

        if self.bias:
            bias_neurons = []
            y = 0
            for i in range(len(self.layers) - 1):
                neuron = Neuron(-distance_between_neurons, y)
                neuron.draw(neuron_radius)
                bias_neurons.append(neuron)
                y += distance_between_layers
            for bias_inx, (bias_neuron, neurons_in_layer) in enumerate(zip(bias_neurons, self.layer_neurons[1:])):
                for layer_neuron in neurons_in_layer:
                    self.__line_between_two_neurons(neuron_radius, layer_neuron, bias_neuron,
                                                    self.__sigmoid(biases[bias_inx]),
                                                    weight > 0)

        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title('Neural Network architecture', fontsize=15)
        pyplot.draw()
        pyplot.pause(0.001)

    def __initialize_layer(self, y, number_of_neurons, distance_between_neurons, widest_layer):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons, distance_between_neurons, widest_layer)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, y)
            neurons.append(neuron)
            x += distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons, distance_between_neurons, widest_layer):
        return distance_between_neurons * (widest_layer - number_of_neurons) / 2

    def __line_between_two_neurons(self, neuron_radius, neuron1, neuron2, weight, positive):
        if positive:
            col = (0, weight, 0)
        else:
            col = (weight, 0, 0)
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment),
                             (neuron1.y - y_adjustment, neuron2.y + y_adjustment), color=col)
        pyplot.gca().add_line(line)

    def __sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    sizes = [2, 8, 8, 10, 1]
    network = NetworkVisualizer(sizes, True)

    for i in range(100):
        weights = [np.random.randn(y, x)
                   for x, y in zip(sizes[:-1], sizes[1:])]
        biases = np.random.rand(len(sizes[:-1]))
        network.update(weights, biases)
