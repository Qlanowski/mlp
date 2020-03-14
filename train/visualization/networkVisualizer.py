from train.visualization.neuralNetwork import NeuralNetwork
from train.visualization.visualizer import Visualizer


class NetworkVisualizer(Visualizer):
    def __init__(self, layers, bias):
        self.layers = layers
        self.bias = bias

    def update(self, weights):
        widest_layer = max(self.layers)
        network = NeuralNetwork(widest_layer)
        for l in self.layers:
            network.add_layer(l)
        network.draw()


if __name__ == "__main__":
    network = NetworkVisualizer([2, 8, 8, 1], True)
    network.update([])
