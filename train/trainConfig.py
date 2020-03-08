class TrainConfig:

    def __init__(self, layers, activation_function, bias, batch_size, number_of_iterations, learning_rate, momentum,
                 problem, input_file):
        self.layers = layers
        self.activation_function = activation_function
        self.bias = bias
        self.batch_size = batch_size
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.problem = problem
        self.input_file = input_file
