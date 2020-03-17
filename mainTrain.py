import getopt
import sys
import pandas as pd
import numpy as np
import train.parser as parser
from train.functions.identity import Identity
from train.functions.sigmoid import Sigmoid
from train.mlp import MLP
from train.trainConfig import TrainConfig
from train.visualization.networkVisualizer import NetworkVisualizer
from train.cost_functions import QuadraticCostFunction
from train.accuracy import get_classification_accuracy


def main(argv):
    help_txt = 'mainTrain.py ' \
               '-l <"1,2,1"> ' \
               '-f <"0" - ReLU | "1" -... |> ' \
               '-b <bias> ' \
               '-s <batch_size> ' \
               '-n <number_of_iterations> ' \
               '-r <learning_rate> ' \
               '-m <momentum> -p <0 - cls | 1 - reg> ' \
               '-i <input_file>' \
               '-t <test_input_file' \
               '-d seed -1 random' \
               '-v visualizer 1 or 0'
    try:
        opts, args = getopt.getopt(argv, "hl:f:b:s:n:r:m:p:i:t:d:v:",
                                   ["help", "layers=", "activation_function=", "bias=", "batch_size=",
                                    "number_of_iterations=", "learning_rate=", "momentum=", "problem=", "input=",
                                    "test=", "seed=", "visualizer="])
    except getopt.GetoptError:
        print(help_txt)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_txt)
            sys.exit()
        elif opt in ("-l", "--layers"):
            layers = [int(x) for x in arg.split(",")]
        elif opt in ("-f", "--activation_function"):
            activation_function = int(arg)
        elif opt in ("-b", "--bias"):
            bias = int(arg)
        elif opt in ("-s", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-n", "--number_of_iterations"):
            number_of_iterations = int(arg)
        elif opt in ("-r", "--learning_rate"):
            learning_rate = float(arg)
        elif opt in ("-m", "--momentum"):
            momentum = float(arg)
        elif opt in ("-p", "--problem"):
            problem = int(arg)
        elif opt in ("-i", "--input"):
            input_file = arg
        elif opt in ("-t", "--test"):
            test_file = arg
        elif opt in ("-d", "--seed"):
            seed = int(arg)
        elif opt in ("-v", "--visualizer"):
            visualizer = int(arg)

    return TrainConfig(layers, activation_function, bias, batch_size, number_of_iterations, learning_rate, momentum,
                       problem, input_file, test_file, seed, visualizer)


if __name__ == "__main__":
    # read config
    config = main(sys.argv[1:])
    # load data
    df = pd.read_csv(config.input_file)

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    if config.problem == 0:
        y, classes = parser.split_y_classes(y)

    test_df = pd.read_csv(config.test_file)
    x_test = df.iloc[:, :-1]
    y_test = df.iloc[:, -1:]

    visualizer = NetworkVisualizer(config.layers, True)

    if config.problem == 1:
        activation_functions = [config.activation_function] * (len(config.layers) - 2) + [Identity()]
    else:
        activation_functions = [config.activation_function] * (len(config.layers) - 2) + [Sigmoid()]

    # train
    mlp = MLP(
        network_size=config.layers,
        is_bias=config.bias,
        activation_functions=activation_functions,
        cost_function=QuadraticCostFunction(),
        visualizer=config.visualizer
    )

    mlp.train(
        x,
        y,
        iterations=config.number_of_iterations,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        seed=config.seed
    )

    y_result = mlp.predict(x_test)
    if config.problem == 0:
        y_result = parser.merge_y_classes(y_result, classes)
        print('accuracy:', get_classification_accuracy(y_result, y_test))

    print(y_result)
    print('cost:', np.sum(mlp.cost_function.function(y_result.to_numpy(), y_test.to_numpy())))

    # save net
