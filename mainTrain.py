import getopt
import sys
import pandas as pd
import numpy as np
import train.parser as parser
from train.functions.sigmoid import Sigmoid
from train.mlp import MLP
from train.trainConfig import TrainConfig


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
               '-d seed -1 random'
    try:
        opts, args = getopt.getopt(argv, "hl:f:b:s:n:r:m:p:i:t:d:",
                                   ["help", "layers=", "activation_function=", "bias=", "batch_size=",
                                    "number_of_iterations=", "learning_rate=", "momentum=", "problem=", "input=", "test=", "seed="])
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

    return TrainConfig(layers, activation_function, bias, batch_size, number_of_iterations, learning_rate, momentum,
                       problem, input_file, test_file, seed)


if __name__ == "__main__":
    # read config
    config = main(sys.argv[1:])
    # load data
    df = pd.read_csv(config.input_file)
    x = df.iloc[:, :-1]
    y, classes = parser.split_y_classes(df.iloc[:, -1:])

    test_df = pd.read_csv(config.test_file)
    x_test = df.iloc[:, :-1]

    # train
    mlp = MLP(
        network_size=config.layers,
        is_bias=config.bias,
        activation_function=Sigmoid()
    )

    mlp.train(
        x,
        y,
        iterations= config.number_of_iterations // config.batch_size,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        momentum=config.momentum
    )

    y_result = parser.merge_y_classes(mlp.predict(x_test), classes)
    print(y_result)

    # save net
