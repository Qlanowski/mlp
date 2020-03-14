import getopt
import sys
import pandas as pd
import numpy as np

from train.network import Network
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
               '-t <test_input_file'
    try:
        opts, args = getopt.getopt(argv, "hl:f:b:s:n:r:m:p:i:t:",
                                   ["help", "layers=", "activation_function=", "bias=", "batch_size=",
                                    "number_of_iterations=", "learning_rate=", "momentum=", "problem=", "input=", "test="])
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

    return TrainConfig(layers, activation_function, bias, batch_size, number_of_iterations, learning_rate, momentum,
                       problem, input_file, test_file)


if __name__ == "__main__":
    # read config
    config = main(sys.argv[1:])
    # load data
    df = pd.read_csv(config.input_file)
    test_df = pd.read_csv(config.test_file)

    if not config.problem:
        classes = np.sort(df.cls.unique())

        train_tuples = [
            (np.array(x[:-1]).reshape(-1, 1),
             np.array([int(x[-1] == i) for i in classes]).reshape(-1, 1))
            for x in df.to_numpy()]

        test_tuples = [
            (np.array(x[:-1]).reshape(-1, 1),
             np.array([int(x[-1] == i) for i in classes]).reshape(-1, 1))
            for x in test_df.to_numpy()]

    else:
        train_tuples = [
            (np.array([x[:-1]]).reshape(-1, 1),
             np.array([x[-1]]).reshape(-1, 1))
            for x in df.to_numpy()]

        test_tuples = [
            (np.array([x[:-1]]).reshape(-1, 1),
             np.array([x[-1]]).reshape(-1, 1))
            for x in test_df.to_numpy()]

    # initialize trainer
    net = Network(config.layers)
    # train
    net.SGD(
        training_data=train_tuples,
        epochs=config.number_of_iterations // config.batch_size,
        mini_batch_size=config.batch_size,
        eta=config.learning_rate
    )

    print(net.evaluate(train_tuples))
    print(net.weights)
    print(net.biases)

    # save net
