import os
import getopt
import sys
from train.cost_functions import QuadraticCostFunction
from train.functions.identity import Identity
from train.functions.relu import ReLU
from train.functions.sigmoid import Sigmoid
from train.functions.tanh import Tanh
import train.parser as pr
import numpy as np
import  pandas as pd
import tests.test_visualisation as tv
from train.network import Network
from train.trainConfig import TrainConfig
from train.visualization.networkVisualizer import NetworkVisualizer
from train.visualization.visualizer import Visualizer


class TestResult:
    def __init__(self, iterations, test_score, train_score):
        self.iterations = iterations
        self.test_score = test_score
        self.train_score = train_score


class ClassificationConfiguration:
    def __init__(self, train_file, test_file, iterations, hid_layers, act_func,
                 cost_func, is_bias, batch_size, lr, momentum, seed, collect_results_for_iterations, visualizer):
        self.train_file = train_file
        self.test_file = test_file
        self.iterations = iterations
        self.hidden_layers = hid_layers
        self.activation_function = act_func
        self.cost_function = cost_func
        self.is_bias = is_bias
        self.batch_size = batch_size
        self.learning_rate = lr
        self.momentum = momentum
        self.seed = seed
        self.collect_results_for_iterations = collect_results_for_iterations
        self.results = []
        self.visualizer = visualizer

    def get_test_title(self):
        return f'Classification for {type(self.activation_function).__name__}'

    def get_score_name(self):
        return "Accuracy %"

    def get_name(self):
        return (f'{type(self.activation_function).__name__.lower()}_'
                f'{"-".join(str(l) for l in self.hidden_layers)}l_'
                f'{"" if self.is_bias else "n"}b_'
                f'{type(self.cost_function).__name__.lower()}_'
                f'{self.batch_size}bs_'
                f'{self.learning_rate}lr_'
                f'{self.momentum}m_'
                f'{self.seed}s_'
                f'{os.path.basename(os.path.splitext(self.train_file)[0])}')

    def perform_test(self):
        train_data = self.__get_train_data(self.train_file)
        test_data = self.__get_test_data(self.test_file)
        train_data_for_evaluation = self.__get_test_data(self.train_file)

        layers = self.hidden_layers + [train_data[0][1].shape[0]]
        layers.insert(0, train_data[0][0].shape[0])
        activation_functions = [self.activation_function] * (len(layers) - 2) + [Sigmoid()]
        if self.visualizer:
            visualizer = NetworkVisualizer(layers, self.is_bias)
        else:
            visualizer = Visualizer(layers, self.is_bias)
        net = Network(layers,
                      is_bias=self.is_bias,
                      activation_functions=activation_functions,
                      cost_function=self.cost_function,
                      visualizer=visualizer)
        net.train(train_data,
                  iterations=self.iterations,
                  mini_batch_size=self.batch_size,
                  learning_rate=self.learning_rate,
                  momentum=self.momentum,
                  seed=self.seed,
                  iteration_function=self.__run_scoring,
                  iteration_train_data=train_data_for_evaluation,
                  iteration_test_data=test_data)

        return self.results

    @staticmethod
    def __get_train_data(filename):
        x, y = pr.load_data(filename)
        y = pr.split_y_classes(y)
        x = np.array(x)
        y = np.array(y[0])
        return [(np.array(_x).reshape(-1, 1), np.array(_y).reshape(-1, 1)) for _x, _y in zip(x, y)]

    @staticmethod
    def __get_test_data(filename):
        x, y = pr.load_data(filename)
        x = np.array(x)
        y = np.array(y)
        return [(np.array(_x).reshape(-1, 1), _y[0]) for _x, _y in zip(x, y)]

    def __run_scoring(self, iteration, iterations, network, train_data, test_data):
        if iteration % self.collect_results_for_iterations:
            return

        test_results = [(np.argmax(network.predict(x)) + 1, y) for (x, y) in test_data]
        test_result = sum(int(x == y) for (x, y) in test_results)
        test_acc = test_result / len(test_results)

        train_results = [(np.argmax(network.predict(x)) + 1, y) for (x, y) in train_data]
        train_result = sum(int(x == y) for (x, y) in train_results)
        train_acc = train_result / len(train_results)
        self.results += [TestResult(iteration, test_acc * 100, train_acc * 100)]
        print(f'Iteration {iteration}/{iterations} completed; Test acc: {test_acc:.4f}; Train acc: {train_acc:.4f}')


class KaggleConfiguration:
    def __init__(self, train_file, test_file, iterations, hid_layers, act_func,
                 cost_func, is_bias, batch_size, lr, momentum, seed, collect_results_for_iterations, split, visualizer):
        self.train_file = train_file
        self.test_file = test_file
        self.iterations = iterations
        self.hidden_layers = hid_layers
        self.activation_function = act_func
        self.cost_function = cost_func
        self.is_bias = is_bias
        self.batch_size = batch_size
        self.learning_rate = lr
        self.momentum = momentum
        self.seed = seed
        self.collect_results_for_iterations = collect_results_for_iterations
        self.split = split
        self.results = []
        self.visualizer = visualizer

    def get_test_title(self):
        return f'Classification for {type(self.activation_function).__name__}'

    def get_score_name(self):
        return "Accuracy %"

    def get_name(self):
        return (f'{type(self.activation_function).__name__.lower()}_'
                f'{"-".join(str(l) for l in self.hidden_layers)}l_'
                f'{"" if self.is_bias else "n"}b_'
                f'{type(self.cost_function).__name__.lower()}_'
                f'{self.batch_size}bs_'
                f'{self.learning_rate}lr_'
                f'{self.momentum}m_'
                f'{self.seed}s_'
                f'{os.path.basename(os.path.splitext(self.train_file)[0])}')

    def perform_test(self):
        train_df, test_df = self.__train_test(self.train_file, self.split, self.seed)
        train_data = self.__prepare_to_train(train_df)
        test_data = self.__prepare_to_test(test_df)
        train_data_evaluation = self.__prepare_to_test(train_df)

        layers = self.hidden_layers + [train_data[0][1].shape[0]]
        layers.insert(0, train_data[0][0].shape[0])
        activation_functions = [self.activation_function] * (len(layers) - 2) + [Sigmoid()]
        if self.visualizer:
            visualizer = NetworkVisualizer(layers, self.is_bias)
        else:
            visualizer = Visualizer(layers, self.is_bias)
        net = Network(layers,
                      is_bias=self.is_bias,
                      activation_functions=activation_functions,
                      cost_function=self.cost_function,
                      visualizer=visualizer)
        net.train(train_data,
                  iterations=self.iterations,
                  mini_batch_size=self.batch_size,
                  learning_rate=self.learning_rate,
                  momentum=self.momentum,
                  seed=self.seed,
                  iteration_function=self.__run_scoring,
                  iteration_train_data=train_data_evaluation,
                  iteration_test_data=test_data)

        return self.results

    @staticmethod
    def __train_test(filename, train_frac, seed):
        df = pd.read_csv(filename)
        train_df = df.sample(frac=train_frac, random_state=seed)  # random state is a seed value
        test_df = df.drop(train_df.index)
        return train_df, test_df

    @staticmethod
    def __prepare_to_train(df):
        x = df.iloc[:, 1:] / 255
        y = df.iloc[:, 0]
        y_vec = np.array([np.zeros(10) for i in y])
        for v, i in zip(y_vec, y):
            v[i] = 1

        x = np.array(x)
        return [(np.array(_x).reshape(-1, 1), np.array(_y).reshape(-1, 1)) for _x, _y in zip(x, y_vec)]

    @staticmethod
    def __prepare_to_test(df):
        x = df.iloc[:, 1:] / 255
        y = df.iloc[:, 0]
        x = np.array(x)
        y = np.array(y)
        return [(np.array(_x).reshape(-1, 1), _y) for _x, _y in zip(x, y)]

    def __run_scoring(self, iteration, iterations, network, train_data, test_data):
        if iteration % self.collect_results_for_iterations:
            return

        test_results = [(np.argmax(network.predict(x)), y) for (x, y) in test_data]
        test_result = sum(int(x == y) for (x, y) in test_results)
        test_acc = test_result / len(test_results)

        train_results = [(np.argmax(network.predict(x)), y) for (x, y) in train_data]
        train_result = sum(int(x == y) for (x, y) in train_results)
        train_acc = train_result / len(train_results)
        self.results += [TestResult(iteration, test_acc * 100, train_acc * 100)]
        print(f'Iteration {iteration}/{iterations} completed; Test acc: {test_acc:.4f}; Train acc: {train_acc:.4f}')


class RegressionConfiguration:
    def __init__(self, train_file, test_file, iterations, hid_layers, act_func,
                 cost_func, is_bias, batch_size, lr, momentum, seed, collect_results_for_iterations, visualizer):
        self.train_file = train_file
        self.test_file = test_file
        self.iterations = iterations
        self.hidden_layers = hid_layers
        self.activation_function = act_func
        self.cost_function = cost_func
        self.is_bias = is_bias
        self.batch_size = batch_size
        self.learning_rate = lr
        self.momentum = momentum
        self.seed = seed
        self.collect_results_for_iterations = collect_results_for_iterations
        self.results = []
        self.visualizer = visualizer

    def get_test_title(self):
        return f'Regression for {type(self.activation_function).__name__}'

    def get_score_name(self):
        return f"Error ({type(self.cost_function).__name__})"

    def get_name(self):
        return (f'{type(self.activation_function).__name__.lower()}_'
                f'{"-".join(str(l) for l in self.hidden_layers)}l_'
                f'{"" if self.is_bias else "n"}b_'
                f'{type(self.cost_function).__name__.lower()}_'
                f'{self.batch_size}bs_'
                f'{self.learning_rate}lr_'
                f'{self.momentum}m_'
                f'{self.seed}s_'
                f'{os.path.basename(os.path.splitext(self.train_file)[0])}')

    def perform_test(self):
        train_data = self.__get_data(self.train_file)
        test_data = self.__get_data(self.test_file)

        layers = self.hidden_layers + [train_data[0][1].shape[0]]
        layers.insert(0, train_data[0][0].shape[0])
        activation_functions = [self.activation_function] * (len(layers) - 2) + [Identity()]
        if self.visualizer:
            visualizer = NetworkVisualizer(layers, self.is_bias)
        else:
            visualizer = Visualizer(layers, self.is_bias)
        net = Network(layers,
                      is_bias=self.is_bias,
                      activation_functions=activation_functions,
                      cost_function=self.cost_function,
                      visualizer=visualizer)
        net.train(train_data,
                  iterations=self.iterations,
                  mini_batch_size=self.batch_size,
                  learning_rate=self.learning_rate,
                  momentum=self.momentum,
                  seed=self.seed,
                  iteration_function=self.__run_scoring,
                  iteration_train_data=train_data,
                  iteration_test_data=test_data)

        return self.results

    @staticmethod
    def __get_data(filename):
        x, y = pr.load_data(filename)
        x = np.array(x)
        y = np.array(y)
        return [(np.array(_x).reshape(-1, 1), np.array(_y).reshape(-1, 1)) for _x, _y in zip(x, y)]

    def __run_scoring(self, iteration, iterations, network, train_data, test_data):
        if iteration % self.collect_results_for_iterations:
            return

        test_results = [(network.predict(x), y) for (x, y) in test_data]
        test_err = sum(self.cost_function.function(x[0][0], y[0][0]) for (x, y) in test_results)
        test_err /= len(test_results)

        train_results = [(network.predict(x), y) for (x, y) in train_data]
        train_err = sum(self.cost_function.function(x[0][0], y[0][0]) for (x, y) in train_results)
        train_err /= len(train_results)

        self.results += [TestResult(iteration, test_err, train_err)]
        print(f'Iteration {iteration}/{iterations} completed; Test err: {test_err:.2f}; Train err: {train_err:.2f}')


def perform_tests_and_save(tester, results_directory, limit_axes):
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    results = tester.perform_test()
    tv.visualise_test_results(
        results=results,
        title=tester.get_test_title(),
        y_label=tester.get_score_name(),
        save=True,
        filename=results_directory + tester.get_name() + '.png',
        limit_axes=limit_axes
    )


def read_configuration(argv):
    help_txt = 'main.py ' \
               '-l <"5,5,5"> ' \
               '-f <"0" - ReLU | "1" -... |> ' \
               '-c <"0" - Quadratic | "1" -Cross Entropy |> ' \
               '-b <bias> ' \
               '-s <batch_size> ' \
               '-n <number_of_iterations> ' \
               '-r <learning_rate> ' \
               '-m <momentum> -p <0 - cls | 1 - reg | 2 - kaggle> ' \
               '-i <input_file>' \
               '-t <test_input_file' \
               '-d seed -1 random' \
               '-v visualizer 1 or 0' \
               '-a collect_results_for_iterations' \
               '-x train split for kaggle dataset'
    try:
        opts, args = getopt.getopt(argv, "hl:f:c:b:s:n:r:m:p:i:t:d:v:a:x:",
                                   ["help", "layers=", "activation_function=", "cost_function=" "bias=", "batch_size=",
                                    "number_of_iterations=", "learning_rate=", "momentum=", "problem=", "input=",
                                    "test=", "seed=", "visualizer=", "--collect_results_for_iterations", "--split"])
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
        elif opt in ("-c", "--cost_function"):
            cost_function = int(arg)
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
        elif opt in ("-a", "--collect_results_for_iterations"):
            collect_results_for_iterations = int(arg)
        elif opt in ("-x", "--split"):
            split = float(arg)

    return TrainConfig(layers, activation_function, cost_function, bias, batch_size, number_of_iterations, learning_rate, momentum,
                       problem, input_file, test_file, seed, visualizer, collect_results_for_iterations, split)


def classification_test(config):
    tester = ClassificationConfiguration(
        train_file=config.input_file,
        test_file=config.test_file,
        iterations=config.number_of_iterations,
        hid_layers=config.layers,
        act_func=config.activation_function,
        cost_func=config.cost_function,
        is_bias=config.bias,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        momentum=config.momentum,
        seed=config.seed,
        collect_results_for_iterations=config.collect_results_for_iterations,
        visualizer=config.visualizer
    )

    perform_tests_and_save(
        tester=tester,
        results_directory='tests/results/classification/',
        limit_axes=True
    )


def kaggle_test(config):
    tester = KaggleConfiguration(
        train_file="kaggle_digits/train.csv",
        test_file="kaggle_digits/test.csv",
        iterations=config.number_of_iterations,
        hid_layers=config.layers,
        act_func=config.activation_function,
        cost_func=config.cost_function,
        is_bias=config.bias,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        momentum=config.momentum,
        seed=config.seed,
        collect_results_for_iterations=config.collect_results_for_iterations,
        split=config.split,
        visualizer=config.visualizer
    )

    perform_tests_and_save(
        tester=tester,
        results_directory='tests/results/kaggle/',
        limit_axes=True
    )


def regression_test(config):
    tester = RegressionConfiguration(
        train_file=config.input_file,
        test_file=config.test_file,
        iterations=config.number_of_iterations,
        hid_layers=config.layers,
        act_func=config.activation_function,
        cost_func=config.cost_function,
        is_bias=config.bias,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        momentum=config.momentum,
        seed=config.seed,
        collect_results_for_iterations=config.collect_results_for_iterations,
        visualizer=config.visualizer
    )

    perform_tests_and_save(
        tester=tester,
        results_directory='tests/results/regression/',
        limit_axes=False
    )


if __name__ == "__main__":
    config = read_configuration(sys.argv[1:])
    if config.problem == 0:
        classification_test(config)
    elif config.problem == 1:
        regression_test(config)
    else:
        kaggle_test(config)
