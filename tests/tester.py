import os
import pandas as pd

import train.parser as pr
import train.scores as sc
from train.functions.sigmoid import Sigmoid
from train.functions.identity import Identity
from train.mlp import MLP
from train.visualization.visualizer import Visualizer


class TestConfig:

    def __init__(self, train_file, test_file, hid_layers, act_func, cost_func, is_bias, batch_size, lr, moment, seed):
        self.train_file = train_file
        self.test_file = test_file
        self.hidden_layers = hid_layers
        self.activation_function = act_func
        self.cost_function = cost_func
        self.is_bias = is_bias
        self.batch_size = batch_size
        self.learning_rate = lr
        self.momentum = moment
        self.seed = seed

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


class TestResult:

    def __init__(self, iterations, test_score, train_score):
        self.iterations = iterations
        self.test_score = test_score
        self.train_score = train_score


class Tester:

    def __init__(self, config):
        self.config = config

    def get_test_title(self):
        return "Base test"

    def get_score_name(self):
        return "None"

    def get_data(self, filename):
        pass

    def get_score(self, expected, result):
        pass

    def get_activation_function_list(self):
        pass

    def get_test_mlp(self, layers, activation_functions):
        return MLP(
            network_size=layers,
            is_bias=self.config.is_bias,
            activation_functions=activation_functions,
            cost_function=self.config.cost_function,
            visualizer=Visualizer(None, None)
        )

    def perform_single_test(self, mlp, iterations, x_train, y_train, x_test, y_test):
        mlp.train(
            x_train,
            y_train,
            iterations=iterations,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            momentum=self.config.momentum,
            seed=self.config.seed
        )
        y_train_result = mlp.predict(x_train)
        y_test_result = mlp.predict(x_test)
        train_score = self.get_score(y_train, y_train_result)
        test_score = self.get_score(y_test, y_test_result)
        return TestResult(
            iterations=iterations,
            train_score=train_score,
            test_score=test_score
        )

    def perform_test_series(self, iteration_list):
        x_train, y_train = self.get_data(self.config.train_file)
        x_test, y_test = self.get_data(self.config.test_file)
        activation_functions = self.get_activation_function_list()
        layers = [len(x_train.columns)] + self.config.hidden_layers + [len(y_train.columns)]
        mlp = self.get_test_mlp(layers, activation_functions)
        results = []
        for iterations in iteration_list:
            result = self.perform_single_test(
                mlp=mlp,
                iterations=iterations,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )
            results.append(result)
        return results


class ClassificationTester(Tester):

    def __init__(self, config):
        super().__init__(config)

    def get_test_title(self):
        return f'Classification for {type(self.config.activation_function).__name__}'

    def get_score_name(self):
        return "Accuracy %"

    def get_data(self, filename):
        x, y = pr.load_data(filename)
        y, _ = pr.split_y_classes(y)
        return x, y

    def get_score(self, expected, result):
        classes = expected.columns
        merged_result = pr.merge_y_classes(result, classes)
        merged_expected = pr.merge_y_classes(expected, classes)
        return sc.get_classification_accuracy(merged_result, merged_expected)

    def get_activation_function_list(self):
        return [self.config.activation_function] * len(self.config.hidden_layers) + [Sigmoid()]


class RegressionTester(Tester):

    def __init__(self, config):
        super().__init__(config)

    def get_test_title(self):
        return f'Regression for {type(self.config.activation_function).__name__}'

    def get_score_name(self):
        return f"Error ({type(self.config.cost_function).__name__})"

    def get_data(self, filename):
        df = pd.read_csv(filename)
        normalized_df = pr.normalize_data(df)
        return normalized_df.iloc[:, :-1], normalized_df.iloc[:, -1:]

    def get_score(self, expected, result):
        return sc.get_regression_score(result, expected, self.config.cost_function)

    def get_activation_function_list(self):
        return [self.config.activation_function] * len(self.config.hidden_layers) + [Identity()]