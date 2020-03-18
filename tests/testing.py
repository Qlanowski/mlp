import os

import tester
import test_visualisation as tv
import train.visualization.dataset_visualization as dv
from train.functions.relu import ReLU
from train.functions.sigmoid import Sigmoid
from train.functions.tanh import Tanh
from train.functions.identity import Identity
from train.cost_functions import QuadraticCostFunction


def perform_tests_and_save(tester_instance, iteration_list, results_directory):
    activation_functions = [ReLU(), Sigmoid(), Tanh(), Identity()]
    config = tester_instance.config
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    for f in activation_functions:
        config.activation_function = f
        results = tester_instance.perform_test_series(iteration_list)
        tv.visualise_test_results(
            results=results,
            title=tester_instance.get_test_name(),
            y_label=tester_instance.get_score_name(),
            save=True,
            filename=results_directory + config.get_name() + '.png'
        )


conf = tester.TestConfig(
        train_file="classification/data.three_gauss.train.500.csv",
        test_file="classification/data.three_gauss.test.500.csv",
        hid_layers=[5, 5],
        act_func=None,
        cost_func=QuadraticCostFunction(),
        is_bias=True,
        batch_size=10,
        lr=0.3,
        moment=0.2,
        seed=123456789
    )
it = range(0, 501, 10)
perform_tests_and_save(
    tester_instance=tester.ClassificationTester(conf),
    iteration_list=it,
    results_directory='tests/results/classification/'
)
