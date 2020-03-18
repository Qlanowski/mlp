import os

import tests.generator as generator
import train.visualization.dataset_visualization as dv
from train.functions.relu import ReLU
from train.functions.sigmoid import Sigmoid
from train.functions.tanh import Tanh
from train.functions.identity import Identity
from train.cost_functions import QuadraticCostFunction


def perform_classification_tests_for_all_af():
    a_functions = [ReLU(), Sigmoid(), Tanh(), Identity()]
    config = generator.TestConfig(
        train_file="../classification/data.three_gauss.train.500.csv",
        test_file="../classification/data.three_gauss.test.500.csv",
        hid_layers=[5, 5],
        act_func=None,
        cost_func=QuadraticCostFunction(),
        is_bias=True,
        batch_size=10,
        lr=0.3,
        moment=0.2,
        seed=123456789
    )
    iterations = range(0, 601, 10)
    directory = 'results/classification/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    dv.visualise_data_set_classes(config.train_file, save=True, filename=directory + 'train_data.png')
    dv.visualise_data_set_classes(config.test_file, save=True, filename=directory + 'test_data.png')
    for func in a_functions:
        config.activation_function = func
        results = generator.get_classification_test_series_result(config, iterations)
        generator.visualise_classification_test_series(
            results=results,
            title=f'Classification Accuracy for {type(func).__name__}',
            save=True,
            filename=directory + generator.get_test_filename(config)
        )


perform_classification_tests_for_all_af()
