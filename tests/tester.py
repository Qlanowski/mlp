import tests.generator as generator
import train.visualization.dataset_visualization as dv
from train.functions.relu import ReLU
from train.cost_functions import QuadraticCostFunction

config = generator.TestConfig(
    train_file="../classification/data.three_gauss.train.500.csv",
    test_file="../classification/data.three_gauss.test.500.csv",
    hid_layers=[5, 5],
    act_func=ReLU(),
    cost_func=QuadraticCostFunction(),
    is_bias=True,
    batch_size=10,
    lr=0.3,
    moment=0.2,
    seed=123456789
)

# dv.visualise_data_set_classes(config.train_file, save=True, filename='train_data.png')
# dv.visualise_data_set_classes(config.test_file, save=True, filename='test_data.png')


results = generator.get_classification_test_series_result(config, range(0, 601, 10))

generator.visualise_classification_test_series(
    results=results,
    title='Classification Accuracy for ReLU',
    save=True,
    filename='results/' + generator.get_test_filename(config)
)
