import tests.generator as generator
import  train.visualization.dataset_visualization as dv
from train.functions.relu import ReLU
from train.cost_functions import QuadraticCostFunction

config = generator.TestConfig(
    train_file="../classification/data.three_gauss.train.100.csv",
    test_file="../classification/data.three_gauss.test.100.csv",
    hid_layers=[10, 10, 10],
    act_func=ReLU(),
    cost_func=QuadraticCostFunction(),
    is_bias=True,
    batch_size=10,
    iterations=None,
    lr=0.3,
    moment=0.2,
    seed=123456789
)

fields = ['iterations']

values = [[value] for value in range(50, 1001, 50)]

# dv.visualise_data_set_classes(config.train_file, save=True, filename='train_data.png')
# dv.visualise_data_set_classes(config.test_file, save=True, filename='test_data.png')

generator.save_classification_plot_to_accuracy(
    config=config,
    fields=fields,
    values=values,
    x_label='iterations',
    y_label='Accuracy [%]',
    title='Classification Accuracy',
    filename='relu.png'
)