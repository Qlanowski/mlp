import tests.generator as generator
from train.functions.sigmoid import Sigmoid
from train.cost_functions import QuadraticCostFunction

config = generator.TestConfig(
    train_file="../classification/data.three_gauss.train.100.csv",
    test_file="../classification/data.three_gauss.test.100.csv",
    hid_layers=[3, 5, 7],
    act_func=Sigmoid(),
    cost_func=QuadraticCostFunction(),
    is_bias=True,
    batch_size=10,
    iterations=1000,
    lr=0.3,
    moment=0.2,
    seed=123456789
)

fields = ['iterations']

values = [[value] for value in range(100, 5000, 500)]

generator.save_classification_plot_to_accuracy(
    config=config,
    fields=fields,
    values=values,
    x_label='iterations',
    y_label='Accuracy [%]',
    title='Classification Accuracy',
    filename='acc_to_iter_classification.png'
)