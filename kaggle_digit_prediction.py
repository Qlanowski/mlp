import numpy as np
import pandas as pd
import os

import train.parser as pr
import train.scores as sc
from train.mlp import MLP
from train.functions.tanh import Tanh
from train.functions.sigmoid import Sigmoid
from train.cost_functions import QuadraticCostFunction
from train.visualization.visualizer import Visualizer


train_set_filename = 'kaggle_digits/train.csv'
test_set_filename = 'kaggle_digits/test.csv'


train_df = pd.read_csv(train_set_filename)
train_x, train_y_unsplitted = train_df.iloc[:, 1:], train_df[['label']]
train_y, classes = pr.split_y_classes(train_y_unsplitted)
input_size = len(train_x.columns)
output_size = len(train_y.columns)
hidden_layers = [5, 5, 5]
activation_functions = [Tanh()] * len(hidden_layers) + [Sigmoid()]

mlp = MLP(
    network_size=[input_size] + hidden_layers + [output_size],
    is_bias=True,
    activation_functions=activation_functions,
    cost_function=QuadraticCostFunction(),
    visualizer=Visualizer(None, None)
)

mlp.train(
    x=train_x,
    y=train_y,
    iterations=1000,
    batch_size=10,
    learning_rate=0.4,
    momentum=0.2,
    seed=123456789
)

test_x = pd.read_csv(test_set_filename)

result_y_unmerged = mlp.predict(test_x)
result_y = pr.merge_y_classes(result_y_unmerged, classes)

submission = pd.DataFrame({
    'ImageId': np.r_[0:len(result_y)],
    'Label': result_y.to_numpy().squeeze()
})

results_directory = 'kaggle_results/'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

submission.to_csv(results_directory + 'submission_1.csv', index=False)
