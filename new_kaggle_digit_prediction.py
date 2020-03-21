from train.network import Network
import numpy as np
import pandas as pd
import os

from train.cost_functions import QuadraticCostFunction
from train.functions.sigmoid import Sigmoid
from train.functions.relu import ReLU

from train.visualization.visualizer import Visualizer


def train_test(filename, train_frac, seed):
    df = pd.read_csv(filename)
    train_df = df.sample(frac=train_frac, random_state=seed)  # random state is a seed value
    test_df = df.drop(train_df.index)
    return train_df, test_df


def prepare_to_train(df):
    x = df.iloc[:, 1:] / 255
    y = df.iloc[:, 0]
    y_vec = np.array([np.zeros(10) for i in y])
    for v, i in zip(y_vec, y):
        v[i] = 1

    x = np.array(x)
    return [(np.array(_x).reshape(-1, 1), np.array(_y).reshape(-1, 1)) for _x, _y in zip(x, y_vec)]


def prepare_to_test(df):
    x = df.iloc[:, 1:] / 255
    y = df.iloc[:, 0]
    x = np.array(x)
    y = np.array(y)
    return [(np.array(_x).reshape(-1, 1), _y) for _x, _y in zip(x, y)]


def load_submission_test_data(filename):
    df = pd.read_csv(filename)
    x = np.array(df / 255)
    return [np.array(_x).reshape(-1, 1) for _x in x]


def save_submission(filename, network):
    submission_test_data = load_submission_test_data(filename)
    result_y = [[inx + 1, int(np.argmax(network.predict(x)))] for inx, x in enumerate(submission_test_data)]
    submission = pd.DataFrame(result_y, columns=['ImageId', 'Label'])

    results_directory = 'kaggle_results/'
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    submission.to_csv(results_directory + 'submission_1.csv', index=False)


def run_scoring(iteration, iterations, network, train_data, test_data):
    if iteration % 500:
        return

    if len(test_data):
        test_results = [(np.argmax(network.predict(x)), y) for (x, y) in test_data]
        test_result = sum(int(x == y) for (x, y) in test_results)
        test_acc = test_result / len(test_results)
    else:
        test_acc = -1

    if len(train_data):
        train_results = [(np.argmax(network.predict(x)), y) for (x, y) in train_data]
        train_result = sum(int(x == y) for (x, y) in train_results)
        train_acc = train_result / len(train_results)
    else:
        train_acc = -1

    print(f'Iteration {iteration}/{iterations} completed; Test acc: {test_acc}; Train acc: {train_acc}')


def main():
    seed = 12345
    train_fraction = 0.7
    train_set_filename = r'kaggle_digits/train.csv'
    test_set_filename = r'kaggle_digits/test.csv'

    train_df, test_df = train_test(train_set_filename, train_fraction, seed)
    train_data = prepare_to_train(train_df)
    test_data = prepare_to_test(test_df)
    train_data_evaluation = prepare_to_test(train_df)

    layers = [train_data[0][0].shape[0], 10, train_data[0][1].shape[0]]
    batch_size = 10
    iterations = len(train_data) / batch_size * 2
    learning_rate = 3
    activation_functions = [Sigmoid()] * (len(layers) - 2) + [Sigmoid()]
    cost_function = QuadraticCostFunction()
    is_bias = True
    seed = 1000
    momentum = 0.01
    visualizer = Visualizer(layers, is_bias)

    net = Network(layers,
                  is_bias=is_bias,
                  activation_functions=activation_functions,
                  cost_function=cost_function,
                  visualizer=visualizer)
    net.train(train_data,
              iterations=iterations,
              mini_batch_size=batch_size,
              learning_rate=learning_rate,
              momentum=momentum,
              seed=seed,
              iteration_function=run_scoring,
              iteration_train_data=train_data_evaluation,
              iteration_test_data=test_data)

    save_submission(test_set_filename, net)


if __name__ == "__main__":
    main()
