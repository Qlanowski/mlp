import numpy as np


def get_classification_accuracy(result, expected):
    accurate = 0.0
    for res, exp in zip(result.iloc[:, 0], expected.iloc[:, 0]):
        accurate += int(res == exp)
    return accurate / len(expected) * 100.0


def get_regression_score(result, expected, cost_function):
    return np.sum(cost_function.function(expected.to_numpy(), result.to_numpy()).squeeze())
