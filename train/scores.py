def get_classification_accuracy(result, expected):
    accurate = 0.0
    for res, exp in zip(result.iloc[:, 0], expected.iloc[:, 0]):
        accurate += int(res == exp)
    return accurate / len(expected)

