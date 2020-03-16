import numpy as np
import pandas as pd


def split_y_classes(y):
    classes = y.iloc[:, 0].unique()
    result = pd.DataFrame(
        np.zeros((len(y.index), len(classes))),
        columns=["cls%s" % cls for cls in classes],
    )
    for col, cls in zip(result.columns, classes):
        result[col] = [int(val == cls) for val in y.iloc[:, 0]]
    return result
