import numpy as np
import pandas as pd


def load_data(filename):
    df = pd.read_csv(filename)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    return x, y


def split_y_classes(y):
    classes = y.iloc[:, 0].unique()
    result = pd.DataFrame(
        np.zeros((len(y.index), len(classes))),
        columns=[cls for cls in classes],
    )
    for col, cls in zip(result.columns, classes):
        result[col] = [int(val == cls) for val in y.iloc[:, 0]]
    return result, classes


def merge_y_classes(y, classes):
    y_to_np = y.to_numpy()
    return pd.DataFrame(
        classes[np.argmax(row)]
        for row in y_to_np
    )


def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())