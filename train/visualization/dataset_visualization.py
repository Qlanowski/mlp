import matplotlib.pyplot as plt


colors = ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"]


def visualise_data_set_classes(x, y, save=False, filename=None):
    classes = y.iloc[:, 0].unique()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    global colors
    for cls, col in zip(classes, colors):
        x_cls = x[y.iloc[:, 0] == cls]
        sp = ax.scatter(
            x_cls.iloc[:, 0],
            x_cls.iloc[:, 1],
            alpha=0.5,
            c=col,
            edgecolors='none',
            s=50,
            label="class %d" % cls
        )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Dataset visualisation")
    ax.legend(loc=2)
    if save and filename:
        fig.savefig(filename)
    else:
        fig.show()