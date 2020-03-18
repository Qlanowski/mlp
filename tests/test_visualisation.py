from matplotlib import pyplot as plt


def visualise_test_results(results, title, y_label, save=False, filename=None):
    train_scores = [res.train_score for res in results]
    test_scores = [res.test_score for res in results]
    iterations = [res.iterations for res in results]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 100, 5))
    _ = ax.plot(iterations, train_scores, 'r', label='training data')
    _ = ax.plot(iterations, test_scores, 'b', label='test data')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc='lower right')
    if save:
        fig.savefig(filename)
    else:
        fig.show()
