import matplotlib.pyplot as plt
import numpy as np


def plot_samples(X, pred_points, upper_limit):
    """
    Function to plot samples

    Inputs:
    X: observed functional data
    pred_points: prediction points
    upper_limit: largest point up to which predictions are made
    """
    plt.figure(figsize=(12, 8))
    plt.xlabel(r'$t$', size=20)
    plt.ylabel(r'Samples', size=20)
    plt.plot(pred_points[pred_points <= upper_limit], X.T)
    return plt.show()


# confidence intervals
def confidence(y, n_tests):
    interval = 1.96 * np.sqrt(y*(1-y)/n_tests)
    return interval


def plot_power(type_II_errors, n_samples, a_list, n_tests, test, K_list):
    """
    Function to plot test power results

    Inputs:
    type_II_errors: type-II error rate over various sample sizes and kernels
    n_samples: (array) number of samples
    a_list: list of dependence factors a (or a' for conditional independence test)
    n_tests: number of tests
    test: (str) which test to perform ('marginal', 'conditional', or 'joint')
    K_list: list of kernels
    """

    colors = ['midnightblue', 'seagreen', 'orangered']
    linestyles = ['dashed', 'solid', 'dashdot']

    plt.figure(figsize=(12, 8))
    if test == 'conditional':
        plt.xlabel(r"$a'$", size=20)
    else:
        plt.xlabel(r'$a$', size=20)
    plt.ylabel(r'Test power', size=20)
    plt.ylim(0, 1.02)
    plt.xlim(np.min(a_list)-0.02, np.max(a_list)+0.02)
    plt.hlines(y=1, xmin=np.min(a_list), xmax=np.max(a_list), colors='k', linestyles='dotted')
    plt.hlines(y=0.05, xmin=np.min(a_list), xmax=np.max(a_list), colors='k', linestyles='dotted')
    if test=='marginal':
        title = 'Marginal'
    elif test=='conditional':
        title = 'Conditional'
    elif test=='joint':
        title = 'Joint'
    else:
        raise ValueError('Test not implemented')
    plt.title('{} independence test'.format(title), size=22, pad=10)

    for col1, n_sample in enumerate(n_samples):
        for col2, K in enumerate(K_list):
            if col2==0:
                plt.plot(a_list, type_II_errors[n_sample][K], colors[col1], marker='o', lw=3,
                         linestyle=linestyles[col2], label='n = {}'.format(int(n_sample)))
            else:
                plt.plot(a_list, type_II_errors[n_sample][K], colors[col1], marker='o', lw=3,
                         linestyle=linestyles[col2])

            error = confidence(np.asarray(type_II_errors[n_sample][K]), n_tests=n_tests)
            plt.fill_between(a_list, type_II_errors[n_sample][K] - error, type_II_errors[n_sample][K] + error,
                             interpolate=True, alpha=0.25, color=colors[col1])

    plt.legend(fontsize=18, framealpha=1, shadow=True)
    plt.yticks(size=18)
    plt.xticks(size=18)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.tight_layout()

    plt.savefig('results/{}/test_power_{}.png'.format(test, test), format='png')
    return plt.show()

