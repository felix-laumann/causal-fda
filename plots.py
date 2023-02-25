import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


plt.rcParams.update({'axes.facecolor': 'none', 'legend.facecolor': 'inherit'})


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
def confidence(y, n_trials):
    interval = 1.96 * np.sqrt(y*(1-y)/n_trials)
    return interval


def plot_power(type_II_errors, n_samples, a_list, n_trials, test, periods, n_vars=1, lamb_opts=None):
    """
    Function to plot test power results

    Inputs:
    type_II_errors: type-II error rate over various sample sizes and kernels
    n_samples: (array) number of samples
    a_list: list of dependence factors a (or a' for conditional independence test)
    n_trials: number of trials
    test: (str) which test to perform ('marginal', 'conditional', or 'joint')
    periods: list of number of periods
    n_vars: number of conditional variables
    lamb_opts: list of optimal lambda values
    """
    if lamb_opts is None:
        lamb_opts = {1: {1: '1'}}
    colors = ['royalblue', 'seagreen', 'orangered']
    linestyles = ['dashed', 'solid', 'dashdot']

    for p in lamb_opts.keys():
        for n_sample in lamb_opts[p]:
            if lamb_opts[p][n_sample] == 1e-5:
                lamb_opts[p][n_sample] = r'$10^{-5}$'
            elif lamb_opts[p][n_sample] == 1e-4:
                lamb_opts[p][n_sample] = r'$10^{-4}$'
            elif lamb_opts[p][n_sample] == 1e-3:
                lamb_opts[p][n_sample] = r'$10^{-3}$'
            else:
                pass

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

    if test == 'conditional':
        plt.title('{} independence test with {} conditional variables'.format(title, n_vars), size=22, pad=10)
    else:
        plt.title('{} independence test'.format(title), size=22, pad=10)

    for col1, n_sample in enumerate(n_samples):
        for col2, p in enumerate(periods):
            if test == 'conditional':
                plt.plot(a_list, type_II_errors[p][n_sample], colors[col1], marker='o', lw=3,
                         linestyle=linestyles[col2], label=r'n = {}, $\lambda^*=$ {}'.format(int(n_sample), lamb_opts[p][n_sample]))
            else:
                if col2==0:
                    plt.plot(a_list, type_II_errors[p][n_sample], colors[col1], marker='o', lw=3,
                             linestyle=linestyles[col2], label='n = {}'.format(int(n_sample)))
                else:
                    plt.plot(a_list, type_II_errors[p][n_sample], colors[col1], marker='o', lw=3,
                             linestyle=linestyles[col2])

            error = confidence(np.asarray(type_II_errors[p][n_sample]), n_trials=n_trials)
            plt.fill_between(a_list, type_II_errors[p][n_sample] - error, type_II_errors[p][n_sample] + error,
                             interpolate=True, alpha=0.25, color=colors[col1])

    legend = plt.legend(fontsize=16, framealpha=1, shadow=True, loc=2)
    legend.get_frame().set_facecolor('white')
    plt.yticks(size=18)
    plt.xticks(size=18)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.tight_layout()

    if test=='conditional':
        plt.savefig('{}/test_power_{}_{}.png'.format(test, test, n_vars), format='png')
    else:
        plt.savefig('results/{}/test_power_{}.png'.format(test, test), format='png')
    return plt.show()


def plot_type_I_errors(type_I_errors, lambs, n_samples, n_trials, test):
    """
    Function to plot test power results

    Inputs:
    type_I_errors: type-I error rate over various sample sizes
    lambs: range of possible lambda values
    n_samples: (array) number of samples
    test: independence test
    """

    colors = ['royalblue', 'seagreen', 'orangered']
    plt.figure(figsize=(12, 8))
    plt.title('Type-I error rate for conditional independence test', size=22, pad=10)
    plt.hlines(y=0.05, xmin=np.min(n_samples) - 4, xmax=np.max(n_samples) + 4, colors='k', linestyles='dotted')
    plt.ylim(0.00, 0.16)
    plt.xlim(np.min(n_samples) - 6, np.max(n_samples) + 6)
    plt.xlabel(r"Sample size", size=20)
    plt.ylabel(r"Type-I error rate", size=20)

    type_I_errors_dict = {}
    for i, lamb in enumerate(lambs):
        type_I_errors_dict[lamb] = [list(type_I_errors[n_sample].values())[0][i] for n_sample in n_samples]

    for i, lamb in enumerate(lambs):
        plt.plot(n_samples, type_I_errors_dict[lamb], colors[i], marker='o', lw=3, linestyle='dashed', label=r'$\lambda =$ %s' % lamb)
        error = confidence(np.asarray(type_I_errors_dict[lamb]), n_trials=n_trials)
        plt.fill_between(n_samples, type_I_errors_dict[lamb] - error, type_I_errors_dict[lamb] + error,
                         interpolate=True, alpha=0.25, color=colors[i])

    legend = plt.legend(fontsize=18, framealpha=1, shadow=True, loc=2)
    legend.get_frame().set_facecolor('white')
    plt.yticks(ticks=[0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15], size=18)
    plt.xticks(ticks=[100, 150, 200], size=18)
    plt.grid(True, linestyle=':', linewidth=1)

    plt.tight_layout()

    plt.savefig('results/{}/type_I_errors_{}.png'.format(test, test), format='png')
    return plt.show()


def type_I_boxplot(df, test='conditional'):
    """
    Plotting a boxplot over the various sample sizes and values for lambda

    Inputs:
    df: dataframe as prepared in saves_type_I_cond.ipynb
    test: the independence test (default: 'conditional')

    Returns:
    boxplot
    """
    #colors = ['#4169e1', '#2e8b57', '#ff4500']
    colors = ['royalblue', 'seagreen', 'orangered']
    plt.figure(figsize=(12, 8))

    # sns_plot = sns.barplot(data=df, x='Sample size', y='p_value', hue='lambda', n_boot=100)
    sns_plot = sns.boxplot(data=df, x='Sample size', y=r'$p$-value', hue=r'$\lambda$', palette=colors, fliersize=2)

    plt.title(r'Evaluation of $\lambda$-values in the conditional independence test', fontsize=22, pad=10)
    plt.xlabel(xlabel='Sample size', fontsize=20, labelpad=10)
    plt.xticks(fontsize=18)
    plt.ylabel(ylabel=r'Type-I error rates', fontsize=20, labelpad=10)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.hlines(y=0.05, xmin=-0.8, xmax=2.8,  colors='k', linestyles='dotted')
    handles, _ = sns_plot.get_legend_handles_labels()
    sns_plot.legend(handles, [r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$'], prop={'size': 16})

    plt.tight_layout()

    fig = sns_plot.get_figure()
    fig.savefig('tests/results/{}/type_I_boxplot_{}.png'.format(test, test), format='png')
    return plt.show()


def plot_SHD(SHD_avg_list, n_samples, a_list, n_nodes):
    """
    Function to plot test power results

    Inputs:
    SHD_avg_list: list of the average SHD over the number of trials
    n_samples: (array) number of samples
    a_list: list of dependence factors a (or a' for conditional independence test)
    n_nodes: the number of nodes in the DAG
    """

    colors = ['royalblue', 'seagreen', 'orangered']

    plt.figure(figsize=(12, 8))

    plt.xlabel(r'$a$', size=20)
    plt.ylabel(r'Structural Hemming distance', size=20)
    plt.ylim(-0.02, np.around(1.5 * max(SHD_avg_list[0][min(SHD_avg_list[0].keys())].values()), 1) + 0.02)
    plt.xlim(np.min(a_list)-0.02, np.max(a_list)+0.02)

    plt.title('Causal structure learning of DAGs with {} nodes'.format(n_nodes), size=22, pad=10)

    for col1, n_sample in enumerate(n_samples):
        for i, SHD_avg in enumerate(SHD_avg_list):
            if i==0:
                plt.plot(a_list, SHD_avg[n_sample].values(), colors[col1], marker='o', lw=3,
                         label='n = {}'.format(int(n_sample)))
            else:
                plt.plot(a_list, SHD_avg[n_sample].values(), colors[col1], marker='o', lw=3, linestyle='dashed')

    legend = plt.legend(fontsize=18, framealpha=1, shadow=True)
    legend.get_frame().set_facecolor('white')
    plt.yticks(size=18)
    plt.xticks(ticks=[0.1, 0.5, 1], size=18)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.tight_layout()

    plt.savefig('causal/SHD_power_{}.png'.format(n_nodes), format='png')
    return plt.show()

