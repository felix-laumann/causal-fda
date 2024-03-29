import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import seaborn as sns


plt.rcParams.update({'axes.facecolor': 'none', 'legend.facecolor': 'inherit'})


def plot_samples(X, Y, pred_points, upper_limit, period):
    """
    Function to plot samples

    Inputs:
    X, Y: observed functional data
    pred_points: prediction points
    upper_limit: largest point up to which predictions are made
    period: period T
    """
    plt.figure(figsize=(12, 8))
    plt.xlabel(r'$t$', size=20)
    plt.ylabel(r'Samples', size=20)
    plt.title(r'Functional samples of $X$ and $Y$ with $T=${}'.format(period), size=22, pad=10)
    plt.plot(pred_points[pred_points <= upper_limit], X.T, 'grey', label=r'$X$')
    plt.plot(pred_points[pred_points <= upper_limit], Y.T, 'blue', label=r'$Y$')
    legend = plt.legend(fontsize=16, framealpha=1, shadow=True, loc=1)
    legend.get_frame().set_facecolor('white')
    plt.tight_layout()
    plt.savefig('results/plot_samples_{}.png'.format(period), format='png')
    return plt.show()


def plot_cross_corr(X, Y, period):
    """
    Function to plot cross correlation of X and Y

    Inputs:
    X, Y: observed functional data
    period: period T
    """
    plt.figure(figsize=(12, 8))
    plt.xlabel(r't', size=20)
    plt.ylabel(r'Cross-correlation', size=20)
    plt.title(r'Cross-correlation', size=22, pad=10)

    lags = signal.correlation_lags(len(X), len(Y))
    corr = signal.correlate(X, Y, method='direct')
    corr /= np.max(corr)

    plt.plot(lags / len(X), corr, 'black')
    plt.tight_layout()
    plt.savefig('results/plot_cross_corr_{}.png'.format(period), format='png')
    return plt.show()


def plot_delay(X, Y, period):
    """
    Function to plot X and Y with delay

    Inputs:
    X, Y: observed functional data
    lag: most informative lag
    period: period T
    """
    plt.figure(figsize=(12, 8))
    plt.xlabel(r'$X$', size=20)
    plt.ylabel(r'$Y$', size=20)
    plt.title(r'Correlation', size=22, pad=10)

    plt.scatter(list(X), list(Y))
    plt.tight_layout()
    plt.savefig('results/plot_XY_corr_{}.png'.format(period), format='png')
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

                error = confidence(np.asarray(type_II_errors[p][n_sample]), n_trials=n_trials)
                plt.fill_between(a_list, type_II_errors[p][n_sample] - error, type_II_errors[p][n_sample] + error,
                                 interpolate=True, alpha=0.25, color=colors[col1])
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
        plt.savefig('results/{}/test_power_{}_{}.png'.format(test, test, n_vars), format='png')
    else:
        plt.savefig('results/{}/test_power_{}.png'.format(test, test), format='png')
    return plt.show()


def plot_lambda_opts(lamb_opts):
    """
    Function to plot optimal lambda values

    Inputs:
    type_I_errors: type-I error rate over various sample sizes
    lambs: range of possible lambda values
    n_samples: (array) number of samples
    test: independence test
    """

    n_vars = list(lamb_opts.keys())
    n_samples = list(lamb_opts[n_vars[0]].keys())
    colors = ['royalblue', 'seagreen', 'orangered']
    plt.figure(figsize=(12, 8))
    plt.title(r'$\lambda^*$ values in conditional independence test', size=22, pad=10)
    plt.yscale('log')
    plt.ylim(1e-4, 1e-1)
    plt.xlim(np.min(n_vars), np.max(n_vars))
    plt.xlabel(r"Number of conditional variables", size=20)
    plt.ylabel(r"$\lambda^*$", size=20)

    for i, n_sample in enumerate(n_samples):
        plt.plot(n_vars, [lamb_opts[k][n_sample] for k in n_vars], colors[i], marker='o', lw=3, label=r'$n =$ %s' % n_sample)

    legend = plt.legend(fontsize=18, framealpha=1, shadow=True, loc=2)
    legend.get_frame().set_facecolor('white')
    plt.yticks(ticks=[1e-1, 1e-2, 1e-3, 1e-4], size=18)
    plt.xticks(ticks=n_vars, size=18)
    plt.grid(True, linestyle=':', linewidth=1)

    plt.tight_layout()

    plt.savefig('results/conditional/lamb_opts_curve.png', format='png')
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


def plot_SHD(SHD_avg_list, SHD_std_list, n_samples, n_nodes, cd_type='constraint', norm=True, std=True):
    """
    Function to plot SHD results on constraint-based causal discovery

    Inputs:
    SHD_avg_list: list of the average SHD over the number of trials
    SHD_std_list: list of the standard deviations of SHD over the number of trials
    n_samples: (array) number of samples
    n_nodes: the number of nodes in the DAG
    cd_type: method of causal discovery (either 'constraint' or 'combined')
    norm: (boolean) whether to plot the normalised SHD or not
    """
    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['royalblue', 'seagreen', 'orangered', 'khaki']

    if norm:
        plt.ylim(-0.02, 0.72)
        plt.ylabel(r'Normalised structural Hamming distance', size=20)
    else:
        plt.ylim(-0.02, 15.02)
        plt.ylabel(r'Structural Hamming distance', size=20)

    plt.xlabel(r"Sample size", size=20)

    plt.xlim(np.min(n_samples) - 1, np.max(n_samples) + 1)

    if cd_type == 'constraint':
        plt.title(r'Constraint-based learning of DAGs', size=22, pad=10)
        methods = ['Constraint']
    elif cd_type == 'combined':
        methods = ['Combined', 'Granger', 'PCMCI']
        plt.title(r'Combined learning of DAGs', size=22, pad=10)
    else:
        raise ValueError("Only 'constraint' and 'combined' supported.")

    for meth in methods:
        for col, d in zip(colors, n_nodes):
            if meth == 'Combined':
                plt.plot(n_samples, SHD_avg_list[meth][d], col, marker='o', lw=3, linestyle='dashed',
                         label='d = {}'.format(d))
                if std:
                    error = SHD_std_list[meth][d]
                    plt.fill_between(n_samples, np.asarray(SHD_avg_list[meth][d]) - error,
                                     np.asarray(SHD_avg_list[meth][d]) + error,
                                     interpolate=True, alpha=0.25, color=col)

            elif meth == 'Granger':
                plt.plot(n_samples, SHD_avg_list[meth][d], col, marker='o', lw=3, linestyle='dashdot')
                if std:
                    error = SHD_std_list[meth][d]
                    plt.fill_between(n_samples, np.asarray(SHD_avg_list[meth][d]) - error,
                                     np.asarray(SHD_avg_list[meth][d]) + error,
                                     interpolate=True, alpha=0.25, color=col)

            elif meth == 'Constraint':
                plt.plot(n_samples, SHD_avg_list[d], col, marker='o', lw=3, linestyle='dashed',
                         label='d = {}'.format(d))
                if std:
                    error = SHD_std_list[d]
                    plt.fill_between(n_samples, np.asarray(SHD_avg_list[d]) - error,
                                     np.asarray(SHD_avg_list[d]) + error,
                                     interpolate=True, alpha=0.25, color=col)

            else:
                plt.plot(n_samples, SHD_avg_list[meth][d], col, marker='o', lw=3, linestyle='solid')
                if std:
                    error = SHD_std_list[meth][d]
                    plt.fill_between(n_samples, np.asarray(SHD_avg_list[meth][d]) - error,
                                     np.asarray(SHD_avg_list[meth][d]) + error,
                                     interpolate=True, alpha=0.25, color=col)

    if cd_type == 'combined':
        plt.plot([], [], ' ', lw=3, linestyle='dashed', label='Combined', c='grey')
        plt.plot([], [], ' ', lw=3, linestyle='dashdot', label='Granger', c='grey')
        plt.plot([], [], ' ', lw=3, linestyle='solid', label='PCMCI', c='grey')

    legend = plt.legend(fontsize=18, framealpha=1, shadow=True, loc=3, ncol=2)
    legend.get_frame().set_facecolor('white')
    plt.yticks(size=18)
    plt.xticks(ticks=n_samples, size=18)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.tight_layout()

    if norm:
        plt.savefig('results/causal/SHD_norm_{}.png'.format(cd_type), format='png')
    else:
        plt.savefig('results/causal/SHD_{}.png'.format(cd_type), format='png')
    return plt.show()


def plot_SHD_regression(SHD_avg_list, SHD_std_list, n_samples, n_nodes, std=True):
    """
    Function to plot SHD results on regression-based causal discovery

    Inputs:
    SHD_avg_list: list of the average SHD over the number of trials
    SHD_std_list: list of the standard deviations of SHD over the number of trials
    n_samples: (array) number of samples
    n_nodes: the number of nodes in the DAG
    """
    plt.figure(figsize=(12, 8))
    colors = ['royalblue', 'seagreen', 'orangered']
    if n_nodes == 2:
        methods = ['Regression', 'Granger', 'CCM']
        plt.ylim(-0.02, 1.02)
    elif n_nodes == 3:
        methods = ['Regression', 'Granger', 'PCMCI']
        plt.ylim(-0.02, 4.02)
    else:
        methods = 0

    plt.xlabel(r"Sample size", size=20)
    plt.ylabel(r'Structural Hamming distance', size=20)

    plt.xlim(np.min(n_samples)-1, np.max(n_samples)+1)

    plt.title(r'Regression-based learning of DAGs with {} nodes'.format(n_nodes), size=22, pad=10)

    for col, meth in zip(colors, methods):
        plt.plot(n_samples, SHD_avg_list[meth], col, marker='o', lw=3, linestyle='dashed',
                 label='{}'.format(meth))

        if std:
            error = SHD_std_list[meth]
            plt.fill_between(n_samples, np.asarray(SHD_avg_list[meth]) - error,
                             np.asarray(SHD_avg_list[meth]) + error,
                             interpolate=True, alpha=0.25, color=col)

    legend = plt.legend(fontsize=18, framealpha=1, shadow=True, loc=7)
    legend.get_frame().set_facecolor('white')
    plt.yticks(size=18)
    plt.xticks(ticks=n_samples, size=18)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.tight_layout()

    plt.savefig('results/causal/SHD_regression_{}.png'.format(n_nodes), format='png')
    return plt.show()


def plot_precision_recall(avg_list, std_list, n_samples, cd_type, std=True, metric='precision'):
    """
    Function to plot precision and recall results

    Inputs:
    avg_list: list of the average precision/recall over the number of trials
    std_list: list of the average precision/recall over the number of trials
    n_samples: (array) number of samples
    a_list: list of dependence factors a (or a' for conditional independence test)
    n_nodes: the number of nodes in the DAG
    cd_type: 'regression', 'constraint', or 'combined'
    std: whether to plot standard deviation
    metric: 'precision' or 'recall
    """

    colors = ['royalblue', 'seagreen', 'orangered', 'khaki']

    n_nodes = list(avg_list.keys())

    plt.figure(figsize=(12, 8))

    if metric=='precision':
        plt.ylabel(r'Precision', size=20)
    elif metric=='recall':
        plt.ylabel(r'Recall', size=20)

    if cd_type=='regression':
        plt.xlabel(r"$a$", size=20)
    else:
        plt.xlabel(r"Sample size", size=20)

    plt.ylim(-0.02, 1.02)
    plt.xlim(np.min(n_samples)-1, np.max(n_samples)+1)

    if cd_type=='regression':
        plt.title(r'Regression-based learning of DAGs with {} nodes'.format(n_nodes), size=22, pad=10)
    elif cd_type=='constraint':
        plt.title(r'Constraint-based learning of DAGs', size=22, pad=10)
    elif cd_type=='combined':
        plt.title(r'Combined learning of DAGs', size=22, pad=10)
    else:
        raise ValueError("Only 'regression', 'constraint', and 'combined' is accepted.")

    for col, n_node in enumerate(n_nodes):
        plt.plot(n_samples, avg_list[n_node], colors[col], marker='o', lw=3, linestyle='dashed',
                 label='d = {}'.format(int(n_node)))

        if std:
            error = np.asarray(std_list[n_node])
            plt.fill_between(n_samples, avg_list[n_node] - error,
                             avg_list[n_node] + error,
                             interpolate=True, alpha=0.25, color=colors[col])

    legend = plt.legend(fontsize=18, framealpha=1, shadow=True, loc=7)
    legend.get_frame().set_facecolor('white')
    plt.yticks(size=18)
    plt.xticks(ticks=n_samples, size=18)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.tight_layout()

    plt.savefig('results/causal/{}_{}.png'.format(metric, cd_type), format='png')
    return plt.show()


def plot_comparison(regr_results, regr_results_std, comp_results, comp_results_std, n_samples, r_list, comp_method='Granger', std=True):
    """
    Function to plot results on regression-based causal discovery in comparison with Granger and CCM

    Inputs:
    comp_results: SHD results of comparison method (Granger or CCM)
    regr_results: SHD results of regression-based causal discovery
    n_samples: (array) number of samples
    r_list: list of evaluated values for r
    comp_method: either 'Granger' or 'CCM'
    """
    plt.figure(figsize=(12, 8))
    colors = ['royalblue', 'seagreen', 'orangered']

    plt.xlabel(r"r", size=20)
    plt.ylabel(r'Structural Hamming distance', size=20)

    plt.xlim(np.min(r_list)-0.01, np.max(r_list)+0.01)
    plt.ylim(-0.02, 2.02)

    plt.title(r'Comparison of {} and regression-based causal discovery'.format(comp_method), size=22, pad=10)

    for n in n_samples:
        if comp_method == 'Granger':
            col = colors[1]
        elif comp_method == 'CCM':
            col = colors[2]
        else:
            raise ValueError("Choose between 'Granger' and 'CCM'.")

        plt.plot(np.asarray(r_list), regr_results[n], colors[0], marker='o', lw=3, linestyle='dashed', label='Regression')
        plt.plot(np.asarray(r_list), comp_results[n], col, marker='o', lw=3, linestyle='dashed', label='{}'.format(comp_method))

        if std:
            error_regr = regr_results_std[n]
            plt.fill_between(np.asarray(r_list), np.asarray(regr_results[n]) - error_regr,
                             np.asarray(regr_results[n]) + error_regr,
                             interpolate=True, alpha=0.25, color=colors[0])

            error_comp = comp_results_std[n]
            plt.fill_between(np.asarray(r_list), np.asarray(comp_results[n]) - error_comp,
                             np.asarray(comp_results[n]) + error_comp,
                             interpolate=True, alpha=0.25, color=col)

    legend = plt.legend(fontsize=18, framealpha=1, shadow=True, loc=1)
    legend.get_frame().set_facecolor('white')
    plt.yticks(size=18)
    plt.xticks(ticks=r_list, size=18)
    plt.grid(True, linestyle=':', linewidth=1)
    plt.tight_layout()

    plt.savefig('results/causal/{}_regression_comparison.png'.format(comp_method), format='png')
    return plt.show()
