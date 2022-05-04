# CONSTRAINT-BASED CD
# 1. call marginal and conditional indep test functions
# 2. Implement PC algorithm to smartly iterate over conditional sets (find implementation online?)

# # write wrapper function that has arguments type (=regression/constraint/combined),
# data (computes number of variables by itself?), kernel,


import numpy as np
from kernels import K_ID, K_dft, K_dft2, K_dct, K_CEXP
from independence import joint_indep_test
from graph_generation import generate_DAGs, generate_DAGs_pd, n_DAGs
from regression import hist_linear


def RESIT(X_array, pred_points, _DAG, n_intervals, n_perms, alpha, make_K):
    """
    Regression with subsequent independence test

    Inputs:
    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in dictionary edges
    pred_points: prediction points
    _DAG: candidate directed acyclic graph
    n_intervals: number of intervals for the historical linear model
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint independence test
    alpha: rejection threshold of the joint independence test
    make_K: function called to construct the kernel matrix

    Returns:
    reject: 1 if null rejected and candidate DAG is rejected, 0 if null accepted and candidate DAG is accepted
    p_value: p-value of joint independence test
    """

    n_nodes, n_samples, n_preds = X_array.shape
    desc_hat = np.zeros((n_nodes, n_samples, n_preds))
    residuals = np.zeros((n_nodes, n_samples, n_preds))

    for k, v in _DAG:
        if v==[]:
            residuals[k] = X_array[k]
        else:
            desc_hat[k] = hist_linear(n_intervals, X_array[v], X_array[k], pred_points)
            residuals[k] = X_array[k] - desc_hat[k]

    reject, p_value = joint_indep_test(residuals, n_perms, alpha, make_K)

    return reject, p_value


def eval_candidate_DAGs(X_array, pred_points, n_intervals, n_perms, alpha, make_K, pd_graph=None):
    """
    Evaluate all possible candidate directed acyclic graphs given the number of variables
    and (optional) a partially directed graph

    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in dictionary edges
    pred_points: prediction points
    _DAG: candidate directed acyclic graph
    n_intervals: number of intervals for the historical linear model
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint independence test
    alpha: rejection threshold of the joint independence test
    make_K: function called to construct the kernel matrix
    pd_graph:

    Returns:
    candidate DAG with largest p-value of joint independence test
    """

    n_nodes, n_samples, n_preds = X_array.shape

    # calculate the number of DAGs that can be constructed with n nodes
    _n_DAGs = n_DAGs(n_nodes)

    rejects = np.zeros(_n_DAGs)
    p_values = np.zeros(_n_DAGs)

    if pd_graph==None:
        # generate candidate DAGs without partially directed graph
        print('Generating {} DAGs...'.format(_n_DAGs))
        DAGs_dict = generate_DAGs(n_nodes)
        print('...Done.')

    else:
        # generate candidate DAGs considering partially directed graph
        DAGs_dict = generate_DAGs_pd(pd_graph)

    # iterate over each candidate DAG
    for i, _DAG in enumerate(DAGs_dict):
        rejects[i], p_values[i] = RESIT(X_array, pred_points, _DAG, n_intervals, n_perms, alpha, make_K)

    # return candidate DAG with greatest p-value
    return DAGs_dict[np.argmax(p_values)], np.max(p_values)


def causal_power(X_dict, edges_dict, pred_points=np.linspace(0, 1, 100), n_intervals=8, n_trials=200, n_perms=1000,
                 alpha=0.05, K='K_ID'):
    """
    Calculate the rate of correctly classified causal graphs given data that was synthetically generated according to
    a random DAG

    Inputs:
    X_dict: dictionary of data sets that were generated according to random DAG for each trial
    edges_dict: dictionary of edges for each trial; each value in the dictionary is another dictionary of the form
                key: descendant and value: parents
    pred_points: prediction points
    n_intervals: number of intervals for the historical linear model
    n_trials: number of trials to compute percentage of rejections over
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint independence test
    alpha: rejection threshold of the joint independence test
    make_K: function called to construct the kernel matrix

    Returns:
    c_power: rate of correctly classified causal graphs over all trials
    """
    corrects = np.zeros(n_trials)
    DAGs = {}
    p_values = np.zeros(n_trials)

    for i in range(n_trials):
        if K == 'K_ID':
            make_K = K_ID
        elif K == 'K_dft':
            make_K = K_dft
        elif K == 'K_dft2':
            make_K = K_dft2
        elif K == 'K_dct':
            make_K = K_dct
        elif K == 'K_CEXP':
            make_K = K_CEXP
        else:
            raise ValueError('Kernel not implemented')

        DAGs[i], p_values[i] = eval_candidate_DAGs(X_dict[i], pred_points, n_intervals, n_perms, alpha, make_K)

        # evaluate whether candidate DAG with the greatest p-value is same as synthetically generated DAG
        if DAGs[i]==edges_dict[i]:
            corrects[i] = 1
        else:
            corrects[i] = 0

    # calculate the rate of correctly classified causal graphs
    c_power = np.mean(corrects)

    return c_power
