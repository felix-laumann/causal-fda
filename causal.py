import numpy as np
from kernels import K_ID, K_dft, K_dft1, K_dft2, K_dct, K_dwt, K_CEXP
from independence import joint_indep_test
from graph_generation import generate_DAGs, generate_DAGs_pd, sparsify_graph, partially_direct
from graph_utils import n_DAGs
from regression import hist_linear


def PC_alg(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, find_lamb=True):
    """
    PC algorithm that returns partially directed according to Meek's orientation rules given some data X_array

    Inputs:
    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in causal graph
    lambs: range to iterate over for optimal value for regularisation of kernel ridge regression to compute HSCIC
           (only for conditional independence test)
    n_pretests: number of tests to find optimal value for lambda
    n_perms: number of permutations performed when bootstrapping the null distribution
    n_steps: number of MC iterations in the CPT
    alpha: rejection threshold of the test
    make_K: function called to construct the kernel matrix
    find_lamb: (boolean) whether optimal lambda should be searched for or not; if not, set to value specified in lambs

    Returns:
    pd_graph: graph that is partially directed according to Meek's orientation rules
    """

    sparse_graph = sparsify_graph(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, find_lamb)
    pd_graph = partially_direct(sparse_graph)

    return pd_graph


def RESIT(X_array, pred_points, _DAG, n_intervals, n_perms, alpha, make_K):
    """
    Regression with subsequent independence test

    Inputs:
    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in causal graph
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

    for desc, parents in _DAG.items():
        if parents == []:
            residuals[desc] = X_array[desc]
        else:
            for p in parents:
                desc_hat[desc] += hist_linear(n_intervals, X_array[p], X_array[desc], pred_points)
            residuals[desc] = X_array[desc] - desc_hat[desc]

    reject, p_value = joint_indep_test(residuals, n_perms, alpha, make_K)

    return reject, p_value


def eval_candidate_DAGs(X_array, pred_points, n_intervals, n_perms, alpha, make_K, pd_graph=None):
    """
    Evaluate all possible candidate directed acyclic graphs given the number of variables
    and (optional) a partially directed graph

    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in causal graph
    pred_points: prediction points
    n_intervals: number of intervals for the historical linear model
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint independence test
    alpha: rejection threshold of the joint independence test
    make_K: function called to construct the kernel matrix
    pd_graph: partially directed graph based on constraint-based approach with Meek's orientation rules

    Returns:
    candidate DAG with largest p-value of joint independence test
    """

    n_nodes, n_samples, n_preds = X_array.shape

    # calculate the number of DAGs that can be constructed with n nodes
    _n_DAGs = n_DAGs(n_nodes, y=1)

    rejects = np.zeros(_n_DAGs)
    p_values = np.zeros(_n_DAGs)

    if pd_graph is None:
        # generate candidate DAGs without partially directed graph
        DAGs_dict = generate_DAGs(n_nodes)
    else:
        # generate candidate DAGs considering partially directed graph
        DAGs_dict = generate_DAGs_pd(pd_graph)

    # iterate over each candidate DAG
    for i, _DAG in DAGs_dict.items():
        print('Evaluating DAG #{}: {}'.format(i, _DAG))
        rejects[i], p_values[i] = RESIT(X_array, pred_points, _DAG, n_intervals, n_perms, alpha, make_K)

    # return candidate DAG with greatest p-value
    return DAGs_dict[np.argmax(p_values)], np.max(p_values)


def causal_discovery(cd_type, X_array, pred_points, n_intervals, n_perms, alpha, make_K, lambs, find_lamb, n_pretests, n_steps):
    """
    Wrapper function to discover causal graph by constraint-based and regression-based approaches (as specified in cd_type)

    Inputs:
    cd_type: approach of causal discovery (regression, constraint, combined)
    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in dictionary edges
    pred_points: prediction points
    n_intervals: number of intervals for the historical linear model
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint and conditional independence tests
    alpha: rejection threshold of the joint and conditional independence tests
    make_K: function called to construct the kernel matrix
    lambs: range to iterate over for optimal value for regularisation of kernel ridge regression to compute HSCIC
           (only for conditional independence test)
    find_lamb: (boolean) whether optimal lambda should be searched for or not; if not, set to value specified in lambs
    n_pretests: number of tests to find optimal value for lambda
    n_steps: number of MC iterations in the CPT

    Returns:
    _DAG: causal graph that is fully directed when 'regression' or 'combined' is used,
          and partially directed when 'constraint' is used
    p_value: p-value of joint independence test of estimated causal graph
    """

    if cd_type=='regression':
        # RESIT
        _DAG, p_value = eval_candidate_DAGs(X_array, pred_points, n_intervals, n_perms, alpha, make_K, pd_graph=None)
    elif cd_type=='constraint':
        # PC algorithm
        _DAG = PC_alg(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, find_lamb)
        p_value = np.nan
    elif cd_type=='combined':
        # combined: first PC algorithm, then RESIT on result of PC algorithm (i.e., equivalence class)
        pd_graph = PC_alg(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, find_lamb)
        _DAG, p_value = eval_candidate_DAGs(X_array, pred_points, n_intervals, n_perms, alpha, make_K, pd_graph=pd_graph)
    else:
        _DAG = {}
        raise ValueError('Chosen causal discovery method not implemented. Choose between "regression", "constraint", and "combined".')

    return _DAG, p_value


def causal_power(cd_type, X_dict, edges_dict, pred_points=np.linspace(0, 1, 100), n_intervals=8, n_trials=200, n_perms=1000,
                 alpha=0.05, K='K_ID', lambs=np.array([0, 1e-4, 1e-2, 1]), find_lamb=False, n_pretests=100, n_steps=50):
    """
    Calculate the rate of correctly classified causal graphs given data that was synthetically generated according to
    a random DAG

    Inputs:
    cd_type: approach of causal discovery (regression, constraint, combined)
    X_dict: dictionary of data sets that were generated according to random DAG for each trial
    edges_dict: dictionary of edges for each trial; each value in the dictionary is another dictionary of the form
                key: descendant and value: parents
    pred_points: prediction points
    n_intervals: number of intervals for the historical linear model
    n_trials: number of old_trials to compute percentage of rejections over
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint independence test
    alpha: rejection threshold of the joint independence test
    K: function called to construct the kernel matrix
    lambs: range to iterate over for optimal value for regularisation of kernel ridge regression to compute HSCIC
           (only for conditional independence test)
    find_lamb: (boolean) whether optimal lambda should be searched for or not; if not, set to value specified in lambs
    n_pretests: number of tests to find optimal value for lambda
    n_steps: number of MC iterations in the CPT

    Returns:
    c_power: rate of correctly classified causal graphs over all old_trials
    """
    if lambs is None:
        lambs = [0, 1e-4, 1e-2, 1]
    corrects = np.zeros(n_trials)
    DAGs = {}
    p_values = np.zeros(n_trials)

    for i in range(n_trials):
        if K == 'K_ID':
            make_K = K_ID
        elif K == 'K_dft':
            make_K = K_dft
        elif K == 'K_dft1':
            make_K = K_dft1
        elif K == 'K_dft2':
            make_K = K_dft2
        elif K == 'K_dct':
            make_K = K_dct
        elif K == 'K_dwt':
            make_K = K_dwt
        elif K == 'K_CEXP':
            make_K = K_CEXP
        else:
            raise ValueError('Kernel not implemented')

        # DAGs[i], p_values[i] = eval_candidate_DAGs(X_dict[i], pred_points, n_intervals, n_perms, alpha, make_K)
        DAGs[i], p_values[i] = causal_discovery(cd_type, X_dict[i], pred_points, n_intervals, n_perms, alpha, make_K,
                                                lambs, find_lamb, n_pretests, n_steps)

        # evaluate whether candidate DAG with the greatest p-value is same as synthetically generated DAG
        if DAGs[i]==edges_dict[i]:
            corrects[i] = 1
        else:
            corrects[i] = 0

    # calculate the rate of correctly classified causal graphs
    c_power = np.mean(corrects)

    return c_power
