import numpy as np
from tqdm.notebook import tqdm
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


def RESIT(X_array, pred_points, _DAG, n_intervals, n_perms, alpha, make_K, analyse):
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
    analyse: (boolean) whether the regression wants to be analysed for performance; if True, R-squared value is returned
             with plots of 10 first functional samples and corresponding model predictions

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
                desc_hat[desc] += hist_linear(n_intervals, X_array[p], X_array[desc], pred_points, analyse)
            residuals[desc] = X_array[desc] - desc_hat[desc]

    reject, p_value = joint_indep_test(residuals, n_perms, alpha, make_K)

    return reject, p_value


def eval_candidate_DAGs(X_array, pred_points, n_intervals, n_perms, alpha, make_K, analyse, pd_graph=None):
    """
    Evaluate all possible candidate directed acyclic graphs given the number of variables
    and (optional) a partially directed graph

    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in causal graph
    pred_points: prediction points
    n_intervals: number of intervals for the historical linear model
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint independence test
    alpha: rejection threshold of the joint independence test
    make_K: function called to construct the kernel matrix
    analyse: (boolean) whether the regression wants to be analysed for performance; if True, R-squared value is returned
             with plots of 10 first functional samples and corresponding model predictions
    pd_graph: partially directed graph based on constraint-based approach with Meek's orientation rules

    Returns:
    candidate DAG with largest p-value of joint independence test
    """

    n_nodes, n_samples, n_preds = X_array.shape

    # calculate the number of DAGs that can be constructed with n nodes
    _n_DAGs = n_DAGs(n_nodes)

    rejects = np.zeros(_n_DAGs)
    p_values = np.zeros(_n_DAGs)

    if pd_graph is None:
        # generate candidate DAGs without partially directed graph
        DAGs_dict = generate_DAGs(n_nodes)
    else:
        # generate candidate DAGs considering partially directed graph
        DAGs_dict = generate_DAGs_pd(pd_graph)

    ### HERE: CAN THIS FOR-LOOP BE COMPUTED IN PARALLEL? With multiprocessing?

    # iterate over each candidate DAG
    for i, _DAG in tqdm(DAGs_dict.items()):
        rejects[i], p_values[i] = RESIT(X_array, pred_points, _DAG, n_intervals, n_perms, alpha, make_K, analyse)
        if analyse is True:
            print('Evaluating DAG #{}: {}'.format(i, _DAG))
            print(p_values[i])

    ### END

    # return candidate DAG with greatest p-value
    return DAGs_dict[np.argmax(p_values)], np.max(p_values)


def causal_discovery(cd_type, X_array, pred_points, n_intervals, n_perms, alpha, make_K, lambs, find_lamb, n_pretests, n_steps, analyse):
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
    analyse: (boolean) whether the regression wants to be analysed for performance; if True, R-squared value is returned
             with plots of 10 first functional samples and corresponding model predictions

    Returns:
    _DAG: causal graph that is fully directed when 'regression' or 'combined' is used,
          and partially directed when 'constraint' is used
    p_value: p-value of joint independence test of estimated causal graph
    """

    if cd_type=='regression':
        # RESIT
        _DAG, p_value = eval_candidate_DAGs(X_array, pred_points, n_intervals, n_perms, alpha, make_K, analyse, pd_graph=None)
    elif cd_type=='constraint':
        # PC algorithm
        _DAG = PC_alg(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, find_lamb)
        p_value = np.nan
    elif cd_type=='combined':
        # combined: first PC algorithm, then RESIT on result of PC algorithm (i.e., on Markov equivalence class)
        pd_graph = PC_alg(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, find_lamb)
        _DAG, p_value = eval_candidate_DAGs(X_array, pred_points, n_intervals, n_perms, alpha, make_K, analyse, pd_graph=pd_graph)
    else:
        _DAG = {}
        raise ValueError('Chosen causal structure learning method not implemented. Choose between "regression", "constraint", and "combined".')

    return _DAG, p_value


# EVALUATION METRICS
def precision(true_dag, cpdag):
    tp = len(np.argwhere((true_dag + cpdag - cpdag.T) == 2))
    fp = len(np.argwhere((true_dag + cpdag.T - cpdag) == 2))
    return tp / (tp + fp) if (tp + fp) > 0 else 1


def recall(true_dag, cpdag):
    tp = len(np.argwhere((true_dag + cpdag - cpdag.T) == 2))
    return tp / np.sum(true_dag)


def f1_score(pre, rec):
    return 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0


def causal_eval(cd_type, X_dict, edges_dict, pred_points=np.linspace(0, 1, 100), n_intervals=8, n_trials=200, n_perms=1000,
                alpha=0.05, K='K_ID', lambs=np.array([1, 1e-2, 1e-4]), find_lamb=True, n_pretests=100, n_steps=50, analyse=False):
    """
    Evaluates the causal discovery algorithms based on precision, recall and F1-score

    Inputs:
    cd_type: approach of causal discovery ('regression', 'constraint', 'combined')
    X_dict: dictionary of data sets that were generated according to random DAG for each trial
    edges_dict: dictionary of "ground truth" edges for each trial that is taken as a base to generate the data X_dict;
                each value in the dictionary is a DAG represented by one list for each node, where parents are written
                behind a "|" sign (for example, "[0|1]" means that node 0 is the descendant and node 1 its parent)
    pred_points: prediction points
    n_intervals: number of intervals for the historical linear model
    n_trials: number of old_trials to compute percentage of rejections over
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint independence test
    alpha: rejection threshold of the joint independence test
    K: function called to construct the kernel matrix
    lambs: range to iterate over for optimal value for regularisation of kernel ridge regression to compute HSCIC
           (only for constraint-based causal discovery)
    find_lamb: (boolean) whether optimal lambda should be searched for or not; if not, set to value specified in lambs
    n_pretests: number of tests to find optimal value for lambda
    n_steps: number of MC iterations in the CPT
    analyse: (boolean) whether the algorithm including regression wants to be analysed for performance;
             if True, multiple values are printed including the true underlying DAG, the R-squared value
             and plots of 10 first functional samples with corresponding model predictions

    Returns:
    c_eval: tuple of precision, recall and F1-score
    """
    if lambs is None:
        lambs = [1, 1e-2, 1e-4]
    CPDAGs = {}
    p_values = np.zeros(n_trials)
    precisions = np.zeros(n_trials)
    recalls = np.zeros(n_trials)
    f1_scores = np.zeros(n_trials)

    for i in tqdm(range(n_trials)):
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

        # transform DAG into adjacency matrix where rows are cause and columns are effect variables
        DAG_adj = np.zeros((len(edges_dict[i].to_nx().nodes()), len(edges_dict[i].to_nx().nodes())))
        for edge in edges_dict[i].to_nx().edges():
            DAG_adj[edge] = 1

        CPDAGs[i], p_values[i] = causal_discovery(cd_type, X_dict[i], pred_points, n_intervals, n_perms, alpha, make_K,
                                                  lambs, find_lamb, n_pretests, n_steps, analyse)

        # transform CPDAG into adjacency matrix where rows are cause and columns are effect variables
        CPDAG_adj = np.zeros((len(edges_dict[i].to_nx().nodes()), len(edges_dict[i].to_nx().nodes())))
        for d, p in CPDAGs[i].items():
            CPDAG_adj[p, d] = 1

        # calculate precision and recall
        precisions[i] = precision(DAG_adj, CPDAG_adj)
        recalls[i] = recall(DAG_adj, CPDAG_adj)

        # calculate F1-score
        f1_scores[i] = f1_score(precisions[i], recalls[i])

    return precisions, recalls, f1_scores, CPDAGs, p_values
