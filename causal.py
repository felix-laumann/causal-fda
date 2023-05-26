import numpy as np
import pandas as pd

from causal_ccm.causal_ccm import ccm
#from tigramite.pcmci import PCMCI
#from tigramite.independence_tests import GPDC
#import tigramite.data_processing as dp
from teaspoon.parameter_selection.FNN_n import FNN_n
from teaspoon.parameter_selection.MI_delay import MI_for_delay
from statsmodels.tsa.stattools import grangercausalitytests

from kernels import K_ID, K_dft, K_dft1, K_dft2, K_dct, K_dwt, K_CEXP
from graph_generation import generate_DAGs, generate_DAGs_pd, sparsify_graph, partially_direct
from graph_utils import n_DAGs
from regression import hist_linear, knn_regressor
from independence import joint_indep_test
from multiprocessing import cpu_count, get_context

from warnings import filterwarnings
filterwarnings('ignore')


# CONSTRAINT-BASED CAUSAL STRUCTURE LEARNING
def PC_alg(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, analyse, l_cond, r_opts, init, find_lambda):
    """
    PC algorithm that returns partially directed according to Meek's orientation rules given some data X_array

    Inputs:
    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in causal graph
    lambs: range to iterate over for optimal value for regularisation of kernel ridge regression to compute HSCIC
           (only for conditional independence test), or single value that's given
    n_pretests: number of tests to find optimal value for lambda
    n_perms: number of permutations performed when bootstrapping the null distribution
    n_steps: number of MC iterations in the CPT
    alpha: rejection threshold of the test
    make_K: function called to construct the kernel matrix
    l_cond: array of optimal lambda values for each size of conditional sets
    r_opts: rejection rate with optimal lambda
    init: with which method the undirected skeleton graph shall be directed ('cond_set', 'cond_indep', 'max_p')
    find_lambda: (boolean) whether lambda values are learnt newly or taken from previous experiments

    Returns:
    pd_graph: graph that is partially directed according to Meek's orientation rules
    """

    sparse_graph, full_results, sepsets_results, lamb_cond, rejects_opts = sparsify_graph(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, l_cond, r_opts, analyse, find_lambda=find_lambda)

    if analyse:
        print('Full results:', full_results)

    pd_graph = partially_direct(sparse_graph, full_results, sepsets_results, init, analyse)

    return pd_graph, lamb_cond, rejects_opts


# REGRESSION-BASED CAUSAL STRUCTURE LEARNING
def RESIT(X_array, pred_points, _DAG, n_intervals, n_neighbours, n_perms, alpha, make_K, analyse, regressor):
    """
    Regression with subsequent independence test

    Inputs:
    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in causal graph
    pred_points: prediction points
    _DAG: candidate directed acyclic graph
    n_intervals: number of intervals for the historical linear model
    n_neighbours: number of neighbours for kNN regression
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint independence test
    alpha: rejection threshold of the joint independence test
    make_K: function called to construct the kernel matrix
    analyse: (boolean) whether the regression wants to be analysed for performance; if True, R-squared value is returned
             with plots of 10 first functional samples and corresponding model predictions
    regressor: 'hist' or 'knn' to choose which regressor function to use

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
                if regressor == 'hist':
                    desc_hat[desc] += hist_linear(n_intervals, X_array[p], X_array[desc], pred_points, analyse)
                elif regressor == 'knn':
                    desc_hat[desc] += knn_regressor(n_neighbours, X_array[p], X_array[desc], pred_points, analyse)
            residuals[desc] = X_array[desc] - desc_hat[desc]

    reject, p_value = joint_indep_test(residuals, n_perms, alpha, make_K)

    return reject, p_value


def eval_candidate_DAGs(X_array, pred_points, n_intervals, n_neighbours, n_perms, alpha, make_K, analyse, regressor, pd_graph=None):
    """
    Evaluate all possible candidate directed acyclic graphs given the number of variables
    and (optional) a partially directed graph

    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in causal graph
    pred_points: prediction points
    n_intervals: number of intervals for the historical linear model
    n_neighbours: number of neighbours for kNN regression
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint independence test
    alpha: rejection threshold of the joint independence test
    make_K: function called to construct the kernel matrix
    analyse: (boolean) whether the regression wants to be analysed for performance; if True, R-squared value is returned
             with plots of 10 first functional samples and corresponding model predictions
    regressor: 'hist' or 'knn' to choose which regressor function to use
    pd_graph: partially directed graph based on constraint-based approach with Meek's orientation rules

    Returns:
    candidate DAG with largest p-value of joint independence test
    """

    n_nodes, n_samples, n_preds = X_array.shape

    if pd_graph is None:
        # generate candidate DAGs without given partially directed graph
        DAGs_dict = generate_DAGs(n_nodes, analyse)
    else:
        # generate candidate DAGs considering given partially directed graph
        DAGs_dict = generate_DAGs_pd(pd_graph)

    p_values = np.zeros(len(DAGs_dict))

    # iterate over each candidate DAG
    with get_context('spawn').Pool(8) as pool:
        jobs = [(i, _DAG, pool.apply_async(RESIT, (X_array, pred_points, _DAG, n_intervals, n_neighbours, n_perms, alpha, make_K, analyse, regressor)))
                for i, _DAG in DAGs_dict.items()]

        for i, _DAG, job in jobs:
            p_values[i] = job.get()[-1]
            if analyse:
                print('Evaluating DAG #{}: {}'.format(i, _DAG))
                print('Independence test p-value:', p_values[i])

    # return candidate DAG with greatest p-value
    max_p_val_i = np.argmax(p_values)
    return DAGs_dict[max_p_val_i], p_values[max_p_val_i]


# ALTERNATIVE METHODS
def ccm_bivariate(x_array, alpha):
    """
    Computes the convergent cross mapping according to Sugihara et al (2012): https://www.science.org/doi/10.1126/science.1227079

    Inputs:
    x_array: the input series of shape (variables, time points)
    alpha: significance level for correlation test

    Returns:
    DAG: learnt DAG
    corr_coef: the correlation coefficient
    p_value: p-value of correlation test for first variable causing second variable in x_array
    """
    n_nodes, n_obs = x_array.shape

    # take first local minimum of mutual information as lag
    x_lags = [MI_for_delay(x_array[i]) for i in range(n_nodes)]
    lag = int(np.sum(x_lags) / n_nodes)

    # use false nearest neighbour to find embedding library
    output_1, output_2 = [FNN_n(x_array[i], lag) for i in range(n_nodes)]
    x_embed_dim = int(np.sum([output_1[-1], output_2[-1]])/2)

    # test whether X causes Y
    ccm_x0x1 = ccm(x_array[0], x_array[1], lag, x_embed_dim, n_obs)
    corr_coef_x0x1, p_value_x0x1 = ccm_x0x1.causality()

    # and whether Y causes X
    ccm_x1x0 = ccm(x_array[1], x_array[0], lag, x_embed_dim, n_obs)
    corr_coef_x1x0, p_value_x1x0 = ccm_x1x0.causality()

    if p_value_x0x1 < alpha and p_value_x1x0 < alpha:
        cause = np.argmin([np.abs(1 - corr_coef_x0x1), np.abs(1 - corr_coef_x1x0)])
        if cause == 0:
            p_value = p_value_x0x1
            corr_coef = corr_coef_x0x1
            DAG = {0: [], 1: 0}
        elif cause == 1:
            p_value = p_value_x1x0
            corr_coef = corr_coef_x1x0
            DAG = {0: 1, 1: []}
        else:
            p_value = 1
            corr_coef = 0
            DAG = {0: [], 1: []}
    elif p_value_x1x0 < alpha:
        p_value = p_value_x1x0
        corr_coef = corr_coef_x1x0
        DAG = {0: 1, 1: []}
    elif p_value_x0x1 < alpha:
        p_value = p_value_x0x1
        corr_coef = corr_coef_x0x1
        DAG = {0: [], 1: 0}
    else:
        if corr_coef_x0x1 > corr_coef_x1x0:
            p_value = p_value_x0x1
            corr_coef = corr_coef_x0x1
            DAG = {0: [], 1: 0}
        elif corr_coef_x1x0 > corr_coef_x0x1:
            p_value = p_value_x1x0
            corr_coef = corr_coef_x1x0
            DAG = {0: 1, 1: []}
        else:
            p_value = 1
            corr_coef = 0
            DAG = {0: [], 1: []}

    return DAG, corr_coef, p_value, lag


def granger(x_array, alpha):
    """
    Two-way Granger-causality test to find out whether X (x_array[0]) causes Y (x_array[1]) or vice-versa

    Inputs:
    x_array: the input series of shape (variables, time points)
    alpha: confidence level

    Returns:
    DAG: learnt DAG
    p_value_x0x1: p-value of correlation test for x0 causes x1
    p_value_x1x0: p-value of correlation test for x1 causes x0
    """
    n_nodes, n_obs = x_array.shape
    df = pd.DataFrame(columns=['x0', 'x1'])
    x_lags = [MI_for_delay(x_array[i]) for i in range(n_nodes)]
    lag = int(np.sum(x_lags) / n_nodes)

    # 1. test whether Y causes X
    df['x0'] = x_array[0]
    df['x1'] = x_array[1]
    ftest_statistic_x0x1, p_value_x1x0 = grangercausalitytests(df[['x0', 'x1']], maxlag=[lag], verbose=False)[lag][0]['ssr_ftest'][0:2]

    # 2. Test whether X causes Y
    df['x0'] = x_array[1]
    df['x1'] = x_array[0]
    ftest_statistic_x1x0, p_value_x0x1 = grangercausalitytests(df[['x0', 'x1']], maxlag=[lag], verbose=False)[lag][0]['ssr_ftest'][0:2]

    if p_value_x0x1 < alpha and p_value_x1x0 < alpha:
        p_value = 1
        ftest_stat = 0
        DAG = {0: [], 1: []}
    elif p_value_x1x0 < alpha:
        p_value = p_value_x1x0
        ftest_stat = ftest_statistic_x1x0
        DAG = {0: 1, 1: []}
    elif p_value_x0x1 < alpha:
        p_value = p_value_x0x1
        ftest_stat = ftest_statistic_x0x1
        DAG = {0: [], 1: 0}
    else:
        p_value = 1
        ftest_stat = 0
        DAG = {0: [], 1: []}

    return DAG, ftest_stat, p_value, lag


def pcmci_graph(x_array, cond_indep_test):
    """
    Performs the PCMCI method as proposed by Runge et al (2018): https://arxiv.org/abs/1702.07007

    Inputs:
    x_array: the input series of shape (variables, time points)
    cond_indep_test: the conditional independence test

    Returns:
    pd_graph: the partially directed graph resulting from the PCMCI method
    """
    n_nodes, n_samples, n_obs = x_array.shape

    # prepare the data
    x_array_T = x_array.T
    #dataframe = dp.DataFrame(x_array_T, analysis_mode='multiple')

    # initialise the class
    #pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_indep_test)

    # find the optimal lag
    x_lags = [MI_for_delay(x_array[i]) for i in range(n_nodes)]
    lag = int(np.sum(x_lags) / n_nodes)

    # perform the PCMCI method
    #results = pcmci.run_pcmci(tau_max=lag, pc_alpha=None)
    #parents = pcmci.return_parents_dict(results['graph'], results['val_matrix'])

    return 0 #parents, results['p_matrix'], results['val_matrix']


# WRAPPER FUNCTION TO CALL ALL CAUSAL DISCOVERY METHODS
def causal_discovery(cd_type, X_array, pred_points, n_intervals, n_neighbours, n_perms, alpha, make_K, lambs, n_pretests,
                     n_steps, analyse, regressor, l_cond, r_opts, init, find_lambda):
    """
    Wrapper function to discover causal graph by constraint-based and regression-based approaches (as specified in cd_type)

    Inputs:
    cd_type: approach of causal discovery ('regression', 'constraint', 'combined', 'CCM', 'PCMCI', 'Granger')
    X_array: (n_nodes, n_samples, n_preds) array with data according to dependencies specified in dictionary edges
    pred_points: prediction points
    n_intervals: number of intervals for the historical linear model
    n_neighbours: number of neighbours for kNN regression
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint and conditional independence tests
    alpha: rejection threshold of the joint and conditional independence tests
    make_K: function called to construct the kernel matrix
    lambs: range of optimal lambda based on pre-tests
    n_pretests: number of tests to find optimal value for lambda
    n_steps: number of MC iterations in the CPT
    analyse: (boolean) whether the regression wants to be analysed for performance; if True, R-squared value is returned
             with plots of 10 first functional samples and corresponding model predictions
    regressor: 'hist' or 'knn' to choose which regressor function to use
    l_cond: array of optimal lambda values for each size of conditional sets
    r_opts: rejection rate with optimal lambda
    init: with which method the undirected skeleton graph shall be directed ('cond_set', 'cond_indep', 'max_p')
    find_lambda: (boolean) whether lambda values are learnt newly or taken from previous experiments

    Returns:
    _DAG: causal graph that is fully directed when 'regression' or 'combined' is used,
          and partially directed when 'constraint' is used
    p_value: p-value of joint independence test of estimated causal graph
    lamb_cond: array of potential lambda values for each size of conditional sets
    rejects_opts: rejection rate with optimal lambda
    """

    if cd_type == 'regression':
        # RESIT
        _DAGs, p_values = eval_candidate_DAGs(X_array, pred_points, n_intervals, n_neighbours, n_perms, alpha, make_K, analyse, regressor, pd_graph=None)
        lamb_cond, rejects_opts, lags, corr_values = 0, 0, 0, 0
    elif cd_type == 'constraint':
        # PC algorithm
        _DAGs, lamb_cond, rejects_opts = PC_alg(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, analyse, l_cond, r_opts, init, find_lambda)
        p_values, lags, corr_values = np.nan, 0, 0
    elif cd_type == 'combined':
        # combined: first PC algorithm, then RESIT on result of PC algorithm (i.e., on Markov equivalence class)
        pd_graph, lamb_cond, rejects_opts = PC_alg(X_array, lambs, n_pretests, n_perms, n_steps, alpha, make_K, analyse, l_cond, r_opts, init, find_lambda)
        _DAGs, p_values = eval_candidate_DAGs(X_array, pred_points, n_intervals, n_neighbours, n_perms, alpha, make_K, analyse, regressor, pd_graph=pd_graph)
        lags, corr_values = 0, 0
    elif cd_type == 'PCMCI':
        # PCMCI method
        _DAGs, p_values, corr_values = pcmci_graph(X_array, cond_indep_test=0) #GPDC())   # Conditional independence test based on Gaussian Process and Distance Correlation
        lags, lamb_cond, rejects_opts = 0, 0, 0
    elif cd_type == 'CCM':
        # convergent cross mapping
        _DAGs = []
        p_values = []
        lags = []
        for i in range(X_array.shape[1]):
            dag, corr_coef, p_value, lag = ccm_bivariate(X_array[:, i, :], alpha)
            _DAGs.append(dag)
            p_values.append(p_value)
            lags.append(lag)
        lamb_cond, rejects_opts, corr_values = 0, 0, 0
    elif cd_type == 'Granger':
        # Granger-causality test
        _DAGs = []
        p_values = []
        corr_values = []
        lags = []
        lamb_cond, rejects_opts = 0, 0
        for i in range(X_array.shape[1]):
            dag, corr_value, p_value, lag = granger(X_array[:, i, :], alpha)
            _DAGs.append(dag)
            corr_values.append(corr_value)
            p_values.append(p_value)
            lags.append(lag)
    else:
        raise ValueError('Chosen causal structure learning method not implemented. Choose between "regression", "constraint", and "combined".')

    return _DAGs, p_values, lamb_cond, rejects_opts, lags, corr_values


# EVALUATION METRICS
def precision(true_dag, cpdag):
    """
    Measures the fraction of directed edges that are correctly oriented; if no edges are oriented, return 1.
    A true positive is orienting an edge correctly, and a false positive is orienting an edge incorrectly.

    Inputs:
    true_dag: adjacency matrix of the true DAG
    cpdag: adjacency matrix of the learnt DAG

    Returns:
    precision: metric between 0 and 1
    """
    tp = len(np.argwhere((true_dag + cpdag - cpdag.T) == 2))
    fp = len(np.argwhere((true_dag + cpdag.T - cpdag) == 2))
    return tp / (tp + fp) if (tp + fp) > 0 else 1


def recall(true_dag, cpdag):
    """
    Measures the fraction of all edges that are oriented; if no edges are oriented, return 0

    Inputs:
    true_dag: adjacency matrix of the true DAG
    cpdag: adjacency matrix of the learnt DAG

    Returns:
    recall: metric between 0 and 1
    """
    tp = len(np.argwhere((true_dag + cpdag - cpdag.T) == 2))
    return tp / np.sum(true_dag)


def f1_score(pre, rec):
    return 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0


def shd(true_dag, dag):
    """
    Compute Structural Hemming Distance

    Inputs:
    true_dag: the underlying DAG that was taken to generate the data
    dag: the DAG that was learnt by the causal structure learning algorithm

    Returns:
    sum(diff): the sum of differences between the underlying true DAG and the learnt DAG
    """
    diff = np.abs(true_dag - dag)
    return np.sum(diff)


def norm_shd(true_dag, dag):
    """
    Compute Structural Hemming Distance normalised by maximum number of mistaken edges

    Inputs:
    true_dag: the underlying DAG that was taken to generate the data
    dag: the DAG that was learnt by the causal structure learning algorithm

    Returns:
    sum(diff): the sum of differences between the underlying true DAG and the learnt DAG
    """
    diff = np.abs(true_dag - dag)
    norm = true_dag.shape[0] * (true_dag.shape[0] - 1)
    return np.sum(diff) / norm


def global_learning(true_dag, dag):
    """
    Compute the Global Learning (Colace et al., 2004)

    Inputs:
    true_dag: the underlying DAG that was taken to generate the data
    dag: the DAG that was learnt by the causal structure learning algorithm

    Returns:
    the Global Learning metric (which is normalised by definition)
    """
    # correctly oriented ddges
    correct_edges = 0
    # incorrect edges (missing or wrongly oriented)
    wrong_edges = 0
    # number of nodes
    n_nodes = np.shape(dag)[0]

    for i in range(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            if (not (dag[i][j] != true_dag[i][j])) and (not (dag[j][i] != true_dag[j][i])):

                if dag[i][j] or dag[j][i]:
                    correct_edges += 1
            else:
                wrong_edges += 1
    dist = correct_edges / (correct_edges + wrong_edges)
    return dist


def causal_eval(cd_type, X_dict, edges_dict, upper_limit=1, n_preds=100, n_intervals=8, n_neighbours=5, n_trials=200, n_perms=1000,
                alpha=0.05, K='K_ID', lambs=[1e-4, 1e-3], n_pretests=100, n_steps=50, analyse=False, regressor='hist',
                init='cond_set', find_lambda=False):
    """
    Evaluates the causal discovery algorithms based on precision, recall and F1-score

    Inputs:
    cd_type: approach of causal discovery ('regression', 'constraint', 'combined', 'CCM', 'PCMCI', 'Granger')
    X_dict: dictionary of data sets that were generated according to random DAG for each trial
    edges_dict: dictionary of "ground truth" edges for each trial that is taken as a base to generate the data X_dict;
                each value in the dictionary is a DAG represented by one list for each node, where parents are written
                behind a "|" sign (for example, "[0|1]" means that node 0 is the descendant and node 1 its parent)
    upper_limit: upper limit of data
    n_preds: number of prediction points
    n_intervals: number of intervals for the historical linear model
    n_neighbours: number of neighbours for kNN regression
    n_trials: number of old_trials to compute percentage of rejections over
    n_perms: number of permutations performed when bootstrapping the null distribution of the joint independence test
    alpha: rejection threshold of the joint independence test
    K: function called to construct the kernel matrix
    lambs: range of optimal lambda based on pre-tests
    n_pretests: number of tests to find optimal value for lambda
    n_steps: number of MC iterations in the CPT
    analyse: (boolean) whether the algorithm including regression wants to be analysed for performance;
             if True, multiple values are printed including the true underlying DAG, the R-squared value
             and plots of 10 first functional samples with corresponding model predictions
    regressor: 'hist' or 'knn' to choose which regressor function to use
    init: with which method the undirected skeleton graph shall be directed ('cond_set', 'cond_indep', 'max_p')
    find_lambda: (boolean) whether lambda values are learnt newly or taken from previous experiments

    Returns:
    precisions, recalls, f1_scores, SHDs, CPDAGs, p_values
    """
    n_vars, n_samples, n_obs = X_dict[0].shape
    pred_points = np.linspace(0, upper_limit, n_preds)

    CPDAG = {}
    p_value = np.zeros(n_trials, dtype=object)
    corr_values = {}
    precisions = np.zeros(n_trials)
    recalls = np.zeros(n_trials)
    f1_scores = np.zeros(n_trials)
    SHDs = np.zeros((n_trials, n_samples))
    GLs = np.zeros((n_trials, n_samples))

    l_cond = np.zeros(n_vars - 2)
    r_opts = np.zeros(n_vars - 2)

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

        if (X_dict[i].shape[0] == 2) and (cd_type == 'constraint' or cd_type == 'combined'):
            raise ValueError('Constraint and combined causal structure learning '
                             'only feasible with more than 2 variables.')

        if analyse:
            print('-------------')
            print('True DAG:', edges_dict[i])

        CPDAG[i], p_value[i], lamb_cond, rejects_opts, lags, corr_values[i] = causal_discovery(cd_type, X_dict[i], pred_points, n_intervals,
                                                                                               n_neighbours, n_perms, alpha, make_K, lambs,
                                                                                               n_pretests, n_steps, analyse, regressor,
                                                                                               l_cond, r_opts, init, find_lambda)

        # updating l_cond and r_opts with learnt optimal lambda values
        l_cond, r_opts = lamb_cond, rejects_opts

        # transform true DAG and learnt CPDAG into adjacency matrix where rows are cause and columns are effect variables
        DAG_adj = np.zeros((len(edges_dict[i].to_nx().nodes()), len(edges_dict[i].to_nx().nodes())))
        for edge in edges_dict[i].to_nx().edges():
            DAG_adj[edge] = 1

        if cd_type == 'CCM' or cd_type == 'Granger':
            SHDs_list = []
            GLs_list = []
            # iterate over all n_samples causal structures that were learnt with CCM or Granger causality
            for cpdag in CPDAG[i]:
                CPDAG_adj = np.zeros((len(edges_dict[i].to_nx().nodes()), len(edges_dict[i].to_nx().nodes())))

                for d, p in cpdag.items():
                    CPDAG_adj[p, d] = 1

                # calculate SHD
                SHDs_list.append(shd(DAG_adj, CPDAG_adj))
                GLs_list.append(global_learning(DAG_adj, CPDAG_adj))

            SHDs[i] = np.asarray(SHDs_list)
            GLs[i] = np.asarray(GLs_list)
            if analyse:
                print('SHD:', SHDs[i])

        elif cd_type == 'PCMCI':
            CPDAG_adj = np.zeros((len(edges_dict[i].to_nx().nodes()), len(edges_dict[i].to_nx().nodes())))
            strengths = {}
            for d, p in CPDAG[i].items():
                strengths[d] = {}
                for p_i in p:
                    if p_i[0] != d:
                        strengths[d][p_i[0]] = 0
                for p_i in p:
                    if p_i[0] != d:   # ignore self-loops
                        strengths[d][p_i[0]] += corr_values[i][p_i[0]][d][abs(p_i[1])]

            for d_i in strengths.keys():
                for d_j in strengths.keys():
                    if d_i == d_j:
                        continue
                    else:
                        if (d_j in strengths[d_i].keys()) and (d_i in strengths[d_j].keys()):
                            if strengths[d_i][d_j] > strengths[d_j][d_i]:
                                CPDAG_adj[d_j, d_i] = 1
                            elif strengths[d_j][d_i] > strengths[d_i][d_j]:
                                CPDAG_adj[d_i, d_j] = 1
                            else:
                                CPDAG_adj[d_i, d_j] = 0
                        elif (d_j in strengths[d_i].keys()) and (d_i not in strengths[d_j].keys()):
                            CPDAG_adj[d_j, d_i] = 1
                        elif (d_j not in strengths[d_i].keys()) and (d_i in strengths[d_j].keys()):
                            CPDAG_adj[d_i, d_j] = 1
                        else:
                            CPDAG_adj[d_i, d_j] = 0
                            CPDAG_adj[d_j, d_i] = 0

            # calculate SHD
            SHDs[i] = shd(DAG_adj, CPDAG_adj)
            GLs[i] = global_learning(DAG_adj, CPDAG_adj)
            if analyse:
                print('SHD:', SHDs[i][0])

        else:
            CPDAG_adj = np.zeros((len(edges_dict[i].to_nx().nodes()), len(edges_dict[i].to_nx().nodes())))
            for d, p in CPDAG[i].items():
                CPDAG_adj[p, d] = 1

            # calculate SHD
            SHDs[i] = shd(DAG_adj, CPDAG_adj)
            GLs[i] = global_learning(DAG_adj, CPDAG_adj)
            if analyse:
                if cd_type == 'regression':
                    print('Learned DAG:', CPDAG[i])
                    print('SHD:', SHDs[i][0])

        # calculate precision and recall
        precisions[i] = precision(DAG_adj, CPDAG_adj)
        recalls[i] = recall(DAG_adj, CPDAG_adj)

        # calculate F1-score
        f1_scores[i] = f1_score(precisions[i], recalls[i])

        if analyse:
            if cd_type == 'constraint':
                print('Precision:', precisions[i])
                print('Recall:', recalls[i])
                print('SHD:', SHDs[i][0])

    return precisions, recalls, f1_scores, SHDs, GLs, CPDAG, p_value
