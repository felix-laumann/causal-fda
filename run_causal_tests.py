import argparse
import sys
sys.path.append('..')
import os
import shutil
import numpy as np
import pickle
from synthetic_data import generate_data
from causal import causal_eval

from warnings import filterwarnings
filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--period', type=float)
    parser.add_argument('--test', type=str)
    parser.add_argument('--n_vars', type=int)
    parser.add_argument('--cd_type', type=str)
    parser.add_argument('--a', type=float)

    args = parser.parse_args()

    # number of trials and permutations
    n_trials = args.n_trials
    n_perms = 1000

    # number of samples and number of points functional data samples are (randomly) observed and discretised
    n_samples = args.n_samples
    n_obs = 100
    n_preds = 100

    # define discretised mesh of points
    upper_limit = 1
    pred_points = np.linspace(0, upper_limit, n_preds)

    # data paramterers
    period = args.period
    n_basis = 3
    sd = 1

    # statistical significance level
    alpha = 0.05

    # create folders to save results
    if not os.path.exists('results'):
        os.mkdir('results')

    if not os.path.exists('results/causal'):
        os.mkdir('results/causal')

    # Parameters specific for evaluation on synthetic data
    test = args.test
    n_vars = args.n_vars
    prob = 0.5
    cd_type = args.cd_type

    # historical dependence is easier to detect the higher a is
    a = args.a

    # regression parameters
    n_intervals = 12
    analyse = False

    # constraint parameters
    lambs = [1e-5, 1e-4, 1e-3]
    #lambs = 2.5e-4
    n_steps = 50
    n_pretests = 100

    log_sys = False

    linear = 0


    # saving evaluation metrics
    precisions_dict = {}
    recalls_dict = {}
    f1_scores_dict = {}
    SHDs_dict = {}
    GLs_dict = {}
    averages_dict = {}

    # saving DAGs + p-values
    edges_dict = {}
    DAGs_dict = {}
    p_values_dict = {}


    print('Period T:', period)
    precisions_dict[period] = {}
    recalls_dict[period] = {}
    f1_scores_dict[period] = {}
    SHDs_dict[period] = {}
    GLs_dict[period] = {}
    averages_dict[period] = {}
    edges_dict[period] = {}
    DAGs_dict[period] = {}
    p_values_dict[period] = {}

    print('Sample size:', n_samples)
    precisions_dict[period][n_samples] = {}
    recalls_dict[period][n_samples] = {}
    f1_scores_dict[period][n_samples] = {}
    SHDs_dict[period][n_samples] = {}
    GLs_dict[period][n_samples] = {}
    averages_dict[period][n_samples] = {}
    edges_dict[period][n_samples] = {}
    DAGs_dict[period][n_samples] = {}
    p_values_dict[period][n_samples] = {}

    precisions_dict[period][n_samples][a] = []
    recalls_dict[period][n_samples][a] = []
    f1_scores_dict[period][n_samples][a] = []
    SHDs_dict[period][n_samples][a] = []
    GLs_dict[period][n_samples][a] = []
    averages_dict[period][n_samples][a] = []
    edges_dict[period][n_samples][a] = []
    DAGs_dict[period][n_samples][a] = []
    p_values_dict[period][n_samples][a] = []

    # generate synthetic data
    edges, X = generate_data(dep=test, n_samples=n_samples, n_trials=n_trials, n_obs=n_obs, n_preds=n_preds, period=period, n_vars=n_vars, a=a, upper_limit=upper_limit, n_basis=n_basis, sd=sd, prob=prob, linear=linear, log_sys=log_sys, analyse=analyse)
    # conduct n trials
    precisions, recalls, f1_scores, SHDs, GLs, CPDAGs, p_values = causal_eval(cd_type=cd_type, X_dict=X, edges_dict=edges, upper_limit=upper_limit, n_preds=n_preds, n_intervals=n_intervals, n_trials=n_trials, n_perms=n_perms, alpha=alpha, K='K_ID', lambs=lambs, analyse=analyse)
    precisions_dict[period][n_samples][a].append(precisions)
    recalls_dict[period][n_samples][a].append(recalls)
    f1_scores_dict[period][n_samples][a].append(f1_scores)
    SHDs_dict[period][n_samples][a].append(SHDs)
    GLs_dict[period][n_samples][a].append(GLs)

    edges_dict[period][n_samples][a].append(edges)
    DAGs_dict[period][n_samples][a].append(CPDAGs)
    p_values_dict[period][n_samples][a].append(p_values)

    # calculate average precision, recall and F1-score
    avg_precision = np.mean(precisions_dict[period][n_samples][a])
    avg_recall = np.mean(recalls_dict[period][n_samples][a])
    avg_f1_score = np.mean(f1_scores_dict[period][n_samples][a])
    avg_SHDs = np.mean(SHDs_dict[period][n_samples][a])
    avg_GLs = np.mean(GLs_dict[period][n_samples][a])

    averages_dict[period][n_samples][a].append([avg_precision, avg_recall, avg_f1_score, avg_SHDs, avg_GLs])

    print('Average SHD:', avg_SHDs)
    print('Average Precision:', avg_precision)
    print('Average Recall:', avg_recall)
    print('----------')

    precision_causal = open('results/causal/precision_{}_{}_{}.pkl'.format(cd_type, n_vars, n_samples), 'wb')
    pickle.dump(precisions_dict, precision_causal)
    precision_causal.close()
    recall_causal = open('results/causal/recall_{}_{}_{}.pkl'.format(cd_type, n_vars, n_samples), 'wb')
    pickle.dump(recalls_dict, recall_causal)
    recall_causal.close()
    f1_causal = open('results/causal/f1_{}_{}_{}.pkl'.format(cd_type, n_vars, n_samples), 'wb')
    pickle.dump(f1_scores_dict, f1_causal)
    f1_causal.close()
    SHD_causal = open('results/causal/shd_{}_{}_{}.pkl'.format(cd_type, n_vars, n_samples), 'wb')
    pickle.dump(SHDs_dict, SHD_causal)
    SHD_causal.close()
    GL_causal = open('results/causal/GL_{}_{}_{}.pkl'.format(cd_type, n_vars, n_samples), 'wb')
    pickle.dump(GLs_dict, GL_causal)
    GL_causal.close()
    averages_causal = open('results/causal/averages_{}_{}_{}.pkl'.format(cd_type, n_vars, n_samples), 'wb')
    pickle.dump(averages_dict, averages_causal)
    averages_causal.close()

    edges_causal = open('results/causal/edges_{}_{}_{}.pkl'.format(cd_type, n_vars, n_samples), 'wb')
    pickle.dump(edges_dict, edges_causal)
    edges_causal.close()
    DAGs_causal = open('results/causal/DAGs_{}_{}_{}.pkl'.format(cd_type, n_vars, n_samples), 'wb')
    pickle.dump(DAGs_dict, DAGs_causal)
    DAGs_causal.close()
    pvalues_causal = open('results/causal/p_values_{}_{}_{}.pkl'.format(cd_type, n_vars, n_samples), 'wb')
    pickle.dump(p_values_dict, pvalues_causal)
    pvalues_causal.close()

    #shutil.copytree('results', '~/results_1', dirs_exist_ok=True)
