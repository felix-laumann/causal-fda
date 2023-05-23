import argparse
import sys
sys.path.append('..')
import numpy as np
import pickle
import os
import pandas as pd

from synthetic_data import generate_data
from independence import opt_lambda, test_power
from plots import plot_samples, plot_power, type_I_boxplot, plot_cross_corr, plot_delay

from warnings import filterwarnings
filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--period', type=float)
    parser.add_argument('--test', type=str)
    parser.add_argument('--n_vars', type=int)

    args = parser.parse_args()

    test = args.test

    # number of trials and permutations
    n_trials = args.n_trials
    n_perms = 1000

    # number of samples and number of points functional data samples are (randomly) observed and discretised
    n_samples = args.n_samples
    n_obs = 100
    n_preds = 100

    # historical dependence is easier to detect the higher a is
    a_list = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # define discretised period
    upper_limit = 1
    period = args.period
    pred_points = np.linspace(0, upper_limit, n_preds)

    # number of Fourier basis functions and std of normal distribution of sampled coefficients
    n_basis = 3
    sd = 1

    # statistical significance level
    alpha = 0.05

    # number of variables in network (for joint independence test)
    n_vars = args.n_vars

    # create folders to save results
    if not os.path.exists('results'):
        os.mkdir('results')

    if not os.path.exists('results/marginal'):
        os.mkdir('results/marginal')

    if not os.path.exists('results/joint'):
        os.mkdir('results/joint')

    if not os.path.exists('results/conditional'):
        os.mkdir('results/conditional')

    # marginal independence test
    if test == 'marginal':

        type_II_errors = {}

        print('Period T:', period)
        type_II_errors[period] = {}
        for n_sample in n_samples:
            print('Sample size:', int(n_sample))
            type_II_errors[period][int(n_sample)] = []
            for a in a_list:
                print('a:', a)
                # generate synthetic data
                X, Y = generate_data(dep=test, n_samples=int(n_sample), n_trials=n_trials, n_obs=n_obs, n_preds=n_preds, period=period, a=a, upper_limit=upper_limit, n_basis=n_basis, sd=sd)

                # conduct n tests
                power = test_power(X=X, Y=Y, n_trials=n_trials, n_perms=n_perms, alpha=alpha, K='K_ID', test=test)
                type_II_errors[period][n_sample].append(power)
                print('Test power:', power)
                print('----------')
            print('----------')

        power_hist = open('results/{}/test_power_hist_{}.pkl'.format(test, test), 'wb')
        pickle.dump(type_II_errors, power_hist)
        power_hist.close()

    # joint independence test
    if test == 'joint':

        type_II_errors = {}

        print('Period T:', period)
        type_II_errors[period] = {}
        for n_sample in n_samples:
            print('Sample size:', int(n_sample))
            type_II_errors[period][int(n_sample)] = []
            for a in a_list:
                print('a:', a)
                # generate synthetic data
                edges_dict, X_dict = generate_data(dep=test, n_samples=int(n_sample), n_trials=n_trials, n_obs=n_obs, n_preds=n_preds, period=period, n_vars=n_vars, a=a, upper_limit=upper_limit, n_basis=n_basis, sd=sd)

                # conduct n trials
                power = test_power(X=X_dict, edges_dict=edges_dict, n_trials=n_trials, n_perms=n_perms, alpha=alpha, K='K_ID', test=test)
                type_II_errors[period][n_sample].append(power)
                print('Test power:', power)
                print('----------')
            print('----------')

        power_hist = open('results/{}/test_power_hist_{}.pkl'.format(test, test), 'wb')
        pickle.dump(type_II_errors, power_hist)
        power_hist.close()

    # conditional independence test
    if test == 'conditional':

        # range of possible values for lambda
        lambs = [5e-4, 7e-4, 2e-3, 5e-3, 7e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 7e-2]

        n_pretests = 100
        n_steps = 50

        a_prime_list = a_list

        type_II_errors = {}
        lamb_opts = {}

        print('Period T:', period)
        type_II_errors[period] = {}
        lamb_opts[period] = {}

        print('Sample size:', int(n_samples))
        type_II_errors[period][int(n_samples)] = {}
        lamb_opts[period][int(n_samples)] = {}

        print('Number of conditional variables:', n_vars)
        type_II_errors[period][int(n_samples)][n_vars] = []
        lamb_opts[period][int(n_samples)][n_vars] = []
        for i_a, a_prime in enumerate(a_prime_list):
            print("a':", a_prime)
            # generate synthetic data
            X, Y, Z = generate_data(dep=test, n_samples=int(n_samples), n_trials=n_trials, n_obs=n_obs, n_preds=n_preds, period=period, n_vars=n_vars, a=1, a_prime=a_prime, upper_limit=upper_limit, n_basis=n_basis, sd=sd)

            lamb_opts = pickle.load(open('lambs_opt_conditional.pkl', 'rb'))
            lamb_opt = lamb_opts[n_vars][n_samples]

            # conduct n trials
            power = test_power(X=X, Y=Y, Z=Z, n_trials=n_trials, n_perms=n_perms, alpha=alpha, K='K_ID', test=test, lamb_opt=lamb_opt)
            type_II_errors[period][n_samples][n_vars].append(power)
            lamb_opts[period][n_samples][n_vars].append(lamb_opt)
            print('Test power:', power)
            print('----------')
        print('----------')

        power_hist = open('results/{}/test_power_hist_{}_{}_{}.pkl'.format(test, test, n_vars, n_samples), 'wb')
        pickle.dump(type_II_errors, power_hist)
        power_hist.close()
        lambs_opt_hist = open('results/{}/lambs_opt_{}_{}_{}.pkl'.format(test, test, n_vars, n_samples), 'wb')
        pickle.dump(lamb_opts, lambs_opt_hist)
        lambs_opt_hist.close()

