import argparse
import numpy as np
import pandas as pd
from causal import causal_discovery
from kernels import K_ID
from synthetic_data import spline_multi_sample
import pickle
import os

from warnings import filterwarnings
filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cd_type', type=str)
    parser.add_argument('--find_lambda', type=bool)

    args = parser.parse_args()

    # data
    df_SDG = pd.read_excel(r'../data/sdgindexdataset.xlsx', sheet_name='Raw Data - Trend Indicators').drop(columns=['id', 'indexreg', 'Population'])

    indicators = df_SDG.drop(columns=['Country', 'Year']).columns.tolist()
    no_countries = ['Upper-middle-income Countries', 'Western Balkans',
                    'Small Island Developing States', 'OECD members', 'Oceania',
                    'Middle East and North Africa', 'Low-income Countries',
                    'Lower-middle-income Countries', 'Latin America and the Caribbean',
                    'High-income Countries', 'East and South Asia',
                    'Eastern Europe and Central Asia', 'Sub-Saharan Africa']

    countries = []
    for country in list(pd.unique(df_SDG['Country'])):
        if country not in no_countries:
            countries.append(country)

    years = pd.unique(df_SDG['Year']).tolist()

    # interpolate and save data in array
    np_SDG_ind = np.zeros((len(indicators), len(countries), len(years)))

    for i_ind, ind in enumerate(indicators):
        for i_c, country in enumerate(countries):
            if 1 < sum(df_SDG[df_SDG['Country'] == country][ind].isna() == False) <= 2:
                m = 'spline'
                o = 1
            elif sum(df_SDG[df_SDG['Country'] == country][ind].isna() == False) <= 1:
                m = 'linear'
                o = 2
            else:
                m = 'spline'
                o = 2

            # standardise data
            # df_series = (df_SDG[df_SDG['Country']==country][ind] - df_SDG[df_SDG['Country']==country][ind].mean()) / df_SDG[df_SDG['Country']==country][ind].std()

            # unstandardised
            df_series = df_SDG[df_SDG['Country'] == country][ind]

            s = df_series.interpolate(method=m, order=o, limit_direction='both',
                                      limit=len(df_SDG[df_SDG['Country'] == country][ind])).tolist()
            np_SDG_ind[i_ind, i_c] = s

    # define the indicators that belong to the same SDG
    dict_indicators = {}

    for goal in np.arange(1, 18):
        dict_indicators[goal] = []
        for ind in indicators:
            if ind.startswith(('sdg' + str(goal) + '_')):
                dict_indicators[goal].append(ind)

    # concatenate indicator data that belong to the same SDG, resulting in shape (17, n_countries, n_years*n_indicators),
    dict_goals = {}

    for goal in np.arange(1, 18):
        dict_goals[goal] = {}
        for i_c, country in enumerate(countries):
            dict_goals[goal][country] = []
            for i_ind, ind in enumerate(dict_indicators[goal]):
                dict_goals[goal][country].extend(list(np_SDG_ind[i_ind, i_c]))

    dict_g = {}

    for goal in np.arange(1, 18):
        dict_g[goal] = {}
        for i_c, country in enumerate(countries):
            if np.isnan(dict_goals[goal][country]).all():
                s = np.zeros((len(dict_goals[goal][country])))
            else:
                s = pd.Series(dict_goals[goal][country]).interpolate(method='linear', order=2, limit_direction='both', limit=len(dict_goals[goal][country])).tolist()
            dict_g[goal][country] = s

    # finding countries that do not have any data for one of the indicators
    countries_to_del = []
    for i_c, country in enumerate(countries):
        for goal in np.arange(1, 18):
            if len(dict_g[goal][country]) == 0:
                countries_to_del.append(country)

    countries_to_del = list(np.unique(countries_to_del))

    # deleting those countries that do not have any data for one of the indicators
    for country in countries_to_del:
        for goal in np.arange(1, 18):
            dict_g[goal].pop(country)

    countries_left = list(dict_g[1].keys())
    n_vars = len(dict_g.keys())
    n_samples = len(countries_left)

    # experiment parameters
    cd_type = args.cd_type
    pred_points = np.linspace(0, 1, 100)
    n_intervals = 12
    n_neighbours = 3
    n_perms = 1000
    alpha = 0.05
    make_K = K_ID
    lambs = [1e-3, 1e-2, 1e-1]
    n_pretests = 100
    n_steps = 50
    analyse = True
    regressor = 'hist'
    init = 'cond_set'
    find_lambda = args.find_lambda

    l_cond = np.zeros(n_vars - 2)
    r_opts = np.zeros(n_vars - 2)

    # data preparation
    g_pred = np.zeros((n_vars, n_samples, len(pred_points)))

    for sdg in range(g_pred.shape[0]):
        obs_points = np.tile(np.linspace(0, 1, len(dict_g[sdg + 1]['France'])), (n_samples, 1))

        np_g = np.zeros((n_samples, len(dict_g[sdg + 1]['France'])))
        for i_c, c in enumerate(countries_left):
            np_g[i_c] = dict_g[sdg + 1][c]

        g_pred[sdg] = spline_multi_sample(np_g, obs_points, pred_points).evaluate(pred_points).squeeze()

    sparse_g, _DAGs, p_values, lamb_cond, rejects_opts, lags, corr_values = causal_discovery(cd_type, g_pred, pred_points, n_intervals, n_neighbours, n_perms, alpha, make_K, lambs, n_pretests,
                                                                                             n_steps, analyse, regressor, l_cond, r_opts, init, find_lambda, pd_graph=None)
