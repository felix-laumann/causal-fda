import numpy as np
from scipy.integrate import cumulative_trapezoid
from skfda.representation.basis import Fourier, FDataBasis
from skfda.representation.grid import FDataGrid
from skfda.representation.interpolation import SplineInterpolation

from graph_generation import generate_DAGs


# define meshes
def sample_points(n_samples, n_obs, upper_limit, random_state=None):
    """
    Randomly samples the observation points for each function sample

    Inputs:
    n_samples: number of function samples
    n_obs: number of observations per function sample
    random_state: random seed

    Returns:
    obs_points: observation point locations for each random function

    """
    rng = np.random.RandomState(random_state)
    obs_grid = rng.uniform(0, upper_limit, (n_samples, n_obs))

    obs_grid.sort(axis=1)
    return obs_grid


# interpolation with splines
def spline_single_sample(X, obs_points, pred_points, int_order, smooth_param):
    """
    Given a single sample of functional data X at locations obs_points, a spline is fit and predictions are
    given at locations pred_points

    Inputs:
    X: single function sample
    obs_points: function sample observation locations
    pred_points: prediction locations
    smooth_param: smoothness penalty

    Returns:
    The predictions of a spline with smoothness parameter at locations pred_points
    """
    X_fd = FDataGrid(data_matrix=X, sample_points=obs_points)
    X_fd.interpolation = SplineInterpolation(interpolation_order=int_order, smoothness_parameter=smooth_param)
    return X_fd.evaluate(pred_points)


def spline_multi_sample(X, obs_points, pred_points, int_order=5, smooth_param=0):
    """
    Given multiple samples of functional data X, an array of corresponding observation locations obs_points,
    splines are fit and predictions are given at locations pred_points

    Inputs:
    X: multiple function samples
    obs_points: function samples observation locations
    pred_points: prediction locations
    smooth_param: smoothness penalty

    Returns:
    The predictions of splines with smoothness penalty at locations pred_points
    """
    X_reg = np.zeros((X.shape[0], len(pred_points)))
    for j in range(X.shape[0]):
        X_reg[j, :] = spline_single_sample(X[j, :], obs_points[j, :], pred_points, int_order, smooth_param).squeeze()
    return FDataGrid(data_matrix=X_reg, sample_points=pred_points.T)


# using skfda to generate functional data
def skfda_basis(n_samples, upper_limit, period, n_basis, sd):
    basis = Fourier((0, upper_limit), n_basis=n_basis, period=period)
    coefficients = np.random.normal(0, sd, size=(n_samples, n_basis))

    return FDataBasis(basis, coefficients)


# mean function for historically dependent data
def mean_func(t, mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4):
    if (mll1==-1) and (mul1==-1) and (mll2==-1) and (mul2==-1) and (mll3==-1) and (mul3==-1) and (mll4==-1) and (mul4==-1):
        mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4 = 1, 2, 4, 8, 1, 2, 4, 8

    a = np.random.uniform(mll1, mul1)
    b = np.random.uniform(mll2, mul2)
    c = np.random.uniform(mll3, mul3)
    d = np.random.uniform(mll4, mul4)
    return a*np.sin(b*t) + c*np.cos(d*t)


# design matrix for historically dependent data
def beta(s, t, bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4, linear):
    if (bll1==-1) and (bul1==-1) and (bll2==-1) and (bul2==-1) and (bll3==-1)and (bul3==-1) and (bll4==-1) and (bul4==-1):
        bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4 = 1, 2, 8, 16, 1, 2, 8, 16

    a = np.random.uniform(bll1, bul1)
    b = np.random.uniform(bll2, bul2)
    c = np.random.uniform(bll3, bul3)
    d = np.random.uniform(bll4, bul4)
    return a*np.sin(b*s) + c*np.cos(d*t) + linear


# historically dependent data
def hist_data(X, upper_limit, a, pred_points, bll1=-1, bul1=-1, bll2=-1, bul2=-1, bll3=-1, bul3=-1, bll4=-1, bul4=-1, linear=0):
    """
    Function to generate historically dependent data

    Inputs:
    X: (n_vars, n_samples * n_tests, n_obs) array of samples
    upper_limit: upper limit for predictions
    a: strength of dependence
    pred_points: prediction points
    linear: extend of linear dependence

    Returns:
    Y: (n_samples * n_tests, n_obs) response variable that is historically dependent on X
    """
    if len(X.shape)==2:
        X_arr = X.reshape(1, X.shape[0], X.shape[1])
    else:
        X_arr = X
    Y = np.zeros((X_arr.shape[1], len(pred_points[pred_points<=upper_limit])))
    for i in range(X_arr.shape[1]):  # looping over samples
        sum_integ = 0
        for p in range(X_arr.shape[0]):  # looping over parent variables
            beta_p = beta(pred_points[pred_points<=upper_limit], upper_limit, bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4, linear)

            integ = cumulative_trapezoid(X_arr[p, i, np.arange(0, len(pred_points[pred_points<=upper_limit]))] * beta_p,
                                         pred_points[pred_points<=upper_limit], initial=0)
            sum_integ += integ

        Y[i] = a*sum_integ

    return Y


def DAG_hist(n_nodes, n_samples, n_obs, n_preds, a, upper_limit, period, n_basis, sd, prob,
             bll1=-1, bul1=-1, bll2=-1, bul2=-1, bll3=-1, bul3=-1, bll4=-1, bul4=-1, linear=0):
    """
    Generates data with dependencies according to random DAG and dependence according to historical model
    for multiple covariates

    Inputs:
    n_nodes: number of nodes in DAG
    n_samples: number of samples drawn per node
    n_obs: number of observations per sample
    n_preds: number of equally spaced observation points in [0, 1]
    upper_limit: upper limit of time
    a: strength of dependence
    n_basis: number of basis functions for nodes without parents
    sd: standard deviation for normal distribution where Fourier basis function coefficients are drawn from
    prob: probability of edge creation

    Returns:
    dict_edges: dictionary of form key: descendent, value: parents
    X_fd_list: (n_nodes, n_samples, n_preds) array with data according to dependencies specified by edges
    """
    X_fd_list = np.empty((n_nodes, n_samples, n_preds))

    _DAG = set()
    DAG_accepted = 0
    while DAG_accepted < 1:
        _DAG = generate_DAGs(n_nodes, prob, discover=False)[0]
        if _DAG.sources() == _DAG.sinks():
            continue
        else:
            DAG_accepted += 1

    pred_points = np.linspace(0, upper_limit, n_preds).reshape(-1, 1)

    for node in _DAG.topological_sort():
        # first generate data for all nodes that do not have any parents
        if _DAG.parents_of(node) == set():
            obs_points_X_desc = sample_points(n_samples, n_obs, upper_limit)

            X_mat_desc = skfda_basis(n_samples, upper_limit=upper_limit, period=period, n_basis=n_basis,
                                     sd=sd).evaluate(obs_points_X_desc, aligned=False).squeeze()
            X_fd_list[node] = spline_multi_sample(X_mat_desc, obs_points_X_desc,
                                                  pred_points).evaluate(pred_points).squeeze() + \
                              np.random.normal(0, sd, size=(n_samples, n_preds))

        else:
            # then generate data for all nodes that have parents
            X_fd_list[node] = hist_data(X_fd_list[list(_DAG.parents_of(node))], upper_limit, a, pred_points,
                                        bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4, linear=linear) + \
                              np.random.normal(0, sd, size=(n_samples, n_preds))

    return _DAG, X_fd_list


def generate_data(dep, n_samples, n_trials, n_obs, n_preds, period=1, n_vars=1, a=1, a_prime=10, upper_limit=1, n_basis=3, sd=1, prob=0.5,
                  bll1=-1, bul1=-1, bll2=-1, bul2=-1, bll3=-1, bul3=-1, bll4=-1, bul4=-1, linear=0):
    """
    Parent function to generate synthetic data

    Inputs:
    dep: type of dependencies for synthetic data ('marginal', 'conditional', or 'joint')
    n_samples: number of samples drawn per node
    n_trials: number of trials to evaluate test power
    n_obs: number of observations per sample
    n_preds: number of equally spaced observation points in [0, 1]
    period: T for Fourier basis functions
    n_vars: number of nodes in DAG (for joint independence),
            or number of variables in conditional set (for conditional independence)
    a: strength of dependence in type 'marginal', 'joint',
       and influence of conditional variable Z on X and Y in type 'conditional'
    a_prime: strength of dependence between X and Y in type 'conditional'
    upper_limit: upper limit of time
    n_basis: number of basis functions for nodes without parents
    sd: standard deviation for normal distribution where Fourier basis function coefficients are drawn from
    prob: probability of edge creation (only for dep='joint; default 0.5)
    bll1, ..., bul4: coefficients for beta function
    linear: coefficient if relationship between variables is meant to be linear

    Returns:
    marginal:
        X, Y: observed functional data where Y is historically dependent on X
    conditional:
        X, Y, Z: observed functional data where both X and Y is historically dependent on Z and
                 Y is historically dependent on X
    joint:
        dict_edges: dictionary of form key: descendent, value: parents
        X_dict: (n_nodes, n_samples, n_preds) dictionary with data according to dependencies specified in dict_edges

    """
    pred_points = np.linspace(0, upper_limit, n_preds)
    if dep == 'marginal':
        obs_points_X = sample_points(n_samples * n_trials, n_obs, upper_limit=upper_limit)
        X_mat = skfda_basis(n_samples * n_trials, upper_limit, period, n_basis, sd).evaluate(obs_points_X,
                                                                                             aligned=False).squeeze()
        X = spline_multi_sample(X_mat, obs_points_X, pred_points).evaluate(pred_points).squeeze() + \
            np.random.normal(0, sd, size=(n_samples * n_trials, n_preds))
        Y = hist_data(X, upper_limit, a, pred_points, bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4, linear=linear) + \
            np.random.normal(0, sd, size=(n_samples * n_trials, n_preds))

        return X, Y

    elif dep == 'conditional':
        Z = np.zeros((n_vars, n_samples*n_trials, len(pred_points)))
        for var in range(n_vars):
            obs_points_Z_i = sample_points(n_samples*n_trials, n_obs, upper_limit=upper_limit)
            Z_mat_i = skfda_basis(n_samples*n_trials, upper_limit, period, n_basis, sd).evaluate(obs_points_Z_i, aligned=False).squeeze()
            Z_i = spline_multi_sample(Z_mat_i, obs_points_Z_i, pred_points).evaluate(pred_points).squeeze() + \
                  np.random.normal(0, sd/6, size=(n_samples * n_trials, n_preds))
            Z[var] = Z_i
        X = hist_data(Z, upper_limit, a, pred_points, bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4, linear=linear) + \
            np.random.normal(0, sd/6, size=(n_samples * n_trials, n_preds))
        Y = hist_data(Z, upper_limit, a, pred_points, bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4, linear=linear) + \
            hist_data(X, upper_limit, a_prime, pred_points, bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4, linear=linear) + \
            np.random.normal(0, sd/6, size=(n_samples * n_trials, n_preds))

        return X, Y, Z

    elif dep == 'joint':
        edges_dict = {}
        X_dict = {}
        for trial in range(n_trials):    # generating n_trials different DAGs with data distributed accordingly
            edges_dict[trial], X_dict[trial] = DAG_hist(n_vars, n_samples, n_obs, n_preds, a, upper_limit, period,
                                                        n_basis, sd, prob, bll1=bll1, bul1=bul1, bll2=bll2, bul2=bul2,
                                                        bll3=bll3, bul3=bul3, bll4=bll4, bul4=bul4, linear=linear)

        return edges_dict, X_dict

    else:
        raise ValueError("Generating synthetic data is only possible with dependence types 'marginal', 'conditional', or 'joint'.")

