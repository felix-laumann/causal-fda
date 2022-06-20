import numpy as np
import networkx as nx
from scipy.integrate import cumulative_trapezoid
from skfda.representation.basis import Fourier, FDataBasis
from skfda.representation.grid import FDataGrid
from skfda.representation.interpolation import SplineInterpolation

from graph_generation import generate_DAGs


# define meshes
def sample_points(n_samples, n_obs, random_state=None):
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
    obs_grid = rng.uniform(0, 1, (n_samples, n_obs))

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


def spline_multi_sample(X, obs_points, pred_points, int_order=3, smooth_param=1.5):
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
def skfda_basis(n_samples, upper_limit, n_basis, sd):
    basis = Fourier((0, upper_limit), n_basis=n_basis)
    coefficients = np.random.normal(0, sd, size=(n_samples, n_basis))
    return FDataBasis(basis, coefficients)


# mean function for historically dependent data
def mean_func(t, mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4):
    if (mll1==0) and (mul1==0) and (mll2==0) and (mul2==0) and (mll3==0) and (mul3==0) and (mll4==0) and (mul4==0):
        mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4 = 1, 2, 4, 8, 1, 2, 4, 8

    a = np.random.uniform(mll1, mul1)
    b = np.random.uniform(mll2, mul2)
    c = np.random.uniform(mll3, mul3)
    d = np.random.uniform(mll4, mul4)
    return a*np.sin(b*t) + c*np.cos(d*t)


# design matrix for historically dependent data
def beta(s, t, bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4):
    if (bll1==0) and (bul1==0) and (bll2==0) and (bul2==0) and (bll3==0)and (bul3==0) and (bll4==0) and (bul4==0):
        bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4 = 1, 2, 8, 16, 1, 2, 8, 16

    a = np.random.uniform(bll1, bul1)
    b = np.random.uniform(bll2, bul2)
    c = np.random.uniform(bll3, bul3)
    d = np.random.uniform(bll4, bul4)
    return a*np.sin(b*s) + c*np.cos(d*t)


# functional noise
def epsilon(t, n_basis=3, sd=1):
    return skfda_basis(1, upper_limit=np.max(t), n_basis=n_basis, sd=sd).evaluate(t).squeeze()


# historically dependent data
def hist_data(X, t, a, pred_points, mean, mll1=0, mul1=0, mll2=0, mul2=0, mll3=0, mul3=0, mll4=0, mul4=0,
              bll1=0, bul1=0, bll2=0, bul2=0, bll3=0, bul3=0, bll4=0, bul4=0):
    """
    Function to generate historically dependent data

    Inputs:
    X: (n_vars, n_samples * n_tests, n_obs) array of samples
    t: upper limit for predictions
    a: strength of dependence
    pred_points: prediction points
    mean: (Boolean) whether mean function should be included or not. True: included (for marginal);
          False: not included (for conditional)

    Returns:
    Y: (n_samples * n_tests, n_obs) response variable that is historically dependent on X
    """
    if mean:
        mu = mean_func(pred_points[pred_points<=t], mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4)
    else:
        mu = 0

    if len(X.shape)==2:
        X_arr = X.reshape(1, X.shape[0], X.shape[1])
    else:
        X_arr = X
    Y = np.zeros((X_arr.shape[1], len(pred_points[pred_points<=t])))
    for i in range(X_arr.shape[1]):  # looping over samples
        sum_integ = 0
        for p in range(X_arr.shape[0]):  # looping over parent variables
            beta_p = beta(pred_points[pred_points<=t], t, bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4)

            integ = cumulative_trapezoid(X_arr[p, i, np.arange(0, len(pred_points[pred_points<=t]))] * beta_p,
                                         pred_points[pred_points<=t], initial=0)
            sum_integ += integ

        Y[i] = mu + a*sum_integ + epsilon(pred_points[pred_points<=t])
    return Y


def DAG_hist(n_nodes, n_samples, n_obs, n_preds, mean, a, upper_limit, n_basis, sd, prob):
    """
    Generates data with dependencies according to random DAG and dependence according to historical model
    for multiple covariates

    Inputs:
    n_nodes: number of nodes in DAG
    n_samples: number of samples drawn per node
    n_obs: number of observations per sample
    n_preds: number of equally spaced observation points in [0, 1]
    mean: (Boolean) whether a mean function is included or not
    upper_limit: upper limit of time
    a: strength of dependence
    n_basis: number of basis functions for nodes without parents
    sd: standard deviation for normal distribution where Fourier basis function coefficients are drawn from
    prob: probability of edge creation

    Returns:
    dict_edges: dictionary of form key: descendent, value: parents
    X_fd_list: (n_nodes, n_samples, n_preds) array with data according to dependencies specified by edges
    """
    DAG = set()
    DAG_accepted = 0
    while DAG_accepted < 1:
        DAG = generate_DAGs(n_nodes, prob, discover=False)[0]
        if DAG.sources() == DAG.sinks():
            continue
        else:
            DAG_accepted += 1

    pred_points = np.linspace(0, upper_limit, n_preds).reshape(-1, 1)
    X_fd_list = np.empty((n_nodes, n_samples, n_preds))

    for node in DAG.topological_sort():
        # first generate data for all nodes that do not have any parents
        if DAG.parents_of(node) == set():
            obs_points_X_desc = sample_points(n_samples, n_obs)
            X_mat_desc = skfda_basis(n_samples, upper_limit=upper_limit, n_basis=n_basis,
                                     sd=sd).evaluate(obs_points_X_desc, aligned=False).squeeze()
            X_fd_list[node] = spline_multi_sample(X_mat_desc, obs_points_X_desc,
                                                  pred_points).evaluate(pred_points).squeeze()

        else:
            # then generate data for all nodes that have parents
            X_fd_list[node] = hist_data(X_fd_list[list(DAG.parents_of(node))], upper_limit, a, pred_points, mean)

    return DAG, X_fd_list


def generate_data(dep, n_samples, n_trials, n_obs, n_preds, n_vars=1, a=1, a_prime=10, upper_limit=1, n_basis=3, sd=1, prob=0.5,
                  mll1=0, mul1=0, mll2=0, mul2=0, mll3=0, mul3=0, mll4=0, mul4=0,
                  bll1=0, bul1=0, bll2=0, bul2=0, bll3=0, bul3=0, bll4=0, bul4=0):
    """
    Parent function to generate synthetic data

    Inputs:
    dep: type of dependencies for synthetic data ('marginal', 'conditional', or 'joint')
    n_samples: number of samples drawn per node
    n_trials: number of old_trials to evaluate test power
    n_obs: number of observations per sample
    n_preds: number of equally spaced observation points in [0, 1]
    n_vars: number of nodes in DAG (for joint independence),
            or number of variables in conditional set (for conditional independence)
    a: strength of dependence in type 'marginal', 'joint',
       and influence of conditional variable Z on X and Y in type 'conditional'
    a_prime: strength of dependence between X and Y in type 'conditional'
    upper_limit: upper limit of time
    n_basis: number of basis functions for nodes without parents
    sd: standard deviation for normal distribution where Fourier basis function coefficients are drawn from
    prob: probability of edge creation (only for dep='joint; default 0.5)
    mll1, ..., bul4: coefficients for mean function and beta

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
        mean = True
        obs_points_X = sample_points(n_samples*n_trials, n_obs)
        X_mat = skfda_basis(n_samples*n_trials, upper_limit, n_basis, sd).evaluate(obs_points_X, aligned=False).squeeze()
        X = spline_multi_sample(X_mat, obs_points_X, pred_points).evaluate(pred_points).squeeze()
        Y = hist_data(X, upper_limit, a, pred_points, mean, mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4,
                      bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4)
        return X, Y

    elif dep == 'conditional':
        mean = False
        Z = np.zeros((n_vars, n_samples*n_trials, len(pred_points)))
        for var in range(n_vars):
            obs_points_Z_i = sample_points(n_samples*n_trials, n_obs)
            Z_mat_i = skfda_basis(n_samples*n_trials, upper_limit, n_basis, sd).evaluate(obs_points_Z_i, aligned=False).squeeze()
            Z_i = spline_multi_sample(Z_mat_i, obs_points_Z_i, pred_points).evaluate(pred_points).squeeze()
            Z[var] = Z_i
        mu_X = mean_func(pred_points[pred_points <= upper_limit], mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4)
        mu_Y = mean_func(pred_points[pred_points <= upper_limit], mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4)
        X = mu_X + hist_data(Z, upper_limit, a, pred_points, mean, mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4,
                             bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4)
        Y = mu_Y + hist_data(Z, upper_limit, a, pred_points, mean, mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4,
                             bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4) + \
            hist_data(X, upper_limit, a_prime, pred_points, mean, mll1, mul1, mll2, mul2, mll3, mul3, mll4, mul4,
                      bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4)
        return X, Y, Z

    elif dep == 'joint':
        mean = True
        edges_dict = {}
        X_dict = {}
        for trial in range(n_trials):    # generating n_trials different DAGs with data distributed accordingly
            edges_dict[trial], X_dict[trial] = DAG_hist(n_vars, n_samples, n_obs, n_preds, mean, a, upper_limit, n_basis, sd, prob)

        return edges_dict, X_dict

    else:
        raise ValueError("Generating synthetic data is only possible with dependence types 'marginal', 'conditional', "
                         "or 'joint'.")
