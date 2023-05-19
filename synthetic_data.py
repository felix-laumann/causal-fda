import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
import causaldag
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
def spline_single_sample(X, obs_points, pred_points, int_order):
    """
    Given a single sample of functional data X at locations obs_points, a spline is fit and predictions are
    given at locations pred_points

    Inputs:
    X: single function sample
    obs_points: function sample observation locations
    pred_points: prediction locations

    Returns:
    The predictions of a spline with smoothness parameter at locations pred_points
    """
    X_fd = FDataGrid(data_matrix=X, sample_points=obs_points)
    X_fd.interpolation = SplineInterpolation(interpolation_order=int_order)
    return X_fd.evaluate(pred_points)


def spline_multi_sample(X, obs_points, pred_points, int_order=5):
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
        X_reg[j, :] = spline_single_sample(X[j, :], obs_points[j, :], pred_points, int_order).squeeze()
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
def beta_old(s, t, bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4, linear):
    if (bll1==-1) and (bul1==-1) and (bll2==-1) and (bul2==-1) and (bll3==-1)and (bul3==-1) and (bll4==-1) and (bul4==-1):
        bll1, bul1, bll2, bul2, bll3, bul3, bll4, bul4 = 1, 2, 8, 16, 1, 2, 8, 16

    a = np.random.uniform(bll1, bul1)
    b = np.random.uniform(bll2, bul2)
    c = np.random.uniform(bll3, bul3)
    d = np.random.uniform(bll4, bul4)
    return a*np.sin(b*s) + c*np.cos(d*t) + linear


# design matrix for historically dependent data
def beta(s, t, linear):
    c_1 = np.random.uniform(0.25, 0.75)
    c_2 = np.random.uniform(0.25, 0.75)
    b = 8 * (s - c_1) ** 2 - 8 * (t - c_2) ** 2

    return b + linear


# historically dependent data
def hist_data(X, upper_limit, a, pred_points, linear=0):
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
    Y = np.zeros((X_arr.shape[1], len(pred_points[pred_points <= upper_limit])))
    s, t = np.meshgrid(pred_points[pred_points <= upper_limit], pred_points[pred_points <= upper_limit])
    for i in range(X_arr.shape[1]):  # looping over samples
        sum_y = np.zeros(len(pred_points[pred_points <= upper_limit]))
        for p in range(X_arr.shape[0]):  # looping over parent variables
            beta_p = beta(s, t, linear)
            y = np.zeros(len(pred_points[pred_points <= upper_limit]))
            for i_t, t in enumerate(pred_points[pred_points <= upper_limit]):  # looping over time points of y
                #y[i_t] = trapezoid(X_arr[p, i][:i_t+1] * beta_p[:i_t+1, i_t])
                if i_t > 0:
                    y[i_t] = np.sum(X_arr[p, i][:i_t + 1] * beta_p[:i_t + 1, i_t]) / i_t
                else:
                    y[i_t] = np.sum(X_arr[p, i][:i_t + 1] * beta_p[:i_t + 1, i_t])

            sum_y += y
            Y[i] = a*sum_y

    return Y


def two_log(X0, Y0, r_x, r_y, B_xy, B_yx, n_samples, n_preds):
    """
    Function to generate data according to a coupled two-species nonlinear logistic difference system with chaotic dynamics
    Inputs:
    X0: initial value for X
    Y0: initial value for X
    r_x: system parameter (set between 3 and 4)
    r_y: system parameter (set between 3 and 4)
    B_xy: effect of Y on X
    B_yx: effect of X on Y

    Returns:
    X_fd_list
    """
    X_fd_list = np.empty((2, n_samples, n_preds))
    t = n_preds * n_samples * 2

    X = [X0]
    Y = [Y0]
    for i_t in range(t-1):
        X_ = X[-1] * (r_x - r_x * X[-1] - B_xy * Y[-1])
        Y_ = Y[-1] * (r_y - r_y * Y[-1] - B_yx * X[-1])
        X.append(X_)
        Y.append(Y_)

    for n_s in range(n_samples):
        X_fd_list[0, n_s] = np.asarray(X)[(2*n_s) * n_preds:((2*n_s) + 1) * n_preds]
        X_fd_list[1, n_s] = np.asarray(Y)[(2*n_s) * n_preds:((2*n_s) + 1) * n_preds]

    return X_fd_list


def DAG_hist(n_nodes, n_samples, n_obs, n_preds, a, upper_limit, period, n_basis, sd, prob, linear=0):
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
        _DAG = generate_DAGs(n_nodes, analyse=False, prob=prob, discover=False)[0]
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
            X_fd_list[node] = spline_multi_sample(X_mat_desc, obs_points_X_desc, pred_points).evaluate(pred_points).squeeze() + \
                              np.random.normal(0, sd, size=(n_samples, n_preds))

        else:
            # then generate data for all nodes that have parents
            X_fd_list[node] = hist_data(X_fd_list[list(_DAG.parents_of(node))], upper_limit, a, pred_points, linear=linear) + \
                              np.random.normal(0, sd, size=(n_samples, n_preds))

    return _DAG, X_fd_list


def generate_data(dep, n_samples, n_trials, n_obs, n_preds, period=1, n_vars=1, a=1, a_prime=1, upper_limit=1, n_basis=3, sd=1, prob=0.5,
                  linear=0, log_sys=False, analyse=False):
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
    c_1, c_2: coefficients for beta function
    linear: coefficient if relationship between variables is meant to be linear
    log_sys: (boolean) whether to generate data according to two_log function

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
        Y = hist_data(X, upper_limit, a_prime, pred_points, linear=linear) + \
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
        X = hist_data(Z, upper_limit, a, pred_points, linear=linear) + np.random.normal(0, sd/6, size=(n_samples * n_trials, n_preds))
        Y = hist_data(Z, upper_limit, a, pred_points, linear=linear) + \
            hist_data(X, upper_limit, a_prime, pred_points, linear=linear) + np.random.normal(0, sd/6, size=(n_samples * n_trials, n_preds))

        return X, Y, Z

    elif dep == 'joint':
        edges_dict = {}
        X_dict = {}
        for trial in range(n_trials):    # generating n_trials different DAGs with data distributed accordingly
            if not log_sys:
                if trial == 0 and analyse:
                    print('Generate data according to historical model')
                edges_dict[trial], X_dict[trial] = DAG_hist(n_vars, n_samples, n_obs, n_preds, a, upper_limit, period,
                                                            n_basis, sd, prob, linear=linear)

            else:
                if trial == 0 and analyse:
                    print('Generate data according to two-species logistic model')
                g = causaldag.DAG({0, 1})
                g.add_arc(0, 1)
                edges_dict[trial] = g
                X0 = 0.2
                Y0 = 0.4
                r_x = 3.8
                r_y = 3.5
                B_xy = 0.02
                B_yx = a
                X_dict[trial] = two_log(X0, Y0, r_x, r_y, B_xy, B_yx, n_samples=n_samples, n_preds=n_preds)

        return edges_dict, X_dict

    else:
        raise ValueError("Generating synthetic data is only possible with dependence types 'marginal', 'conditional', "
                         "or 'joint'.")

