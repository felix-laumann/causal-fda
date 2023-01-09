import numpy as np
from skfda.ml.regression import HistoricalLinearRegression, KNeighborsRegressor
from skfda import FDataGrid
from sklearn.metrics import r2_score


def hist_linear(n_intervals, ind_var, dep_var, pred_points, analyse):
    """
    Historical linear model that regressed dependent variable (dep_var) on independent variable (ind_var)

    Inputs:
    args: includes:
        n_intervals: number of intervals to solve the integral
        ind_var: observed functional data for first distribution
        dep_var: observed functional data for second distribution
        pred_points: points over which X and Y are predicted
        analyse: (boolean) whether to calculate R-squared score of regression

    Returns:
    dep_var_hat: prediction of dep_var regressed on ind_var
    """
    upper_limit = np.max(pred_points)
    hist = HistoricalLinearRegression(n_intervals=n_intervals)
    _ = hist.fit(FDataGrid(ind_var, domain_range=(0, upper_limit)).to_grid(grid_points=pred_points),
                 FDataGrid(dep_var, domain_range=(0, upper_limit)).to_grid(grid_points=pred_points))
    dep_var_hat = hist.predict(FDataGrid(ind_var, domain_range=(0, upper_limit)).to_grid(grid_points=pred_points)).evaluate(pred_points).squeeze()

    if analyse is True:
        R_squared = r2_score(FDataGrid(dep_var, domain_range=(0, upper_limit)).to_grid(grid_points=pred_points).evaluate(pred_points).squeeze(),
                             dep_var_hat)
        print('R-squared:', R_squared)

    return dep_var_hat


def knn_regressor(n_neighbors, ind_var, dep_var, pred_points, analyse, weights='uniform', regressor='mean', alg='auto', metric='l2_distance'):
    """
    kNN regression model that regressed dependent variable (dep_var) on independent variable (ind_var)

    Inputs:
    n_neighbors: number of neighbors to predict value of dependent variable
    weights: weight function used in prediction; for example, 'uniform', ’distance’
    regressor: function to perform the local regression in the functional response case; for example, 'mean'
    algorithm: algorithm used to compute the nearest neighbors; for example 'ball_tree', ’brute’, 'auto'
    metric: the distance metric to use for the tree; for example, 'l2_distance'
    ind_var: observed functional data for first distribution
    dep_var: observed functional data for second distribution
    pred_points: points over which X and Y are predicted
    analyse: (boolean) whether to calculate R-squared score of regression

    Returns:
    dep_var_hat: prediction of dep_var regressed on ind_var
    """
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, regressor=regressor, algorithm=alg, metric=metric)
    _ = knn.fit(FDataGrid(ind_var).to_grid(grid_points=pred_points), FDataGrid(dep_var).to_grid(grid_points=pred_points))
    dep_var_hat = knn.predict(FDataGrid(ind_var).to_grid(grid_points=pred_points)).evaluate(pred_points).squeeze()

    if analyse is True:
        R_squared = r2_score(FDataGrid(dep_var).to_grid(grid_points=pred_points).evaluate(pred_points).squeeze(),
                             dep_var_hat)
        print('R-squared:', R_squared)

    return dep_var_hat
