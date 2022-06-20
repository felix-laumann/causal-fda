from skfda.ml.regression import HistoricalLinearRegression
from skfda import FDataGrid


def hist_linear(n_intervals, ind_var, dep_var, pred_points):
    """
    Historical linear model that regressed dependent variable (dep_var) on independent variable (ind_var)

    Inputs:
    n_intervals: number of intervals to solve the integral
    ind_var: observed functional data for first distribution
    dep_var: observed functional data for second distribution
    pred_points: points over which X and Y are predicted

    Returns:
    dep_var_hat: prediction of dep_var regressed on ind_var
    """

    hist = HistoricalLinearRegression(n_intervals=n_intervals)
    _ = hist.fit(FDataGrid(ind_var).to_grid(grid_points=pred_points), FDataGrid(dep_var).to_grid(grid_points=pred_points))
    dep_var_hat = hist.predict(FDataGrid(ind_var).to_grid(grid_points=pred_points)).evaluate(pred_points).squeeze()
    return dep_var_hat
