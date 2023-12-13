import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import statsmodels.api as sm

def calculate_fitness(chromosome, data, outcome_index, objective_function="AIC", log_outcome=True, regression_type="OLS"):
    """
    Calculate the fitness of a chromosome.

    Parameters:
    - chromosome (list of bool): Binary representation of predictors.
    - data (pd.DataFrame): Input data with predictors.
    - outcome_index (int): Index of outcome column (0 index).
    - objective_function (str): Default is AIC, other options are "BIC", "Adjusted R-squared, "MSE", 
    "Mallows CP". 
    - log_outcome (bool): True if the outcome variable is on the log scale, False else.
    - regression_type (str): Default is OLS, other options are "Lasso" and "Rigde".

    Returns:
    - float: fitness value.
    """
    assert all(bit in {0, 1} for bit in chromosome), "Chromosome must be a binary list of 0s and 1s"
    assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
    assert data.shape[1] >= 2, "Data must have at least 2 columns."
    assert data.shape[0] >= 20, "Data must have at least 20 rows."
    assert not data.isnull().any().any(), "Data must not contain missing values."

    assert isinstance(outcome_index, int) and outcome_index >= 0, "Outcome index must be a non-negative integer"
    assert objective_function in ["AIC", "BIC", "Adjusted R-squared", "MSE", "Mallows CP"], \
    "Objective function options are limited to: 'AIC', 'BIC', 'Adjusted R-squared', 'MSE', 'Mallows CP.' If not from list, default calculations are AIC."
    assert isinstance(log_outcome, bool), "log_outcome must be either True or False"
    assert regression_type in ["OLS","Ridge","Lasso"], "Regression type options are limited to: 'OLS','Ridge','Lasso'. If not from list, default calculations are OLS."

    if log_outcome == True:
        outcome = pd.Series(np.log(data.iloc[:, outcome_index].astype(float)))
        outcome_array = np.asarray(outcome)
    else:
        outcome = pd.Series(data.iloc[:, outcome_index].astype(float))
        outcome_array = np.asarray(outcome)
        
    # Select the predictors according to the chromosome
    if outcome_index > 0:
        predictors_1 = data.iloc[:, :outcome_index]
        predictors = pd.concat([predictors_1, data.iloc[:, outcome_index+1:]], axis=1)
    else:
        predictors = data.iloc[:, outcome_index+1:]
    
    selected_predictors = predictors.loc[:, [bool(x) for x in chromosome]]
    selected_predictors = sm.add_constant(selected_predictors)

    # Convert selected predictors to a NumPy array
    predictors_array = np.asarray(selected_predictors.astype(float))

    # Fit linear regression model
    if regression_type == "Lasso":
        model  = Lasso(alpha=1).fit(predictors_array, outcome_array)
    elif regression_type == "Ridge":
        model  = Ridge(alpha=1).fit(predictors_array, outcome_array)
    else:
        model = sm.OLS(outcome_array, predictors_array).fit()

    # Calculate objective function inputs
    rss = model.ssr
    tss = np.sum((outcome - np.mean(outcome))**2)
    s = np.sum(chromosome)  # Number of predictors
    k = s + 2  # Number of parameters including intercept
    n = len(outcome) 
    mse = np.mean((outcome - model.predict(predictors_array))**2)
    
    # Return fitness functions based on objective function input
    if objective_function == "BIC":
        return n * np.log(rss / n) + k * np.log(n)
    elif objective_function == "Adjusted R-squared":
        r_squared = 1 - (rss / tss)
        return 1 - (1 - (r_squared)) * (n - 1) / (n - s - 1)
    elif objective_function == "MSE":
        return mse
    elif objective_function == "Mallows CP":
        return rss + 2 * s * mse / (n - s)
    else:
        # default is AIC
        return n * np.log(rss / n) + 2 * k