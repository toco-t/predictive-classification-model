"""
Feature selection methods.

This module contains functions that perform feature selection on a dataset.
The following methods are implemented:
    - Recursive Feature Elimination
    - Forward Feature Selection
    - Chi-squared Test

@author: Toco Tachibana
"""
import pandas as pd
from sklearn.feature_selection import RFE, f_regression, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression


def recursive_feature_elimination(*, data_frame: pd.DataFrame, target: str):
    """
    Select significant predictors using recursive feature elimination.

    :param data_frame: pandas DataFrame containing the dataset
    :param target: name of the target variable
    :return: list of selected predictors
    """

    X = data_frame.drop(target, axis=1)
    y = data_frame[target]

    # Perform recursive feature elimination
    model = LogisticRegression(solver="newton-cholesky", max_iter=1000)
    rfe = RFE(estimator=model, n_features_to_select=15).fit(X, y)

    selected_predictors = X.columns[rfe.support_].tolist()
    return selected_predictors


def forward_feature_selection(*, data_frame: pd.DataFrame, target: str):
    """
    Select significant predictors using forward feature selection.

    :param data_frame: pandas DataFrame containing the dataset
    :param target: name of the target variable
    :return: list of selected predictors
    """

    X = data_frame.drop(target, axis=1)
    y = data_frame[target]

    # Perform forward feature selection
    ffs = SelectKBest(score_func=f_regression, k=15)
    ffs.fit_transform(X, y)

    # Get indices of the selected features
    selected_indices = ffs.get_support(indices=True)

    # Get the column names of the selected features
    selected_predictors = X.columns[selected_indices].tolist()

    return selected_predictors


def chi_squared_test(*, data_frame: pd.DataFrame, target: str):
    """
    Perform chi-squared test to select features.

    :param data_frame: pandas DataFrame containing the dataset
    :param target: name of the target variable
    :return: pandas DataFrame with selected features
    """

    predictors = data_frame.drop(columns=target)
    target_variable = data_frame[target]

    chi2_selector = SelectKBest(chi2, k=15)
    chi2_selector.fit(predictors, target_variable)

    selected_features = predictors.columns[chi2_selector.get_support()].tolist()
    return selected_features
