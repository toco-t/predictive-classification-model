"""
Visual exploration of data.

This module contains functions for visual exploration of data.

@author: Toco Tachibana
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def side_by_side(*, data_frame: pd.DataFrame, target: str):
    """
    Produce side-by-side box plots for each number column.

    :param data_frame: pandas DataFrame containing the dataset
    :param target: target variable as a string
    """
    churned = data_frame[data_frame[target] == 1]
    not_churned = data_frame[data_frame[target] == 0]

    for column in data_frame.select_dtypes(include="number").columns:
        if column == target:
            continue

        # Produce side-by-side box plots
        # plt.figure()
        # plt.title(column)
        # plt.boxplot([churned[column], not_churned[column]])
        # plt.xticks([1, 2], ["Churned", "Not Churned"])
        # plt.show()

        # Produce side-by-side histograms
        plt.figure()
        plt.title(column)
        plt.hist([churned[column], not_churned[column]])
        plt.legend(["Churned", "Not Churned"])
        plt.show()


def grouped_plots(*, data_frame: pd.DataFrame, target: str, title: str, predictors: list[str]):
    """
    Produce grouped box plots and histograms for each number column.

    :param data_frame: pandas DataFrame containing the dataset
    :param target: target variable as a string
    :param title: title of the plot as a string
    :param predictors: list of predictors as strings
    """
    churned = data_frame[data_frame[target] == 1]
    not_churned = data_frame[data_frame[target] == 0]

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    fig.tight_layout(pad=3.0)
    fig.suptitle(title)

    for idx, column in enumerate(predictors):
        axs[idx // 3, idx % 3].boxplot(
            [churned[column], not_churned[column]],
            labels=["Churned", "Not Churned"]
        )
        axs[idx // 3, idx % 3].set_title(column)

    plt.savefig(f"{title}.png")
    plt.show()


def visual_exploration(*, data_frame: pd.DataFrame, target: str) -> list[str]:
    """
    Visual exploration of data.

    :param data_frame: pandas DataFrame containing the dataset
    :param target: target variable as a string
    :return: list of predictors with correlation greater than 0
    """

    correlations = data_frame.corr(numeric_only=True).round(1)
    sns.heatmap(data=correlations.round(1))
    plt.tight_layout()
    plt.savefig("correlation_heatmap_outliers.png")
    plt.show()

    return [predictor for predictor in correlations[target].index.values
            if abs(correlations.loc[target, predictor]) != 0
            and predictor != target]
