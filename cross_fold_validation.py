"""
Split dataset into subgroups using cross-fold validation to train and test.

This module contains functions for splitting the dataset into subgroups using
cross-fold validation to train and test the model.

@author: Toco Tachibana
"""
from collections import Counter

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler


def train_and_test(
        *, data_frame: pd.DataFrame, target: str, predictors: list[str]
):
    cross_fold = KFold(n_splits=10, shuffle=True)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for index, (train_indexes, test_indexes) \
            in enumerate(cross_fold.split(data_frame)):
        X_train = data_frame.iloc[train_indexes, :][predictors]
        X_test = data_frame.iloc[test_indexes, :][predictors]
        y_train = data_frame.iloc[train_indexes, :][target]
        y_test = data_frame.iloc[test_indexes, :][target]

        smote = SMOTETomek()
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train, y_train
        )

        print("Original class distribution:", Counter(y_train))
        print("SMOTETomek class distribution:", Counter(y_train_resampled))

        # Scale continuous variables
        # scaler = StandardScaler()
        # X_train_resampled = scaler.fit_transform(X_train_resampled)
        # X_test = scaler.transform(X_test)

        logistic_regression = LogisticRegression(solver="liblinear", max_iter=1000)
        logistic_regression.fit(X_train_resampled, y_train_resampled)

        print(f"\nK-FOLD VALIDATION: {index + 1}")
        for feature, coefficient in zip(predictors, logistic_regression.coef_[0]):
            print(f"{feature} -> {coefficient}")

        predictions = logistic_regression.predict(X_test)

        confusion_matrix = pd.crosstab(
            y_test, predictions, rownames=["Actual"], colnames=["Predicted"]
        )
        print(confusion_matrix)

        accuracy = metrics.accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)
        print(f"Accuracy -> {accuracy}")

        precision = metrics.precision_score(y_test, predictions)
        precision_scores.append(precision)
        print(f"Precision -> {precision}")

        recall = metrics.recall_score(y_test, predictions)
        recall_scores.append(recall)
        print(f"Recall -> {recall}")

        f1 = metrics.f1_score(y_test, predictions)
        f1_scores.append(f1)
        print(f"F1 -> {f1}")

    average_accuracy, accuracy_standard_deviation = calculate_stats(
        scores=accuracy_scores
    )
    average_precision, precision_standard_deviation = calculate_stats(
        scores=precision_scores
    )
    average_recall, recall_standard_deviation = calculate_stats(
        scores=recall_scores
    )
    average_f1, f1_standard_deviation = calculate_stats(scores=f1_scores)

    print(f"\nAverage Accuracy -> {average_accuracy}")
    print(f"Accuracy Standard Deviation -> {accuracy_standard_deviation}")
    print(f"Average Precision -> {average_precision}")
    print(f"Precision Standard Deviation -> {precision_standard_deviation}")
    print(f"Average Recall -> {average_recall}")
    print(f"Recall Standard Deviation -> {recall_standard_deviation}")
    print(f"Average F1 -> {average_f1}")
    print(f"F1 Standard Deviation -> {f1_standard_deviation}")


def calculate_stats(scores: list[float]) -> tuple[float, float]:
    statistic = sum(scores) / len(scores)
    standard_deviation = \
        sum((x - statistic) ** 2 for x in scores) / len(scores)

    return statistic, standard_deviation
