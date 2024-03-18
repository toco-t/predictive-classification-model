"""
Logistic Regression model for customer churn prediction.

@author: Toco Tachibana (A01279235)
"""
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from imblearn.over_sampling import SMOTE


def predictor_variables():
    predictors = [
        "AccountAge",
        "MonthlyCharges_[0-9)",
        "MonthlyCharges_[9-20)",
        "ViewingHoursPerWeek",
        "AverageViewingDuration",
        "ContentDownloadsPerMonth",
        "SupportTicketsPerMonth",
        "SubscriptionType_Basic",
        "SubscriptionType_Premium",
        "PaymentMethod_Credit card",
        "GenrePreference_Sci-Fi",
        "GenrePreference_Comedy"
    ]
    return predictors


def binned_variables():
    binned_features = {"MonthlyCharges": [0, 9, 20]}
    label = ["[0-9)", "[9-20)"]

    return binned_features, label


def categorical_variables():
    categorical = [
        "SubscriptionType",
        "PaymentMethod",
        "PaperlessBilling",
        "ContentType",
        "MultiDeviceAccess",
        "DeviceRegistered",
        "GenrePreference",
        "Gender",
        "ParentalControl",
        "SubtitlesEnabled"
    ]
    return categorical


def numerical_variables():
    numerical = [
        "AccountAge",
        "MonthlyCharges",
        "TotalCharges",
        "ViewingHoursPerWeek",
        "AverageViewingDuration",
        "ContentDownloadsPerMonth",
        "UserRating",
        "SupportTicketsPerMonth",
        "WatchlistSize"
    ]
    return numerical


def target_variable():
    return "Churn"


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

        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train, y_train
        )

        logistic_regression = LogisticRegression(
            solver="liblinear",
            max_iter=1000
        )
        logistic_regression.fit(X_train_resampled, y_train_resampled)

        print(f"\nK-FOLD VALIDATION: {index + 1}")

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


def bin_features(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Bin features in the dataset.

    :param data_frame: pandas DataFrame containing the dataset
    :return: pandas DataFrame with binned features
    """
    criteria, labels = binned_variables()

    binned_data_frame = data_frame.copy()[criteria.keys()]
    for column, interval in criteria.items():
        binned_data_frame[column] = pd.cut(
            x=data_frame[column],
            bins=interval,
            labels=labels
        )

        binned_data_frame = replace_with_dummies(
            data_frame=binned_data_frame, categorical=[column]
        )

    binned_data_frame = pd.concat([data_frame, binned_data_frame], axis=1)
    return binned_data_frame


def replace_with_dummies(
        *, data_frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """
    Replace categorical variables with dummies.

    :param data_frame: pandas DataFrame containing the dataset
    :param categorical: list of categorical variables
    :return: pandas DataFrame with dummy variables
    """

    data_frame_with_dummies = pd.get_dummies(
        data_frame.copy(), columns=categorical, dtype=int
    )
    return data_frame_with_dummies


def impute_missing_values(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the dataset using KNNImputer.

    :param data_frame: pandas DataFrame containing the dataset
    :return: pandas DataFrame with imputed values
    """

    data_frame_with_dummies = replace_with_dummies(
        data_frame=data_frame, categorical=categorical_variables())

    imputer = KNNImputer(n_neighbors=10)
    imputed_data_frame = pd.DataFrame(
        imputer.fit_transform(data_frame_with_dummies),
        columns=data_frame_with_dummies.columns
    )

    return imputed_data_frame


def build_customer_churn_prediction_model(*, data_frame: pd.DataFrame):
    """
    Bild a logistic regression model for customer churn prediction.

    :param data_frame: pandas DataFrame containing the dataset
    """

    # Exclude CustomerID from the dataset as it is not a predictor
    data_frame = data_frame.drop(columns=["CustomerID"])

    imputed_data_frame = impute_missing_values(data_frame=data_frame)
    binned_data_frame = bin_features(data_frame=imputed_data_frame)

    train_and_test(
        data_frame=binned_data_frame,
        target=target_variable(),
        predictors=predictor_variables()
    )


def main():
    dataset_path = \
        "/Users/toco/Desktop/COMP3948/Assignment2_Data/CustomerChurn.csv"
    data_frame = pd.read_csv(dataset_path)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    build_customer_churn_prediction_model(data_frame=data_frame)


if __name__ == "__main__":
    main()
