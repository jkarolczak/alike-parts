from typing import (Literal,
                    TypeAlias)

import pandas as pd

TDataset: TypeAlias = Literal[
    "apple_quality", "australia_rain", "breast_cancer", "diabetes", "passenger_satisfaction", "wine_quality"]


def apple_quality(
        train_size: float = 0.6,
        valid_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load apple quality dataset and split it into train, validation and test sets.

    https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality

    Args:
        train_size (float): Proportion of the dataset to include in the train subset.
        valid_size (float): Proportion of the dataset to include in the validation subset.
        test_size (float): Proportion of the dataset to include in the test subset.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train, validation and test sets.
    """

    assert train_size + valid_size + test_size == 1.0

    df = pd.read_csv("data/apple-quality.csv", index_col=0)
    df = df.sample(frac=1.0, random_state=random_state)
    x = df.drop(columns="Quality")
    y = (df["Quality"] == "good").astype(int)
    train_max_idx = int(len(x) * train_size)
    test_max_idx = int(len(x) * (train_size + test_size))
    return (
        x[:train_max_idx].reset_index(drop=True), y[:train_max_idx].reset_index(drop=True),
        x[test_max_idx:].reset_index(drop=True), y[test_max_idx:].reset_index(drop=True),
        x[train_max_idx:test_max_idx].reset_index(drop=True), y[train_max_idx:test_max_idx].reset_index(drop=True),
    )


def australia_rain(
        train_size: float = 0.6,
        valid_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load australia rain dataset and split it into train, validation and test sets.

    https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

    Args:
        train_size (float): Proportion of the dataset to include in the train subset.
        valid_size (float): Proportion of the dataset to include in the validation subset.
        test_size (float): Proportion of the dataset to include in the test subset.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train, validation and test sets.
    """

    assert train_size + valid_size + test_size == 1.0

    df = pd.read_csv("data/australia-rain.csv")
    df = df[df["RainTomorrow"].notna()]
    df = df.sample(frac=0.05, random_state=random_state)
    x = df.drop(columns=["Date", "RainTomorrow", "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "Date"])
    x["RainToday"] = (x["RainToday"] == "Yes").astype(int)
    y = (df["RainTomorrow"] == "Yes").astype(int)
    train_max_idx = int(len(x) * train_size)
    test_max_idx = int(len(x) * (train_size + test_size))
    return (
        x[:train_max_idx].reset_index(drop=True), y[:train_max_idx].reset_index(drop=True),
        x[test_max_idx:].reset_index(drop=True), y[test_max_idx:].reset_index(drop=True),
        x[train_max_idx:test_max_idx].reset_index(drop=True), y[train_max_idx:test_max_idx].reset_index(drop=True),
    )


def breast_cancer(
        train_size: float = 0.6,
        valid_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the breast cancer dataset and split it into train, validation and test sets.

    https://www.kaggle.com/datasets/rahmasleam/breast-cancer

    Args:
        train_size (float): Proportion of the dataset to include in the train subset.
        valid_size (float): Proportion of the dataset to include in the validation subset.
        test_size (float): Proportion of the dataset to include in the test subset.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train, validation and test sets.
    """

    assert train_size + valid_size + test_size == 1.0

    df = pd.read_csv("data/breast-cancer.csv", index_col=0).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=random_state)
    x = df.drop(columns="diagnosis")
    y = (df["diagnosis"] == "M").astype(int)
    train_max_idx = int(len(x) * train_size)
    test_max_idx = int(len(x) * (train_size + test_size))
    return (
        x[:train_max_idx].reset_index(drop=True), y[:train_max_idx].reset_index(drop=True),
        x[test_max_idx:].reset_index(drop=True), y[test_max_idx:].reset_index(drop=True),
        x[train_max_idx:test_max_idx].reset_index(drop=True), y[train_max_idx:test_max_idx].reset_index(drop=True),
    )


def diabetes(
        train_size: float = 0.6,
        valid_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load diabetes dataset and split it into train, validation and test sets.

    https://www.kaggle.com/datasets/mathchi/diabetes-data-set

    Args:
        train_size (float): Proportion of the dataset to include in the train subset.
        valid_size (float): Proportion of the dataset to include in the validation subset.
        test_size (float): Proportion of the dataset to include in the test subset.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train, validation and test sets.
    """

    assert train_size + valid_size + test_size == 1.0

    df = pd.read_csv("data/diabetes.csv")
    df = df.sample(frac=1.0, random_state=random_state)
    x = df.drop(columns="Outcome")
    y = df["Outcome"]
    train_max_idx = int(len(x) * train_size)
    test_max_idx = int(len(x) * (train_size + test_size))
    return (
        x[:train_max_idx].reset_index(drop=True), y[:train_max_idx].reset_index(drop=True),
        x[test_max_idx:].reset_index(drop=True), y[test_max_idx:].reset_index(drop=True),
        x[train_max_idx:test_max_idx].reset_index(drop=True), y[train_max_idx:test_max_idx].reset_index(drop=True),
    )


def passenger_satisfaction(
        train_size: float = 0.6,
        valid_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load passenger satisfaction dataset and split it into train, validation and test sets.

    https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

    Args:
        train_size (float): Proportion of the dataset to include in the train subset.
        valid_size (float): Proportion of the dataset to include in the validation subset.
        test_size (float): Proportion of the dataset to include in the test subset.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train, validation and test sets.
    """

    assert train_size + valid_size + test_size == 1.0

    df = pd.read_csv("data/passenger-satisfaction.csv", index_col=0)
    df = df.sample(frac=0.005, random_state=random_state)
    x = df.drop(columns=["id", "satisfaction"])
    x["Gender"] = (x["Gender"] == "Male").astype(int)
    x["Customer Type"] = (x["Customer Type"] == "Loyal Customer").astype(int)
    x["Type of Travel"] = (x["Type of Travel"] == "Business travel").astype(int)
    x["Class"] = (x["Class"] == "Business").astype(int)
    y = (df["satisfaction"] == "satisfied").astype(int)
    train_max_idx = int(len(x) * train_size)
    test_max_idx = int(len(x) * (train_size + test_size))
    return (
        x[:train_max_idx].reset_index(drop=True), y[:train_max_idx].reset_index(drop=True),
        x[test_max_idx:].reset_index(drop=True), y[test_max_idx:].reset_index(drop=True),
        x[train_max_idx:test_max_idx].reset_index(drop=True), y[train_max_idx:test_max_idx].reset_index(drop=True),
    )


def wine_quality(
        train_size: float = 0.6,
        valid_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load wine dataset and split it into train, validation and test sets.

    https://www.kaggle.com/datasets/taweilo/wine-quality-dataset-balanced-classification

    Args:
        train_size (float): Proportion of the dataset to include in the train subset.
        valid_size (float): Proportion of the dataset to include in the validation subset.
        test_size (float): Proportion of the dataset to include in the test subset.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train, validation and test sets.
    """

    assert train_size + valid_size + test_size == 1.0

    df = pd.read_csv("data/wine_quality.csv")
    df = df.sample(frac=0.1, random_state=random_state)
    x = df.drop(columns="quality")
    y = df["quality"]
    y = (y >= 7).astype(int)
    train_max_idx = int(len(x) * train_size)
    test_max_idx = int(len(x) * (train_size + test_size))
    return (
        x[:train_max_idx].reset_index(drop=True), y[:train_max_idx].reset_index(drop=True),
        x[test_max_idx:].reset_index(drop=True), y[test_max_idx:].reset_index(drop=True),
        x[train_max_idx:test_max_idx].reset_index(drop=True), y[train_max_idx:test_max_idx].reset_index(drop=True),
    )


def get_dataset(name: Literal["diabetes"], random_state: int = 42
                ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load dataset and split it into train, validation and test sets.

    Args:
        name (Literal["diabetes"]): Name of the dataset.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train, validation and test sets.
    """

    return globals()[name](random_state=random_state)
