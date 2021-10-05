import numpy as np
import pandas as pd


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalizes a 1D matrix to have 0 mean and 1 std.

    :param X: Matrix to be normalized.
    :return: Normalized matrix.
    """

    return (X - X.mean()) / X.std()


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes all dataframe columns to have 0 mean and 1 std.

    :param df: Dataframe to be normalized.
    :return: Normalized dataframe.
    """

    for col in df.columns:
        df[col] = normalize(df[col])
    return df