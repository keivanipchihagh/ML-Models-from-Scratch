import numpy as np


def MSE(y: np.array, y_pred: np.array) -> float:
    '''
    Calculate the Mean Squared Error

    :param y: the actual values
    :param y_pred: the predicted values
    :return: the MSE
    '''

    return np.mean((y_pred - y) ** 2)