import numpy as np
from Regression import Regression


class LinearRegression(Regression):
    '''
    Linear regression model
    '''

    def __init__(self) -> None:
        super().__init__()


    def fit(self, X: np.array, y: np.array) -> None:
        '''
        Fit the model weight and bias to the training data X using batch optimization by Least Squares

        :param X: training data
        :param y: target data
        :return: None
        '''

        # Calculating means
        X_mean = np.mean(X)
        Y_mean = np.mean(y)

        num = 0
        den = 0

        # Calculating numerator and denominator
        for i in range(len(X)):
            num += (X[i] - X_mean) * (y[i] - Y_mean)
            den += (X[i] - X_mean) ** 2

        # Calculating weight
        self.weight = num / den

        # Calculating bias
        self.bias = Y_mean - self.weight * X_mean