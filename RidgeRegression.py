import numpy as np
from Regression import Regression


class L2_Regularizer(object):
    ''' L2 Regularizer for Ridge Regression '''

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.sum(w ** 2)

    def grad(self, w):
        return 2 * self.alpha * w


class RidgeRegression(Regression):
    '''
    Ridge regression model
    '''

    def __init__(self, alpha: float) -> None:
        
        super().__init__()
        self.regularization = L2_Regularizer(alpha)
    

    def fit(self, X: np.array, y: np.array, epochs: int = 100, learning_rate: int = 0.0001) -> None:
        '''
        Fit the model to the training data X using Gradient Descent and L2 regularization

        :param X: training data
        :param y: target data
        :param epochs: number of epochs (Iterations) in case gradient decent is used (Default: 100)
        :param learning_rate: learning rate for gradient decent (Deafult: 0.0001)
        :return: None
        '''

        self.epochs = epochs
        self.learning_rate = learning_rate

        # Initialize weight
        self.initialize_weight(n_features = len(X))

        n_features = len(X)

        for _ in range(self.epochs):

            # Make prediction
            y_pred = np.dot(X, self.weight) + self.bias

            # Calculate error
            error = y - y_pred

            # Calculate gradient descent
            D_m = -(2 / n_features) * sum(X * (error))     # Partial Derivative for M (Weights)
            D_c = -(2 / n_features) * sum(error)           # Partial Derivative for C (Bias)

            # Apply L2 regularization (Ridge)
            D_m += self.regularization.grad(self.weight)

            # Update weights and bias
            self.weight -= (learning_rate * D_m)
            self.bias -= (learning_rate * D_c)